#!/usr/bin/env python3
# KTP_eval_transreid_baseline.py
#
# TransReID-only KTP evaluation baseline:
#   - uses KTP 2D GT annotations only to initialize new identities
#   - applies the same prompt filtering rules as the ReID-SAMURAI evaluation
#   - uses YOLO person detections as frame-by-frame candidate boxes
#   - uses TransReID embeddings + online gallery for identity assignment
#   - exports MOT-style GT/pred txt files for TrackEval
#
# Expected KTP structure:
#   KTP/
#     images/
#       Arc/rgb/*.jpg
#       Rotation/rgb/*.jpg
#       Still/rgb/*.jpg
#       Translation/rgb/*.jpg
#     ground_truth/
#       Arc_gt2D.txt
#       Rotation_gt2D.txt
#       Still_gt2D.txt
#       Translation_gt2D.txt
#
# GT format:
#   <timestamp>: <id> <x> <y> <w> <h>, <id> <x> <y> <w> <h>, ...

import sys
import re
import csv
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "Ultralytics is required for this baseline. Install it with:\n"
        "  pip install ultralytics"
    ) from e


# ---------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "demo" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from sam2.reid_backends.transreid_backend import TransReIDBackend


# ---------------------------------------------------------------------
# TransReID feature extractor adapter
# ---------------------------------------------------------------------
class TransReIDExtractor:
    """
    Adapter around the same TransReID backend used by ReID-SAMURAI.

    Input:
        RGB crop as numpy array.

    Internally:
        Converts RGB to BGR because TransReIDBackend.embed_crop_bgr()
        expects BGR crops.

    Output:
        L2-normalized embedding as a NumPy array.
    """

    def __init__(self, device: str = "cuda"):
        self.backend = TransReIDBackend(device=device)

    @torch.inference_mode()
    def extract(self, rgb_crop: np.ndarray) -> Optional[np.ndarray]:
        if rgb_crop is None or rgb_crop.size == 0:
            return None

        crop_bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
        feat = self.backend.embed_crop_bgr(crop_bgr)

        if feat is None:
            return None

        if torch.is_tensor(feat):
            feat = feat.detach().cpu().numpy()

        feat = np.asarray(feat, dtype=np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        return feat


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
PALETTE_RGB = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
    (255, 0, 128),
    (0, 255, 128),
]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _id_to_rgb(obj_id: int):
    return PALETTE_RGB[int(obj_id) % len(PALETTE_RGB)]


def _rgb_to_bgr(color_rgb):
    r, g, b = color_rgb
    return int(b), int(g), int(r)


def rotate_frame(bgr: np.ndarray, rot_deg: int) -> np.ndarray:
    rot_deg = int(rot_deg) % 360

    if rot_deg == 0:
        return bgr
    if rot_deg == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if rot_deg == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if rot_deg == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError("--rotate must be one of {0,90,180,270}")


_TS_LEAD_NUM = re.compile(r"^(\d+(?:\.\d+)?)")


def ts_from_filename_robust(path: Path) -> Optional[str]:
    m = _TS_LEAD_NUM.match(path.stem)
    if not m:
        return None
    return m.group(1)


def bbox_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def xyxy_to_xywh(bb: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bb
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def clamp_bbox_xyxy(bb: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bb

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1

    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter

    return float(inter / denom) if denom > 0 else 0.0


def crop_rgb(rgb: np.ndarray, bb: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = clamp_bbox_xyxy(bb, w, h)

    if x2 <= x1 or y2 <= y1:
        return None

    return rgb[y1:y2, x1:x2].copy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def estimate_sequence_fps(
    frames: List[Path],
    ts_by_path: Dict[Path, str],
    fallback_fps: float = 15.0,
) -> float:
    vals = []

    for p in frames:
        ts = ts_by_path.get(p, None)
        if ts is None:
            continue
        try:
            vals.append(float(ts))
        except Exception:
            pass

    if len(vals) < 2:
        return float(fallback_fps)

    diffs = []
    for i in range(1, len(vals)):
        dt = vals[i] - vals[i - 1]
        if dt > 1e-9:
            diffs.append(dt)

    if not diffs:
        return float(fallback_fps)

    med_dt = float(np.median(diffs))

    if med_dt <= 1e-9:
        return float(fallback_fps)

    fps = 1.0 / med_dt
    return float(max(1.0, min(120.0, fps)))


# ---------------------------------------------------------------------
# GT parsing
# ---------------------------------------------------------------------
def parse_gt2d_file(gt_path: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    gt_map: Dict[str, List[Tuple[int, float, float, float, float]]] = {}

    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    with gt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            ts_part, rest = line.split(":", 1)
            ts = ts_part.strip()
            rest = rest.strip()

            dets = []

            for raw_det in [r.strip() for r in rest.split(",") if r.strip()]:
                parts = raw_det.split()

                if len(parts) < 5:
                    continue

                try:
                    gid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    dets.append((gid, x, y, w, h))
                except Exception:
                    continue

            gt_map[ts] = dets

    return gt_map


def load_ordered_frames(img_dir: Path, stride: int = 1, max_frames: int = -1):
    frames_all = list(img_dir.glob("*.jpg"))

    if not frames_all:
        raise RuntimeError(f"No .jpg frames found in {img_dir}")

    items = []

    for p in frames_all:
        ts_str = ts_from_filename_robust(p)

        if ts_str is None:
            continue

        try:
            ts_float = float(ts_str)
        except Exception:
            continue

        items.append((ts_float, ts_str, p))

    if not items:
        raise RuntimeError(f"No parseable timestamped frames found in {img_dir}")

    items.sort(key=lambda t: t[0])

    frames = []
    ts_by_path = {}
    seen_ts = set()

    for _, ts_str, path in items:
        if ts_str in seen_ts:
            continue

        seen_ts.add(ts_str)
        frames.append(path)
        ts_by_path[path] = ts_str

    if stride > 1:
        frames = frames[::stride]

    if max_frames > 0:
        frames = frames[:max_frames]

    return frames, ts_by_path


# ---------------------------------------------------------------------
# Online ReID gallery
# ---------------------------------------------------------------------
class OnlineGallery:
    def __init__(self, max_embeddings_per_id: int = 20):
        self.max_embeddings_per_id = int(max_embeddings_per_id)
        self.gallery: Dict[int, List[np.ndarray]] = {}

    def add(self, identity: int, embedding: np.ndarray) -> None:
        identity = int(identity)
        emb = l2_normalize(np.asarray(embedding, dtype=np.float32))

        if identity not in self.gallery:
            self.gallery[identity] = []

        self.gallery[identity].append(emb)

        if len(self.gallery[identity]) > self.max_embeddings_per_id:
            self.gallery[identity] = self.gallery[identity][-self.max_embeddings_per_id:]

    def identities(self) -> List[int]:
        return sorted(self.gallery.keys())

    def best_similarity(self, embedding: np.ndarray, identity: int) -> float:
        if identity not in self.gallery or len(self.gallery[identity]) == 0:
            return -1.0

        emb = l2_normalize(np.asarray(embedding, dtype=np.float32))
        return max(cosine_similarity(emb, g) for g in self.gallery[identity])

    def similarity_matrix(self, embeddings: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        ids = self.identities()

        if len(embeddings) == 0 or len(ids) == 0:
            return np.zeros((len(embeddings), len(ids)), dtype=np.float32), ids

        sim = np.zeros((len(embeddings), len(ids)), dtype=np.float32)

        for i, emb in enumerate(embeddings):
            for j, identity in enumerate(ids):
                sim[i, j] = self.best_similarity(emb, identity)

        return sim, ids


# ---------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------
def run_yolo_person_detector(
    yolo_model,
    rgb: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
    person_class_id: int = 0,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Returns:
        list of (x1, y1, x2, y2, confidence)
    """
    results = yolo_model.predict(
        source=rgb,
        conf=float(conf),
        imgsz=int(imgsz),
        verbose=False,
    )

    detections = []

    if not results:
        return detections

    r = results[0]

    if r.boxes is None:
        return detections

    boxes = r.boxes.xyxy.detach().cpu().numpy()
    confs = r.boxes.conf.detach().cpu().numpy()
    classes = r.boxes.cls.detach().cpu().numpy().astype(int)

    h, w = rgb.shape[:2]

    for bb, score, cls_id in zip(boxes, confs, classes):
        if int(cls_id) != int(person_class_id):
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bb.tolist()]
        x1, y1, x2, y2 = clamp_bbox_xyxy((x1, y1, x2, y2), w, h)

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append((x1, y1, x2, y2, float(score)))

    return detections


# ---------------------------------------------------------------------
# Matching detections to gallery identities
# ---------------------------------------------------------------------
def assign_detections_to_gallery(
    det_embeddings: List[np.ndarray],
    gallery: OnlineGallery,
    reid_thr: float,
) -> Dict[int, int]:
    """
    Returns:
        det_idx -> predicted_identity

    Hungarian matching is used on cosine distance, followed by thresholding.
    """
    sim_mat, gallery_ids = gallery.similarity_matrix(det_embeddings)

    if sim_mat.shape[0] == 0 or sim_mat.shape[1] == 0:
        return {}

    cost_mat = 1.0 - sim_mat
    row_ind, col_ind = linear_sum_assignment(cost_mat)

    assignments: Dict[int, int] = {}

    for r, c in zip(row_ind, col_ind):
        sim = float(sim_mat[r, c])

        if sim >= float(reid_thr):
            assignments[int(r)] = int(gallery_ids[c])

    return assignments


# ---------------------------------------------------------------------
# Evaluation matching, same style as ReID-SAMURAI script
# ---------------------------------------------------------------------
def match_frame_hungarian(
    gt_bb_by_id: Dict[int, Tuple[int, int, int, int]],
    pred_bbox_by_id: Dict[int, Tuple[int, int, int, int]],
    iou_match_thr: float,
):
    gt_ids = list(gt_bb_by_id.keys())
    pred_ids = list(pred_bbox_by_id.keys())

    gt_to_pred: Dict[int, Optional[int]] = {gid: None for gid in gt_ids}
    gt_to_iou: Dict[int, float] = {gid: 0.0 for gid in gt_ids}
    matched_gt_ids = set()
    matched_pred_ids = set()

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return gt_to_pred, gt_to_iou, matched_gt_ids, matched_pred_ids

    iou_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

    for i, gid in enumerate(gt_ids):
        for j, pid in enumerate(pred_ids):
            iou_mat[i, j] = iou_xyxy(gt_bb_by_id[gid], pred_bbox_by_id[pid])

    cost_mat = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost_mat)

    for r, c in zip(row_ind, col_ind):
        iou_val = float(iou_mat[r, c])

        if iou_val >= float(iou_match_thr):
            gid = gt_ids[r]
            pid = pred_ids[c]
            gt_to_pred[gid] = pid
            gt_to_iou[gid] = iou_val
            matched_gt_ids.add(gid)
            matched_pred_ids.add(pid)

    return gt_to_pred, gt_to_iou, matched_gt_ids, matched_pred_ids


@dataclass
class GTState:
    prev_pred: Optional[int] = None
    in_gap: bool = False
    gap_len: int = 0


@dataclass
class SeqMetrics:
    frames: int = 0
    total_gt_boxes: int = 0
    eligible_gt_boxes: int = 0
    matches: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    id_switches: int = 0
    reacq_events: int = 0
    reacq_gaps: List[int] = None
    iou_sum: float = 0.0
    iou_count: int = 0
    seed_skipped_overlap: int = 0
    seed_skipped_small: int = 0
    seed_failed: int = 0
    total_unique_gt_ids: int = 0
    seeded_ids_count: int = 0

    def __post_init__(self):
        if self.reacq_gaps is None:
            self.reacq_gaps = []


def summarize_reacq(gaps: List[int]):
    if not gaps:
        return 0, 0.0, 0.0, 0

    gaps_sorted = sorted(gaps)
    n = len(gaps_sorted)
    mean = float(sum(gaps_sorted) / n)

    if n % 2 == 1:
        med = float(gaps_sorted[n // 2])
    else:
        med = float(0.5 * (gaps_sorted[n // 2 - 1] + gaps_sorted[n // 2]))

    mx = int(max(gaps_sorted))
    return n, mean, med, mx


def metrics_to_row(
    run_prefix: str,
    label: str,
    seq: str,
    reid_thr: float,
    yolo_model_name: str,
    yolo_conf: float,
    met: SeqMetrics,
    out_csv: str,
    gt_mot_path: str,
    pred_mot_path: str,
):
    reacq_n, reacq_mean, reacq_med, reacq_max = summarize_reacq(met.reacq_gaps)

    denom_gt = met.eligible_gt_boxes
    misses = met.false_negatives

    match_rate = met.matches / denom_gt if denom_gt > 0 else 0.0
    miss_rate = misses / denom_gt if denom_gt > 0 else 0.0
    mean_iou = met.iou_sum / met.iou_count if met.iou_count > 0 else 0.0
    id_switches_per_match = met.id_switches / met.matches if met.matches > 0 else 0.0
    id_switches_per_gt = met.id_switches / denom_gt if denom_gt > 0 else 0.0
    seed_coverage = met.seeded_ids_count / met.total_unique_gt_ids if met.total_unique_gt_ids > 0 else 0.0

    precision = met.matches / (met.matches + met.false_positives) if (met.matches + met.false_positives) > 0 else 0.0
    recall = met.matches / (met.matches + met.false_negatives) if (met.matches + met.false_negatives) > 0 else 0.0
    mota = 1.0 - ((met.false_negatives + met.false_positives + met.id_switches) / denom_gt) if denom_gt > 0 else 0.0

    return {
        "run": run_prefix,
        "label": label,
        "seq": seq,
        "baseline": "transreid_yolo_gallery",
        "reid_thr": float(reid_thr),
        "yolo_model": yolo_model_name,
        "yolo_conf": float(yolo_conf),

        "frames": met.frames,
        "total_gt_boxes": met.total_gt_boxes,
        "eligible_gt_boxes": met.eligible_gt_boxes,
        "matches": met.matches,
        "misses": misses,

        "false_positives": met.false_positives,
        "false_negatives": met.false_negatives,
        "precision": precision,
        "recall": recall,
        "mota": mota,

        "match_rate": match_rate,
        "miss_rate": miss_rate,

        "id_switches": met.id_switches,
        "id_switches_per_match": id_switches_per_match,
        "id_switches_per_gt": id_switches_per_gt,

        "reacq_events": reacq_n,
        "reacq_mean_frames": reacq_mean,
        "reacq_median_frames": reacq_med,
        "reacq_max_frames": reacq_max,

        "mean_iou_when_matched": mean_iou,

        "total_unique_gt_ids": met.total_unique_gt_ids,
        "seeded_ids_count": met.seeded_ids_count,
        "seed_coverage": seed_coverage,
        "seed_skipped_small": met.seed_skipped_small,
        "seed_skipped_overlap": met.seed_skipped_overlap,
        "seed_failed": met.seed_failed,

        "out_csv": out_csv,
        "gt_mot_path": gt_mot_path,
        "pred_mot_path": pred_mot_path,
    }


# ---------------------------------------------------------------------
# Main sequence runner
# ---------------------------------------------------------------------
@torch.inference_mode()
def run_sequence(
    seq_name: str,
    ktp_root: Path,
    yolo_model,
    reid_extractor: TransReIDExtractor,
    out_csv_path: Path,
    mot_gt_path: Path,
    mot_pred_path: Path,
    reid_thr: float,
    yolo_conf: float,
    yolo_imgsz: int,
    max_gallery_embeddings: int,
    update_gallery: bool,
    rotate_deg: int = 0,
    stride: int = 1,
    max_frames: int = -1,
    visible_area_frac: float = 0.02,
    visible_min_h: int = 120,
    visible_min_w: int = 0,
    seed_overlap_iou_max: float = 0.10,
    iou_match_thr: float = 0.30,
    eval_seed_frame: bool = False,
    save_video: bool = True,
) -> SeqMetrics:
    img_dir = ktp_root / "images" / seq_name / "rgb"
    gt_path = ktp_root / "ground_truth" / f"{seq_name}_gt2D.txt"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_map = parse_gt2d_file(gt_path)
    frames, ts_by_path = load_ordered_frames(img_dir, stride=stride, max_frames=max_frames)

    bgr0 = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)

    if bgr0 is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")

    bgr0 = rotate_frame(bgr0, rotate_deg)
    height, width = bgr0.shape[:2]
    video_fps = estimate_sequence_fps(frames, ts_by_path, fallback_fps=15.0)

    all_gt_ids = set()

    for dets in gt_map.values():
        for gid, *_ in dets:
            all_gt_ids.add(int(gid))

    gallery = OnlineGallery(max_embeddings_per_id=max_gallery_embeddings)
    seeded = set()
    gt_states: Dict[int, GTState] = {}

    metrics = SeqMetrics()
    metrics.total_unique_gt_ids = len(all_gt_ids)

    safe_mkdir(out_csv_path.parent)
    safe_mkdir(mot_gt_path.parent)
    safe_mkdir(mot_pred_path.parent)

    fcsv = out_csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)

    f_gt_mot = mot_gt_path.open("w", encoding="utf-8")
    f_pred_mot = mot_pred_path.open("w", encoding="utf-8")

    writer.writerow(["# baseline: transreid_yolo_gallery"])
    writer.writerow([f"# reid_thr: {reid_thr}"])
    writer.writerow([f"# yolo_conf: {yolo_conf}"])
    writer.writerow([
        f"# seed_rules: visible_area_frac={visible_area_frac}, visible_min_h={visible_min_h}, "
        f"visible_min_w={visible_min_w}, seed_overlap_iou_max={seed_overlap_iou_max}, "
        f"iou_match_thr={iou_match_thr}, stride={stride}, max_frames={max_frames}, "
        f"eval_seed_frame={eval_seed_frame}, update_gallery={update_gallery}"
    ])
    writer.writerow([
        "seq", "frame_idx", "ts",
        "gt_id", "gt_x", "gt_y", "gt_w", "gt_h", "gt_area_frac",
        "eligible", "seeded_now", "seeded_already", "seed_skip_reason",
        "pred_id", "match_iou", "id_switch_event", "reacq_event", "gap_len"
    ])

    video_writer = None

    if save_video:
        video_out_path = out_csv_path.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, video_fps, (width, height))

    for fidx, fp in enumerate(frames):
        ts = ts_by_path.get(fp, None)

        if ts is None:
            continue

        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)

        if bgr is None:
            continue

        bgr = rotate_frame(bgr, rotate_deg)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gt_dets = gt_map.get(ts, [])

        metrics.frames += 1
        metrics.total_gt_boxes += len(gt_dets)

        gt_bb_by_id_all: Dict[int, Tuple[int, int, int, int]] = {}

        for gid, x, y, w, h in gt_dets:
            gt_bb_by_id_all[int(gid)] = clamp_bbox_xyxy(
                bbox_xywh_to_xyxy(x, y, w, h),
                width,
                height,
            )

        seeded_now_ids = set()
        seed_skip_reason_by_gid: Dict[int, str] = {}

        # -------------------------------------------------------------
        # 1) Initialize new GT identities using the same filtering rules.
        # -------------------------------------------------------------
        for gid, x, y, w, h in gt_dets:
            gid = int(gid)

            if gid not in gt_states:
                gt_states[gid] = GTState()

            if gid in seeded:
                seed_skip_reason_by_gid[gid] = ""
                continue

            bb = gt_bb_by_id_all[gid]
            bw = max(0, bb[2] - bb[0])
            bh = max(0, bb[3] - bb[1])
            area = bw * bh
            area_frac = area / float(width * height + 1e-9)

            visible_ok = (area_frac >= float(visible_area_frac)) and (bh >= int(visible_min_h))

            if visible_min_w and int(visible_min_w) > 0:
                visible_ok = visible_ok and (bw >= int(visible_min_w))

            if not visible_ok:
                seed_skip_reason_by_gid[gid] = "small"
                metrics.seed_skipped_small += 1
                continue

            max_iou_other = 0.0

            for other_gid, other_bb in gt_bb_by_id_all.items():
                if other_gid == gid:
                    continue
                max_iou_other = max(max_iou_other, iou_xyxy(bb, other_bb))

            if max_iou_other > float(seed_overlap_iou_max):
                seed_skip_reason_by_gid[gid] = f"overlap(max_iou={max_iou_other:.3f})"
                metrics.seed_skipped_overlap += 1
                continue

            # IMPORTANT:
            # Crop from RGB, because TransReIDExtractor.extract() expects RGB.
            gt_crop = crop_rgb(rgb, bb)

            if gt_crop is None:
                seed_skip_reason_by_gid[gid] = "seed_failed"
                metrics.seed_failed += 1
                continue

            emb = reid_extractor.extract(gt_crop)

            if emb is None:
                seed_skip_reason_by_gid[gid] = "seed_failed"
                metrics.seed_failed += 1
                continue

            gallery.add(gid, emb)

            # Debug seed crop.
            debug_crop_dir = out_csv_path.parent / "debug_crops" / seq_name / f"id_{gid}"
            safe_mkdir(debug_crop_dir)
            gt_crop_bgr = cv2.cvtColor(gt_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_crop_dir / f"seed_frame_{fidx:05d}.jpg"), gt_crop_bgr)

            seeded.add(gid)
            seeded_now_ids.add(gid)
            metrics.seeded_ids_count = len(seeded)
            seed_skip_reason_by_gid[gid] = ""

        # -------------------------------------------------------------
        # 2) Define eligible GT boxes for evaluation.
        # -------------------------------------------------------------
        eligible_gt_ids = set()

        for gid, *_ in gt_dets:
            gid = int(gid)

            if gid in seeded:
                if (gid in seeded_now_ids) and (not eval_seed_frame):
                    continue

                eligible_gt_ids.add(gid)

        gt_bb_by_id_eval = {gid: gt_bb_by_id_all[gid] for gid in eligible_gt_ids}
        metrics.eligible_gt_boxes += len(gt_bb_by_id_eval)

        mot_frame_idx = fidx + 1

        for gid in sorted(gt_bb_by_id_eval.keys()):
            x1, y1, w, h = xyxy_to_xywh(gt_bb_by_id_eval[gid])
            f_gt_mot.write(f"{mot_frame_idx},{gid},{x1},{y1},{w},{h},1,1,1,1\n")

        # -------------------------------------------------------------
        # 3) YOLO detections + TransReID embeddings.
        # -------------------------------------------------------------
        yolo_dets = run_yolo_person_detector(
            yolo_model=yolo_model,
            rgb=rgb,
            conf=yolo_conf,
            imgsz=yolo_imgsz,
        )

        det_bboxes: List[Tuple[int, int, int, int]] = []
        det_embeddings: List[np.ndarray] = []

        for x1, y1, x2, y2, conf in yolo_dets:
            det_bb = (x1, y1, x2, y2)

            # IMPORTANT:
            # Crop from RGB, because TransReIDExtractor.extract() expects RGB.
            det_crop = crop_rgb(rgb, det_bb)

            if det_crop is None:
                continue

            emb = reid_extractor.extract(det_crop)

            if emb is None:
                continue

            det_bboxes.append(det_bb)
            det_embeddings.append(emb)

            # Debug: save YOLO crops that overlap GT 4.
            if 4 in gt_bb_by_id_all:
                iou_with_4 = iou_xyxy(det_bb, gt_bb_by_id_all[4])

                if iou_with_4 > 0.3:
                    debug_crop_dir = out_csv_path.parent / "debug_crops" / seq_name / "id_4"
                    safe_mkdir(debug_crop_dir)

                    det_crop_bgr = cv2.cvtColor(det_crop, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(
                        str(debug_crop_dir / f"det_frame_{fidx:05d}_iou{iou_with_4:.2f}.jpg"),
                        det_crop_bgr,
                    )

        # -------------------------------------------------------------
        # 4) Assign YOLO detections to initialized identities.
        # -------------------------------------------------------------
        det_to_identity = assign_detections_to_gallery(
            det_embeddings=det_embeddings,
            gallery=gallery,
            reid_thr=reid_thr,
        )

        sim_mat, gallery_ids = gallery.similarity_matrix(det_embeddings)

        det_best_match = {}

        for det_idx in range(len(det_embeddings)):
            if len(gallery_ids) == 0 or sim_mat.shape[1] == 0:
                det_best_match[det_idx] = {
                    "best_id": "none",
                    "best_sim": -1.0,
                    "gallery_ids": [],
                    "sim_by_id": {},
                }
            else:
                best_col = int(np.argmax(sim_mat[det_idx]))
                best_id = int(gallery_ids[best_col])
                best_sim = float(sim_mat[det_idx, best_col])

                sim_by_id = {
                    int(gallery_ids[j]): float(sim_mat[det_idx, j])
                    for j in range(len(gallery_ids))
                }

                det_best_match[det_idx] = {
                    "best_id": best_id,
                    "best_sim": best_sim,
                    "gallery_ids": [int(x) for x in gallery_ids],
                    "sim_by_id": sim_by_id,
                }

        pred_bbox_by_id: Dict[int, Tuple[int, int, int, int]] = {}

        for det_idx, pred_id in det_to_identity.items():
            pred_bbox_by_id[int(pred_id)] = det_bboxes[int(det_idx)]

            if update_gallery:
                gallery.add(int(pred_id), det_embeddings[int(det_idx)])

        for pred_id in sorted(pred_bbox_by_id.keys()):
            x1, y1, w, h = xyxy_to_xywh(pred_bbox_by_id[pred_id])
            f_pred_mot.write(f"{mot_frame_idx},{pred_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")

        # -------------------------------------------------------------
        # 5) Frame-level matching for diagnostic CSV.
        # TrackEval will compute the final HOTA/CLEAR/Identity metrics.
        # -------------------------------------------------------------
        gt_to_pred, gt_to_iou, matched_gt_ids, matched_pred_ids = match_frame_hungarian(
            gt_bb_by_id=gt_bb_by_id_eval,
            pred_bbox_by_id=pred_bbox_by_id,
            iou_match_thr=iou_match_thr,
        )

        num_matches = len(matched_gt_ids)
        num_fn = len(gt_bb_by_id_eval) - num_matches
        num_fp = len(pred_bbox_by_id) - len(matched_pred_ids)

        metrics.matches += num_matches
        metrics.false_negatives += num_fn
        metrics.false_positives += num_fp

        for gid in matched_gt_ids:
            metrics.iou_sum += gt_to_iou[gid]
            metrics.iou_count += 1

        for gid, x, y, w, h in gt_dets:
            gid = int(gid)
            st = gt_states.get(gid, GTState())
            eligible = gid in eligible_gt_ids
            cur = gt_to_pred.get(gid, None) if eligible else None

            idsw = 0
            reacq = 0

            if eligible:
                if cur is None:
                    if st.prev_pred is not None and not st.in_gap:
                        st.in_gap = True
                        st.gap_len = 1
                    elif st.in_gap:
                        st.gap_len += 1
                else:
                    if st.in_gap:
                        reacq = 1
                        metrics.reacq_events += 1
                        metrics.reacq_gaps.append(st.gap_len)
                        st.in_gap = False
                        st.gap_len = 0

                    if st.prev_pred is not None and cur != st.prev_pred:
                        idsw = 1
                        metrics.id_switches += 1

                    st.prev_pred = cur

            gt_states[gid] = st

            gt_bb = gt_bb_by_id_all[gid]
            gt_area = max(0, gt_bb[2] - gt_bb[0]) * max(0, gt_bb[3] - gt_bb[1])
            gt_area_frac = gt_area / float(width * height + 1e-9)

            writer.writerow([
                seq_name,
                fidx,
                ts,
                gid,
                f"{x:.3f}",
                f"{y:.3f}",
                f"{w:.3f}",
                f"{h:.3f}",
                f"{gt_area_frac:.6f}",
                int(eligible),
                int(gid in seeded_now_ids),
                int(gid in seeded),
                seed_skip_reason_by_gid.get(gid, ""),
                cur if cur is not None else "",
                f"{gt_to_iou.get(gid, 0.0):.6f}",
                idsw,
                reacq,
                gt_states[gid].gap_len if gt_states[gid].in_gap else 0,
            ])

        # -------------------------------------------------------------
        # 6) Optional visualization video.
        # -------------------------------------------------------------
        if video_writer is not None:
            vis = bgr.copy()

            # Draw GT boxes.
            for gid, bb in gt_bb_by_id_all.items():
                color = (255, 255, 255) if gid in eligible_gt_ids else (120, 120, 120)
                cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.putText(
                    vis,
                    f"GT {gid}",
                    (bb[0], max(0, bb[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Draw raw YOLO detections before TransReID assignment.
            for det_idx, (x1, y1, x2, y2) in enumerate(det_bboxes):
                info = det_best_match.get(det_idx, {})
                best_id = info.get("best_id", "none")
                best_sim = info.get("best_sim", -1.0)
                sim_by_id = info.get("sim_by_id", {})

                sim4 = sim_by_id.get(4, None)

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

                text1 = f"YOLO best={best_id} sim={best_sim:.2f}"
                cv2.putText(
                    vis,
                    text1,
                    (x1, max(0, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if sim4 is not None:
                    text2 = f"sim(ID4)={sim4:.2f}"
                else:
                    text2 = "ID4 not in gallery"

                cv2.putText(
                    vis,
                    text2,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw final predicted boxes after TransReID assignment.
            for pred_id, bb in pred_bbox_by_id.items():
                color = _rgb_to_bgr(_id_to_rgb(pred_id))
                cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.putText(
                    vis,
                    f"PR {pred_id}",
                    (bb[0], bb[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            video_writer.write(vis)

    fcsv.close()
    f_gt_mot.close()
    f_pred_mot.close()

    if video_writer is not None:
        video_writer.release()

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ktp_root", type=str, required=True)
    ap.add_argument("--sequences", type=str, default="Arc,Rotation,Still,Translation")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="transreid_yolo_baseline")

    ap.add_argument("--yolo_model", type=str, default="yolov8s.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_imgsz", type=int, default=640)

    ap.add_argument("--reid_thr", type=float, default=0.80)
    ap.add_argument("--max_gallery_embeddings", type=int, default=20)
    ap.add_argument("--no_update_gallery", action="store_true")

    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=-1)

    ap.add_argument("--visible_area_frac", type=float, default=0.02)
    ap.add_argument("--visible_min_h", type=int, default=120)
    ap.add_argument("--visible_min_w", type=int, default=0)
    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.10)

    ap.add_argument("--iou_match_thr", type=float, default=0.30)
    ap.add_argument("--eval_seed_frame", action="store_true")

    ap.add_argument("--save_video", action="store_true")

    args = ap.parse_args()

    ktp_root = Path(args.ktp_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(out_dir)

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    print("[setup]")
    print("  repo root :", REPO_ROOT)
    print("  ktp root  :", ktp_root)
    print("  out dir   :", out_dir)
    print("  sequences :", seqs)
    print("  yolo model:", args.yolo_model)
    print("  yolo conf :", args.yolo_conf)
    print("  reid thr  :", args.reid_thr)
    print("  cuda      :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("  gpu       :", torch.cuda.get_device_name(0))

    yolo_model = YOLO(args.yolo_model)
    reid_extractor = TransReIDExtractor(device="cuda")

    run_id = time.strftime("%Y%m%d_%H%M%S")

    label = (
        f"transreid_yolo"
        f"__rthr{args.reid_thr:g}"
        f"_yolo{Path(args.yolo_model).stem}"
        f"_conf{args.yolo_conf:g}"
    )

    run_prefix = f"{args.run_name}_{label}_{run_id}"

    mot_dir = out_dir / "mot_exports" / run_prefix
    safe_mkdir(mot_dir)

    summary_rows = []
    per_sequence_rows = []
    all_metrics = SeqMetrics()

    for seq in seqs:
        out_csv = out_dir / f"{run_prefix}__{seq}.csv"
        gt_mot_path = mot_dir / f"{seq}_gt.txt"
        pred_mot_path = mot_dir / f"{seq}_pred.txt"

        met = run_sequence(
            seq_name=seq,
            ktp_root=ktp_root,
            yolo_model=yolo_model,
            reid_extractor=reid_extractor,
            out_csv_path=out_csv,
            mot_gt_path=gt_mot_path,
            mot_pred_path=pred_mot_path,
            reid_thr=args.reid_thr,
            yolo_conf=args.yolo_conf,
            yolo_imgsz=args.yolo_imgsz,
            max_gallery_embeddings=args.max_gallery_embeddings,
            update_gallery=not args.no_update_gallery,
            rotate_deg=args.rotate,
            stride=args.stride,
            max_frames=args.max_frames,
            visible_area_frac=args.visible_area_frac,
            visible_min_h=args.visible_min_h,
            visible_min_w=args.visible_min_w,
            seed_overlap_iou_max=args.seed_overlap_iou_max,
            iou_match_thr=args.iou_match_thr,
            eval_seed_frame=args.eval_seed_frame,
            save_video=args.save_video,
        )

        row = metrics_to_row(
            run_prefix=run_prefix,
            label=label,
            seq=seq,
            reid_thr=args.reid_thr,
            yolo_model_name=args.yolo_model,
            yolo_conf=args.yolo_conf,
            met=met,
            out_csv=str(out_csv),
            gt_mot_path=str(gt_mot_path),
            pred_mot_path=str(pred_mot_path),
        )

        summary_rows.append(row)
        per_sequence_rows.append(row)

        all_metrics.frames += met.frames
        all_metrics.total_gt_boxes += met.total_gt_boxes
        all_metrics.eligible_gt_boxes += met.eligible_gt_boxes
        all_metrics.matches += met.matches
        all_metrics.false_positives += met.false_positives
        all_metrics.false_negatives += met.false_negatives
        all_metrics.id_switches += met.id_switches
        all_metrics.reacq_events += met.reacq_events
        all_metrics.reacq_gaps.extend(met.reacq_gaps)
        all_metrics.iou_sum += met.iou_sum
        all_metrics.iou_count += met.iou_count
        all_metrics.seed_skipped_small += met.seed_skipped_small
        all_metrics.seed_skipped_overlap += met.seed_skipped_overlap
        all_metrics.seed_failed += met.seed_failed
        all_metrics.total_unique_gt_ids += met.total_unique_gt_ids
        all_metrics.seeded_ids_count += met.seeded_ids_count

        print(
            f"[{seq}] "
            f"mota={row['mota']:.3f} "
            f"match_rate={row['match_rate']:.3f} "
            f"idsw={row['id_switches']} "
            f"fp={row['false_positives']} "
            f"fn={row['false_negatives']} "
            f"mean_iou={row['mean_iou_when_matched']:.3f}"
        )

    row_all = metrics_to_row(
        run_prefix=run_prefix,
        label=label,
        seq="ALL",
        reid_thr=args.reid_thr,
        yolo_model_name=args.yolo_model,
        yolo_conf=args.yolo_conf,
        met=all_metrics,
        out_csv="",
        gt_mot_path="",
        pred_mot_path="",
    )

    summary_rows.append(row_all)

    summary_json_path = out_dir / f"{run_prefix}__summary.json"

    payload = {
        "run": run_prefix,
        "created_at": run_id,
        "repo_root": str(REPO_ROOT),
        "ktp_root": str(ktp_root),
        "baseline": "transreid_yolo_gallery",
        "label": label,
        "sequences": seqs,
        "settings": {
            "yolo_model": args.yolo_model,
            "yolo_conf": args.yolo_conf,
            "yolo_imgsz": args.yolo_imgsz,
            "reid_thr": args.reid_thr,
            "max_gallery_embeddings": args.max_gallery_embeddings,
            "update_gallery": not args.no_update_gallery,
            "rotate": args.rotate,
            "stride": args.stride,
            "max_frames": args.max_frames,
            "visible_area_frac": args.visible_area_frac,
            "visible_min_h": args.visible_min_h,
            "visible_min_w": args.visible_min_w,
            "seed_overlap_iou_max": args.seed_overlap_iou_max,
            "iou_match_thr": args.iou_match_thr,
            "eval_seed_frame": args.eval_seed_frame,
        },
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "mot_exports_dir": str(mot_dir),
        "per_sequence": per_sequence_rows,
        "overall": row_all,
        "rows_flat": summary_rows,
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n[ALL]")
    print(
        f"mota={row_all['mota']:.3f} "
        f"match_rate={row_all['match_rate']:.3f} "
        f"idsw={row_all['id_switches']} "
        f"fp={row_all['false_positives']} "
        f"fn={row_all['false_negatives']} "
        f"mean_iou={row_all['mean_iou_when_matched']:.3f}"
    )

    print("\n[done]")
    print("  summary json:", summary_json_path)
    print("  MOT exports :", mot_dir)
    print(
        "\nNext step: copy the MOT exports to TrackEval format using the same "
        "prepare_ktp_for_trackeval.py script you used for ReID-SAMURAI."
    )


if __name__ == "__main__":
    main()