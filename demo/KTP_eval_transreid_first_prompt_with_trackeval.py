#!/usr/bin/env python3
# KTP_eval_transreid_baseline.py
#
# TransReID + YOLO baseline for the KTP dataset.
#
# This version matches the evaluation logic used in the updated KTP_eval_run.py:
#   - GT identities are used only for initialization/seeding.
#   - A GT identity is eligible only after it has been seeded.
#   - Seed filtering checks visibility and overlap with all GT boxes in the frame.
#   - YOLO detects all people in each frame.
#   - TransReID embeds each YOLO person crop.
#   - Hungarian assignment matches detections to the online identity gallery.
#   - Standard MOT-style matching is still computed for continuity with older runs.
#   - Strict same-ID diagnostic matching is also computed: GT i can only match PR i.
#   - Optional annotation-limited evaluation can ignore predictions whose same-ID GT
#     annotation is missing or has very low overlap.
#   - MOTChallenge GT/pred files are exported for TrackEval.

import argparse
import csv
import json
import re
import sys
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("Install ultralytics first: pip install ultralytics") from e


# ---------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "demo" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from sam2.reid_backends.transreid_backend import TransReIDBackend


# ---------------------------------------------------------------------
# Constants / drawing helpers
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

_TS_LEAD_NUM = re.compile(r"^(\d+(?:\.\d+)?)")


def parse_paper_frame_ranges(spec: str):
    """
    Parse a string like:
        "Arc:90-120;Rotation:345-365"

    Returns:
        {
            "Arc": [(90, 120)],
            "Rotation": [(345, 365)]
        }
    """
    ranges = {}

    if spec is None or str(spec).strip() == "":
        return ranges

    parts = [p.strip() for p in spec.split(";") if p.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid paper frame range: {part}")

        seq_name, range_part = part.split(":", 1)
        seq_name = seq_name.strip()

        seq_ranges = []
        for r in range_part.split(","):
            r = r.strip()
            if "-" not in r:
                raise ValueError(f"Invalid frame range: {r}")
            a, b = r.split("-", 1)
            seq_ranges.append((int(a), int(b)))

        ranges[seq_name] = seq_ranges

    return ranges


def should_save_paper_frame(seq_name: str, frame_idx: int, paper_frame_ranges: dict) -> bool:
    if not paper_frame_ranges:
        return False

    if seq_name not in paper_frame_ranges:
        return False

    for start, end in paper_frame_ranges[seq_name]:
        if start <= int(frame_idx) <= end:
            return True

    return False


def save_paper_frame(paper_frames_dir, model_name: str, seq_name: str, frame_idx: int, image_bgr):
    if paper_frames_dir is None:
        return

    out_dir = Path(paper_frames_dir) / model_name / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{seq_name}_f{int(frame_idx):06d}.jpg"
    cv2.imwrite(str(out_path), image_bgr)
                

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _id_to_rgb(obj_id: int):
    return PALETTE_RGB[int(obj_id) % len(PALETTE_RGB)]


def _rgb_to_bgr(color_rgb):
    r, g, b = color_rgb
    return int(b), int(g), int(r)


def draw_text_with_bg(img, text, org, fg=(255, 255, 255), bg=(0, 0, 0), scale=0.55, thickness=1):
    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(str(text), font, float(scale), int(thickness))
    x1 = max(0, x - 2)
    y1 = max(0, y - th - baseline - 3)
    x2 = min(img.shape[1] - 1, x + tw + 2)
    y2 = min(img.shape[0] - 1, y + baseline + 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(img, str(text), (x, y), font, float(scale), fg, int(thickness), cv2.LINE_AA)


def draw_text_plain(img, text, org, color=(255, 255, 255), scale=0.65, thickness=2):
    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        str(text),
        (x, y),
        font,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


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


def ts_from_filename_robust(path: Path) -> Optional[str]:
    m = _TS_LEAD_NUM.match(path.stem)
    return m.group(1) if m else None


def bbox_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


def xyxy_to_xywh(bb: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bb]
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def clamp_bbox_xyxy(bb: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bb]
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
    return np.asarray(x, dtype=np.float32) / (np.linalg.norm(x) + 1e-12)


# ---------------------------------------------------------------------
# KTP loading
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
            dets = []
            for raw_det in [r.strip() for r in rest.strip().split(",") if r.strip()]:
                parts = raw_det.split()
                if len(parts) < 5:
                    continue
                try:
                    dets.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
                except Exception:
                    continue
            gt_map[ts] = dets
    return gt_map


def find_sequence_paths(ktp_root: Path, seq_name: str):
    img_dir_candidates = [
        ktp_root / "images" / seq_name / "rgb",
        ktp_root / seq_name / "rgb",
        ktp_root / seq_name / "images",
        ktp_root / seq_name,
    ]
    gt_candidates = [
        ktp_root / "ground_truth" / f"{seq_name}_gt2D.txt",
        ktp_root / seq_name / f"{seq_name}_gt2D.txt",
        ktp_root / seq_name / "gt2D.txt",
    ]

    img_dir = next((p for p in img_dir_candidates if p.exists()), None)
    gt_path = next((p for p in gt_candidates if p.exists()), None)

    if img_dir is None:
        raise FileNotFoundError(f"Could not find image directory for {seq_name} under {ktp_root}")
    if gt_path is None:
        raise FileNotFoundError(f"Could not find GT file for {seq_name} under {ktp_root}")
    return img_dir, gt_path


def load_ordered_frames(img_dir: Path, stride: int = 1, max_frames: int = -1):
    frames_all = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not frames_all:
        raise RuntimeError(f"No image frames found in {img_dir}")

    items = []
    for p in frames_all:
        ts_str = ts_from_filename_robust(p)
        if ts_str is None:
            continue
        try:
            items.append((float(ts_str), ts_str, p))
        except Exception:
            continue

    if not items:
        raise RuntimeError(f"No parseable timestamped frames found in {img_dir}")

    items.sort(key=lambda t: t[0])
    frames = []
    ts_by_path = {}
    seen = set()
    for _, ts_str, path in items:
        if ts_str in seen:
            continue
        seen.add(ts_str)
        frames.append(path)
        ts_by_path[path] = ts_str

    stride = max(1, int(stride))
    if stride > 1:
        frames = frames[::stride]
    if max_frames and int(max_frames) > 0:
        frames = frames[: int(max_frames)]
    return frames, ts_by_path


def estimate_sequence_fps(frames: List[Path], ts_by_path: Dict[Path, str], fallback_fps: float = 15.0) -> float:
    vals = []
    for p in frames:
        try:
            vals.append(float(ts_by_path[p]))
        except Exception:
            pass
    if len(vals) < 2:
        return float(fallback_fps)
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals)) if vals[i] - vals[i - 1] > 1e-9]
    if not diffs:
        return float(fallback_fps)
    return float(max(1.0, min(120.0, 1.0 / float(np.median(diffs)))))


# ---------------------------------------------------------------------
# TransReID + online gallery
# ---------------------------------------------------------------------
class TransReIDExtractor:
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
        return l2_normalize(np.asarray(feat, dtype=np.float32))


class FirstPromptGallery:
    """
    Gallery used for the fair TransReID baseline.

    Each identity is represented only by its first N accepted seed crops.
    After an identity reaches N embeddings, its gallery is frozen. Detections
    are compared against those fixed embeddings; no later predictions are added.
    This avoids turning the appearance baseline into an online tracker.
    """

    def __init__(self, required_embeddings_per_id: int = 10):
        self.required_embeddings_per_id = max(1, int(required_embeddings_per_id))
        self.gallery: Dict[int, List[np.ndarray]] = {}

    def add_seed(self, identity: int, embedding: np.ndarray) -> bool:
        identity = int(identity)
        if self.is_complete(identity):
            return False
        emb = l2_normalize(embedding)
        self.gallery.setdefault(identity, []).append(emb)
        return True

    def count(self, identity: int) -> int:
        return len(self.gallery.get(int(identity), []))

    def is_complete(self, identity: int) -> bool:
        return self.count(identity) >= self.required_embeddings_per_id

    def identities(self) -> List[int]:
        # Only identities with a complete first-prompt gallery can produce predictions.
        return sorted([gid for gid in self.gallery.keys() if self.is_complete(gid)])

    def best_similarity(self, embedding: np.ndarray, identity: int) -> float:
        refs = self.gallery.get(int(identity), [])
        if not refs:
            return -1.0
        emb = l2_normalize(embedding)
        return max(cosine_similarity(emb, ref) for ref in refs)

    def similarity_matrix(self, embeddings: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        ids = self.identities()
        sim = np.zeros((len(embeddings), len(ids)), dtype=np.float32)
        if len(embeddings) == 0 or len(ids) == 0:
            return sim, ids
        for i, emb in enumerate(embeddings):
            for j, identity in enumerate(ids):
                sim[i, j] = self.best_similarity(emb, identity)
        return sim, ids


class FixedInitialGallery:
    """
    Fixed-gallery TransReID baseline.

    Each identity is represented only by its first N accepted seed crops.
    For the first-prompt baseline, N should be 1.

    After an identity reaches N embeddings, its gallery is frozen.
    No later detections are added to the gallery.
    """

    def __init__(self, required_embeddings_per_id: int = 1):
        self.required_embeddings_per_id = max(1, int(required_embeddings_per_id))
        self.gallery: Dict[int, List[np.ndarray]] = {}

    def add_seed(self, identity: int, embedding: np.ndarray) -> bool:
        identity = int(identity)

        if self.is_complete(identity):
            return False

        emb = l2_normalize(embedding)
        self.gallery.setdefault(identity, []).append(emb)
        return True

    def count(self, identity: int) -> int:
        return len(self.gallery.get(int(identity), []))

    def is_complete(self, identity: int) -> bool:
        return self.count(identity) >= self.required_embeddings_per_id

    def identities(self) -> List[int]:
        return sorted([
            gid for gid in self.gallery.keys()
            if self.is_complete(gid)
        ])

    def best_similarity(self, embedding: np.ndarray, identity: int) -> float:
        refs = self.gallery.get(int(identity), [])
        if not refs:
            return -1.0

        emb = l2_normalize(embedding)
        return max(cosine_similarity(emb, ref) for ref in refs)

    def similarity_matrix(self, embeddings: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        ids = self.identities()
        sim = np.zeros((len(embeddings), len(ids)), dtype=np.float32)

        if len(embeddings) == 0 or len(ids) == 0:
            return sim, ids

        for i, emb in enumerate(embeddings):
            for j, identity in enumerate(ids):
                sim[i, j] = self.best_similarity(emb, identity)

        return sim, ids
    

# ---------------------------------------------------------------------
# Detection / assignment
# ---------------------------------------------------------------------
def run_yolo_person_detector(yolo_model, rgb: np.ndarray, conf: float = 0.25, imgsz: int = 640, person_class_id: int = 0):
    results = yolo_model.predict(source=rgb, conf=float(conf), imgsz=int(imgsz), classes=[int(person_class_id)], verbose=False)
    detections = []
    if not results or results[0].boxes is None:
        return detections
    boxes = results[0].boxes.xyxy.detach().cpu().numpy()
    confs = results[0].boxes.conf.detach().cpu().numpy()
    h, w = rgb.shape[:2]
    for bb, score in zip(boxes, confs):
        x1, y1, x2, y2 = [int(round(v)) for v in bb.tolist()]
        x1, y1, x2, y2 = clamp_bbox_xyxy((x1, y1, x2, y2), w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append((x1, y1, x2, y2, float(score)))
    return detections


def assign_detections_to_gallery(det_embeddings: List[np.ndarray], gallery, reid_thr: float) -> Dict[int, int]:
    sim_mat, gallery_ids = gallery.similarity_matrix(det_embeddings)
    if sim_mat.shape[0] == 0 or sim_mat.shape[1] == 0:
        return {}
    row_ind, col_ind = linear_sum_assignment(1.0 - sim_mat)
    assignments: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        sim = float(sim_mat[r, c])
        if sim >= float(reid_thr):
            assignments[int(r)] = int(gallery_ids[c])
    return assignments


# ---------------------------------------------------------------------
# Evaluation matching
# ---------------------------------------------------------------------
def match_frame_hungarian(gt_bb_by_id, pred_bbox_by_id, iou_match_thr: float):
    gt_ids = list(gt_bb_by_id.keys())
    pred_ids = list(pred_bbox_by_id.keys())
    gt_to_pred = {gid: None for gid in gt_ids}
    gt_to_iou = {gid: 0.0 for gid in gt_ids}
    matched_gt_ids = set()
    matched_pred_ids = set()

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return gt_to_pred, gt_to_iou, matched_gt_ids, matched_pred_ids

    iou_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, gid in enumerate(gt_ids):
        for j, pid in enumerate(pred_ids):
            iou_mat[i, j] = iou_xyxy(gt_bb_by_id[gid], pred_bbox_by_id[pid])

    row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
    for r, c in zip(row_ind, col_ind):
        val = float(iou_mat[r, c])
        if val >= float(iou_match_thr):
            gid = gt_ids[r]
            pid = pred_ids[c]
            gt_to_pred[gid] = pid
            gt_to_iou[gid] = val
            matched_gt_ids.add(gid)
            matched_pred_ids.add(pid)
    return gt_to_pred, gt_to_iou, matched_gt_ids, matched_pred_ids


def match_frame_same_id_strict(gt_bb_by_id, pred_bbox_by_id, iou_match_thr: float):
    strict_gt_to_pred = {int(gid): None for gid in gt_bb_by_id.keys()}
    strict_gt_to_iou = {int(gid): 0.0 for gid in gt_bb_by_id.keys()}
    strict_matched_gt_ids = set()
    strict_matched_pred_ids = set()
    wrong_id_pred_for_gt = {int(gid): None for gid in gt_bb_by_id.keys()}
    wrong_id_iou_for_gt = {int(gid): 0.0 for gid in gt_bb_by_id.keys()}
    thr = float(iou_match_thr)

    for gid, gt_bb in gt_bb_by_id.items():
        gid = int(gid)
        same_bb = pred_bbox_by_id.get(gid, None)
        if same_bb is not None:
            same_iou = iou_xyxy(gt_bb, same_bb)
            if same_iou >= thr:
                strict_gt_to_pred[gid] = gid
                strict_gt_to_iou[gid] = float(same_iou)
                strict_matched_gt_ids.add(gid)
                strict_matched_pred_ids.add(gid)

        best_wrong_pid = None
        best_wrong_iou = 0.0
        for pid, pred_bb in pred_bbox_by_id.items():
            pid = int(pid)
            if pid == gid:
                continue
            val = iou_xyxy(gt_bb, pred_bb)
            if val > best_wrong_iou:
                best_wrong_iou = val
                best_wrong_pid = pid
        if best_wrong_iou >= thr:
            wrong_id_pred_for_gt[gid] = best_wrong_pid
            wrong_id_iou_for_gt[gid] = float(best_wrong_iou)

    return (
        strict_gt_to_pred,
        strict_gt_to_iou,
        strict_matched_gt_ids,
        strict_matched_pred_ids,
        wrong_id_pred_for_gt,
        wrong_id_iou_for_gt,
    )


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

    strict_matches: int = 0
    strict_false_positives: int = 0
    strict_false_negatives: int = 0
    strict_reacq_events: int = 0
    strict_reacq_gaps: List[int] = None
    strict_iou_sum: float = 0.0
    strict_iou_count: int = 0
    wrong_id_overlaps: int = 0
    wrong_id_reacq_events: int = 0

    seed_skipped_overlap: int = 0
    seed_skipped_small: int = 0
    seed_failed: int = 0
    total_unique_gt_ids: int = 0
    seeded_ids_count: int = 0
    ignored_predictions_no_gt: int = 0

    def __post_init__(self):
        if self.reacq_gaps is None:
            self.reacq_gaps = []
        if self.strict_reacq_gaps is None:
            self.strict_reacq_gaps = []


def summarize_reacq(gaps: List[int]):
    if not gaps:
        return 0, 0.0, 0.0, 0
    gaps = sorted(gaps)
    n = len(gaps)
    med = float(gaps[n // 2]) if n % 2 else float(0.5 * (gaps[n // 2 - 1] + gaps[n // 2]))
    return n, float(sum(gaps) / n), med, int(max(gaps))


def metrics_to_row(run_prefix, label, seq, reid_thr, yolo_model_name, yolo_conf, yolo_imgsz, initial_gallery_frames, met, out_csv, gt_mot_path, pred_mot_path):
    denom_gt = met.eligible_gt_boxes
    reacq_n, reacq_mean, reacq_med, reacq_max = summarize_reacq(met.reacq_gaps)
    strict_reacq_n, strict_reacq_mean, strict_reacq_med, strict_reacq_max = summarize_reacq(met.strict_reacq_gaps)

    precision = met.matches / (met.matches + met.false_positives) if (met.matches + met.false_positives) > 0 else 0.0
    recall = met.matches / (met.matches + met.false_negatives) if (met.matches + met.false_negatives) > 0 else 0.0
    mota = 1.0 - ((met.false_negatives + met.false_positives + met.id_switches) / denom_gt) if denom_gt > 0 else 0.0

    strict_precision = met.strict_matches / (met.strict_matches + met.strict_false_positives) if (met.strict_matches + met.strict_false_positives) > 0 else 0.0
    strict_recall = met.strict_matches / (met.strict_matches + met.strict_false_negatives) if (met.strict_matches + met.strict_false_negatives) > 0 else 0.0
    strict_mota = 1.0 - ((met.strict_false_negatives + met.strict_false_positives) / denom_gt) if denom_gt > 0 else 0.0

    return {
        "run": run_prefix,
        "label": label,
        "seq": seq,
        "baseline": "transreid_yolo_fixed_initial_gallery",
        "reid_thr": float(reid_thr),
        "yolo_model": yolo_model_name,
        "yolo_conf": float(yolo_conf),
        "yolo_imgsz": int(yolo_imgsz),
        "initial_gallery_frames": int(initial_gallery_frames),
        "update_gallery": False,
        "frames": met.frames,
        "total_gt_boxes": met.total_gt_boxes,
        "eligible_gt_boxes": met.eligible_gt_boxes,
        "matches": met.matches,
        "misses": met.false_negatives,
        "false_positives": met.false_positives,
        "false_negatives": met.false_negatives,
        "ignored_predictions_no_gt": met.ignored_predictions_no_gt,
        "precision": precision,
        "recall": recall,
        "mota": mota,
        "match_rate": met.matches / denom_gt if denom_gt > 0 else 0.0,
        "miss_rate": met.false_negatives / denom_gt if denom_gt > 0 else 0.0,
        "id_switches": met.id_switches,
        "id_switches_per_match": met.id_switches / met.matches if met.matches > 0 else 0.0,
        "id_switches_per_gt": met.id_switches / denom_gt if denom_gt > 0 else 0.0,
        "reacq_events": reacq_n,
        "reacq_rate_per_gt": reacq_n / denom_gt if denom_gt > 0 else 0.0,
        "reacq_mean_frames": reacq_mean,
        "reacq_median_frames": reacq_med,
        "reacq_max_frames": reacq_max,
        "mean_iou_when_matched": met.iou_sum / met.iou_count if met.iou_count > 0 else 0.0,
        "strict_matches": met.strict_matches,
        "strict_false_positives": met.strict_false_positives,
        "strict_false_negatives": met.strict_false_negatives,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_mota": strict_mota,
        "strict_match_rate": met.strict_matches / denom_gt if denom_gt > 0 else 0.0,
        "strict_miss_rate": met.strict_false_negatives / denom_gt if denom_gt > 0 else 0.0,
        "strict_reacq_events": strict_reacq_n,
        "strict_reacq_mean_frames": strict_reacq_mean,
        "strict_reacq_median_frames": strict_reacq_med,
        "strict_reacq_max_frames": strict_reacq_max,
        "strict_mean_iou_when_matched": met.strict_iou_sum / met.strict_iou_count if met.strict_iou_count > 0 else 0.0,
        "wrong_id_overlaps": met.wrong_id_overlaps,
        "wrong_id_overlap_rate": met.wrong_id_overlaps / denom_gt if denom_gt > 0 else 0.0,
        "wrong_id_reacq_events": met.wrong_id_reacq_events,
        "total_unique_gt_ids": met.total_unique_gt_ids,
        "seeded_ids_count": met.seeded_ids_count,
        "seed_coverage": met.seeded_ids_count / met.total_unique_gt_ids if met.total_unique_gt_ids > 0 else 0.0,
        "seed_skipped_small": met.seed_skipped_small,
        "seed_skipped_overlap": met.seed_skipped_overlap,
        "seed_failed": met.seed_failed,
        "out_csv": str(out_csv),
        "gt_mot_path": str(gt_mot_path),
        "pred_mot_path": str(pred_mot_path),
    }


def save_review_frame(review_root: Path, category: str, seq_name: str, fidx: int, pred_id: int, vis_bgr: np.ndarray):
    out_dir = review_root / category / seq_name / f"id_{int(pred_id)}"
    safe_mkdir(out_dir)
    cv2.imwrite(str(out_dir / f"frame_{int(fidx):06d}_pr{int(pred_id)}.jpg"), vis_bgr)


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
    initial_gallery_frames: int,
    rotate_deg: int = 0,
    stride: int = 1,
    max_frames: int = -1,
    visible_area_frac: float = 0.02,
    visible_min_h: int = 120,
    visible_min_w: int = 0,
    seed_overlap_iou_max: float = 0.10,
    iou_match_thr: float = 0.30,
    eval_seed_frame: bool = False,
    ignore_predictions_without_gt: bool = False,
    ignore_predictions_without_gt_iou: float = 0.01,
    save_review_frames: bool = False,
    save_video: bool = True,
    paper_frames_dir=None,
    paper_frame_ranges=None,
    paper_model_name="",
) -> SeqMetrics:
    img_dir, gt_path = find_sequence_paths(ktp_root, seq_name)
    gt_map = parse_gt2d_file(gt_path)
    frames, ts_by_path = load_ordered_frames(img_dir, stride=stride, max_frames=max_frames)

    bgr0 = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    bgr0 = rotate_frame(bgr0, rotate_deg)
    height, width = bgr0.shape[:2]
    video_fps = estimate_sequence_fps(frames, ts_by_path, fallback_fps=15.0)

    all_gt_ids = {int(gid) for dets in gt_map.values() for gid, *_ in dets}
    gallery = FirstPromptGallery(required_embeddings_per_id=initial_gallery_frames)
    seeded = set()
    gt_states: Dict[int, GTState] = {}
    strict_gt_states: Dict[int, GTState] = {}

    metrics = SeqMetrics(total_unique_gt_ids=len(all_gt_ids))

    safe_mkdir(out_csv_path.parent)
    safe_mkdir(mot_gt_path.parent)
    safe_mkdir(mot_pred_path.parent)

    review_root = out_csv_path.parent / "annotation_limited_review"

    fcsv = out_csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)
    f_gt_mot = mot_gt_path.open("w", encoding="utf-8")
    f_pred_mot = mot_pred_path.open("w", encoding="utf-8")

    writer.writerow(["# baseline: transreid_yolo_fixed_initial_gallery"])
    writer.writerow([f"# reid_thr: {reid_thr}"])
    writer.writerow([f"# yolo_conf: {yolo_conf}"])
    writer.writerow([
        f"# seed_rules: visible_area_frac={visible_area_frac}, visible_min_h={visible_min_h}, "
        f"visible_min_w={visible_min_w}, seed_overlap_iou_max={seed_overlap_iou_max}, "
        f"iou_match_thr={iou_match_thr}, stride={stride}, max_frames={max_frames}, "
        f"eval_seed_frame={eval_seed_frame}, initial_gallery_frames={initial_gallery_frames}, update_gallery=False, "
        f"ignore_predictions_without_gt={ignore_predictions_without_gt}, "
        f"ignore_predictions_without_gt_iou={ignore_predictions_without_gt_iou}"
    ])
    writer.writerow([
        "seq", "frame_idx", "ts",
        "gt_id", "gt_x", "gt_y", "gt_w", "gt_h", "gt_area_frac",
        "eligible", "seeded_now", "seeded_already", "seed_skip_reason",
        "pred_id", "match_iou", "id_switch_event", "reacq_event", "gap_len",
        "strict_pred_id", "strict_match_iou", "strict_reacq_event", "strict_gap_len",
        "wrong_id_pred_id", "wrong_id_iou",
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
            gt_bb_by_id_all[int(gid)] = clamp_bbox_xyxy(bbox_xywh_to_xyxy(x, y, w, h), width, height)

        seeded_now_ids = set()
        seed_skip_reason_by_gid: Dict[int, str] = {}

        # 1) Build a first-prompt gallery from GT crops.
        #    Unlike the online TransReID baseline, this does NOT update the gallery
        #    with later predictions. For each identity, we collect only the first
        #    accepted GT crop and then freeze that ID.
        for gid, x, y, w, h in gt_dets:
            gid = int(gid)
            if gid not in gt_states:
                gt_states[gid] = GTState()

            if gallery.is_complete(gid):
                if gid not in seeded:
                    seeded.add(gid)
                    metrics.seeded_ids_count = len(seeded)
                seed_skip_reason_by_gid[gid] = ""
                continue

            bb = gt_bb_by_id_all[gid]
            bw = max(0, bb[2] - bb[0])
            bh = max(0, bb[3] - bb[1])
            area_frac = (bw * bh) / float(width * height + 1e-9)
            visible_ok = area_frac >= float(visible_area_frac) and bh >= int(visible_min_h)
            if visible_min_w and int(visible_min_w) > 0:
                visible_ok = visible_ok and bw >= int(visible_min_w)
            if not visible_ok:
                seed_skip_reason_by_gid[gid] = f"collecting_gallery_{gallery.count(gid)}/{initial_gallery_frames}:small"
                metrics.seed_skipped_small += 1
                continue

            # Important: check overlap against all GT identities in the current frame,
            # not only identities already seeded. This avoids using ambiguous crops
            # as fixed identity references.
            max_iou_other = 0.0
            for other_gid, other_bb in gt_bb_by_id_all.items():
                if int(other_gid) == gid:
                    continue
                max_iou_other = max(max_iou_other, iou_xyxy(bb, other_bb))
            if max_iou_other > float(seed_overlap_iou_max):
                seed_skip_reason_by_gid[gid] = (
                    f"collecting_gallery_{gallery.count(gid)}/{initial_gallery_frames}:"
                    f"overlap(max_iou={max_iou_other:.3f})"
                )
                metrics.seed_skipped_overlap += 1
                continue

            gt_crop = crop_rgb(rgb, bb)
            emb = reid_extractor.extract(gt_crop) if gt_crop is not None else None
            if emb is None:
                seed_skip_reason_by_gid[gid] = f"collecting_gallery_{gallery.count(gid)}/{initial_gallery_frames}:seed_failed"
                metrics.seed_failed += 1
                continue

            gallery.add_seed(gid, emb)
            n = gallery.count(gid)
            if gallery.is_complete(gid):
                seeded.add(gid)
                seeded_now_ids.add(gid)
                metrics.seeded_ids_count = len(seeded)
                seed_skip_reason_by_gid[gid] = f"gallery_complete_{n}/{initial_gallery_frames}"
            else:
                seed_skip_reason_by_gid[gid] = f"collecting_gallery_{n}/{initial_gallery_frames}"

        # 2) Eligible GT boxes.
        eligible_gt_ids = set()
        for gid, *_ in gt_dets:
            gid = int(gid)
            if gid in seeded:
                if gid in seeded_now_ids and not eval_seed_frame:
                    continue
                eligible_gt_ids.add(gid)
        gt_bb_by_id_eval = {gid: gt_bb_by_id_all[gid] for gid in eligible_gt_ids}
        metrics.eligible_gt_boxes += len(gt_bb_by_id_eval)

        mot_frame_idx = fidx + 1
        for gid in sorted(gt_bb_by_id_eval.keys()):
            x1, y1, ww, hh = xyxy_to_xywh(gt_bb_by_id_eval[gid])
            f_gt_mot.write(f"{mot_frame_idx},{gid},{x1},{y1},{ww},{hh},1,1,1,1\n")

        # 3) YOLO detections + ReID embeddings.
        yolo_dets = run_yolo_person_detector(yolo_model, rgb, conf=yolo_conf, imgsz=yolo_imgsz)
        det_bboxes: List[Tuple[int, int, int, int]] = []
        det_embeddings: List[np.ndarray] = []
        for x1, y1, x2, y2, conf in yolo_dets:
            det_bb = (x1, y1, x2, y2)
            crop = crop_rgb(rgb, det_bb)
            emb = reid_extractor.extract(crop) if crop is not None else None
            if emb is None:
                continue
            det_bboxes.append(det_bb)
            det_embeddings.append(emb)

        # 4) Assign detections to gallery IDs.
        det_to_identity = assign_detections_to_gallery(det_embeddings, gallery, reid_thr=reid_thr)
        sim_mat, gallery_ids = gallery.similarity_matrix(det_embeddings)

        det_best_match = {}
        for det_idx in range(len(det_embeddings)):
            if len(gallery_ids) == 0 or sim_mat.shape[1] == 0:
                det_best_match[det_idx] = {"best_id": "none", "best_sim": -1.0}
            else:
                best_col = int(np.argmax(sim_mat[det_idx]))
                det_best_match[det_idx] = {
                    "best_id": int(gallery_ids[best_col]),
                    "best_sim": float(sim_mat[det_idx, best_col]),
                }

        pred_bbox_by_id_raw: Dict[int, Tuple[int, int, int, int]] = {}
        for det_idx, pred_id in det_to_identity.items():
            # Fixed-gallery baseline: predictions are never added back into the gallery.
            pred_id = int(pred_id)
            pred_bbox_by_id_raw[pred_id] = det_bboxes[int(det_idx)]

        # 5) Optional annotation-limited filtering.
        suspicious_pred_ids = set()
        suspicious_pred_reason: Dict[int, str] = {}
        same_id_iou_thr = float(ignore_predictions_without_gt_iou)

        for pid, pbb in pred_bbox_by_id_raw.items():
            pid = int(pid)
            same_gt_bb = gt_bb_by_id_all.get(pid, None)
            if same_gt_bb is None:
                best_other_gid = None
                best_other_iou = 0.0
                for gid2, gbb2 in gt_bb_by_id_all.items():
                    val = iou_xyxy(pbb, gbb2)
                    if val > best_other_iou:
                        best_other_iou = val
                        best_other_gid = int(gid2)
                suspicious_pred_ids.add(pid)
                if best_other_gid is None or best_other_iou < same_id_iou_thr:
                    suspicious_pred_reason[pid] = "no_same_id_gt_no_other_overlap"
                else:
                    suspicious_pred_reason[pid] = f"no_same_id_gt_overlaps_gt{best_other_gid}_iou{best_other_iou:.3f}"
            else:
                same_iou = iou_xyxy(pbb, same_gt_bb)
                if same_iou < same_id_iou_thr:
                    suspicious_pred_ids.add(pid)
                    suspicious_pred_reason[pid] = f"same_id_gt_low_iou_{same_iou:.3f}"

        if ignore_predictions_without_gt:
            pred_bbox_by_id_eval = {pid: bb for pid, bb in pred_bbox_by_id_raw.items() if int(pid) not in suspicious_pred_ids}
            metrics.ignored_predictions_no_gt += len(suspicious_pred_ids)
        else:
            pred_bbox_by_id_eval = dict(pred_bbox_by_id_raw)

        for pred_id in sorted(pred_bbox_by_id_eval.keys()):
            x1, y1, ww, hh = xyxy_to_xywh(pred_bbox_by_id_eval[pred_id])
            f_pred_mot.write(f"{mot_frame_idx},{pred_id},{x1},{y1},{ww},{hh},1,-1,-1,-1\n")

        # 6) Match standard and strict.
        gt_to_pred, gt_to_iou, matched_gt_ids, matched_pred_ids = match_frame_hungarian(
            gt_bb_by_id_eval, pred_bbox_by_id_eval, iou_match_thr
        )
        metrics.matches += len(matched_gt_ids)
        metrics.false_negatives += len(gt_bb_by_id_eval) - len(matched_gt_ids)
        metrics.false_positives += len(pred_bbox_by_id_eval) - len(matched_pred_ids)
        for gid in matched_gt_ids:
            metrics.iou_sum += gt_to_iou[gid]
            metrics.iou_count += 1

        (
            strict_gt_to_pred,
            strict_gt_to_iou,
            strict_matched_gt_ids,
            strict_matched_pred_ids,
            wrong_id_pred_for_gt,
            wrong_id_iou_for_gt,
        ) = match_frame_same_id_strict(gt_bb_by_id_eval, pred_bbox_by_id_eval, iou_match_thr)

        metrics.strict_matches += len(strict_matched_gt_ids)
        metrics.strict_false_negatives += len(gt_bb_by_id_eval) - len(strict_matched_gt_ids)
        metrics.strict_false_positives += len(pred_bbox_by_id_eval) - len(strict_matched_pred_ids)
        for gid in strict_matched_gt_ids:
            metrics.strict_iou_sum += strict_gt_to_iou[gid]
            metrics.strict_iou_count += 1
        metrics.wrong_id_overlaps += sum(1 for gid in gt_bb_by_id_eval.keys() if wrong_id_pred_for_gt.get(int(gid), None) is not None)

        # 7) Visualization base, also used for optional review frames.
        #
        # Clean thesis/debug visualization:
        #   - GT boxes are shown without black label backgrounds.
        #   - Only assigned PR boxes are shown.
        #   - Raw YOLO detections are not shown, to avoid clutter.
        #   - Suspicious/ignored predictions remain orange, but only show "PR <id>".
        vis = bgr.copy()

        # Draw GT boxes.
        for gid, bb in gt_bb_by_id_all.items():
            color = (255, 255, 255) if gid in eligible_gt_ids else (120, 120, 120)
            cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
            draw_text_plain(
                vis,
                f"GT {gid}",
                (bb[0], max(18, bb[1] - 6)),
                color=color,
                scale=0.65,
                thickness=2,
            )

        # Draw normal evaluated predictions.
        # These are predictions that remain after annotation-limited filtering.
        for pred_id, bb in pred_bbox_by_id_eval.items():
            pred_id = int(pred_id)
            color = _rgb_to_bgr(_id_to_rgb(pred_id))
            cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
            draw_text_plain(
                vis,
                f"PR {pred_id}",
                (bb[0], min(height - 5, bb[1] + 22)),
                color=color,
                scale=0.65,
                thickness=2,
            )

        # Draw suspicious predictions in orange.
        # These remain visible even if they are ignored for the annotation-limited metric.
        review_items = []

        for pred_id, bb in pred_bbox_by_id_raw.items():
            pred_id = int(pred_id)
            if pred_id not in suspicious_pred_ids:
                continue

            color = (0, 165, 255)  # orange in BGR
            cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
            draw_text_plain(
                vis,
                f"PR {pred_id}",
                (bb[0], min(height - 5, bb[1] + 22)),
                color=color,
                scale=0.65,
                thickness=2,
            )

            if save_review_frames:
                reason = suspicious_pred_reason.get(pred_id, "unknown")
                if reason.startswith("no_same_id_gt_no_other_overlap"):
                    cat = "no_same_id_gt_no_other_overlap"
                elif reason.startswith("no_same_id_gt_overlaps"):
                    cat = "no_same_id_gt_overlaps_other_gt"
                else:
                    cat = "same_id_gt_low_iou"

                review_items.append((cat, pred_id))

        # Save review frames after all GT/PR boxes have been drawn.
        if save_review_frames:
            for cat, pred_id in review_items:
                save_review_frame(review_root, cat, seq_name, fidx, int(pred_id), vis)

        if should_save_paper_frame(seq_name, int(fidx), paper_frame_ranges):
            save_paper_frame(
                paper_frames_dir=paper_frames_dir,
                model_name=paper_model_name,
                seq_name=seq_name,
                frame_idx=int(fidx),
                image_bgr=vis,
            )

        if video_writer is not None:
            video_writer.write(vis)

        # 8) Per-GT diagnostic CSV.
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

            strict_st = strict_gt_states.get(gid, GTState())
            strict_cur = strict_gt_to_pred.get(gid, None) if eligible else None
            strict_reacq = 0
            wrong_cur = wrong_id_pred_for_gt.get(gid, None) if eligible else None
            wrong_iou = wrong_id_iou_for_gt.get(gid, 0.0) if eligible else 0.0

            if eligible:
                if strict_cur is None:
                    if strict_st.prev_pred is not None and not strict_st.in_gap:
                        strict_st.in_gap = True
                        strict_st.gap_len = 1
                    elif strict_st.in_gap:
                        strict_st.gap_len += 1
                else:
                    if strict_st.in_gap:
                        strict_reacq = 1
                        metrics.strict_reacq_events += 1
                        metrics.strict_reacq_gaps.append(strict_st.gap_len)
                        strict_st.in_gap = False
                        strict_st.gap_len = 0
                    strict_st.prev_pred = strict_cur
                if strict_cur is None and wrong_cur is not None and strict_st.in_gap:
                    metrics.wrong_id_reacq_events += 1
            strict_gt_states[gid] = strict_st

            gt_bb = gt_bb_by_id_all[gid]
            gt_area = max(0, gt_bb[2] - gt_bb[0]) * max(0, gt_bb[3] - gt_bb[1])
            gt_area_frac = gt_area / float(width * height + 1e-9)

            writer.writerow([
                seq_name, fidx, ts,
                gid, f"{x:.3f}", f"{y:.3f}", f"{w:.3f}", f"{h:.3f}", f"{gt_area_frac:.6f}",
                int(eligible),
                1 if gid in seeded_now_ids else 0,
                1 if gid in seeded else 0,
                seed_skip_reason_by_gid.get(gid, ""),
                cur if cur is not None else "",
                f"{gt_to_iou.get(gid, 0.0):.6f}",
                idsw,
                reacq,
                gt_states[gid].gap_len if gt_states[gid].in_gap else 0,
                strict_cur if strict_cur is not None else "",
                f"{strict_gt_to_iou.get(gid, 0.0):.6f}",
                strict_reacq,
                strict_gt_states[gid].gap_len if strict_gt_states[gid].in_gap else 0,
                wrong_cur if wrong_cur is not None else "",
                f"{wrong_iou:.6f}",
            ])

    fcsv.close()
    f_gt_mot.close()
    f_pred_mot.close()
    if video_writer is not None:
        video_writer.release()

    return metrics


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# TrackEval integration
# ---------------------------------------------------------------------
def count_max_frame_in_mot(mot_file: Path) -> int:
    """Return the largest frame index found in a MOT-format text file."""
    max_frame = 0
    if not Path(mot_file).exists():
        return max_frame
    with Path(mot_file).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if not parts:
                continue
            try:
                max_frame = max(max_frame, int(float(parts[0])))
            except Exception:
                pass
    return int(max_frame)


def write_trackeval_seqinfo(seq_dir: Path, seq_name: str, seq_length: int, fps: float, width: int, height: int) -> None:
    """Write seqinfo.ini in the format expected by TrackEval's MOTChallenge loader."""
    seq_dir.mkdir(parents=True, exist_ok=True)
    seqinfo_path = seq_dir / "seqinfo.ini"
    seqinfo_path.write_text(
        "[Sequence]\n"
        f"name={seq_name}\n"
        "imDir=img1\n"
        f"frameRate={fps:g}\n"
        f"seqLength={int(seq_length)}\n"
        f"imWidth={int(width)}\n"
        f"imHeight={int(height)}\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )


def prepare_mot_exports_for_trackeval(
    mot_export_dir: Path,
    trackeval_root: Path,
    tracker_name: str,
    benchmark: str,
    split: str,
    sequences: List[str],
    fps: float,
    width: int = 640,
    height: int = 480,
    overwrite: bool = True,
) -> Dict[str, str]:
    """
    Copy this script's MOT exports into TrackEval's MOTChallenge directory layout.

    Expected input files in mot_export_dir:
        <seq>_gt.txt
        <seq>_pred.txt
    """
    mot_export_dir = Path(mot_export_dir).resolve()
    trackeval_root = Path(trackeval_root).resolve()

    if not mot_export_dir.exists():
        raise FileNotFoundError(f"MOT export directory not found: {mot_export_dir}")
    if not trackeval_root.exists():
        raise FileNotFoundError(f"TrackEval root not found: {trackeval_root}")

    benchmark_split = f"{benchmark}-{split}"
    gt_root = trackeval_root / "data" / "gt" / "mot_challenge" / benchmark_split
    seqmaps_dir = gt_root / "seqmaps"
    tracker_root = trackeval_root / "data" / "trackers" / "mot_challenge" / benchmark_split / tracker_name
    tracker_data_dir = tracker_root / "data"

    seqmaps_dir.mkdir(parents=True, exist_ok=True)
    tracker_data_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for seq in sequences:
        seq_safe = seq.replace("/", "_")
        if not (mot_export_dir / f"{seq_safe}_gt.txt").exists():
            missing.append(str(mot_export_dir / f"{seq_safe}_gt.txt"))
        if not (mot_export_dir / f"{seq_safe}_pred.txt").exists():
            missing.append(str(mot_export_dir / f"{seq_safe}_pred.txt"))
    if missing:
        raise FileNotFoundError("Missing required MOT files:\n" + "\n".join(f"  - {m}" for m in missing))

    seqmap_path = seqmaps_dir / f"{benchmark_split}.txt"
    seqmap_path.write_text("name\n" + "\n".join(seq.replace("/", "_") for seq in sequences) + "\n", encoding="utf-8")

    print("\n[TrackEval setup]")
    print("  MOT export dir :", mot_export_dir)
    print("  TrackEval root :", trackeval_root)
    print("  benchmark/split:", benchmark_split)
    print("  tracker name   :", tracker_name)
    print("  seqmap         :", seqmap_path)

    for seq in sequences:
        seq_safe = seq.replace("/", "_")
        src_gt = mot_export_dir / f"{seq_safe}_gt.txt"
        src_pred = mot_export_dir / f"{seq_safe}_pred.txt"

        dst_gt_dir = gt_root / seq_safe / "gt"
        dst_gt_dir.mkdir(parents=True, exist_ok=True)
        dst_gt = dst_gt_dir / "gt.txt"
        dst_pred = tracker_data_dir / f"{seq_safe}.txt"

        if not overwrite and (dst_gt.exists() or dst_pred.exists()):
            raise FileExistsError(
                f"TrackEval files already exist for sequence {seq_safe}. "
                "Use --trackeval_overwrite to replace them."
            )

        shutil.copyfile(src_gt, dst_gt)
        shutil.copyfile(src_pred, dst_pred)

        seq_len = max(count_max_frame_in_mot(src_gt), count_max_frame_in_mot(src_pred))
        write_trackeval_seqinfo(
            seq_dir=gt_root / seq_safe,
            seq_name=seq_safe,
            seq_length=seq_len,
            fps=fps,
            width=width,
            height=height,
        )

        print(f"  [ok] {seq_safe}: len={seq_len}, GT -> {dst_gt}, Pred -> {dst_pred}")

    return {
        "benchmark_split": benchmark_split,
        "gt_root": str(gt_root),
        "tracker_root": str(tracker_root),
        "tracker_data_dir": str(tracker_data_dir),
        "seqmap_path": str(seqmap_path),
    }


def run_trackeval_now(
    trackeval_root: Path,
    tracker_name: str,
    benchmark: str,
    split: str,
    threshold: float = 0.5,
) -> subprocess.CompletedProcess:
    """Run TrackEval's MOTChallenge evaluator immediately and stream the output."""
    trackeval_root = Path(trackeval_root).resolve()
    script_path = trackeval_root / "scripts" / "run_mot_challenge.py"
    if not script_path.exists():
        raise FileNotFoundError(f"TrackEval script not found: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--BENCHMARK", str(benchmark),
        "--SPLIT_TO_EVAL", str(split),
        "--TRACKERS_TO_EVAL", str(tracker_name),
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1",
        "--THRESHOLD", str(float(threshold)),
    ]

    print("\n[TrackEval run]")
    print("  cwd:", trackeval_root)
    print("  cmd:", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    print("")

    return subprocess.run(cmd, cwd=str(trackeval_root), check=True)


def parse_sequences(seq_arg: str) -> List[str]:
    return [s.strip() for s in str(seq_arg).split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktp_root", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--run_name", default="transreid_yolo_baseline", type=str)
    ap.add_argument("--sequences", default="Arc,Rotation,Still,Translation", type=str)

    ap.add_argument("--yolo_model", default="yolov8n.pt", type=str)
    ap.add_argument("--yolo_conf", default=0.25, type=float)
    ap.add_argument("--yolo_imgsz", default=640, type=int)

    ap.add_argument("--reid_thr", default=0.80, type=float)
    ap.add_argument(
        "--initial_gallery_frames",
        default=10,
        type=int,
        help="Number of accepted initial GT crops stored per identity before that identity becomes evaluable. The gallery is then frozen.",
    )
    # Kept only so older commands do not crash; ignored by this fixed-gallery baseline.
    ap.add_argument("--max_gallery_embeddings", default=None, type=int)
    ap.add_argument("--no_update_gallery", action="store_true")

    ap.add_argument("--rotate", default=0, type=int)
    ap.add_argument("--stride", default=1, type=int)
    ap.add_argument("--max_frames", default=-1, type=int)
    ap.add_argument("--visible_area_frac", default=0.02, type=float)
    ap.add_argument("--visible_min_h", default=120, type=int)
    ap.add_argument("--visible_min_w", default=0, type=int)
    ap.add_argument("--seed_overlap_iou_max", default=0.10, type=float)
    ap.add_argument("--iou_match_thr", default=0.30, type=float)
    ap.add_argument("--eval_seed_frame", action="store_true")

    ap.add_argument("--ignore_predictions_without_gt", action="store_true")
    ap.add_argument("--ignore_predictions_without_gt_iou", default=0.01, type=float)
    ap.add_argument("--save_review_frames", action="store_true")

    ap.add_argument("--save_video", action="store_true")

    # TrackEval integration. When enabled, the script copies the MOT exports
    # into TrackEval's expected folder structure and runs HOTA/CLEAR/Identity
    # immediately from the same output files.
    ap.add_argument("--run_trackeval", action="store_true",
                    help="After this evaluation, prepare files for TrackEval and run TrackEval automatically.")
    ap.add_argument("--trackeval_root", default=None, type=str,
                    help="Path to the TrackEval repository root. Required if --run_trackeval is set.")
    ap.add_argument("--trackeval_tracker_name", default=None, type=str,
                    help="Tracker name to use inside TrackEval. Default: sanitized run_name.")
    ap.add_argument("--trackeval_benchmark", default="KTP-5Hz", type=str)
    ap.add_argument("--trackeval_split", default="train", type=str)
    ap.add_argument("--trackeval_threshold", default=0.5, type=float,
                    help="TrackEval matching threshold. Use 0.5 for standard MOT; optionally 0.3 for your diagnostic setting.")
    ap.add_argument("--trackeval_fps", default=None, type=float,
                    help="FPS written to seqinfo.ini. Default: 30 / stride, so stride 6 gives 5 Hz.")
    ap.add_argument("--trackeval_width", default=640, type=int)
    ap.add_argument("--trackeval_height", default=480, type=int)
    ap.add_argument("--trackeval_overwrite", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to overwrite existing TrackEval files for the same tracker name.")
    
    ap.add_argument("--paper_frames_dir", type=str, default="")
    ap.add_argument("--paper_frame_ranges", type=str, default="")
    ap.add_argument("--paper_model_name", type=str, default="")

    args = ap.parse_args()

    paper_frame_ranges = parse_paper_frame_ranges(args.paper_frame_ranges)
    paper_frames_dir = Path(args.paper_frames_dir) if args.paper_frames_dir else None
    paper_model_name = args.paper_model_name if args.paper_model_name else args.run_name

    # Dedicated first-prompt protocol:
    # even if an old command passes --initial_gallery_frames 10, this script
    # keeps exactly one initial reference crop per identity and then freezes it.
    args.initial_gallery_frames = 1

    ktp_root = Path(args.ktp_root)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    print("[paths]")
    print(f"  REPO_ROOT: {REPO_ROOT}")
    print(f"  KTP_ROOT : {ktp_root}")
    print(f"  OUT_DIR  : {out_dir}")
    print(f"  YOLO     : {args.yolo_model}")
    print(f"  REID     : TransReID")

    print(f"[YOLO] loading {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    reid_extractor = TransReIDExtractor(device=device)

    sequences = parse_sequences(args.sequences)
    run_prefix = (
        f"{args.run_name}_transreid_yolo"
        f"_stride{args.stride}_rthr{args.reid_thr:g}"
        f"_firstprompt"
        f"_yconf{args.yolo_conf:g}_imgsz{args.yolo_imgsz}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    mot_root = out_dir / "mot_exports" / run_prefix
    safe_mkdir(mot_root)

    summary_rows = []
    all_metrics = SeqMetrics()

    settings = {
        "run_name": args.run_name,
        "run_prefix": run_prefix,
        "ktp_root": str(ktp_root),
        "sequences": sequences,
        "baseline": "transreid_yolo_fixed_initial_gallery",
        "yolo_model": args.yolo_model,
        "yolo_conf": args.yolo_conf,
        "yolo_imgsz": args.yolo_imgsz,
        "reid_thr": args.reid_thr,
        "initial_gallery_frames": args.initial_gallery_frames,
        "max_gallery_embeddings_ignored": args.max_gallery_embeddings,
        "update_gallery": False,
        "rotate": args.rotate,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "visible_area_frac": args.visible_area_frac,
        "visible_min_h": args.visible_min_h,
        "visible_min_w": args.visible_min_w,
        "seed_overlap_iou_max": args.seed_overlap_iou_max,
        "iou_match_thr": args.iou_match_thr,
        "eval_seed_frame": args.eval_seed_frame,
        "ignore_predictions_without_gt": args.ignore_predictions_without_gt,
        "ignore_predictions_without_gt_iou": args.ignore_predictions_without_gt_iou,
        "save_review_frames": args.save_review_frames,
        "save_video": args.save_video,
        "run_trackeval": args.run_trackeval,
        "trackeval_root": args.trackeval_root,
        "trackeval_tracker_name": args.trackeval_tracker_name,
        "trackeval_benchmark": args.trackeval_benchmark,
        "trackeval_split": args.trackeval_split,
        "trackeval_threshold": args.trackeval_threshold,
        "trackeval_fps": args.trackeval_fps,
    }
    (out_dir / f"{run_prefix}__settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    for seq in sequences:
        seq_safe = seq.replace("/", "_")
        out_csv = out_dir / f"{run_prefix}__{seq_safe}.csv"
        gt_mot_path = mot_root / f"{seq_safe}_gt.txt"
        pred_mot_path = mot_root / f"{seq_safe}_pred.txt"

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
            initial_gallery_frames=args.initial_gallery_frames,
            rotate_deg=args.rotate,
            stride=args.stride,
            max_frames=args.max_frames,
            visible_area_frac=args.visible_area_frac,
            visible_min_h=args.visible_min_h,
            visible_min_w=args.visible_min_w,
            seed_overlap_iou_max=args.seed_overlap_iou_max,
            iou_match_thr=args.iou_match_thr,
            eval_seed_frame=args.eval_seed_frame,
            ignore_predictions_without_gt=args.ignore_predictions_without_gt,
            ignore_predictions_without_gt_iou=args.ignore_predictions_without_gt_iou,
            save_review_frames=args.save_review_frames,
            save_video=args.save_video,
            paper_frames_dir=paper_frames_dir,
            paper_frame_ranges=paper_frame_ranges,
            paper_model_name=paper_model_name,
        )

        row = metrics_to_row(
            run_prefix=run_prefix,
            label=args.run_name,
            seq=seq,
            reid_thr=args.reid_thr,
            yolo_model_name=args.yolo_model,
            yolo_conf=args.yolo_conf,
            yolo_imgsz=args.yolo_imgsz,
            initial_gallery_frames=args.initial_gallery_frames,
            met=met,
            out_csv=str(out_csv),
            gt_mot_path=str(gt_mot_path),
            pred_mot_path=str(pred_mot_path),
        )
        summary_rows.append(row)

        all_metrics.frames += met.frames
        all_metrics.total_gt_boxes += met.total_gt_boxes
        all_metrics.eligible_gt_boxes += met.eligible_gt_boxes
        all_metrics.matches += met.matches
        all_metrics.false_positives += met.false_positives
        all_metrics.false_negatives += met.false_negatives
        all_metrics.id_switches += met.id_switches
        all_metrics.reacq_gaps.extend(met.reacq_gaps)
        all_metrics.iou_sum += met.iou_sum
        all_metrics.iou_count += met.iou_count
        all_metrics.strict_matches += met.strict_matches
        all_metrics.strict_false_positives += met.strict_false_positives
        all_metrics.strict_false_negatives += met.strict_false_negatives
        all_metrics.strict_reacq_gaps.extend(met.strict_reacq_gaps)
        all_metrics.strict_iou_sum += met.strict_iou_sum
        all_metrics.strict_iou_count += met.strict_iou_count
        all_metrics.wrong_id_overlaps += met.wrong_id_overlaps
        all_metrics.wrong_id_reacq_events += met.wrong_id_reacq_events
        all_metrics.seed_skipped_overlap += met.seed_skipped_overlap
        all_metrics.seed_skipped_small += met.seed_skipped_small
        all_metrics.seed_failed += met.seed_failed
        all_metrics.ignored_predictions_no_gt += met.ignored_predictions_no_gt
        all_metrics.total_unique_gt_ids += met.total_unique_gt_ids
        all_metrics.seeded_ids_count += met.seeded_ids_count

        print(
            f"[seq {seq}] mota={row['mota']:.3f}  match_rate={row['match_rate']:.3f}  "
            f"idsw={row['id_switches']}  fp={row['false_positives']}  fn={row['false_negatives']}  "
            f"mean_iou={row['mean_iou_when_matched']:.3f}"
        )
        print(
            f"           strict_mota={row['strict_mota']:.3f}  "
            f"strict_match_rate={row['strict_match_rate']:.3f}  "
            f"strict_fp={row['strict_false_positives']}  strict_fn={row['strict_false_negatives']}  "
            f"wrong_id_overlaps={row['wrong_id_overlaps']}"
        )

    all_row = metrics_to_row(
        run_prefix=run_prefix,
        label=args.run_name,
        seq="ALL",
        reid_thr=args.reid_thr,
        yolo_model_name=args.yolo_model,
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
        initial_gallery_frames=args.initial_gallery_frames,
        met=all_metrics,
        out_csv="",
        gt_mot_path=str(mot_root),
        pred_mot_path=str(mot_root),
    )
    summary_rows.append(all_row)

    summary_csv = out_dir / f"{run_prefix}__summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    trackeval_info = None
    if args.run_trackeval:
        if not args.trackeval_root:
            raise ValueError("--trackeval_root is required when --run_trackeval is set.")

        if args.trackeval_tracker_name is not None and str(args.trackeval_tracker_name).strip():
            tracker_name = str(args.trackeval_tracker_name).strip()
        else:
            tracker_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name).strip("_")
            if not tracker_name:
                tracker_name = "transreid_fixed_gallery"

        trackeval_fps = (
            float(args.trackeval_fps)
            if args.trackeval_fps is not None
            else (30.0 / max(1, int(args.stride)))
        )

        trackeval_info = prepare_mot_exports_for_trackeval(
            mot_export_dir=mot_root,
            trackeval_root=Path(args.trackeval_root),
            tracker_name=tracker_name,
            benchmark=args.trackeval_benchmark,
            split=args.trackeval_split,
            sequences=sequences,
            fps=trackeval_fps,
            width=args.trackeval_width,
            height=args.trackeval_height,
            overwrite=bool(args.trackeval_overwrite),
        )

        trackeval_summary = {
            "tracker_name": tracker_name,
            "benchmark": args.trackeval_benchmark,
            "split": args.trackeval_split,
            "threshold": args.trackeval_threshold,
            "fps": trackeval_fps,
            **trackeval_info,
        }
        (out_dir / f"{run_prefix}__trackeval.json").write_text(
            json.dumps(trackeval_summary, indent=2),
            encoding="utf-8",
        )

        run_trackeval_now(
            trackeval_root=Path(args.trackeval_root),
            tracker_name=tracker_name,
            benchmark=args.trackeval_benchmark,
            split=args.trackeval_split,
            threshold=args.trackeval_threshold,
        )

    print(
        f"[ALL] mota={all_row['mota']:.3f}  match_rate={all_row['match_rate']:.3f}  "
        f"idsw={all_row['id_switches']}  fp={all_row['false_positives']}  "
        f"fn={all_row['false_negatives']}  mean_iou={all_row['mean_iou_when_matched']:.3f}"
    )
    print(
        f"      strict_mota={all_row['strict_mota']:.3f}  "
        f"strict_match_rate={all_row['strict_match_rate']:.3f}  "
        f"strict_fp={all_row['strict_false_positives']}  strict_fn={all_row['strict_false_negatives']}  "
        f"wrong_id_overlaps={all_row['wrong_id_overlaps']}"
    )
    print(
        f"      seed_coverage={all_row['seed_coverage']:.3f}  "
        f"seed_skipped_small={all_row['seed_skipped_small']}  "
        f"seed_skipped_overlap={all_row['seed_skipped_overlap']}  "
        f"ignored_predictions_no_gt={all_row['ignored_predictions_no_gt']}"
    )
    print(f"[summary] {summary_csv}")
    print(f"[mot] {mot_root}")
    if trackeval_info is not None:
        print(f"[TrackEval tracker] {trackeval_info.get('tracker_root')}")


if __name__ == "__main__":
    main()
