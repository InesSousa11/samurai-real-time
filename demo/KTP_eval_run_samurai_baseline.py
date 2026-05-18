#!/usr/bin/env python3
# KTP_eval_run_samurai_baseline.py
# Single-config KTP evaluation for the SAMURAI baseline:
#   - evaluates one fixed configuration across one or more sequences
#   - saves per-sequence frame CSVs
#   - saves one summary JSON
#   - exports GT/pred in MOT-style txt for later HOTA / IDF1 / MOTA evaluation
#   - optional quick PNG plots
#
# KTP structure expected:
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
# GT line format assumed:
#   <timestamp>: <id> <x> <y> <w> <h>, <id> <x> <y> <w> <h>, ...

import sys
import re
import csv
import json
import time
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

# ---------------- Repo root + imports ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "demo") else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_camera_predictor

# ---------------- Hard-coded checkpoint/config ----------------
CKPT_PATH = (REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt").resolve()
# In this older SAMURAI baseline branch, Hydra expects the config name
# relative to the sam2 package, not an absolute filesystem path.
CFG_NAME = "configs/samurai/sam2.1_hiera_s.yaml"
CFG_PATH = (REPO_ROOT / "sam2" / CFG_NAME).resolve()

# ---------------- Shared color palette ----------------
PALETTE_RGB = [
    (255,   0,   0),
    (  0, 255,   0),
    (  0,   0, 255),
    (255, 255,   0),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 128,   0),
    (128,   0, 255),
    (  0, 128, 255),
    (128, 255,   0),
    (255,   0, 128),
    (  0, 255, 128),
    (180,  60,  60),
    ( 60, 180,  60),
    ( 60,  60, 180),
    (255, 180,  60),
]

def _id_to_rgb(obj_id: int):
    return PALETTE_RGB[int(obj_id) % len(PALETTE_RGB)]

def _rgb_to_bgr(color_rgb):
    r, g, b = color_rgb
    return (int(b), int(g), int(r))

def draw_mask_overlay(rgb_frame, mask_bool_by_id, alpha=0.5):
    if rgb_frame is None:
        return rgb_frame

    h, w = rgb_frame.shape[:2]
    overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for obj_id, mask_bool in mask_bool_by_id.items():
        if mask_bool is None:
            continue
        if mask_bool.shape[:2] != (h, w):
            continue
        overlay_rgb[mask_bool] = _id_to_rgb(int(obj_id))

    return cv2.addWeighted(rgb_frame, 1.0, overlay_rgb, float(alpha), 0.0)

# ---------------- Helpers ----------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

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

def ts_from_filename_robust(p: Path) -> Optional[str]:
    m = _TS_LEAD_NUM.match(p.stem)
    if not m:
        return None
    return m.group(1)

def estimate_sequence_fps(frames: List[Path], ts_by_path: Dict[Path, str], fallback_fps: float = 15.0) -> float:
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
    fps = max(1.0, min(120.0, fps))
    return float(fps)

def iou_xyxy(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def bbox_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[int,int,int,int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return (x1, y1, x2, y2)

def xyxy_to_xywh(bb: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = bb
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

def clamp_bbox_xyxy(bb: Tuple[int,int,int,int], W: int, H: int) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = bb
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return (x1, y1, x2, y2)

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    if mask is None:
        return None

    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])

    if mask.ndim != 2:
        return None

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)

def logits_to_mask_bbox(logits: torch.Tensor) -> Optional[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    if logits is None or (not torch.is_tensor(logits)):
        return None

    lg = logits.detach()

    while lg.ndim > 2 and lg.shape[0] == 1:
        lg = lg[0]

    while lg.ndim > 2:
        lg = lg[0]

    if lg.ndim != 2:
        return None

    for thr in (0.0, -2.0, -4.0):
        m = (lg > thr)
        m_np = m.cpu().numpy().astype(bool)
        bb = mask_to_bbox(m_np)
        if bb is not None:
            return (m_np, bb)

    return None

def sync_reid_threshold(predictor, reid_thr: Optional[float]) -> None:
    if reid_thr is None:
        return
    try:
        predictor.reid_thr = float(reid_thr)
    except Exception:
        pass

    try:
        cs = getattr(predictor, "condition_state", None)
        if isinstance(cs, dict):
            cs["reid_thr"] = float(reid_thr)
    except Exception:
        pass

def _set_attr_and_state(predictor, name: str, value) -> None:
    """
    Set a runtime parameter both as a predictor attribute and inside
    condition_state when condition_state already exists.
    """
    if value is None:
        return
    try:
        setattr(predictor, name, value)
    except Exception:
        pass
    try:
        cs = getattr(predictor, "condition_state", None)
        if isinstance(cs, dict):
            cs[name] = value
    except Exception:
        pass


def sync_runtime_thresholds_to_state(predictor) -> None:
    """
    Copy relevant runtime thresholds from predictor attributes to
    condition_state. This is useful after load_first_frame(), because
    condition_state may not exist before that point.
    """
    keys = [
        "stable_frames_threshold",
        "stable_ious_threshold",
        "min_obj_score_logits",
        "kf_score_weight",
        "memory_bank_iou_threshold",
        "memory_bank_obj_score_threshold",
        "memory_bank_kf_score_threshold",
    ]
    try:
        cs = getattr(predictor, "condition_state", None)
        if not isinstance(cs, dict):
            return
        for key in keys:
            if hasattr(predictor, key):
                cs[key] = getattr(predictor, key)
    except Exception:
        pass


def set_predictor_thresholds(
    predictor,
    stable_frames_threshold: Optional[int] = None,
    stable_ious_threshold: Optional[float] = None,
    min_obj_score_logits: Optional[float] = None,
    kf_score_weight: Optional[float] = None,
    memory_bank_iou_threshold: Optional[float] = None,
    memory_bank_obj_score_threshold: Optional[float] = None,
    memory_bank_kf_score_threshold: Optional[float] = None,
):
    _set_attr_and_state(predictor, "stable_frames_threshold", int(stable_frames_threshold) if stable_frames_threshold is not None else None)
    _set_attr_and_state(predictor, "stable_ious_threshold", float(stable_ious_threshold) if stable_ious_threshold is not None else None)
    _set_attr_and_state(predictor, "min_obj_score_logits", float(min_obj_score_logits) if min_obj_score_logits is not None else None)
    _set_attr_and_state(predictor, "kf_score_weight", float(kf_score_weight) if kf_score_weight is not None else None)
    _set_attr_and_state(predictor, "memory_bank_iou_threshold", float(memory_bank_iou_threshold) if memory_bank_iou_threshold is not None else None)
    _set_attr_and_state(predictor, "memory_bank_obj_score_threshold", float(memory_bank_obj_score_threshold) if memory_bank_obj_score_threshold is not None else None)
    _set_attr_and_state(predictor, "memory_bank_kf_score_threshold", float(memory_bank_kf_score_threshold) if memory_bank_kf_score_threshold is not None else None)

    sync_runtime_thresholds_to_state(predictor)

def print_predictor_thresholds(predictor):
    keys = [
        "stable_frames_threshold",
        "stable_ious_threshold",
        "min_obj_score_logits",
        "kf_score_weight",
        "memory_bank_iou_threshold",
        "memory_bank_obj_score_threshold",
        "memory_bank_kf_score_threshold",
        "samurai_mode",
    ]
    print("\n[predictor attrs]")
    for k in keys:
        print(f"  {k}: {getattr(predictor, k, None)}")
    print("")

# ---------------- GT parsing ----------------
def parse_gt2d_file(gt_path: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    d: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
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

            dets_raw = [r.strip() for r in rest.split(",") if r.strip()]
            dets: List[Tuple[int, float, float, float, float]] = []
            for dr in dets_raw:
                parts = dr.split()
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
            d[ts] = dets
    return d

# ---------------- MOT-style matching ----------------
def match_frame_hungarian(
    gt_bb_by_id: Dict[int, Tuple[int, int, int, int]],
    pred_bbox_by_id: Dict[int, Tuple[int, int, int, int]],
    iou_match_thr: float,
) -> Tuple[Dict[int, Optional[int]], Dict[int, float], set, set]:
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

# ---------------- Metrics ----------------
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

# ---------------- Derived metrics ----------------
def summarize_reacq(gaps: List[int]) -> Tuple[int, float, float, int]:
    if not gaps:
        return (0, 0.0, 0.0, 0)
    gaps_sorted = sorted(gaps)
    n = len(gaps_sorted)
    mean = float(sum(gaps_sorted) / n)
    med = float(gaps_sorted[n // 2]) if (n % 2 == 1) else float(0.5 * (gaps_sorted[n // 2 - 1] + gaps_sorted[n // 2]))
    mx = int(max(gaps_sorted))
    return (n, mean, med, mx)

def metrics_to_row(
    run_prefix: str,
    label: str,
    seq: str,
    reid_backend: str,
    sf: int,
    si: float,
    mo: float,
    kf: float,
    mbi: float,
    mbo: float,
    mbkf: float,
    met: SeqMetrics,
    out_csv: str,
    gt_mot_path: str,
    pred_mot_path: str,
) -> dict:
    reacq_n, reacq_mean, reacq_med, reacq_max = summarize_reacq(met.reacq_gaps)

    misses = met.false_negatives
    denom_gt = met.eligible_gt_boxes

    match_rate = (met.matches / denom_gt) if denom_gt > 0 else 0.0
    miss_rate = (misses / denom_gt) if denom_gt > 0 else 0.0
    mean_iou = (met.iou_sum / met.iou_count) if met.iou_count > 0 else 0.0
    id_switches_per_match = (met.id_switches / met.matches) if met.matches > 0 else 0.0
    id_switches_per_gt = (met.id_switches / denom_gt) if denom_gt > 0 else 0.0
    reacq_rate_per_gt = (reacq_n / denom_gt) if denom_gt > 0 else 0.0
    seed_coverage = (met.seeded_ids_count / met.total_unique_gt_ids) if met.total_unique_gt_ids > 0 else 0.0

    precision = (met.matches / (met.matches + met.false_positives)) if (met.matches + met.false_positives) > 0 else 0.0
    recall = (met.matches / (met.matches + met.false_negatives)) if (met.matches + met.false_negatives) > 0 else 0.0
    mota = 1.0 - ((met.false_negatives + met.false_positives + met.id_switches) / denom_gt) if denom_gt > 0 else 0.0

    return {
        "run": run_prefix,
        "label": label,
        "seq": seq,
        "reid_backend": reid_backend,
        "stable_frames_threshold": sf,
        "stable_ious_threshold": si,
        "min_obj_score_logits": mo,
        "kf_score_weight": kf,
        "memory_bank_iou_threshold": mbi,
        "memory_bank_obj_score_threshold": mbo,
        "memory_bank_kf_score_threshold": mbkf,

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
        "reacq_rate_per_gt": reacq_rate_per_gt,
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

# ---------------- Core run for one sequence ----------------
@torch.inference_mode()
def run_sequence(
    seq_name: str,
    ktp_root: Path,
    predictor,
    out_csv_path: Path,
    reid_backend_name: str,
    mot_gt_path: Path,
    mot_pred_path: Path,
    rotate_deg: int = 0,
    stride: int = 1,
    max_frames: int = -1,
    visible_area_frac: float = 0.02,
    visible_min_h: int = 120,
    visible_min_w: int = 0,
    seed_overlap_iou_max: float = 0.10,
    iou_match_thr: float = 0.30,
    eval_seed_frame: bool = False,
    no_display: bool = True,
    display_scale: float = 1.0,
    save_video: bool = True,
    save_video_fps: Optional[float] = None,
    alpha: float = 0.5,
) -> SeqMetrics:
    img_dir = ktp_root / "images" / seq_name / "rgb"
    gt_path = ktp_root / "ground_truth" / f"{seq_name}_gt2D.txt"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_map = parse_gt2d_file(gt_path)

    all_gt_ids = set()
    for dets in gt_map.values():
        for gid, *_ in dets:
            all_gt_ids.add(int(gid))

    frames_all = list(img_dir.glob("*.jpg"))
    if len(frames_all) == 0:
        raise RuntimeError(f"No .jpg frames found in {img_dir}")

    items = []
    for p in frames_all:
        ts_str = ts_from_filename_robust(p)
        if ts_str is None:
            continue
        try:
            ts_f = float(ts_str)
        except Exception:
            continue
        items.append((ts_f, ts_str, p))

    if len(items) == 0:
        raise RuntimeError(f"Found .jpg files in {img_dir} but none had a parseable leading numeric timestamp.")

    items.sort(key=lambda t: t[0])

    frames: List[Path] = []
    ts_by_path: Dict[Path, str] = {}
    seen_ts = set()
    for _, ts_str, p in items:
        if ts_str in seen_ts:
            continue
        seen_ts.add(ts_str)
        frames.append(p)
        ts_by_path[p] = ts_str

    if stride > 1:
        frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]
    if len(frames) == 0:
        raise RuntimeError(f"No frames left after stride/max_frames in {img_dir}")

    bgr0 = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    bgr0 = rotate_frame(bgr0, rotate_deg)
    H, W = bgr0.shape[:2]
    rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)

    seq_fps = float(save_video_fps) if save_video_fps is not None else estimate_sequence_fps(frames, ts_by_path, fallback_fps=15.0)

    video_writer = None
    if save_video:
        video_out_path = out_csv_path.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, seq_fps, (W, H))

    predictor.load_first_frame(rgb0)
    sync_runtime_thresholds_to_state(predictor)

    seeded: set = set()
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

    writer.writerow([f"# reid_backend: {reid_backend_name}"])
    writer.writerow([f"# predictor_internal: {{"
                     f"'stable_frames_threshold':{getattr(predictor,'stable_frames_threshold',None)}, "
                     f"'stable_ious_threshold':{getattr(predictor,'stable_ious_threshold',None)}, "
                     f"'min_obj_score_logits':{getattr(predictor,'min_obj_score_logits',None)}, "
                     f"'kf_score_weight':{getattr(predictor,'kf_score_weight',None)}, "
                     f"'memory_bank_iou_threshold':{getattr(predictor,'memory_bank_iou_threshold',None)}, "
                     f"'memory_bank_obj_score_threshold':{getattr(predictor,'memory_bank_obj_score_threshold',None)}, "
                     f"'memory_bank_kf_score_threshold':{getattr(predictor,'memory_bank_kf_score_threshold',None)}, "
                     f"'memory_bank_reid_threshold':{getattr(predictor,'memory_bank_reid_threshold',None)}, "
                     f"'reid_thr':{getattr(predictor,'reid_thr',None)}, "
                     f"'reid_gallery_max_size':{getattr(predictor,'reid_gallery_max_size',None)}, "
                     f"'reid_gallery_add_sim_threshold':{getattr(predictor,'reid_gallery_add_sim_threshold',None)}, "
                     f"'reid_gallery_add_cooldown':{getattr(predictor,'reid_gallery_add_cooldown',None)}, "
                     f"'reid_gallery_random_replace_prob':{getattr(predictor,'reid_gallery_random_replace_prob',None)}, "
                     f"'reid_gallery_random_replace_if_diverse_prob':{getattr(predictor,'reid_gallery_random_replace_if_diverse_prob',None)}, "
                     f"'reid_gallery_anchor_protect':{getattr(predictor,'reid_gallery_anchor_protect',None)}"
                     f"}}"])
    writer.writerow([f"# seed_rules: visible_area_frac={visible_area_frac}, visible_min_h={visible_min_h}, "
                     f"visible_min_w={visible_min_w}, seed_overlap_iou_max={seed_overlap_iou_max}, "
                     f"iou_match_thr={iou_match_thr}, stride={stride}, max_frames={max_frames}, "
                     f"eval_seed_frame={eval_seed_frame}"])
    writer.writerow([
        "seq","frame_idx","ts","t_sec",
        "gt_id","gt_x","gt_y","gt_w","gt_h","gt_area_frac",
        "eligible","seeded_now","seeded_already","seed_skip_reason",
        "pred_id","match_iou","id_switch_event","reacq_event","gap_len"
    ])

    win = f"KTP {seq_name}" if not no_display else None
    if win is not None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        t0 = float(ts_by_path.get(frames[0], "0.0"))
    except Exception:
        t0 = 0.0

    def seed_bbox(gt_id: int, bbox_xyxy, rgb_frame: np.ndarray, late: bool) -> bool:
        bbox = np.array([[bbox_xyxy[0], bbox_xyxy[1]], [bbox_xyxy[2], bbox_xyxy[3]]], dtype=np.float32)
        try:
            if not late:
                predictor.add_new_prompt(frame_idx=0, obj_id=int(gt_id), bbox=bbox)
                sync_runtime_thresholds_to_state(predictor)
            else:
                predictor.add_conditioning_frame(rgb_frame)
                sync_runtime_thresholds_to_state(predictor)

                predictor.add_new_prompt_during_track(
                    bbox=bbox,
                    if_new_target=True,
                    obj_id=int(gt_id),
                    labels=None,
                    clear_old_points=True,
                )
                sync_runtime_thresholds_to_state(predictor)
            return True
        except Exception as e:
            print(f"[DBG seed:FAIL] gid={gt_id} late={late} err={repr(e)}")
            traceback.print_exc()
            return False

    def get_obj_id_to_idx() -> Dict[int, int]:
        m = None
        if hasattr(predictor, "condition_state"):
            m = predictor.condition_state.get("obj_id_to_idx", None)
        if m is None and hasattr(predictor, "obj_id_to_idx"):
            m = getattr(predictor, "obj_id_to_idx", None)
        if m is None:
            return {}
        try:
            return {int(k): int(v) for k, v in dict(m).items()}
        except Exception:
            return {}

    def logits_for_obj_id(out_mask_logits, obj_id: int) -> Optional[torch.Tensor]:
        obj_id_to_idx = get_obj_id_to_idx()
        if obj_id not in obj_id_to_idx:
            return None
        obj_idx = obj_id_to_idx[obj_id]

        if out_mask_logits is None:
            return None

        if torch.is_tensor(out_mask_logits):
            if out_mask_logits.ndim < 3:
                return None
            if not (0 <= obj_idx < int(out_mask_logits.shape[0])):
                return None
            return out_mask_logits[obj_idx]

        if isinstance(out_mask_logits, (list, tuple)):
            if not (0 <= obj_idx < len(out_mask_logits)):
                return None
            return out_mask_logits[obj_idx] if torch.is_tensor(out_mask_logits[obj_idx]) else None

        return None

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

        gt_bb_by_id_all: Dict[int, Tuple[int,int,int,int]] = {}
        for (gid, x, y, w, h) in gt_dets:
            gt_bb_by_id_all[gid] = clamp_bbox_xyxy(bbox_xywh_to_xyxy(x, y, w, h), W, H)

        seeded_now_ids = set()
        seed_skip_reason_by_gid: Dict[int, str] = {}

        # -------- Seed before track --------
        for (gid, x, y, w, h) in gt_dets:
            if gid not in gt_states:
                gt_states[gid] = GTState()
            if gid in seeded:
                seed_skip_reason_by_gid[gid] = ""
                continue

            bb = gt_bb_by_id_all[gid]
            bw = max(0, bb[2] - bb[0])
            bh = max(0, bb[3] - bb[1])
            area = bw * bh
            area_frac = area / float(W * H + 1e-9)

            visible_ok = (area_frac >= float(visible_area_frac)) and (bh >= int(visible_min_h))
            if visible_min_w and int(visible_min_w) > 0:
                visible_ok = visible_ok and (bw >= int(visible_min_w))

            if not visible_ok:
                seed_skip_reason_by_gid[gid] = "small"
                metrics.seed_skipped_small += 1
                continue

            max_iou_other = 0.0
            for ogid, obb in gt_bb_by_id_all.items():
                if ogid == gid:
                    continue
                max_iou_other = max(max_iou_other, iou_xyxy(bb, obb))

            if max_iou_other > float(seed_overlap_iou_max):
                seed_skip_reason_by_gid[gid] = f"overlap(max_iou={max_iou_other:.3f})"
                metrics.seed_skipped_overlap += 1
                continue

            late = (fidx != 0)
            ok = seed_bbox(gid, bb, rgb_frame=rgb, late=late)
            if ok:
                seeded.add(gid)
                seeded_now_ids.add(gid)
                metrics.seeded_ids_count = len(seeded)
                seed_skip_reason_by_gid[gid] = ""
            else:
                seed_skip_reason_by_gid[gid] = "seed_failed"
                metrics.seed_failed += 1

        # -------- Eligible GT for evaluation --------
        eligible_gt_ids = set()
        for gid, *_ in gt_dets:
            if gid in seeded:
                if (gid in seeded_now_ids) and (not eval_seed_frame):
                    continue
                eligible_gt_ids.add(gid)

        gt_bb_by_id_eval = {gid: gt_bb_by_id_all[gid] for gid in eligible_gt_ids}
        metrics.eligible_gt_boxes += len(gt_bb_by_id_eval)

        # Export eligible GT to MOT txt
        mot_frame_idx = fidx + 1
        for gid in sorted(gt_bb_by_id_eval.keys()):
            x1, y1, w, h = xyxy_to_xywh(gt_bb_by_id_eval[gid])
            f_gt_mot.write(f"{mot_frame_idx},{gid},{x1},{y1},{w},{h},1,1,1,1\n")

        try:
            out_obj_ids, out_mask_logits = predictor.track(rgb)
        except Exception:
            out_obj_ids, out_mask_logits = [], None

        if out_obj_ids is None:
            out_obj_ids = []
        if torch.is_tensor(out_obj_ids):
            out_obj_ids = [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
        elif isinstance(out_obj_ids, (list, tuple)):
            out_obj_ids = [int(x) for x in out_obj_ids]
        else:
            out_obj_ids = [int(out_obj_ids)]

        pred_bbox_by_id: Dict[int, Tuple[int,int,int,int]] = {}
        pred_mask_by_id: Dict[int, np.ndarray] = {}

        if out_mask_logits is not None:
            for oid in out_obj_ids:
                logits = logits_for_obj_id(out_mask_logits, int(oid))
                if logits is None:
                    continue
                res = logits_to_mask_bbox(logits)
                if res is None:
                    continue
                mask_bool, bbp = res
                pred_mask_by_id[int(oid)] = mask_bool
                pred_bbox_by_id[int(oid)] = clamp_bbox_xyxy(bbp, W, H)

        # Export all predictions to MOT txt
        for pid in sorted(pred_bbox_by_id.keys()):
            x1, y1, w, h = xyxy_to_xywh(pred_bbox_by_id[pid])
            f_pred_mot.write(f"{mot_frame_idx},{pid},{x1},{y1},{w},{h},1,-1,-1,-1\n")

        # -------- One-to-one matching on eligible GT only --------
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

        for (gid, x, y, w, h) in gt_dets:
            st = gt_states.get(gid, GTState())
            eligible = gid in eligible_gt_ids
            cur = gt_to_pred.get(gid, None) if eligible else None

            idsw = 0
            reacq = 0

            if eligible:
                if cur is None:
                    if st.prev_pred is not None and (not st.in_gap):
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
            area = max(0, gt_bb[2] - gt_bb[0]) * max(0, gt_bb[3] - gt_bb[1])
            area_frac = area / float(W * H + 1e-9)

            try:
                t_sec = float(ts) - t0
            except Exception:
                t_sec = 0.0

            writer.writerow([
                seq_name, fidx, ts, f"{t_sec:.6f}",
                gid, f"{x:.3f}", f"{y:.3f}", f"{w:.3f}", f"{h:.3f}", f"{area_frac:.6f}",
                int(eligible),
                (1 if gid in seeded_now_ids else 0),
                (1 if gid in seeded else 0),
                seed_skip_reason_by_gid.get(gid, ""),
                (cur if cur is not None else ""),
                f"{gt_to_iou.get(gid, 0.0):.6f}",
                idsw,
                reacq,
                gt_states[gid].gap_len if gt_states[gid].in_gap else 0,
            ])

        vis_rgb = rgb.copy()
        vis_rgb = draw_mask_overlay(vis_rgb, pred_mask_by_id, alpha=alpha)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        for (gid, x, y, w, h) in gt_dets:
            bb = gt_bb_by_id_all[gid]
            color = (255, 255, 255) if gid in eligible_gt_ids else (120, 120, 120)
            cv2.rectangle(vis_bgr, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
            cv2.putText(
                vis_bgr,
                f"GT {gid}",
                (bb[0], max(0, bb[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        for pid, bb in pred_bbox_by_id.items():
            col = _rgb_to_bgr(_id_to_rgb(pid))
            cv2.rectangle(vis_bgr, (bb[0], bb[1]), (bb[2], bb[3]), col, 2)
            cv2.putText(
                vis_bgr,
                f"PR {pid}",
                (bb[0], bb[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                2,
                cv2.LINE_AA,
            )

        if video_writer is not None:
            video_writer.write(vis_bgr)

        if win is not None:
            disp = vis_bgr
            if display_scale != 1.0:
                disp = cv2.resize(disp, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    fcsv.close()
    f_gt_mot.close()
    f_pred_mot.close()

    if video_writer is not None:
        video_writer.release()

    if win is not None:
        cv2.destroyWindow(win)

    return metrics

# ---------------- Optional quick plots ----------------
def make_plots(summary_rows: List[dict], plots_dir: Path, title_prefix: str):
    safe_mkdir(plots_dir)
    if not summary_rows:
        return

    rows = [r for r in summary_rows if r.get("seq") == "ALL"]
    if not rows:
        rows = summary_rows

    labels = []
    match_rate = []
    mota = []
    idsw = []
    reacq_mean = []

    for r in rows:
        labels.append(r["label"])
        match_rate.append(float(r["match_rate"]))
        mota.append(float(r["mota"]))
        idsw.append(int(float(r["id_switches"])))
        reacq_mean.append(float(r["reacq_mean_frames"]))

    x = np.arange(len(labels))

    plt.figure()
    plt.plot(x, match_rate, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("match_rate")
    plt.title(f"{title_prefix} - match_rate")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "match_rate.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(x, mota, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("MOTA")
    plt.title(f"{title_prefix} - mota")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "mota.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(x, idsw, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("ID switches")
    plt.title(f"{title_prefix} - id_switches")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "id_switches.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(idsw, match_rate)
    plt.xlabel("ID switches")
    plt.ylabel("match_rate")
    plt.title(f"{title_prefix} - pareto")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "pareto_idsw_vs_matchrate.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(x, reacq_mean, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("reacq_mean_frames")
    plt.title(f"{title_prefix} - reacq_mean_frames")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "reacq_mean_frames.png"), dpi=150)
    plt.close()

# ---------------- Main ----------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktp_root", type=str, required=True, help="Path to KTP root folder")
    ap.add_argument("--sequences", type=str, default="Arc,Rotation,Still,Translation",
                    help="Comma-separated sequence names")

    ap.add_argument("--stable_frames_threshold", type=int, default=15)
    ap.add_argument("--stable_ious_threshold", type=float, default=0.30)
    ap.add_argument("--min_obj_score_logits", type=float, default=0.5)
    ap.add_argument("--kf_score_weight", type=float, default=0.25)
    ap.add_argument("--memory_bank_iou_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_obj_score_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_kf_score_threshold", type=float, default=0.0)

    ap.add_argument("--rotate", type=int, default=0, help="Rotate frames by {0,90,180,270} degrees")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    ap.add_argument("--max_frames", type=int, default=-1, help="Limit frames per sequence")

    ap.add_argument("--visible_area_frac", type=float, default=0.02)
    ap.add_argument("--visible_min_h", type=int, default=120)
    ap.add_argument("--visible_min_w", type=int, default=0)

    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.10)
    ap.add_argument("--iou_match_thr", type=float, default=0.30)
    ap.add_argument("--eval_seed_frame", action="store_true",
                    help="If set, evaluate GT boxes already on the same frame they are seeded. Default: false")

    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--run_name", type=str, default="samurai_eval", help="Prefix for output files")

    ap.add_argument("--no_display", action="store_true")
    ap.add_argument("--display_scale", type=float, default=1.0)

    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--save_video_fps", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.5)

    ap.add_argument("--make_plots", action="store_true")
    args = ap.parse_args()

    ktp_root = Path(args.ktp_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(out_dir)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")

    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    sf   = args.stable_frames_threshold
    si   = args.stable_ious_threshold
    mo   = args.min_obj_score_logits
    kf   = args.kf_score_weight
    mbi  = args.memory_bank_iou_threshold
    mbo  = args.memory_bank_obj_score_threshold
    mbkf = args.memory_bank_kf_score_threshold
    label = (
        "samurai_baseline"
        f"__sf{sf}"
        f"_si{si:g}"
        f"_mo{mo:g}"
        f"_kf{kf:g}"
        f"_mbi{mbi:g}"
        f"_mbo{mbo:g}"
        f"_mbkf{mbkf:g}"
    )

    print("[paths]")
    print("  REPO_ROOT:", REPO_ROOT)
    print("  CKPT     :", CKPT_PATH)
    print("  CFG      :", CFG_PATH)
    print("  KTP_ROOT :", ktp_root)
    print("  OUT_DIR  :", out_dir)
    print("  MODEL    : SAMURAI baseline")
    print("  RUN_NAME :", args.run_name)
    print(f"[eval] 1 config x {len(seqs)} sequences")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{args.run_name}_{label}_{run_id}"

    autocast_cm = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()

    summary_rows: List[dict] = []
    per_sequence_rows: List[dict] = []

    plots_dir = out_dir / "plots" / run_prefix
    mot_dir = out_dir / "mot_exports" / run_prefix
    safe_mkdir(mot_dir)
    if args.make_plots:
        safe_mkdir(plots_dir)

    all_metrics = SeqMetrics()

    for seq in seqs:
        with autocast_cm:
            predictor = build_sam2_camera_predictor(
                CFG_NAME,
                str(CKPT_PATH),
            )

        set_predictor_thresholds(
            predictor,
            stable_frames_threshold=sf,
            stable_ious_threshold=si,
            min_obj_score_logits=mo,
            kf_score_weight=kf,
            memory_bank_iou_threshold=mbi,
            memory_bank_obj_score_threshold=mbo,
            memory_bank_kf_score_threshold=mbkf,
        )

        print("Applied internal thresholds:",
              getattr(predictor, "stable_frames_threshold", None),
              getattr(predictor, "stable_ious_threshold", None),
              getattr(predictor, "min_obj_score_logits", None),
              getattr(predictor, "kf_score_weight", None),
              getattr(predictor, "memory_bank_iou_threshold", None),
              getattr(predictor, "memory_bank_obj_score_threshold", None),
              getattr(predictor, "memory_bank_kf_score_threshold", None))

        if seq == seqs[0]:
            print("SAMURAI mode:", getattr(predictor, "samurai_mode", None))
            print_predictor_thresholds(predictor)

        out_csv = out_dir / f"{run_prefix}__{seq}.csv"
        gt_mot_path = mot_dir / f"{seq}_gt.txt"
        pred_mot_path = mot_dir / f"{seq}_pred.txt"

        with autocast_cm:
            met = run_sequence(
                seq_name=seq,
                ktp_root=ktp_root,
                predictor=predictor,
                out_csv_path=out_csv,
                reid_backend_name="samurai_baseline",
                mot_gt_path=gt_mot_path,
                mot_pred_path=pred_mot_path,
                rotate_deg=args.rotate,
                stride=args.stride,
                max_frames=args.max_frames,
                visible_area_frac=args.visible_area_frac,
                visible_min_h=args.visible_min_h,
                visible_min_w=args.visible_min_w,
                seed_overlap_iou_max=args.seed_overlap_iou_max,
                iou_match_thr=args.iou_match_thr,
                eval_seed_frame=args.eval_seed_frame,
                no_display=args.no_display,
                display_scale=args.display_scale,
                save_video=args.save_video,
                save_video_fps=args.save_video_fps,
                alpha=args.alpha,
            )

        row = metrics_to_row(
            run_prefix=run_prefix,
            label=label,
            seq=seq,
            reid_backend="samurai_baseline",
            sf=sf, si=si, mo=mo, kf=kf, mbi=mbi, mbo=mbo, mbkf=mbkf,
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
            f"  [seq {seq}] "
            f"mota={row['mota']:.3f}  "
            f"match_rate={row['match_rate']:.3f}  "
            f"idsw={row['id_switches']}  "
            f"fp={row['false_positives']}  "
            f"fn={row['false_negatives']}  "
            f"mean_iou={row['mean_iou_when_matched']:.3f}"
        )

    row_all = metrics_to_row(
        run_prefix=run_prefix,
        label=label,
        seq="ALL",
        reid_backend="samurai_baseline",
        sf=sf, si=si, mo=mo, kf=kf, mbi=mbi, mbo=mbo, mbkf=mbkf,
        met=all_metrics,
        out_csv="",
        gt_mot_path="",
        pred_mot_path="",
    )
    summary_rows.append(row_all)

    print(
        f"  [ALL] "
        f"mota={row_all['mota']:.3f}  "
        f"match_rate={row_all['match_rate']:.3f}  "
        f"idsw={row_all['id_switches']}  "
        f"fp={row_all['false_positives']}  "
        f"fn={row_all['false_negatives']}  "
        f"mean_iou={row_all['mean_iou_when_matched']:.3f}"
    )
    print(
        f"        seed_coverage={row_all['seed_coverage']:.3f}  "
        f"seed_skipped_small={row_all['seed_skipped_small']}  "
        f"seed_skipped_overlap={row_all['seed_skipped_overlap']}"
    )

    summary_json_path = out_dir / f"{run_prefix}__summary.json"
    json_payload = {
        "run": run_prefix,
        "created_at": run_id,
        "repo_root": str(REPO_ROOT),
        "checkpoint": str(CKPT_PATH),
        "config": str(CFG_PATH),
        "ktp_root": str(ktp_root),
        "model": "samurai_baseline",
        "label": label,
        "sequences": seqs,
        "settings": {
            "stable_frames_threshold": sf,
            "stable_ious_threshold": si,
            "min_obj_score_logits": mo,
            "kf_score_weight": kf,
            "memory_bank_iou_threshold": mbi,
            "memory_bank_obj_score_threshold": mbo,
            "memory_bank_kf_score_threshold": mbkf,
            "rotate": args.rotate,
            "stride": args.stride,
            "max_frames": args.max_frames,
            "visible_area_frac": args.visible_area_frac,
            "visible_min_h": args.visible_min_h,
            "visible_min_w": args.visible_min_w,
            "seed_overlap_iou_max": args.seed_overlap_iou_max,
            "iou_match_thr": args.iou_match_thr,
            "eval_seed_frame": args.eval_seed_frame,
            "save_video": args.save_video,
            "save_video_fps": args.save_video_fps,
            "alpha": args.alpha,
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
        json.dump(json_payload, f, indent=2)

    if args.make_plots:
        make_plots(summary_rows, plots_dir=plots_dir, title_prefix=run_prefix)

    print("\n[done]")
    print("  summary json:", summary_json_path)
    print("  MOT exports :", mot_dir)
    if args.make_plots:
        print("  plots dir   :", plots_dir)
    print("Important:")
    print("  - This script now gives trustworthy MOTA-style variables.")
    print("  - For IDF1 and HOTA, use the exported MOT txt files with TrackEval.")

if __name__ == "__main__":
    main()