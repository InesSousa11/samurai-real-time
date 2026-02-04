# 2_ktp_threshold_sweep.py
# KTP sweep (internal SAMURAI thresholds) + ID-switch evaluation
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
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import traceback

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

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
CFG_PATH  = (REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()

# ---------------- Helpers ----------------
def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

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
    """
    Extracts the leading numeric chunk from the filename stem.
    Examples:
      1339072479.587683490.jpg        -> "1339072479.587683490"
      1339072479.587683490(1).jpg     -> "1339072479.587683490"
      1339072525.535337309rro.jpg     -> "1339072525.535337309"
      foo.jpg                          -> None
    """
    m = _TS_LEAD_NUM.match(p.stem)
    if not m:
        return None
    return m.group(1)

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
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def bbox_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[int,int,int,int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return (x1, y1, x2, y2)

def clamp_bbox_xyxy(bb: Tuple[int,int,int,int], W: int, H: int) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = bb
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2))
    y2 = max(0, min(H,   y2))
    if x2 < x1: x2 = x1
    if y2 < y1: y2 = y1
    return (x1,y1,x2,y2)

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min()); x2 = int(xs.max()) + 1
    y1 = int(ys.min()); y2 = int(ys.max()) + 1
    return (x1,y1,x2,y2)

def logits_to_mask_bbox(logits: torch.Tensor) -> Optional[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    """
    Convert predicted mask logits to boolean mask + bbox.

    SAM2/SAMURAI can output negative logits even for valid masks,
    so strict >0 can be empty. We try thresholds: 0, -2, -4.
    """
    if logits is None or (not torch.is_tensor(logits)):
        return None

    if logits.ndim == 3:
        lg = logits[0]
    elif logits.ndim == 2:
        lg = logits
    else:
        return None

    lg = lg.detach()

    for thr in (0.0, -2.0, -4.0):
        m = (lg > thr)
        m_np = m.cpu().numpy().astype(bool)
        bb = mask_to_bbox(m_np)
        if bb is not None:
            return (m_np, bb)

    return None

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
    # Set only if attr exists
    def _set(name, val):
        if val is None:
            return
        if hasattr(predictor, name):
            setattr(predictor, name, val)

    _set("stable_frames_threshold", int(stable_frames_threshold) if stable_frames_threshold is not None else None)
    _set("stable_ious_threshold", float(stable_ious_threshold) if stable_ious_threshold is not None else None)
    _set("min_obj_score_logits", float(min_obj_score_logits) if min_obj_score_logits is not None else None)
    _set("kf_score_weight", float(kf_score_weight) if kf_score_weight is not None else None)
    _set("memory_bank_iou_threshold", float(memory_bank_iou_threshold) if memory_bank_iou_threshold is not None else None)
    _set("memory_bank_obj_score_threshold", float(memory_bank_obj_score_threshold) if memory_bank_obj_score_threshold is not None else None)
    _set("memory_bank_kf_score_threshold", float(memory_bank_kf_score_threshold) if memory_bank_kf_score_threshold is not None else None)

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
    """
    Returns dict: ts_string -> list of (gt_id, x, y, w, h)
    """
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
                    x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                    dets.append((gid, x, y, w, h))
                except Exception:
                    continue
            d[ts] = dets
    return d

# ---------------- Metrics ----------------
@dataclass
class GTState:
    prev_pred: Optional[int] = None
    in_gap: bool = False
    gap_len: int = 0

@dataclass
class SeqMetrics:
    frames: int = 0
    gt_boxes: int = 0
    matches: int = 0
    id_switches: int = 0
    reacq_events: int = 0
    reacq_gaps: List[int] = None
    iou_sum: float = 0.0
    iou_count: int = 0

    seed_skipped_overlap: int = 0
    seed_skipped_small: int = 0

    def __post_init__(self):
        if self.reacq_gaps is None:
            self.reacq_gaps = []

# ---------------- Core run for one sequence ----------------
@torch.inference_mode()
def run_sequence(
    seq_name: str,
    ktp_root: Path,
    predictor,
    out_csv_path: Path,
    rotate_deg: int = 0,
    stride: int = 1,
    max_frames: int = -1,
    visible_area_frac: float = 0.02,
    visible_min_h: int = 120,
    visible_min_w: int = 0,
    seed_overlap_iou_max: float = 0.10,
    iou_match_thr: float = 0.30,
    no_display: bool = True,
    display_scale: float = 1.0,
) -> SeqMetrics:
    """
    Oracle seeding:
      - visible enough: area_frac >= visible_area_frac AND height >= visible_min_h (and optional width)
      - non-overlapping enough: max IoU with other GT bboxes <= seed_overlap_iou_max

    IMPORTANT:
    - Metrics are computed from RAW masks returned by predictor.track (no gating).
    - We map obj_id -> obj_idx using predictor.condition_state["obj_id_to_idx"].
    """
    img_dir = ktp_root / "images" / seq_name / "rgb"
    gt_path = ktp_root / "ground_truth" / f"{seq_name}_gt2D.txt"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_map = parse_gt2d_file(gt_path)

    # ---- collect, sort, dedup frames ----
    frames_all = list(img_dir.glob("*.jpg"))
    if len(frames_all) == 0:
        raise RuntimeError(f"No .jpg frames found in {img_dir}")

    items = []  # (ts_float, ts_str, path)
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

    # ---- Load first frame ----
    bgr0 = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    bgr0 = rotate_frame(bgr0, rotate_deg)
    H, W = bgr0.shape[:2]
    rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)

    predictor.load_first_frame(rgb0)

    seeded: set = set()
    gt_states: Dict[int, GTState] = {}
    metrics = SeqMetrics()

    # output csv
    safe_mkdir(out_csv_path.parent)
    fcsv = out_csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)

    # record predictor internal params in header
    writer.writerow([f"# predictor_internal: {{"
                     f"'stable_frames_threshold':{getattr(predictor,'stable_frames_threshold',None)}, "
                     f"'stable_ious_threshold':{getattr(predictor,'stable_ious_threshold',None)}, "
                     f"'min_obj_score_logits':{getattr(predictor,'min_obj_score_logits',None)}, "
                     f"'kf_score_weight':{getattr(predictor,'kf_score_weight',None)}, "
                     f"'memory_bank_iou_threshold':{getattr(predictor,'memory_bank_iou_threshold',None)}, "
                     f"'memory_bank_obj_score_threshold':{getattr(predictor,'memory_bank_obj_score_threshold',None)}, "
                     f"'memory_bank_kf_score_threshold':{getattr(predictor,'memory_bank_kf_score_threshold',None)}"
                     f"}}"])
    writer.writerow([f"# seed_rules: visible_area_frac={visible_area_frac}, visible_min_h={visible_min_h}, "
                     f"visible_min_w={visible_min_w}, seed_overlap_iou_max={seed_overlap_iou_max}, "
                     f"iou_match_thr={iou_match_thr}, stride={stride}, max_frames={max_frames}"])
    writer.writerow([
        "seq","frame_idx","ts","t_sec",
        "gt_id","gt_x","gt_y","gt_w","gt_h","gt_area_frac",
        "seeded_now","seeded_already","seed_skip_reason",
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
            else:
                predictor.add_conditioning_frame(rgb_frame)
                predictor.add_new_prompt_during_track(
                    bbox=bbox,
                    if_new_target=True,
                    obj_id=int(gt_id),
                    labels=None,
                    clear_old_points=True,
                )
            return True
        except Exception as e:
            print(f"[DBG seed:FAIL] gid={gt_id} late={late} err={repr(e)}")
            traceback.print_exc()
            return False

    # obj_id -> obj_idx mapping
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

        # Tensor: (B,1,H,W) in obj_idx order
        if torch.is_tensor(out_mask_logits):
            if out_mask_logits.ndim < 3:
                return None
            if not (0 <= obj_idx < int(out_mask_logits.shape[0])):
                return None
            return out_mask_logits[obj_idx]

        # list/tuple: assume obj_idx order
        if isinstance(out_mask_logits, (list, tuple)):
            if not (0 <= obj_idx < len(out_mask_logits)):
                return None
            return out_mask_logits[obj_idx] if torch.is_tensor(out_mask_logits[obj_idx]) else None

        return None

    # ---- main loop ----
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
        metrics.gt_boxes += len(gt_dets)

        # clamped GT bboxes for overlap tests
        gt_bb_by_id: Dict[int, Tuple[int,int,int,int]] = {}
        for (gid, x, y, w, h) in gt_dets:
            gt_bb_by_id[gid] = clamp_bbox_xyxy(bbox_xywh_to_xyxy(x, y, w, h), W, H)

        seeded_now_ids = set()
        seed_skip_reason_by_gid: Dict[int, str] = {}

        # ---- oracle seeding with overlap guard ----
        for (gid, x, y, w, h) in gt_dets:
            if gid not in gt_states:
                gt_states[gid] = GTState()
            if gid in seeded:
                seed_skip_reason_by_gid[gid] = ""
                continue

            bb = gt_bb_by_id[gid]
            bw = max(0, bb[2]-bb[0])
            bh = max(0, bb[3]-bb[1])
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
            for ogid, obb in gt_bb_by_id.items():
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
                seed_skip_reason_by_gid[gid] = ""
            else:
                seed_skip_reason_by_gid[gid] = "seed_failed"

        # ---- tracking (RAW output) ----
        try:
            out_obj_ids, out_mask_logits = predictor.track(rgb)
        except Exception:
            out_obj_ids, out_mask_logits = [], None

        # normalize ids list
        if out_obj_ids is None:
            out_obj_ids = []
        if torch.is_tensor(out_obj_ids):
            out_obj_ids = [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
        elif isinstance(out_obj_ids, (list, tuple)):
            out_obj_ids = [int(x) for x in out_obj_ids]
        else:
            out_obj_ids = [int(out_obj_ids)]

        # ---- predicted bboxes keyed by obj_id (RAW, no gating) ----
        pred_bbox_by_id: Dict[int, Tuple[int,int,int,int]] = {}
        if out_mask_logits is not None:
            for oid in out_obj_ids:
                logits = logits_for_obj_id(out_mask_logits, int(oid))
                if logits is None:
                    continue
                res = logits_to_mask_bbox(logits)
                if res is None:
                    continue
                _, bbp = res
                pred_bbox_by_id[int(oid)] = clamp_bbox_xyxy(bbp, W, H)

        active_pred_ids = list(pred_bbox_by_id.keys())

        # ---- GT->pred matching (greedy per GT) ----
        gt_to_pred: Dict[int, Optional[int]] = {}
        gt_to_iou: Dict[int, float] = {}

        for (gid, x, y, w, h) in gt_dets:
            gt_bb = gt_bb_by_id[gid]
            best_pid = None
            best_iou = 0.0
            for pid in active_pred_ids:
                pb = pred_bbox_by_id.get(pid, None)
                if pb is None:
                    continue
                i = iou_xyxy(gt_bb, pb)
                if i > best_iou:
                    best_iou = i
                    best_pid = pid

            if best_pid is not None and best_iou >= float(iou_match_thr):
                gt_to_pred[gid] = best_pid
                gt_to_iou[gid] = best_iou
                metrics.matches += 1
                metrics.iou_sum += best_iou
                metrics.iou_count += 1
            else:
                gt_to_pred[gid] = None
                gt_to_iou[gid] = 0.0

        # ---- ID switches + reacq ----
        for (gid, x, y, w, h) in gt_dets:
            st = gt_states.get(gid, GTState())
            cur = gt_to_pred.get(gid, None)

            idsw = 0
            reacq = 0

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

            gt_bb = gt_bb_by_id[gid]
            area = max(0, gt_bb[2]-gt_bb[0]) * max(0, gt_bb[3]-gt_bb[1])
            area_frac = area / float(W * H + 1e-9)

            try:
                t_sec = float(ts) - t0
            except Exception:
                t_sec = 0.0

            writer.writerow([
                seq_name, fidx, ts, f"{t_sec:.6f}",
                gid, f"{x:.3f}", f"{y:.3f}", f"{w:.3f}", f"{h:.3f}", f"{area_frac:.6f}",
                (1 if gid in seeded_now_ids else 0),
                (1 if gid in seeded else 0),
                seed_skip_reason_by_gid.get(gid, ""),
                (cur if cur is not None else ""),
                f"{gt_to_iou.get(gid, 0.0):.6f}",
                idsw,
                reacq,
                gt_states[gid].gap_len if gt_states[gid].in_gap else 0,
            ])

        # ---- optional display ----
        if win is not None:
            disp = bgr.copy()

            for (gid, x, y, w, h) in gt_dets:
                bb = gt_bb_by_id[gid]
                cv2.rectangle(disp, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 2)
                cv2.putText(disp, f"GT {gid}", (bb[0], max(0, bb[1]-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            for pid, bb in pred_bbox_by_id.items():
                col = (0,255,0)
                cv2.rectangle(disp, (bb[0], bb[1]), (bb[2], bb[3]), col, 2)
                cv2.putText(disp, f"PR {pid}", (bb[0], bb[1]+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            if display_scale != 1.0:
                disp = cv2.resize(disp, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    fcsv.close()
    if win is not None:
        cv2.destroyWindow(win)

    return metrics

def summarize_reacq(gaps: List[int]) -> Tuple[int, float, float, int]:
    if not gaps:
        return (0, 0.0, 0.0, 0)
    gaps_sorted = sorted(gaps)
    n = len(gaps_sorted)
    mean = float(sum(gaps_sorted) / n)
    med = float(gaps_sorted[n//2]) if (n % 2 == 1) else float(0.5*(gaps_sorted[n//2 - 1] + gaps_sorted[n//2]))
    mx = int(max(gaps_sorted))
    return (n, mean, med, mx)

def make_plots(summary_rows: List[dict], plots_dir: Path, title_prefix: str):
    safe_mkdir(plots_dir)
    if not summary_rows:
        return

    rows = [r for r in summary_rows if r.get("seq") == "ALL"]
    if not rows:
        rows = summary_rows

    labels = []
    match_rate = []
    idsw = []
    reacq_mean = []
    for r in rows:
        labels.append(r["label"])
        match_rate.append(float(r["match_rate"]))
        idsw.append(int(float(r["id_switches"])))
        reacq_mean.append(float(r["reacq_mean_frames"]))

    x = np.arange(len(labels))

    plt.figure()
    plt.plot(x, match_rate, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("match_rate (matches/gt_boxes)")
    plt.title(f"{title_prefix} - match_rate")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "match_rate.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(x, idsw, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("ID switches (count)")
    plt.title(f"{title_prefix} - id_switches")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "id_switches.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(idsw, match_rate)
    plt.xlabel("ID switches (lower is better)")
    plt.ylabel("match_rate (higher is better)")
    plt.title(f"{title_prefix} - pareto")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "pareto_idsw_vs_matchrate.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(x, reacq_mean, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("reacq_mean_frames (lower is better)")
    plt.title(f"{title_prefix} - reacq_mean_frames")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "reacq_mean_frames.png"), dpi=150)
    plt.close()

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktp_root", type=str, required=True, help="Path to KTP root folder")
    ap.add_argument("--sequences", type=str, default="Arc,Rotation,Still,Translation",
                    help="Comma-separated sequence names")

    # Internal SAMURAI sweep params
    ap.add_argument("--stable_frames_threshold", type=str, default="15")
    ap.add_argument("--stable_ious_threshold", type=str, default="0.4")
    ap.add_argument("--min_obj_score_logits", type=str, default="0.5")
    ap.add_argument("--kf_score_weight", type=str, default="0.25")
    ap.add_argument("--memory_bank_iou_threshold", type=str, default="0.5")
    ap.add_argument("--memory_bank_obj_score_threshold", type=str, default="0.7")
    ap.add_argument("--memory_bank_kf_score_threshold", type=str, default="0.0")

    ap.add_argument("--rotate", type=int, default=0, help="Rotate frames by {0,90,180,270} degrees")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame (speed)")
    ap.add_argument("--max_frames", type=int, default=-1, help="Limit frames per sequence (debug)")

    # Oracle seed visibility criteria
    ap.add_argument("--visible_area_frac", type=float, default=0.02,
                    help="Seed when GT bbox area >= frac*(W*H)")
    ap.add_argument("--visible_min_h", type=int, default=120,
                    help="Seed when GT bbox height >= this many pixels")
    ap.add_argument("--visible_min_w", type=int, default=0,
                    help="Optional: seed when GT bbox width >= this many pixels (0 disables)")

    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.10,
                    help="Do NOT seed a GT id if its bbox overlaps any other GT bbox by IoU > this value")

    ap.add_argument("--iou_match_thr", type=float, default=0.30, help="IoU threshold for matching GT<->pred")

    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for CSV + plots")
    ap.add_argument("--run_name", type=str, default="samurai_internal_sweep", help="Prefix for output files")

    ap.add_argument("--no_display", action="store_true", help="Do not show frames (faster)")
    ap.add_argument("--display_scale", type=float, default=1.0, help="Scale display window (if enabled)")

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

    stable_frames_list = parse_list_ints(args.stable_frames_threshold)
    stable_ious_list   = parse_list_floats(args.stable_ious_threshold)
    min_obj_list       = parse_list_floats(args.min_obj_score_logits)
    kf_w_list          = parse_list_floats(args.kf_score_weight)
    mb_iou_list        = parse_list_floats(args.memory_bank_iou_threshold)
    mb_obj_list        = parse_list_floats(args.memory_bank_obj_score_threshold)
    mb_kf_list         = parse_list_floats(args.memory_bank_kf_score_threshold)

    combos = []
    for sf in stable_frames_list:
        for si in stable_ious_list:
            for mo in min_obj_list:
                for kf in kf_w_list:
                    for mbi in mb_iou_list:
                        for mbo in mb_obj_list:
                            for mbkf in mb_kf_list:
                                combos.append((sf, si, mo, kf, mbi, mbo, mbkf))

    print("[paths]")
    print("  REPO_ROOT:", REPO_ROOT)
    print("  CKPT     :", CKPT_PATH)
    print("  CFG      :", CFG_PATH)
    print("  KTP_ROOT :", ktp_root)
    print("  OUT_DIR  :", out_dir)
    print(f"[sweep] {len(combos)} combos x {len(seqs)} sequences")

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else None

    sweep_rows: List[dict] = []

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{args.run_name}_{run_id}"

    plots_dir = out_dir / "plots" / run_prefix
    safe_mkdir(plots_dir)

    for ci, (sf, si, mo, kf, mbi, mbo, mbkf) in enumerate(combos):
        label = f"sf{sf}_si{si:g}_mo{mo:g}_kf{kf:g}_mbi{mbi:g}_mbo{mbo:g}_mbkf{mbkf:g}"
        print(f"\n[combo {ci+1}/{len(combos)}] {label}")

        all_metrics = SeqMetrics()

        for seq in seqs:
            # build predictor
            if autocast_ctx is not None:
                with autocast_ctx:
                    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))
            else:
                predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))

            # apply internal thresholds
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

            if ci == 0 and seq == seqs[0]:
                print("SAMURAI mode:", getattr(predictor, "samurai_mode", None))
                print_predictor_thresholds(predictor)

            out_csv = out_dir / f"{run_prefix}__{seq}__{label}.csv"

            if autocast_ctx is not None:
                with autocast_ctx:
                    met = run_sequence(
                        seq_name=seq,
                        ktp_root=ktp_root,
                        predictor=predictor,
                        out_csv_path=out_csv,
                        rotate_deg=args.rotate,
                        stride=args.stride,
                        max_frames=args.max_frames,
                        visible_area_frac=args.visible_area_frac,
                        visible_min_h=args.visible_min_h,
                        visible_min_w=args.visible_min_w,
                        seed_overlap_iou_max=args.seed_overlap_iou_max,
                        iou_match_thr=args.iou_match_thr,
                        no_display=args.no_display,
                        display_scale=args.display_scale,
                    )
            else:
                met = run_sequence(
                    seq_name=seq,
                    ktp_root=ktp_root,
                    predictor=predictor,
                    out_csv_path=out_csv,
                    rotate_deg=args.rotate,
                    stride=args.stride,
                    max_frames=args.max_frames,
                    visible_area_frac=args.visible_area_frac,
                    visible_min_h=args.visible_min_h,
                    visible_min_w=args.visible_min_w,
                    seed_overlap_iou_max=args.seed_overlap_iou_max,
                    iou_match_thr=args.iou_match_thr,
                    no_display=args.no_display,
                    display_scale=args.display_scale,
                )

            reacq_n, reacq_mean, reacq_med, reacq_max = summarize_reacq(met.reacq_gaps)
            match_rate = (met.matches / met.gt_boxes) if met.gt_boxes > 0 else 0.0
            mean_iou = (met.iou_sum / met.iou_count) if met.iou_count > 0 else 0.0

            row = {
                "run": run_prefix,
                "label": label,
                "seq": seq,
                "stable_frames_threshold": sf,
                "stable_ious_threshold": si,
                "min_obj_score_logits": mo,
                "kf_score_weight": kf,
                "memory_bank_iou_threshold": mbi,
                "memory_bank_obj_score_threshold": mbo,
                "memory_bank_kf_score_threshold": mbkf,
                "frames": met.frames,
                "gt_boxes": met.gt_boxes,
                "matches": met.matches,
                "match_rate": match_rate,
                "id_switches": met.id_switches,
                "reacq_events": reacq_n,
                "reacq_mean_frames": reacq_mean,
                "reacq_median_frames": reacq_med,
                "reacq_max_frames": reacq_max,
                "mean_iou_when_matched": mean_iou,
                "seed_skipped_small": met.seed_skipped_small,
                "seed_skipped_overlap": met.seed_skipped_overlap,
                "out_csv": str(out_csv),
            }
            sweep_rows.append(row)

            # accumulate
            all_metrics.frames += met.frames
            all_metrics.gt_boxes += met.gt_boxes
            all_metrics.matches += met.matches
            all_metrics.id_switches += met.id_switches
            all_metrics.reacq_events += met.reacq_events
            all_metrics.reacq_gaps.extend(met.reacq_gaps)
            all_metrics.iou_sum += met.iou_sum
            all_metrics.iou_count += met.iou_count
            all_metrics.seed_skipped_small += met.seed_skipped_small
            all_metrics.seed_skipped_overlap += met.seed_skipped_overlap

            print(f"  [seq {seq}] match_rate={match_rate:.3f}  idsw={met.id_switches}  reacq_events={reacq_n}  mean_iou={mean_iou:.3f}")

        # ALL row
        reacq_n, reacq_mean, reacq_med, reacq_max = summarize_reacq(all_metrics.reacq_gaps)
        match_rate = (all_metrics.matches / all_metrics.gt_boxes) if all_metrics.gt_boxes > 0 else 0.0
        mean_iou = (all_metrics.iou_sum / all_metrics.iou_count) if all_metrics.iou_count > 0 else 0.0

        sweep_rows.append({
            "run": run_prefix,
            "label": label,
            "seq": "ALL",
            "stable_frames_threshold": sf,
            "stable_ious_threshold": si,
            "min_obj_score_logits": mo,
            "kf_score_weight": kf,
            "memory_bank_iou_threshold": mbi,
            "memory_bank_obj_score_threshold": mbo,
            "memory_bank_kf_score_threshold": mbkf,
            "frames": all_metrics.frames,
            "gt_boxes": all_metrics.gt_boxes,
            "matches": all_metrics.matches,
            "match_rate": match_rate,
            "id_switches": all_metrics.id_switches,
            "reacq_events": reacq_n,
            "reacq_mean_frames": reacq_mean,
            "reacq_median_frames": reacq_med,
            "reacq_max_frames": reacq_max,
            "mean_iou_when_matched": mean_iou,
            "seed_skipped_small": all_metrics.seed_skipped_small,
            "seed_skipped_overlap": all_metrics.seed_skipped_overlap,
            "out_csv": "",
        })

        print(f"  [ALL] match_rate={match_rate:.3f}  idsw={all_metrics.id_switches}  reacq_events={reacq_n}  mean_iou={mean_iou:.3f}")
        print(f"        seed_skipped_small={all_metrics.seed_skipped_small}  seed_skipped_overlap={all_metrics.seed_skipped_overlap}")

    # write sweep summary
    summary_path = out_dir / f"{run_prefix}__sweep_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run","label","seq",
            "stable_frames_threshold","stable_ious_threshold","min_obj_score_logits","kf_score_weight",
            "memory_bank_iou_threshold","memory_bank_obj_score_threshold","memory_bank_kf_score_threshold",
            "frames","gt_boxes","matches","match_rate",
            "id_switches",
            "reacq_events","reacq_mean_frames","reacq_median_frames","reacq_max_frames",
            "mean_iou_when_matched",
            "seed_skipped_small","seed_skipped_overlap",
            "out_csv"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)

    make_plots(sweep_rows, plots_dir=plots_dir, title_prefix=run_prefix)

    print("\n[done]")
    print("  sweep summary:", summary_path)
    print("  plots dir     :", plots_dir)
    print("Tip: filter sweep_summary.csv to seq == ALL, then sort by:")
    print("  1) id_switches ascending (lower better)")
    print("  2) match_rate descending (higher better)")
    print("  3) reacq_mean_frames ascending (lower better)")

if __name__ == "__main__":
    main()