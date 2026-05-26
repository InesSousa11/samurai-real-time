#!/usr/bin/env python3
"""
KTP_deep_debug_reid_auto.py

KTP-specific version of video_deep_debug_reid.py.

What this script does:
- Loads KTP image sequences directly from --ktp_root.
- Uses the same automatic GT-box prompting logic as KTP_eval_run.py:
  visible GT persons are prompted once, using the GT bbox, when they become eligible.
- Runs the SAMURAI/ReID predictor through the sequence.
- Saves a debug video with GT boxes, predicted boxes, masks, and internal scores.
- Saves a clean video with only the predicted masks.
- Saves optional debug frames for cases where a prediction exists without a same-ID GT match.

Recommended location:
    demo/KTP_deep_debug_reid_auto.py

Example:
python demo/KTP_deep_debug_reid_auto.py `
  --ktp_root "C:\\Users\\inesg\\OneDrive\\Desktop\\Thesis\\datasets\\KTP" `
  --out_dir "C:\\tmp\\ktp_deep_debug" `
  --sequences Arc,Rotation,Still,Translation `
  --stride 6 `
  --reid_thr 0.80 `
  --memory_bank_reid_threshold 0.65 `
  --min_obj_score_logits 1.0 `
  --reid_gallery_add_sim_threshold 0.85 `
  --stable_frames_threshold 15 `
  --stable_ious_threshold 0.30 `
  --kf_score_weight 0.25 `
  --memory_bank_iou_threshold 0.5 `
  --memory_bank_obj_score_threshold 0.5 `
  --memory_bank_kf_score_threshold 0.0 `
  --save_video `
  --no_display
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'", category=UserWarning)

# repo root: parent of /demo when script is inside demo
REPO_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "demo" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_camera_predictor

CKPT_PATH = (REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt").resolve()
CFG_PATH = (REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp_bbox_xyxy(bb, W: int, H: int):
    x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, 0, W)
    y2 = clamp(y2, 0, H)
    return x1, y1, x2, y2


def xywh_to_xyxy(x, y, w, h, W: int, H: int):
    return clamp_bbox_xyxy((x, y, x + w, y + h), W, H)


def bbox_iou_xyxy(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def mask_to_bbox(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask_bool is None:
        return None
    ys, xs = np.where(mask_bool > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _to_id_list(out_obj_ids):
    if out_obj_ids is None:
        return []
    if isinstance(out_obj_ids, (list, tuple)):
        return [int(x) for x in out_obj_ids]
    if torch.is_tensor(out_obj_ids):
        return [int(x) for x in out_obj_ids.detach().reshape(-1).cpu().tolist()]
    return [int(out_obj_ids)]


def _logits_to_mask_np(logits: torch.Tensor, H: int, W: int) -> Optional[np.ndarray]:
    if logits is None or not torch.is_tensor(logits):
        return None
    lg = logits.detach()
    # Accept common forms: HxW, 1xHxW, 1x1xHxW
    while lg.ndim > 2 and lg.shape[0] == 1:
        lg = lg[0]
    while lg.ndim > 2:
        lg = lg[0]
    if lg.ndim != 2:
        return None
    if tuple(lg.shape[-2:]) != (H, W):
        lg = F.interpolate(lg[None, None].float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return (lg > 0).detach().cpu().numpy().astype(bool)


def masks_from_predictor_output(out_obj_ids, out_mask_logits, H: int, W: int) -> Dict[int, np.ndarray]:
    ids = _to_id_list(out_obj_ids)
    masks: Dict[int, np.ndarray] = {}
    if out_mask_logits is None or len(ids) == 0:
        return masks

    if torch.is_tensor(out_mask_logits):
        logits = out_mask_logits
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.ndim == 3:
            n = min(len(ids), int(logits.shape[0]))
            for i in range(n):
                m = _logits_to_mask_np(logits[i], H, W)
                if m is not None and m.any():
                    masks[int(ids[i])] = m
        return masks

    if isinstance(out_mask_logits, (list, tuple)):
        n = min(len(ids), len(out_mask_logits))
        for i in range(n):
            m = _logits_to_mask_np(out_mask_logits[i], H, W)
            if m is not None and m.any():
                masks[int(ids[i])] = m
    return masks


PALETTE_RGB = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (180, 60, 60), (60, 180, 60), (60, 60, 180), (255, 180, 60),
]


def _id_to_rgb(obj_id: int):
    return PALETTE_RGB[int(obj_id) % len(PALETTE_RGB)]


def _rgb_to_bgr(c):
    return int(c[2]), int(c[1]), int(c[0])


def draw_mask_overlay(rgb_frame: np.ndarray, mask_bool_by_id: Dict[int, np.ndarray], alpha: float = 0.5):
    h, w = rgb_frame.shape[:2]
    overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_id, mask_bool in mask_bool_by_id.items():
        if mask_bool is None or mask_bool.shape[:2] != (h, w):
            continue
        overlay_rgb[mask_bool] = _id_to_rgb(int(obj_id))
    return cv2.addWeighted(rgb_frame, 1.0, overlay_rgb, float(alpha), 0.0)


def _draw_text_with_bg(img, text, org, fg_color=(255,255,255), bg_color=(0,0,0), scale=0.45, thickness=1):
    x, y = int(org[0]), int(org[1])
    (tw, th), base = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x1 = max(0, x - 2)
    y1 = max(0, y - th - base - 2)
    x2 = min(img.shape[1] - 1, x + tw + 2)
    y2 = min(img.shape[0] - 1, y + base + 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg_color, thickness, cv2.LINE_AA)


def _fmt_score(x):
    try:
        if x is None:
            return "na"
        if torch.is_tensor(x):
            x = x.detach().cpu().reshape(-1)
            if x.numel() == 0:
                return "na"
            x = float(x[0].item())
        x = float(x)
        if not np.isfinite(x):
            return "na"
        return f"{x:.2f}"
    except Exception:
        return "na"


def _fmt_bool(x):
    if x is None:
        return "na"
    return "1" if bool(x) else "0"


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
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals)) if vals[i] - vals[i - 1] > 1e-9]
    if not diffs:
        return float(fallback_fps)
    med_dt = float(np.median(diffs))
    if med_dt <= 1e-9:
        return float(fallback_fps)
    return float(max(1.0, min(120.0, 1.0 / med_dt)))


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
            for raw_det in [r.strip() for r in rest.split(",") if r.strip()]:
                parts = raw_det.split()
                if len(parts) < 5:
                    continue
                try:
                    gid = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    dets.append((gid, x, y, w, h))
                except Exception:
                    continue
            gt_map[ts] = dets
    return gt_map


def _timestamp_from_image_path(p: Path) -> str:
    # KTP frame filenames are normally timestamps. Keep only stem.
    return p.stem


def load_ordered_frames(img_dir: Path, stride: int = 1, max_frames: int = -1):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    frames_all: List[Path] = []
    for ext in exts:
        frames_all.extend(img_dir.glob(ext))
    if not frames_all:
        raise RuntimeError(f"No image frames found in: {img_dir}")

    def sort_key(p: Path):
        s = p.stem
        try:
            return (0, float(s))
        except Exception:
            return (1, s)

    frames_all = sorted(frames_all, key=sort_key)
    stride = max(1, int(stride))
    frames = frames_all[::stride]
    if max_frames is not None and int(max_frames) > 0:
        frames = frames[:int(max_frames)]
    ts_by_path = {p: _timestamp_from_image_path(p) for p in frames}
    return frames, ts_by_path


def find_sequence_paths(ktp_root: Path, seq_name: str):
    """
    Find KTP sequence paths using the same folder layout as KTP_eval_run.py.

    Expected structure:
        KTP/
          images/
            Arc/rgb/*.jpg
            Rotation/rgb/*.jpg
            Still/rgb/*.jpg
            Translation/rgb/*.jpg
          ground_truth/
            Arc_gt2D.txt
            Rotation_gt2D.txt
            Still_gt2D.txt
            Translation_gt2D.txt

    Returns:
        seq_dir, img_dir, gt_path
    """
    ktp_root = Path(ktp_root)
    seq_name = str(seq_name).strip()

    # Exact expected paths
    seq_dir = ktp_root / "images" / seq_name
    img_dir = seq_dir / "rgb"
    gt_path = ktp_root / "ground_truth" / f"{seq_name}_gt2D.txt"

    if img_dir.exists() and gt_path.exists():
        return seq_dir, img_dir, gt_path

    # Case-insensitive fallback, useful if folder names differ in capitalization
    images_root = ktp_root / "images"
    gt_root = ktp_root / "ground_truth"

    if images_root.exists():
        for child in images_root.iterdir():
            if child.is_dir() and child.name.lower() == seq_name.lower():
                seq_dir = child
                img_dir = seq_dir / "rgb"
                gt_path = gt_root / f"{child.name}_gt2D.txt"

                if img_dir.exists() and gt_path.exists():
                    return seq_dir, img_dir, gt_path

    # Helpful error message
    raise FileNotFoundError(
        f"Could not find KTP sequence '{seq_name}'.\n"
        f"Expected image directory:\n"
        f"  {ktp_root / 'images' / seq_name / 'rgb'}\n"
        f"Expected GT file:\n"
        f"  {ktp_root / 'ground_truth' / (seq_name + '_gt2D.txt')}\n"
        f"Check that --ktp_root points to the folder that contains 'images' and 'ground_truth'."
    )


# ---------------------------------------------------------------------
# Predictor configuration
# ---------------------------------------------------------------------

def _set_attr_and_state(predictor, name: str, value) -> None:
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
    keys = [
        "stable_frames_threshold", "stable_ious_threshold", "min_obj_score_logits",
        "kf_score_weight", "memory_bank_iou_threshold", "memory_bank_obj_score_threshold",
        "memory_bank_kf_score_threshold", "memory_bank_reid_threshold", "reid_thr",
        "reid_gallery_max_size", "reid_gallery_add_sim_threshold", "reid_gallery_add_cooldown",
        "reid_gallery_random_replace_prob", "reid_gallery_random_replace_if_diverse_prob",
        "reid_gallery_anchor_protect",
    ]
    try:
        cs = getattr(predictor, "condition_state", None)
        if isinstance(cs, dict):
            for key in keys:
                if hasattr(predictor, key):
                    cs[key] = getattr(predictor, key)
    except Exception:
        pass


def set_predictor_thresholds(predictor, args) -> None:
    mapping = {
        "stable_frames_threshold": args.stable_frames_threshold,
        "stable_ious_threshold": args.stable_ious_threshold,
        "min_obj_score_logits": args.min_obj_score_logits,
        "kf_score_weight": args.kf_score_weight,
        "memory_bank_iou_threshold": args.memory_bank_iou_threshold,
        "memory_bank_obj_score_threshold": args.memory_bank_obj_score_threshold,
        "memory_bank_kf_score_threshold": args.memory_bank_kf_score_threshold,
        "memory_bank_reid_threshold": args.memory_bank_reid_threshold,
        "reid_thr": args.reid_thr,
        "reid_gallery_max_size": args.reid_gallery_max_size,
        "reid_gallery_add_sim_threshold": args.reid_gallery_add_sim_threshold,
        "reid_gallery_add_cooldown": args.reid_gallery_add_cooldown,
        "reid_gallery_random_replace_prob": args.reid_gallery_random_replace_prob,
        "reid_gallery_random_replace_if_diverse_prob": args.reid_gallery_random_replace_if_diverse_prob,
        "reid_gallery_anchor_protect": args.reid_gallery_anchor_protect,
    }
    for k, v in mapping.items():
        _set_attr_and_state(predictor, k, v)


def build_predictor(args):
    print("[init] Building SAM2 camera predictor...", flush=True)
    # Your SAM2CameraPredictor constructor already defaults to TransReID in the versions you showed.
    # If your build_sam2_camera_predictor accepts reid_backend_name, this try-block uses it.
    try:
        predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH), reid_backend_name=args.reid_backend)
    except TypeError:
        predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))
        try:
            predictor.reid_backend_name = args.reid_backend
        except Exception:
            pass
    set_predictor_thresholds(predictor, args)
    return predictor


# ---------------------------------------------------------------------
# Prompting helpers: same idea as KTP_eval_run.py
# ---------------------------------------------------------------------

def is_gt_eligible(bb_xyxy, frame_shape, visible_area_frac: float, visible_min_h: int, visible_min_w: int) -> bool:
    H, W = frame_shape[:2]
    x1, y1, x2, y2 = bb_xyxy
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw < int(visible_min_w) or bh < int(visible_min_h):
        return False
    area_frac = (bw * bh) / float(W * H + 1e-9)
    return area_frac >= float(visible_area_frac)


def bbox_prompt_from_xyxy(bb_xyxy):
    x1, y1, x2, y2 = bb_xyxy
    return np.array([[x1, y1], [x2, y2]], dtype=np.float32)


def add_prompt_for_id(predictor, rgb_frame, obj_id: int, bbox_xyxy, is_first_frame: bool):
    bbox_prompt = bbox_prompt_from_xyxy(bbox_xyxy)

    if is_first_frame:
        # Common API in your KTP_eval_run/SAM2 camera predictor.
        try:
            return predictor.add_new_prompt(
                frame_idx=0,
                obj_id=int(obj_id),
                bbox=bbox_prompt,
                labels=None,
                clear_old_points=True,
            )
        except TypeError:
            return predictor.add_new_prompt(
                bbox=bbox_prompt,
                obj_id=int(obj_id),
                labels=None,
                clear_old_points=True,
            )

    # Mid-sequence prompt: this mirrors the KTP_eval_run behavior for late-seeded IDs.
    try:
        predictor.add_conditioning_frame(rgb_frame)
    except Exception:
        pass

    try:
        return predictor.add_new_prompt_during_track(
            bbox=bbox_prompt,
            if_new_target=True,
            obj_id=int(obj_id),
            labels=None,
            clear_old_points=True,
        )
    except TypeError:
        # fallback for slightly different local signatures
        return predictor.add_new_prompt_during_track(
            bbox=bbox_prompt,
            obj_id=int(obj_id),
            labels=None,
            clear_old_points=True,
        )


# ---------------------------------------------------------------------
# Debug extraction / drawing
# ---------------------------------------------------------------------

def extract_debug_for_id(predictor, pid: int) -> Dict[str, Any]:
    info = {}
    try:
        cs = getattr(predictor, "condition_state", {})
        if isinstance(cs, dict):
            reid_last = cs.get("reid_last", {})
            if isinstance(reid_last, dict):
                info.update(reid_last.get(int(pid), reid_last.get(pid, {})) or {})

            live = cs.get("live_debug", {})
            if isinstance(live, dict):
                obj_ids = list(cs.get("obj_ids", []))
                if int(pid) in [int(x) for x in obj_ids]:
                    idx = [int(x) for x in obj_ids].index(int(pid))
                    for key in ["object_score_logits", "object_score_prob", "reid_ok", "best_iou", "kf_score", "reacq_score"]:
                        vals = live.get(key, None)
                        if isinstance(vals, list) and idx < len(vals):
                            info.setdefault(key, vals[idx])
                    reacq_map = live.get("reacquire_mode_per_id", {})
                    if isinstance(reacq_map, dict):
                        info.setdefault("reacquire", bool(reacq_map.get(int(pid), False)))
    except Exception:
        pass
    return info


def draw_debug_boxes(vis_bgr, gt_dets, gt_bb_by_id, pred_bbox_by_id, predictor, eligible_gt_ids):
    # GT boxes
    for gid, x, y, w, h in gt_dets:
        bb = gt_bb_by_id[int(gid)]
        color = (255, 255, 255) if int(gid) in eligible_gt_ids else (120, 120, 120)
        cv2.rectangle(vis_bgr, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
        cv2.putText(vis_bgr, f"GT {int(gid)}", (bb[0], max(0, bb[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Prediction boxes + scores
    for pid, bb in pred_bbox_by_id.items():
        pid = int(pid)
        col = _rgb_to_bgr(_id_to_rgb(pid))
        cv2.rectangle(vis_bgr, (bb[0], bb[1]), (bb[2], bb[3]), col, 2)
        cv2.putText(vis_bgr, f"PR {pid}", (bb[0], bb[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        info = extract_debug_for_id(predictor, pid)
        sim = info.get("sim", None)
        obj_logit = info.get("obj_logit", info.get("object_score_logits", None))
        obj_prob = info.get("obj_prob", info.get("object_score_prob", None))
        best_iou = info.get("best_iou", None)
        kf_score = info.get("kf_score", None)
        accepted = info.get("accepted", None)
        reacquire = info.get("reacquire", None)
        component_cleaned = info.get("component_cleaned", None)
        reacq_score = info.get("reacq_score", None)

        line1 = f"PR{pid} sim={_fmt_score(sim)} obj={_fmt_score(obj_logit)} p={_fmt_score(obj_prob)}"
        line2 = f"iou={_fmt_score(best_iou)} kf={_fmt_score(kf_score)} ok={_fmt_bool(accepted)} R={_fmt_bool(reacquire)} C={_fmt_bool(component_cleaned)}"
        if reacq_score is not None:
            line2 += f" rq={_fmt_score(reacq_score)}"

        y1 = max(14, bb[1] - 22)
        y2 = max(14, bb[1] - 6)
        if bb[1] < 45:
            y1 = min(vis_bgr.shape[0] - 30, bb[3] + 18)
            y2 = min(vis_bgr.shape[0] - 12, bb[3] + 36)
        _draw_text_with_bg(vis_bgr, line1, (bb[0], y1), fg_color=col, bg_color=(0, 0, 0), scale=0.42, thickness=1)
        _draw_text_with_bg(vis_bgr, line2, (bb[0], y2), fg_color=col, bg_color=(0, 0, 0), scale=0.42, thickness=1)


def draw_orange_unmatched_predictions(vis_bgr, pred_bbox_by_id, gt_bb_by_id, same_id_iou_thr: float = 0.01):
    # Orange boxes: predicted ID has no same-ID GT overlap. This is intentionally broad for manual review.
    for pid, pbb in pred_bbox_by_id.items():
        pid = int(pid)
        same_gt = gt_bb_by_id.get(pid, None)
        same_iou = bbox_iou_xyxy(pbb, same_gt) if same_gt is not None else 0.0
        if same_iou < same_id_iou_thr:
            cv2.rectangle(vis_bgr, (pbb[0], pbb[1]), (pbb[2], pbb[3]), (0, 140, 255), 3)
            _draw_text_with_bg(vis_bgr, f"CHECK PR{pid} sameIoU={same_iou:.2f}", (pbb[0], max(14, pbb[1] - 40)), fg_color=(0, 140, 255), bg_color=(0, 0, 0), scale=0.45, thickness=1)


# ---------------------------------------------------------------------
# Main sequence runner
# ---------------------------------------------------------------------

@torch.inference_mode()
def run_sequence(seq_name: str, args) -> None:
    ktp_root = Path(args.ktp_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(out_dir)

    seq_dir, img_dir, gt_path = find_sequence_paths(ktp_root, seq_name)
    frames, ts_by_path = load_ordered_frames(img_dir, stride=args.stride, max_frames=args.max_frames)
    gt_map = parse_gt2d_file(gt_path)

    if not frames:
        raise RuntimeError(f"No frames loaded for sequence {seq_name}")

    rgb0_bgr = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if rgb0_bgr is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")
    rgb0 = cv2.cvtColor(rgb0_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb0.shape[:2]

    seq_fps = float(args.save_video_fps) if args.save_video_fps else estimate_sequence_fps(frames, ts_by_path, fallback_fps=15.0)

    predictor = build_predictor(args)
    predictor.load_first_frame(rgb0)
    set_predictor_thresholds(predictor, args)
    sync_runtime_thresholds_to_state(predictor)

    seq_out_dir = out_dir / seq_name
    safe_mkdir(seq_out_dir)
    review_root = seq_out_dir / "manual_review_frames"
    safe_mkdir(review_root)
    safe_mkdir(review_root / "no_same_id_gt_no_other_overlap")
    safe_mkdir(review_root / "no_same_id_gt_overlaps_other_gt")

    debug_writer = None
    clean_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(str(seq_out_dir / f"{seq_name}_debug.mp4"), fourcc, seq_fps, (W, H))
        clean_writer = cv2.VideoWriter(str(seq_out_dir / f"{seq_name}_clean.mp4"), fourcc, seq_fps, (W, H))

    csv_path = seq_out_dir / f"{seq_name}_debug_log.csv"
    fcsv = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)
    writer.writerow([
        "seq", "local_frame_idx", "global_frame_idx", "timestamp", "seeded_now", "num_pred",
        "pred_id", "pred_x1", "pred_y1", "pred_x2", "pred_y2", "same_id_iou", "max_other_gt_iou",
        "manual_review_bucket", "sim", "obj_logit", "obj_prob", "accepted", "reacquire", "component_cleaned"
    ])

    win = None if args.no_display else f"KTP debug {seq_name}"
    if win is not None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    seeded = set()
    last_out_obj_ids = None
    last_out_masks = None
    did_start_tracking = False

    print(f"[seq {seq_name}] frames={len(frames)} img_dir={img_dir} gt={gt_path}", flush=True)

    for local_idx, frame_path in enumerate(frames):
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ts = ts_by_path.get(frame_path, frame_path.stem)
        gt_dets = gt_map.get(ts, [])

        gt_bb_by_id: Dict[int, Tuple[int, int, int, int]] = {}
        eligible_gt_ids = set()
        for gid, x, y, w, h in gt_dets:
            bb = xywh_to_xyxy(x, y, w, h, W, H)
            gt_bb_by_id[int(gid)] = bb
            if is_gt_eligible(bb, rgb.shape, args.visible_area_frac, args.visible_min_h, args.visible_min_w):
                eligible_gt_ids.add(int(gid))

        seeded_now = []
        # Same automatic prompt logic: each GT ID is prompted once, when visible enough and not overlapping an already seeded person too much.
        for gid in sorted(eligible_gt_ids):
            if gid in seeded:
                continue
            bb = gt_bb_by_id[gid]
            overlap_with_seeded = 0.0
            for sid in seeded:
                if sid in gt_bb_by_id:
                    overlap_with_seeded = max(overlap_with_seeded, bbox_iou_xyxy(bb, gt_bb_by_id[sid]))
            if overlap_with_seeded > float(args.seed_overlap_iou_max):
                continue

            try:
                is_first_prompt_frame = (len(seeded) == 0 and not did_start_tracking)
                out = add_prompt_for_id(predictor, rgb, gid, bb, is_first_frame=is_first_prompt_frame)
                # Try to capture prompt output if returned.
                if isinstance(out, tuple):
                    if len(out) >= 3:
                        _, last_out_obj_ids, last_out_masks = out[:3]
                    elif len(out) == 2:
                        last_out_obj_ids, last_out_masks = out
                seeded.add(gid)
                seeded_now.append(gid)
                print(f"[seq {seq_name}] frame={local_idx} ts={ts} seeded GT id {gid} bbox={bb}", flush=True)
            except Exception as e:
                print(f"[seq {seq_name}] failed to seed id={gid} frame={local_idx}: {repr(e)}", flush=True)

        # Track after at least one object is seeded. For frame 0, if we just prompted, use prompt output.
        if len(seeded) > 0:
            if local_idx == 0 and last_out_masks is not None:
                out_obj_ids, out_masks = last_out_obj_ids, last_out_masks
            else:
                try:
                    out = predictor.track(rgb)
                    did_start_tracking = True
                    if isinstance(out, tuple):
                        if len(out) >= 3:
                            _, out_obj_ids, out_masks = out[:3]
                        elif len(out) == 2:
                            out_obj_ids, out_masks = out
                        else:
                            out_obj_ids, out_masks = None, None
                    else:
                        out_obj_ids, out_masks = None, None
                    last_out_obj_ids, last_out_masks = out_obj_ids, out_masks
                except Exception as e:
                    print(f"[seq {seq_name}] track failed frame={local_idx}: {repr(e)}", flush=True)
                    out_obj_ids, out_masks = None, None
        else:
            out_obj_ids, out_masks = None, None

        pred_mask_by_id = masks_from_predictor_output(out_obj_ids, out_masks, H, W)
        pred_bbox_by_id = {}
        for pid, m in pred_mask_by_id.items():
            bb = mask_to_bbox(m)
            if bb is not None:
                pred_bbox_by_id[int(pid)] = bb

        clean_rgb = draw_mask_overlay(rgb.copy(), pred_mask_by_id, alpha=args.alpha)
        debug_rgb = clean_rgb.copy()
        debug_bgr = cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
        clean_bgr = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)

        draw_debug_boxes(debug_bgr, gt_dets, gt_bb_by_id, pred_bbox_by_id, predictor, eligible_gt_ids)
        if args.draw_manual_review_orange:
            draw_orange_unmatched_predictions(debug_bgr, pred_bbox_by_id, gt_bb_by_id, same_id_iou_thr=args.same_id_iou_thr)

        # Save manual review frames + CSV rows.
        for pid, pbb in pred_bbox_by_id.items():
            same_gt = gt_bb_by_id.get(int(pid), None)
            same_iou = bbox_iou_xyxy(pbb, same_gt) if same_gt is not None else 0.0
            max_other_iou = 0.0
            for gid, gbb in gt_bb_by_id.items():
                if int(gid) == int(pid):
                    continue
                max_other_iou = max(max_other_iou, bbox_iou_xyxy(pbb, gbb))

            bucket = ""
            if same_iou < float(args.same_id_iou_thr):
                if max_other_iou >= float(args.other_gt_iou_thr):
                    bucket = "no_same_id_gt_overlaps_other_gt"
                else:
                    bucket = "no_same_id_gt_no_other_overlap"
                if args.save_manual_review_frames:
                    fname = f"{seq_name}_f{local_idx:06d}_ts{ts}_PR{int(pid)}_same{same_iou:.2f}_other{max_other_iou:.2f}.jpg"
                    cv2.imwrite(str(review_root / bucket / fname), debug_bgr)

            info = extract_debug_for_id(predictor, int(pid))
            writer.writerow([
                seq_name, local_idx, local_idx * int(args.stride), ts, "+".join(map(str, seeded_now)), len(pred_bbox_by_id),
                int(pid), pbb[0], pbb[1], pbb[2], pbb[3], f"{same_iou:.6f}", f"{max_other_iou:.6f}",
                bucket,
                info.get("sim", ""),
                info.get("obj_logit", info.get("object_score_logits", "")),
                info.get("obj_prob", info.get("object_score_prob", "")),
                info.get("accepted", ""), info.get("reacquire", ""), info.get("component_cleaned", ""),
            ])

        # HUD
        hud = f"{seq_name} frame {local_idx+1}/{len(frames)} ts={ts} seeded={sorted(seeded)} new={seeded_now} preds={sorted(pred_bbox_by_id)}"
        _draw_text_with_bg(debug_bgr, hud, (10, 24), fg_color=(255, 255, 255), bg_color=(0, 0, 0), scale=0.55, thickness=1)

        if debug_writer is not None:
            debug_writer.write(debug_bgr)
        if clean_writer is not None:
            clean_writer.write(clean_bgr)

        if win is not None:
            disp = debug_bgr
            if args.display_scale != 1.0:
                disp = cv2.resize(disp, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                cv2.waitKey(0)

    fcsv.close()
    if debug_writer is not None:
        debug_writer.release()
    if clean_writer is not None:
        clean_writer.release()
    if win is not None:
        cv2.destroyWindow(win)

    print(f"[seq {seq_name}] saved to: {seq_out_dir}", flush=True)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_sequences(s: str) -> List[str]:
    if not s:
        return ["Arc", "Rotation", "Still", "Translation"]
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktp_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--sequences", type=str, default="Arc,Rotation,Still,Translation")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=-1)

    # KTP prompt/seed rules copied in spirit from KTP_eval_run.py
    ap.add_argument("--visible_area_frac", type=float, default=0.004)
    ap.add_argument("--visible_min_h", type=int, default=35)
    ap.add_argument("--visible_min_w", type=int, default=15)
    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.50)

    # Predictor/backend
    ap.add_argument("--reid_backend", type=str, default="transreid")
    ap.add_argument("--reid_thr", type=float, default=0.80)
    ap.add_argument("--memory_bank_reid_threshold", type=float, default=None)
    ap.add_argument("--min_obj_score_logits", type=float, default=None)
    ap.add_argument("--reid_gallery_add_sim_threshold", type=float, default=None)
    ap.add_argument("--stable_frames_threshold", type=int, default=None)
    ap.add_argument("--stable_ious_threshold", type=float, default=None)
    ap.add_argument("--kf_score_weight", type=float, default=None)
    ap.add_argument("--memory_bank_iou_threshold", type=float, default=None)
    ap.add_argument("--memory_bank_obj_score_threshold", type=float, default=None)
    ap.add_argument("--memory_bank_kf_score_threshold", type=float, default=None)
    ap.add_argument("--reid_gallery_max_size", type=int, default=None)
    ap.add_argument("--reid_gallery_add_cooldown", type=int, default=None)
    ap.add_argument("--reid_gallery_random_replace_prob", type=float, default=None)
    ap.add_argument("--reid_gallery_random_replace_if_diverse_prob", type=float, default=None)
    ap.add_argument("--reid_gallery_anchor_protect", action="store_true", default=None)

    # Video/debug output
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--save_video_fps", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.50)
    ap.add_argument("--no_display", action="store_true")
    ap.add_argument("--display_scale", type=float, default=1.0)

    # Manual review/debug options
    ap.add_argument("--draw_manual_review_orange", action="store_true", default=True)
    ap.add_argument("--save_manual_review_frames", action="store_true", default=True)
    ap.add_argument("--same_id_iou_thr", type=float, default=0.01)
    ap.add_argument("--other_gt_iou_thr", type=float, default=0.01)

    args = ap.parse_args()

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")

    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    print("[paths]")
    print(f"  REPO_ROOT: {REPO_ROOT}")
    print(f"  CKPT     : {CKPT_PATH}")
    print(f"  CFG      : {CFG_PATH}")
    print(f"  KTP_ROOT : {Path(args.ktp_root).resolve()}")
    print(f"  OUT_DIR  : {Path(args.out_dir).resolve()}")

    for seq in parse_sequences(args.sequences):
        run_sequence(seq, args)


if __name__ == "__main__":
    main()
