#!/usr/bin/env python3
r"""
save_ktp_video_with_overlays.py

Run SAMURAI on a KTP sequence and save an MP4 with:
- per-ID colored masks
- per-ID colored prediction bboxes (same color as mask)

This version uses ORACLE seeding from the KTP ground-truth file:
  KTP/ground_truth/<Seq>_gt2D.txt

So you get deterministic results for advisor demos.

Example:
  py .\\demo\\save_ktp_video_with_overlays.py ^
    --ktp_root "C:\\Users\\inesg\\OneDrive\\Desktop\\Thesis\\datasets\\KTP" ^
    --seq "Rotation" ^
    --out_video "C:\\Users\\inesg\\OneDrive\\Desktop\\Thesis\\datasets\\KTP\\rotation_demo_15fps.mp4" ^
    --fps 15 ^
    --stable_kf_time_sec 1.0 ^
    --stable_ious_threshold 0.30 ^
    --min_obj_score_logits 0.5 ^
    --kf_score_weight 0.25 ^
    --memory_bank_iou_threshold 0.5 ^
    --memory_bank_obj_score_threshold 0.5 ^
    --memory_bank_kf_score_threshold 0.0 ^
    --visible_area_frac 0.02 ^
    --visible_min_h 120 ^
    --seed_overlap_iou_max 0.05 ^
    --stride 1 ^
    --draw_gt ^
    --draw_frame_idx
"""

from __future__ import annotations

import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch

# ---------------- Repo root + imports ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "demo") else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

from sam2.build_sam import build_sam2_camera_predictor

# ---------------- Hard-coded checkpoint/config ----------------
CKPT_PATH = (REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt").resolve()
CFG_PATH  = (REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()

# ---------------- Helpers ----------------
_TS_LEAD_NUM = re.compile(r"^(\d+(?:\.\d+)?)")

def ts_from_filename_robust(p: Path) -> Optional[str]:
    m = _TS_LEAD_NUM.match(p.stem)
    return m.group(1) if m else None

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

def parse_gt2d_file(gt_path: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """
    Returns dict: ts_string -> list of (gt_id, x, y, w, h)
    """
    d: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    with gt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            ts_part, rest = line.split(":", 1)
            ts = ts_part.strip()
            dets_raw = [r.strip() for r in rest.strip().split(",") if r.strip()]
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

def set_predictor_thresholds_seconds(
    predictor,
    fps_sim: float,
    stable_kf_time_sec: float,
    stable_ious_threshold: float,
    min_obj_score_logits: float,
    kf_score_weight: float,
    memory_bank_iou_threshold: float,
    memory_bank_obj_score_threshold: float,
    memory_bank_kf_score_threshold: float,
) -> int:
    """
    Convert stable_kf_time_sec -> stable_frames_threshold using fps_sim, then set predictor attrs.
    """
    stable_frames_threshold = int(round(float(stable_kf_time_sec) * float(fps_sim)))

    def _set(name, val):
        if hasattr(predictor, name):
            setattr(predictor, name, val)

    _set("stable_frames_threshold", stable_frames_threshold)
    _set("stable_ious_threshold", float(stable_ious_threshold))
    _set("min_obj_score_logits", float(min_obj_score_logits))
    _set("kf_score_weight", float(kf_score_weight))
    _set("memory_bank_iou_threshold", float(memory_bank_iou_threshold))
    _set("memory_bank_obj_score_threshold", float(memory_bank_obj_score_threshold))
    _set("memory_bank_kf_score_threshold", float(memory_bank_kf_score_threshold))

    return stable_frames_threshold

def _to_id_list(out_obj_ids):
    if out_obj_ids is None:
        return []
    if isinstance(out_obj_ids, (list, tuple)):
        return [int(x) for x in out_obj_ids]
    if torch.is_tensor(out_obj_ids):
        return [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
    return [int(out_obj_ids)]

def id_to_bgr(obj_id: int) -> Tuple[int,int,int]:
    """
    Deterministic vivid-ish color per id, returned as BGR for OpenCV.
    """
    hue = int((37 * int(obj_id) + 61) % 180)
    hsv = np.uint8([[[hue, 255, 255]]])  # 1x1
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def logits_to_mask(logits_i: torch.Tensor) -> Optional[np.ndarray]:
    """
    Convert mask logits to boolean mask. Use >0 like your demo (consistent),
    but if empty, try slightly lower thresholds (robust).
    """
    if logits_i is None or not torch.is_tensor(logits_i):
        return None

    if logits_i.ndim == 3:
        lg = logits_i[0]
    elif logits_i.ndim == 2:
        lg = logits_i
    else:
        return None

    lg = lg.detach()
    for thr in (0.0, -2.0, -4.0):
        m = (lg > thr).detach().cpu().numpy().astype(bool)
        if m.any():
            return m
    return None

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min()); x2 = int(xs.max()) + 1
    y1 = int(ys.min()); y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)

def overlay_mask_bgr(frame_bgr: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int,int,int], alpha: float) -> None:
    """
    In-place alpha overlay on frame_bgr for mask pixels.
    """
    if mask is None or not mask.any():
        return
    col = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    region = frame_bgr[mask].astype(np.float32)
    blended = (1.0 - alpha) * region + alpha * col
    frame_bgr[mask] = blended.astype(np.uint8)

def put_text_bg(img_bgr: np.ndarray, text: str, org: Tuple[int,int], scale=0.65, thickness=2,
                text_color=(255,255,255), bg_color=(0,0,0)) -> None:
    """
    Draw text with a solid background rectangle for readability.
    """
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img_bgr, (x-4, y-th-6), (x+tw+4, y+base+6), bg_color, -1)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)

# ---------------- Main ----------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktp_root", required=True, type=str)
    ap.add_argument("--seq", required=True, type=str, help="Arc/Rotation/Still/Translation")
    ap.add_argument("--out_video", required=True, type=str, help="Output MP4 path")
    ap.add_argument("--fps", type=float, default=15.0, help="FPS to save the output video (use the original sequence FPS)")
    ap.add_argument("--rotate", type=int, default=0, help="Rotate frames by {0,90,180,270}")
    ap.add_argument("--stride", type=int, default=1)

    # Seeding rules
    ap.add_argument("--visible_area_frac", type=float, default=0.02)
    ap.add_argument("--visible_min_h", type=int, default=120)
    ap.add_argument("--visible_min_w", type=int, default=0)
    ap.add_argument("--seed_overlap_iou_max", type=float, default=0.05)

    # Thresholds (your chosen best set)
    ap.add_argument("--stable_kf_time_sec", type=float, default=1.0)
    ap.add_argument("--stable_ious_threshold", type=float, default=0.30)
    ap.add_argument("--min_obj_score_logits", type=float, default=0.5)
    ap.add_argument("--kf_score_weight", type=float, default=0.25)
    ap.add_argument("--memory_bank_iou_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_obj_score_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_kf_score_threshold", type=float, default=0.0)

    # Visuals
    ap.add_argument("--mask_alpha", type=float, default=0.45)
    ap.add_argument("--draw_gt", action="store_true", help="Also draw GT boxes in white")
    ap.add_argument("--draw_frame_idx", action="store_true",
                    help="Draw fidx (0-based loop index after stride) and timestamp on each frame")
    ap.add_argument("--no_window", action="store_true", help="Do not show OpenCV window")

    args = ap.parse_args()

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {CKPT_PATH}")
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found:\n  {CFG_PATH}")

    ktp_root = Path(args.ktp_root).resolve()
    seq = args.seq

    img_dir = ktp_root / "images" / seq / "rgb"
    gt_path = ktp_root / "ground_truth" / f"{seq}_gt2D.txt"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_map = parse_gt2d_file(gt_path)

    # ---- collect frames sorted by timestamp ----
    frames_all = list(img_dir.glob("*.jpg"))
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
    if not items:
        raise RuntimeError(f"No parseable timestamped .jpg frames in {img_dir}")

    items.sort(key=lambda t: t[0])

    # dedup timestamps
    frames = []
    ts_by_path = {}
    seen = set()
    for _, ts_str, p in items:
        if ts_str in seen:
            continue
        seen.add(ts_str)
        frames.append(p)
        ts_by_path[p] = ts_str

    if args.stride > 1:
        frames = frames[::args.stride]
    if not frames:
        raise RuntimeError("No frames after stride filtering.")

    # ---- load first frame for sizes ----
    bgr0 = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    bgr0 = rotate_frame(bgr0, args.rotate)
    H, W = bgr0.shape[:2]

    # ---- init predictor ----
    print("[init] Building predictor...")
    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))

    stable_frames = set_predictor_thresholds_seconds(
        predictor,
        fps_sim=args.fps,
        stable_kf_time_sec=args.stable_kf_time_sec,
        stable_ious_threshold=args.stable_ious_threshold,
        min_obj_score_logits=args.min_obj_score_logits,
        kf_score_weight=args.kf_score_weight,
        memory_bank_iou_threshold=args.memory_bank_iou_threshold,
        memory_bank_obj_score_threshold=args.memory_bank_obj_score_threshold,
        memory_bank_kf_score_threshold=args.memory_bank_kf_score_threshold,
    )

    print("[thresholds]")
    print(f"  fps_sim={args.fps}")
    print(f"  stable_kf_time_sec={args.stable_kf_time_sec}  -> stable_frames_threshold={stable_frames}")
    print(f"  stable_ious_threshold={args.stable_ious_threshold}")
    print(f"  min_obj_score_logits={args.min_obj_score_logits}")
    print(f"  kf_score_weight={args.kf_score_weight}")
    print(f"  memory_bank_iou_threshold={args.memory_bank_iou_threshold}")
    print(f"  memory_bank_obj_score_threshold={args.memory_bank_obj_score_threshold}")
    print(f"  memory_bank_kf_score_threshold={args.memory_bank_kf_score_threshold}")

    # ---- video writer ----
    out_path = Path(args.out_video).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at: {out_path}")

    # ---- seed rules state ----
    seeded: set[int] = set()
    first_loaded = False

    def should_seed(gid: int, bb_xyxy: Tuple[int,int,int,int], gt_bbs: Dict[int, Tuple[int,int,int,int]]) -> bool:
        bw = max(0, bb_xyxy[2]-bb_xyxy[0])
        bh = max(0, bb_xyxy[3]-bb_xyxy[1])
        area = bw * bh
        area_frac = area / float(W * H + 1e-9)

        visible_ok = (area_frac >= float(args.visible_area_frac)) and (bh >= int(args.visible_min_h))
        if args.visible_min_w and int(args.visible_min_w) > 0:
            visible_ok = visible_ok and (bw >= int(args.visible_min_w))
        if not visible_ok:
            return False

        max_iou_other = 0.0
        for ogid, obb in gt_bbs.items():
            if ogid == gid:
                continue
            max_iou_other = max(max_iou_other, iou_xyxy(bb_xyxy, obb))
        if max_iou_other > float(args.seed_overlap_iou_max):
            return False

        return True

    # optional display
    win = "KTP demo (press q/ESC to quit)" if not args.no_window else None
    if win is not None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else None

    print(f"[run] Frames: {len(frames)}  Output: {out_path}")

    try:
        for fidx, fp in enumerate(frames):
            ts = ts_by_path.get(fp, "")
            bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            bgr = rotate_frame(bgr, args.rotate)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # load first frame once
            if not first_loaded:
                predictor.load_first_frame(rgb)
                first_loaded = True

            # build clamped GT bboxes for this frame (for seeding decisions)
            gt_dets = gt_map.get(ts, [])
            gt_bb_by_id: Dict[int, Tuple[int,int,int,int]] = {}
            for (gid, x, y, w, h) in gt_dets:
                gt_bb_by_id[int(gid)] = clamp_bbox_xyxy(bbox_xywh_to_xyxy(x, y, w, h), W, H)

            # seed new ids when visible and non-overlapping
            seeded_now: List[int] = []
            for (gid, x, y, w, h) in gt_dets:
                gid = int(gid)
                if gid in seeded:
                    continue
                bb = gt_bb_by_id[gid]
                if not should_seed(gid, bb, gt_bb_by_id):
                    continue

                bbox = np.array([[bb[0], bb[1]], [bb[2], bb[3]]], dtype=np.float32)
                try:
                    if fidx == 0:
                        predictor.add_new_prompt(frame_idx=0, obj_id=gid, bbox=bbox)
                    else:
                        predictor.add_conditioning_frame(rgb)
                        predictor.add_new_prompt_during_track(
                            bbox=bbox,
                            if_new_target=True,
                            obj_id=gid,
                            labels=None,
                            clear_old_points=True,
                        )
                    seeded.add(gid)
                    seeded_now.append(gid)
                except Exception:
                    pass

            # track
            if autocast_ctx is not None:
                with autocast_ctx:
                    out_obj_ids, out_mask_logits = predictor.track(rgb)
            else:
                out_obj_ids, out_mask_logits = predictor.track(rgb)

            tracked_ids = _to_id_list(out_obj_ids)

            # render overlays in BGR for video
            out_bgr = bgr.copy()
            visible_ids: List[int] = []

            # draw masks + bboxes per id
            if out_mask_logits is not None:
                if isinstance(out_mask_logits, (list, tuple)):
                    M = len(out_mask_logits)
                    get_logits = lambda i: out_mask_logits[i]
                elif torch.is_tensor(out_mask_logits):
                    M = int(out_mask_logits.shape[0]) if out_mask_logits.ndim >= 1 else 0
                    get_logits = lambda i: out_mask_logits[i]
                else:
                    M = 0
                    get_logits = None

                n = min(len(tracked_ids), M)
                for i in range(n):
                    oid = tracked_ids[i]
                    col = id_to_bgr(oid)

                    logits_i = get_logits(i)
                    mask = logits_to_mask(logits_i)
                    if mask is None:
                        continue

                    bb = mask_to_bbox(mask)
                    if bb is None:
                        continue

                    visible_ids.append(oid)

                    overlay_mask_bgr(out_bgr, mask, col, alpha=float(args.mask_alpha))

                    x1, y1, x2, y2 = clamp_bbox_xyxy(bb, W, H)
                    cv2.rectangle(out_bgr, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(
                        out_bgr, f"ID {oid}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA
                    )

            # optional GT boxes
            if args.draw_gt:
                for (gid, x, y, w, h) in gt_dets:
                    bb = gt_bb_by_id[int(gid)]
                    cv2.rectangle(out_bgr, (bb[0], bb[1]), (bb[2], bb[3]), (255,255,255), 2)
                    cv2.putText(
                        out_bgr, f"GT {int(gid)}",
                        (bb[0], bb[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA
                    )

            # HUD
            known_ids = sorted(list(seeded))
            hud = f"known_ids={known_ids}  seeded_now={seeded_now}  visible_ids={visible_ids}"
            put_text_bg(out_bgr, hud, (10, 30), scale=0.60, thickness=2)

            # Optional frame index + timestamp
            if args.draw_frame_idx:
                put_text_bg(out_bgr, f"fidx={fidx}  ts={ts}", (10, 58), scale=0.62, thickness=2)

            writer.write(out_bgr)

            if win is not None:
                cv2.imshow(win, out_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        writer.release()
        if win is not None:
            cv2.destroyAllWindows()

    print(f"[done] Saved: {out_path}")

if __name__ == "__main__":
    main()