#!/usr/bin/env python3
r"""
save_mp4_with_yolo_first_frame_prompts.py

Goal:
- Load an input MP4
- Use YOLO on the FIRST frame to detect ALL persons (class=0)
- Prompt SAMURAI/SAM2 with one bbox per detected person (obj_id = 1..N)
- Track through the video
- Save an output MP4 at the SAME FPS as the input (unless you override)
- Draw for each ID:
    - colored mask overlay
    - colored predicted bbox (from mask)
    - "ID <n>" text
(No ground-truth.)

Example:
  python .\demo\save_mp4_with_yolo_first_frame_prompts.py ^
    --in_video "C:\path\to\input.mp4" ^
    --out_video "C:\path\to\output_overlay.mp4" ^
    --stable_kf_time_sec 1.0 ^
    --stable_ious_threshold 0.30 ^
    --min_obj_score_logits 0.5 ^
    --kf_score_weight 0.25 ^
    --memory_bank_iou_threshold 0.5 ^
    --memory_bank_obj_score_threshold 0.5 ^
    --memory_bank_kf_score_threshold 0.0 ^
    --yolo_conf 0.35 ^
    --mask_alpha 0.45 ^
    --no_window
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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


# ---------------- Utilities ----------------
def yolo_person_bboxes(bgr_frame: np.ndarray, model: YOLO, conf_thres: float) -> List[Tuple[int,int,int,int,float]]:
    """Returns list of (x1,y1,x2,y2,conf) for class=person, sorted by conf desc."""
    if bgr_frame is None:
        return []
    res = model(bgr_frame, verbose=False, conf=float(conf_thres))[0]
    out: List[Tuple[int,int,int,int,float]] = []
    if res.boxes is None:
        return out
    for det in res.boxes:
        if int(det.cls) == 0:  # person
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item()) if det.conf is not None else 0.0
            out.append((x1, y1, x2, y2, conf))
    out.sort(key=lambda t: t[4], reverse=True)
    return out

def clamp_bbox_xyxy(bb: Tuple[int,int,int,int], W: int, H: int) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = bb
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2))
    y2 = max(0, min(H,   y2))
    if x2 < x1: x2 = x1
    if y2 < y1: y2 = y1
    return (x1,y1,x2,y2)

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

def id_to_bgr(obj_id: int) -> Tuple[int,int,int]:
    """Deterministic vivid-ish color per id, returned as BGR for OpenCV."""
    hue = int((37 * int(obj_id) + 61) % 180)
    hsv = np.uint8([[[hue, 255, 255]]])  # 1x1
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def _to_id_list(out_obj_ids):
    if out_obj_ids is None:
        return []
    if isinstance(out_obj_ids, (list, tuple)):
        return [int(x) for x in out_obj_ids]
    if torch.is_tensor(out_obj_ids):
        return [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
    return [int(out_obj_ids)]

def logits_to_mask(logits_i: torch.Tensor) -> Optional[np.ndarray]:
    """
    Convert mask logits to boolean mask.
    Robust thresholds because SAM2/SAMURAI logits can be negative even for good masks.
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
    """In-place alpha overlay on frame_bgr for mask pixels."""
    if mask is None or not mask.any():
        return
    col = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    region = frame_bgr[mask].astype(np.float32)
    blended = (1.0 - alpha) * region + alpha * col
    frame_bgr[mask] = blended.astype(np.uint8)

def get_logits_accessor(out_mask_logits):
    """Return (M, get_logits(i)) for tensor/list outputs."""
    if isinstance(out_mask_logits, (list, tuple)):
        M = len(out_mask_logits)
        def get_logits(i):
            return out_mask_logits[i]
        return M, get_logits
    if torch.is_tensor(out_mask_logits):
        M = int(out_mask_logits.shape[0]) if out_mask_logits.ndim >= 1 else 0
        def get_logits(i):
            return out_mask_logits[i]
        return M, get_logits
    return 0, None


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_video", required=True, type=str, help="Input MP4/video path")
    ap.add_argument("--out_video", required=True, type=str, help="Output MP4 path")

    ap.add_argument("--yolo_model", type=str, default="yolov8s.pt", help="YOLO model name/path (ultralytics)")
    ap.add_argument("--yolo_conf", type=float, default=0.35, help="YOLO person confidence threshold")
    ap.add_argument("--max_persons", type=int, default=0, help="If >0, keep only top-K persons by conf on first frame")

    ap.add_argument("--mask_alpha", type=float, default=0.45, help="Mask overlay alpha (0..1)")
    ap.add_argument("--draw_yolo_on_first", action="store_true", help="Draw YOLO bboxes on first frame for debugging")
    ap.add_argument("--no_window", action="store_true", help="Do not show OpenCV window")

    # Video FPS handling
    ap.add_argument("--fps_out", type=float, default=0.0, help="Override output FPS (0 = use input FPS)")

    # Internal thresholds (seconds-based stable time)
    ap.add_argument("--stable_kf_time_sec", type=float, default=1.0)
    ap.add_argument("--stable_ious_threshold", type=float, default=0.30)
    ap.add_argument("--min_obj_score_logits", type=float, default=0.5)
    ap.add_argument("--kf_score_weight", type=float, default=0.25)
    ap.add_argument("--memory_bank_iou_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_obj_score_threshold", type=float, default=0.5)
    ap.add_argument("--memory_bank_kf_score_threshold", type=float, default=0.0)

    args = ap.parse_args()

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {CKPT_PATH}")
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found:\n  {CFG_PATH}")

    in_path = Path(args.in_video).resolve()
    out_path = Path(args.out_video).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {in_path}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps_in <= 1e-6:
        # some files report 0; default to 30 in that case
        fps_in = 30.0
    fps_out = float(args.fps_out) if float(args.fps_out) > 0 else fps_in

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if W <= 0 or H <= 0:
        # fallback: read first frame to infer
        ok, bgr0 = cap.read()
        if not ok or bgr0 is None:
            raise RuntimeError("Failed to read first frame to infer size.")
        H, W = bgr0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("[video]")
    print("  in :", in_path)
    print("  out:", out_path)
    print(f"  fps_in={fps_in:.3f}  fps_out={fps_out:.3f}  size={W}x{H}")

    # Init predictor
    print("[init] Building predictor...")
    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))

    stable_frames = set_predictor_thresholds_seconds(
        predictor,
        fps_sim=fps_out,  # use output processing fps as "sim fps" for seconds->frames conversion
        stable_kf_time_sec=args.stable_kf_time_sec,
        stable_ious_threshold=args.stable_ious_threshold,
        min_obj_score_logits=args.min_obj_score_logits,
        kf_score_weight=args.kf_score_weight,
        memory_bank_iou_threshold=args.memory_bank_iou_threshold,
        memory_bank_obj_score_threshold=args.memory_bank_obj_score_threshold,
        memory_bank_kf_score_threshold=args.memory_bank_kf_score_threshold,
    )

    print("[thresholds]")
    print(f"  stable_kf_time_sec={args.stable_kf_time_sec} -> stable_frames_threshold={stable_frames} (using fps_out)")
    print(f"  stable_ious_threshold={args.stable_ious_threshold}")
    print(f"  min_obj_score_logits={args.min_obj_score_logits}")
    print(f"  kf_score_weight={args.kf_score_weight}")
    print(f"  memory_bank_iou_threshold={args.memory_bank_iou_threshold}")
    print(f"  memory_bank_obj_score_threshold={args.memory_bank_obj_score_threshold}")
    print(f"  memory_bank_kf_score_threshold={args.memory_bank_kf_score_threshold}")

    # Init YOLO
    print(f"[init] Loading YOLO ({args.yolo_model})...")
    yolo = YOLO(args.yolo_model)

    # Read first frame
    ok, bgr = cap.read()
    if not ok or bgr is None:
        raise RuntimeError("Failed to read first frame from input video.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # YOLO detect persons on first frame
    cands = yolo_person_bboxes(bgr, yolo, conf_thres=float(args.yolo_conf))
    if args.max_persons and int(args.max_persons) > 0:
        cands = cands[: int(args.max_persons)]

    if not cands:
        raise RuntimeError("YOLO found 0 persons in the first frame. Try lowering --yolo_conf.")

    print(f"[yolo] Found {len(cands)} persons on first frame (conf>={args.yolo_conf}).")

    # Load first frame into predictor and add prompts for every detected person
    predictor.load_first_frame(rgb)

    seeded_ids: List[int] = []
    for i, (x1, y1, x2, y2, conf) in enumerate(cands, start=1):
        bb = clamp_bbox_xyxy((x1, y1, x2, y2), W, H)
        bbox = np.array([[bb[0], bb[1]], [bb[2], bb[3]]], dtype=np.float32)
        obj_id = int(i)
        try:
            predictor.add_new_prompt(frame_idx=0, obj_id=obj_id, bbox=bbox)
            seeded_ids.append(obj_id)
        except Exception as e:
            print(f"[seed] Failed obj_id={obj_id} conf={conf:.2f}: {repr(e)}")

    if not seeded_ids:
        raise RuntimeError("Failed to seed any person into the predictor (all prompts failed).")

    print(f"[seed] Seeded obj_ids: {seeded_ids}")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps_out), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at: {out_path}")

    # Optional display window
    win = None if args.no_window else "SAMURAI MP4 overlay (q/ESC to quit)"
    if win is not None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else None

    # Helper: render overlays for one frame given predictor outputs
    def render_frame(frame_bgr: np.ndarray, out_obj_ids, out_mask_logits) -> np.ndarray:
        out_bgr = frame_bgr.copy()
        ids = _to_id_list(out_obj_ids)

        visible_ids: List[int] = []

        if out_mask_logits is not None:
            M, get_logits = get_logits_accessor(out_mask_logits)
            n = min(len(ids), M)

            for k in range(n):
                oid = ids[k]
                col = id_to_bgr(oid)

                logits_i = get_logits(k)
                mask = logits_to_mask(logits_i)
                if mask is None:
                    continue
                bb = mask_to_bbox(mask)
                if bb is None:
                    continue

                visible_ids.append(oid)

                overlay_mask_bgr(out_bgr, mask, col, alpha=float(args.mask_alpha))
                x1,y1,x2,y2 = clamp_bbox_xyxy(bb, W, H)
                cv2.rectangle(out_bgr, (x1,y1), (x2,y2), col, 2)
                cv2.putText(
                    out_bgr, f"ID {oid}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA
                )

        # Small HUD: only what matters
        hud = f"seeded_ids={seeded_ids}  visible_ids={visible_ids}"
        cv2.putText(out_bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)

        return out_bgr

    # Process & write first frame too (with optional YOLO debug)
    if args.draw_yolo_on_first:
        dbg = bgr.copy()
        for j, (x1,y1,x2,y2,conf) in enumerate(cands):
            cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(dbg, f"Y{j} {conf:.2f}", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

    # Track on first frame once to get outputs (optional, but makes overlay consistent)
    if autocast_ctx is not None:
        with autocast_ctx:
            out_obj_ids, out_mask_logits = predictor.track(rgb)
    else:
        out_obj_ids, out_mask_logits = predictor.track(rgb)

    out_bgr = render_frame(bgr, out_obj_ids, out_mask_logits)
    writer.write(out_bgr)
    if win is not None:
        cv2.imshow(win, out_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            writer.release()
            cap.release()
            cv2.destroyAllWindows()
            print("[done] Early exit.")
            return

    # Main loop (remaining frames)
    frame_idx = 1
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if autocast_ctx is not None:
                with autocast_ctx:
                    out_obj_ids, out_mask_logits = predictor.track(rgb)
            else:
                out_obj_ids, out_mask_logits = predictor.track(rgb)

            out_bgr = render_frame(bgr, out_obj_ids, out_mask_logits)
            writer.write(out_bgr)

            if win is not None:
                cv2.imshow(win, out_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1

    finally:
        writer.release()
        cap.release()
        if win is not None:
            cv2.destroyAllWindows()

    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()