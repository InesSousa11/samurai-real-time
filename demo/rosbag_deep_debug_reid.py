#!/usr/bin/env python3
"""
frames_deep_debug_reid.py

Run SAMURAI/SAM2 + internal ReID on frames extracted from a ROS 2 bag.

Expected input:
    extracted_rosbag_rgb_only/
        rgb/
            frame_000000.png
            frame_000001.png
            ...
        timestamps.csv

Important:
- The model receives each original rosbag frame exactly once.
- No duplicated frames are created for model input.
- Optional --realtime_replay waits according to timestamps.csv.
"""

print("=== frames_deep_debug_reid.py VERSION: ROSBAG-FRAMES-REID-001 ===", flush=True)

import sys
import time
import csv
import json
import math
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'", category=UserWarning)

# repo root: parent of /demo
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_camera_predictor


# ---------------- paths ----------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT2 = SCRIPT_DIR.parent if SCRIPT_DIR.name == "demo" else Path.cwd()

CKPT_PATH = (REPO_ROOT2 / "checkpoints" / "sam2.1_hiera_small.pt").resolve()
CFG_PATH = (REPO_ROOT2 / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()


# ---------------- utils ----------------

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FrameDirectoryCapture:
    """
    Replacement for cv2.VideoCapture that reads extracted rosbag frames.

    Expected:
        frames_dir/
            frame_000000.png
            frame_000001.png
            ...

    Optional:
        timestamps.csv with columns:
            rgb_filename, timestamp_ns, relative_time_s, dt_from_previous_s
    """

    def __init__(self, frames_dir, timestamps_csv=None):
        self.frames_dir = Path(frames_dir).resolve()
        self.timestamps_csv = Path(timestamps_csv).resolve() if timestamps_csv else None

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found:\n  {self.frames_dir}")

        self.frame_paths = sorted(
            list(self.frames_dir.glob("*.png"))
            + list(self.frames_dir.glob("*.jpg"))
            + list(self.frames_dir.glob("*.jpeg"))
        )

        if not self.frame_paths:
            raise RuntimeError(f"No image frames found in:\n  {self.frames_dir}")

        self.index = 0
        self.opened = True

        first = cv2.imread(str(self.frame_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Could not read first frame:\n  {self.frame_paths[0]}")

        self.height, self.width = first.shape[:2]

        self.timestamps_ns = []
        self.relative_times_s = []
        self.dt_from_previous_s = []

        if self.timestamps_csv is not None and self.timestamps_csv.exists():
            self._load_timestamps_csv(self.timestamps_csv)

        if len(self.dt_from_previous_s) != len(self.frame_paths):
            self.dt_from_previous_s = [0.0] * len(self.frame_paths)

        self.estimated_fps = self._estimate_fps()

    def _load_timestamps_csv(self, csv_path):
        rows_by_filename = {}

        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                filename = row.get("rgb_filename", "")
                if filename:
                    rows_by_filename[filename] = row

        for frame_path in self.frame_paths:
            row = rows_by_filename.get(frame_path.name, None)

            if row is None:
                self.timestamps_ns.append(None)
                self.relative_times_s.append(None)
                self.dt_from_previous_s.append(0.0)
                continue

            try:
                self.timestamps_ns.append(int(row["timestamp_ns"]))
            except Exception:
                self.timestamps_ns.append(None)

            try:
                self.relative_times_s.append(float(row["relative_time_s"]))
            except Exception:
                self.relative_times_s.append(None)

            try:
                self.dt_from_previous_s.append(float(row["dt_from_previous_s"]))
            except Exception:
                self.dt_from_previous_s.append(0.0)

    def _estimate_fps(self):
        valid_dt = [
            float(x)
            for x in self.dt_from_previous_s
            if x is not None and np.isfinite(float(x)) and float(x) > 0.0
        ]

        if not valid_dt:
            return 25.0

        median_dt = float(np.median(valid_dt))
        if median_dt <= 0:
            return 25.0

        return 1.0 / median_dt

    def isOpened(self):
        return self.opened

    def read(self):
        if self.index >= len(self.frame_paths):
            return False, None

        path = self.frame_paths[self.index]
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)

        self.index += 1

        if frame is None:
            return False, None

        return True, frame

    def release(self):
        self.opened = False

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.estimated_fps)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return int(len(self.frame_paths))
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return int(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return int(self.height)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return int(self.index)
        return 0

    def current_dt_from_previous(self):
        last_read_index = self.index - 1

        if last_read_index < 0 or last_read_index >= len(self.dt_from_previous_s):
            return 0.0

        return float(self.dt_from_previous_s[last_read_index])


def yolo_person_bboxes(bgr_frame, model, conf_thres=0.25):
    """Returns list of (x1, y1, x2, y2, conf) for class person."""
    if bgr_frame is None:
        return []

    res = model(bgr_frame, verbose=False, conf=conf_thres)[0]
    out = []

    if res.boxes is None:
        return out

    for det in res.boxes:
        if int(det.cls) == 0:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item()) if det.conf is not None else 0.0
            out.append((x1, y1, x2, y2, conf))

    out.sort(key=lambda t: t[4], reverse=True)
    return out


def _to_id_list(out_obj_ids):
    if out_obj_ids is None:
        return []
    if isinstance(out_obj_ids, (list, tuple)):
        return [int(x) for x in out_obj_ids]
    if torch.is_tensor(out_obj_ids):
        return [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
    return [int(out_obj_ids)]


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


def _id_to_rgb(obj_id: int):
    return PALETTE_RGB[int(obj_id) % len(PALETTE_RGB)]


def draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits, alpha=0.5):
    """Overlay segmentation masks on RGB frame."""
    if rgb_frame is None or out_mask_logits is None:
        return rgb_frame

    ids = _to_id_list(out_obj_ids)

    if isinstance(out_mask_logits, (list, tuple)):
        M = len(out_mask_logits)
        get_logits = lambda i: out_mask_logits[i]
    elif torch.is_tensor(out_mask_logits):
        M = int(out_mask_logits.shape[0]) if out_mask_logits.ndim >= 1 else 0
        get_logits = lambda i: out_mask_logits[i]
    else:
        return rgb_frame

    n = max(0, min(len(ids), M))
    if n == 0:
        return rgb_frame

    h, w = rgb_frame.shape[:2]
    overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(n):
        logits_i = get_logits(i)

        if logits_i is None or not torch.is_tensor(logits_i):
            continue

        if logits_i.ndim == 3:
            m = logits_i[0] > 0
        elif logits_i.ndim == 2:
            m = logits_i > 0
        else:
            continue

        m = m.detach().cpu().numpy().astype(bool)
        color = _id_to_rgb(ids[i])
        overlay_rgb[m] = color

    return cv2.addWeighted(rgb_frame, 1.0, overlay_rgb, float(alpha), 0.0)


def _extract_reid_debug_info(predictor, pf_now: int):
    """
    Small HUD debug extractor.
    Keeps this script lighter than the full dump-heavy version.
    """
    out = {
        "object_score_logits": None,
        "object_score_prob": None,
        "object_score_thr": None,
        "reid_ok": None,
        "good_mem_count": 0,
        "good_mem_frames": [],
        "reacquire_mode_per_id": {},
        "any_reacquire": False,
    }

    try:
        cs = getattr(predictor, "condition_state", None)
        if not isinstance(cs, dict):
            return out

        live = cs.get("live_debug", None)

        if isinstance(live, dict) and int(live.get("frame_idx", -999999)) == int(pf_now):
            out["object_score_logits"] = live.get("object_score_logits", None)
            out["object_score_prob"] = live.get("object_score_prob", None)
            out["object_score_thr"] = live.get("object_score_thr", None)
            out["reid_ok"] = live.get("reid_ok", None)
            out["good_mem_count"] = int(live.get("good_mem_count", 0))
            out["good_mem_frames"] = [int(x) for x in live.get("good_mem_frames", [])]
            out["reacquire_mode_per_id"] = {
                int(k): bool(v)
                for k, v in live.get("reacquire_mode_per_id", {}).items()
            }
            out["any_reacquire"] = bool(live.get("any_reacquire", False))

        if out["object_score_thr"] is None:
            try:
                out["object_score_thr"] = float(getattr(predictor, "min_obj_score_logits", None))
            except Exception:
                out["object_score_thr"] = None

    except Exception:
        pass

    return out


def save_run_summary(run_dir: Path, args, input_fps, total_frames, W, H):
    summary = {
        "frames_dir": str(args.frames_dir),
        "timestamps_csv": str(args.timestamps_csv) if args.timestamps_csv else None,
        "input_fps_median_estimate": float(input_fps),
        "total_frames": int(total_frames),
        "width": int(W),
        "height": int(H),
        "realtime_replay": bool(args.realtime_replay),
        "reid_thr": args.reid_thr,
        "yolo_conf_initial": args.yolo_conf,
        "note": (
            "Model receives each extracted rosbag RGB frame exactly once. "
            "No duplicated frames are used as model input."
        ),
    }

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--frames_dir", type=str, required=True, help="Path to extracted RGB frames folder")
    ap.add_argument("--timestamps_csv", type=str, default=None, help="Path to timestamps.csv from extraction script")

    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--out_root", type=str, default=str(REPO_ROOT2 / "debug_cases_frames"))
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--reid_thr", type=float, default=None)
    ap.add_argument("--reid_print", action="store_true")

    ap.add_argument(
        "--realtime_replay",
        action="store_true",
        help="Wait between frames according to timestamps.csv. The model still receives each frame once.",
    )

    args = ap.parse_args()

    frames_dir = Path(args.frames_dir).resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found:\n  {frames_dir}")

    if args.timestamps_csv is not None:
        timestamps_csv = Path(args.timestamps_csv).resolve()
        if not timestamps_csv.exists():
            raise FileNotFoundError(f"timestamps.csv not found:\n  {timestamps_csv}")
    else:
        timestamps_csv = None

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {CKPT_PATH}")

    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found:\n  {CFG_PATH}")

    out_root = Path(args.out_root).resolve()
    safe_mkdir(out_root)

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    input_stem = frames_dir.parent.name + "_" + frames_dir.name
    run_name = f"{input_stem}_{run_ts}"
    run_dir = out_root / run_name
    safe_mkdir(run_dir)

    print("[init] Building SAM2 camera predictor...", flush=True)
    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))

    def _sync_reid_threshold():
        if args.reid_thr is None:
            return

        try:
            predictor.reid_thr = float(args.reid_thr)
        except Exception:
            pass

        cs = getattr(predictor, "condition_state", None)
        if isinstance(cs, dict):
            cs["reid_thr"] = float(args.reid_thr)

    _sync_reid_threshold()

    cs0 = getattr(predictor, "condition_state", {})
    try:
        thr0 = float(cs0.get("reid_thr", getattr(predictor, "reid_thr", float("nan"))))
    except Exception:
        thr0 = getattr(predictor, "reid_thr", None)

    print(f"[init] Internal ReID ON, threshold={thr0}", flush=True)

    print("[init] Loading YOLO yolov8s.pt...", flush=True)
    yolo_model = YOLO("yolov8s.pt")

    cap = FrameDirectoryCapture(
        frames_dir=frames_dir,
        timestamps_csv=timestamps_csv,
    )

    if not cap.isOpened():
        raise RuntimeError(f"Could not open frames folder:\n  {frames_dir}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps is None or input_fps <= 0 or not np.isfinite(input_fps):
        input_fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[input] Frames: {total_frames}", flush=True)
    print(f"[input] Resolution: {W}x{H}", flush=True)
    print(f"[input] Median timestamp FPS estimate: {input_fps:.2f}", flush=True)
    print(f"[input] Realtime replay: {args.realtime_replay}", flush=True)

    save_run_summary(run_dir, args, input_fps, total_frames, W, H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    debug_video_path = run_dir / f"{input_stem}_debug.mp4"
    clean_video_path = run_dir / f"{input_stem}_clean_masks.mp4"

    debug_writer = cv2.VideoWriter(str(debug_video_path), fourcc, float(input_fps), (W, H))
    clean_writer = cv2.VideoWriter(str(clean_video_path), fourcc, float(input_fps), (W, H))

    if not debug_writer.isOpened():
        raise RuntimeError(f"Could not open debug video writer:\n  {debug_video_path}")

    if not clean_writer.isOpened():
        raise RuntimeError(f"Could not open clean video writer:\n  {clean_video_path}")

    state = {
        "first_frame_loaded": False,
        "tracking": False,
        "injecting": False,
        "paused": True,
        "yolo_enabled": True,
        "yolo_conf": float(args.yolo_conf),
        "cands": [],
        "selected_idx": 0,
        "last_rgb": None,
        "next_obj_id": 1,
        "added_obj_ids": [],
        "out_obj_ids": None,
        "out_mask_logits": None,
        "frame_number": 0,
        "eof": False,
    }

    ok, first_bgr = cap.read()
    if not ok or first_bgr is None:
        raise RuntimeError("Failed to read first frame.")

    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    state["last_rgb"] = first_rgb
    state["frame_number"] = 0

    win = (
        "SAMURAI rosbag frames debug "
        "(A add | T start | SPACE pause | P prompt | Y yolo | +/- conf | R reset | Q quit)"
    )
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    noncond_ring: Dict[int, np.ndarray] = {}
    noncond_keys = deque(maxlen=600)

    def _ring_store(global_fidx: int, rgb_img: np.ndarray):
        noncond_ring[int(global_fidx)] = rgb_img.copy()
        noncond_keys.append(int(global_fidx))

        while len(noncond_keys) > noncond_keys.maxlen:
            old = noncond_keys.popleft()
            noncond_ring.pop(int(old), None)

    def reset_all():
        nonlocal predictor, cap, noncond_ring, noncond_keys

        print("[reset] Rebuilding predictor and rewinding frames...", flush=True)

        predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))
        _sync_reid_threshold()

        cap.release()
        cap = FrameDirectoryCapture(
            frames_dir=frames_dir,
            timestamps_csv=timestamps_csv,
        )

        ok2, bgr2 = cap.read()
        if not ok2 or bgr2 is None:
            raise RuntimeError("Failed to read first frame after reset.")

        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

        state.update({
            "first_frame_loaded": False,
            "tracking": False,
            "injecting": False,
            "paused": True,
            "cands": [],
            "selected_idx": 0,
            "last_rgb": rgb2,
            "next_obj_id": 1,
            "added_obj_ids": [],
            "out_obj_ids": None,
            "out_mask_logits": None,
            "frame_number": 0,
            "eof": False,
        })

        noncond_ring = {}
        noncond_keys = deque(maxlen=600)

    def add_prompt_from_selected():
        if not state["cands"]:
            print("[add] No YOLO candidates available.", flush=True)
            return

        if state["last_rgb"] is None:
            print("[add] No frame available.", flush=True)
            return

        idx = clamp(state["selected_idx"], 0, len(state["cands"]) - 1)
        x1, y1, x2, y2, conf = state["cands"][idx]

        bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        obj_id = int(state["next_obj_id"])

        if not state["tracking"]:
            if not state["first_frame_loaded"]:
                predictor.load_first_frame(state["last_rgb"])
                _sync_reid_threshold()
                state["first_frame_loaded"] = True

            try:
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=obj_id,
                    bbox=bbox,
                )

                cs = predictor.condition_state
                cs.setdefault("reacquire_mode_per_id", {})
                cs["reacquire_mode_per_id"][int(obj_id)] = False

                state["out_obj_ids"] = out_obj_ids
                state["out_mask_logits"] = out_mask_logits
                state["added_obj_ids"].append(obj_id)
                state["next_obj_id"] += 1

                print(
                    f"[add] Added object #{obj_id} on initial frame, YOLO conf={conf:.2f}.",
                    flush=True,
                )

            except Exception as e:
                print(f"[add] add_new_prompt failed: {repr(e)}", flush=True)

            return

        try:
            state["injecting"] = True

            predictor.add_conditioning_frame(state["last_rgb"])
            _sync_reid_threshold()

            frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_prompt_during_track(
                bbox=bbox,
                if_new_target=True,
                obj_id=obj_id,
                labels=None,
                clear_old_points=True,
            )

            cs = predictor.condition_state
            cs.setdefault("reacquire_mode_per_id", {})
            cs["reacquire_mode_per_id"][int(obj_id)] = False

            state["out_obj_ids"] = out_obj_ids
            state["out_mask_logits"] = out_mask_logits
            state["added_obj_ids"].append(obj_id)
            state["next_obj_id"] += 1

            print(
                f"[add] Late-added object #{obj_id} at predictor frame_idx={frame_idx}, YOLO conf={conf:.2f}.",
                flush=True,
            )

        except Exception as e:
            print(f"[add] Late add failed: {repr(e)}", flush=True)

        finally:
            state["injecting"] = False

    def start_tracking():
        if not state["added_obj_ids"]:
            print("[track] Add at least one person first using A.", flush=True)
            return

        state["tracking"] = True
        state["paused"] = False

        print(f"[track] Tracking started/resumed. Objects: {state['added_obj_ids']}", flush=True)

    def dump_simple_case():
        cs = getattr(predictor, "condition_state", {}) if predictor is not None else {}
        pf = getattr(predictor, "frame_idx", None)
        pf = int(pf) if pf is not None else -1

        ts = time.strftime("%Y%m%d_%H%M%S")
        case_dir = run_dir / f"case_{ts}_pf{pf:06d}"
        safe_mkdir(case_dir)

        rgb = state["last_rgb"]

        if rgb is not None:
            cv2.imwrite(str(case_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if state["out_mask_logits"] is not None:
                overlay = draw_mask_overlay(
                    rgb.copy(),
                    state["out_obj_ids"],
                    state["out_mask_logits"],
                    alpha=args.alpha,
                )
                cv2.imwrite(str(case_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        summary = {
            "predictor_frame_idx": pf,
            "frame_number": int(state["frame_number"]),
            "added_obj_ids": list(state["added_obj_ids"]),
            "condition_state_obj_ids": cs.get("obj_ids", None) if isinstance(cs, dict) else None,
            "reid_last": cs.get("reid_last", None) if isinstance(cs, dict) else None,
            "reid_thr": cs.get("reid_thr", None) if isinstance(cs, dict) else None,
        }

        with open(case_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"[dump] Saved simple case to: {case_dir}", flush=True)

    last_time = time.time()
    display_fps = 0.0

    print("\nControls:", flush=True)
    print("  Left/Right arrows: select YOLO candidate", flush=True)
    print("  A: add selected candidate as object", flush=True)
    print("  T: start tracking / resume", flush=True)
    print("  SPACE: pause/resume", flush=True)
    print("  P: pause for prompting", flush=True)
    print("  D: dump simple debug case", flush=True)
    print("  Y: toggle YOLO overlay", flush=True)
    print("  +/-: adjust YOLO confidence", flush=True)
    print("  R: reset to first frame", flush=True)
    print("  Q or ESC: quit\n", flush=True)

    try:
        while True:
            current_rgb = state["last_rgb"]

            if current_rgb is None:
                break

            out_rgb_clean = current_rgb.copy()
            out_rgb_debug = current_rgb.copy()

            frame_dbg = {
                "object_score_logits": None,
                "object_score_prob": None,
                "object_score_thr": None,
                "reid_ok": None,
                "good_mem_count": 0,
                "good_mem_frames": [],
                "reacquire_mode_per_id": {},
                "any_reacquire": False,
            }

            if state["tracking"] and not state["paused"] and not state["injecting"] and not state["eof"]:
                out_obj_ids, out_mask_logits = predictor.track(current_rgb)

                cs = predictor.condition_state
                state["out_obj_ids"] = out_obj_ids
                state["out_mask_logits"] = out_mask_logits

                pf_now = int(getattr(predictor, "frame_idx", -1))

                if pf_now >= 0:
                    _ring_store(pf_now, current_rgb)

                frame_dbg = _extract_reid_debug_info(predictor, pf_now)

                if args.reid_print:
                    print("reid keys:", [k for k in cs.keys() if "reid" in str(k)])
                    print("reid_last:", cs.get("reid_last", None))

                out_rgb_clean = draw_mask_overlay(
                    out_rgb_clean,
                    out_obj_ids,
                    out_mask_logits,
                    alpha=args.alpha,
                )
                out_rgb_debug = out_rgb_clean.copy()

                next_ok, next_bgr = cap.read()

                if next_ok and next_bgr is not None:
                    if args.realtime_replay:
                        wait_s = cap.current_dt_from_previous()
                        if wait_s > 0:
                            time.sleep(wait_s)

                    next_rgb = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2RGB)
                    state["last_rgb"] = next_rgb
                    state["frame_number"] += 1

                else:
                    state["eof"] = True
                    state["paused"] = True
                    print("[frames] Reached end of sequence.", flush=True)

            else:
                if state["out_mask_logits"] is not None:
                    out_rgb_clean = draw_mask_overlay(
                        out_rgb_clean,
                        state["out_obj_ids"],
                        state["out_mask_logits"],
                        alpha=args.alpha,
                    )
                    out_rgb_debug = out_rgb_clean.copy()

            disp_bgr = cv2.cvtColor(out_rgb_debug, cv2.COLOR_RGB2BGR)
            clean_bgr = cv2.cvtColor(out_rgb_clean, cv2.COLOR_RGB2BGR)

            if state["yolo_enabled"]:
                cands = yolo_person_bboxes(
                    disp_bgr,
                    yolo_model,
                    conf_thres=state["yolo_conf"],
                )

                state["cands"] = cands

                if cands:
                    state["selected_idx"] = clamp(state["selected_idx"], 0, len(cands) - 1)

                    for j, (x1, y1, x2, y2, conf) in enumerate(cands):
                        is_sel = j == state["selected_idx"]
                        color = (0, 255, 0) if is_sel else (0, 200, 255)
                        thick = 3 if is_sel else 1

                        cv2.rectangle(disp_bgr, (x1, y1), (x2, y2), color, thick)
                        cv2.putText(
                            disp_bgr,
                            f"#{j} {conf:.2f}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

            now = time.time()
            dt = now - last_time
            last_time = now

            if dt > 0:
                display_fps = 0.9 * display_fps + 0.1 * (1.0 / dt)

            pf = int(getattr(predictor, "frame_idx", -1))

            hud = (
                f"FPS:{display_fps:4.1f}  "
                f"pf:{pf}  "
                f"frame:{state['frame_number']}/{max(total_frames - 1, 0)}  "
                f"tracking:{'ON' if state['tracking'] else 'OFF'}  "
                f"paused:{'YES' if state['paused'] else 'NO'}  "
                f"objs:{state['added_obj_ids']}  "
                f"sel:{state['selected_idx']}  "
                f"cands:{len(state['cands'])}"
            )

            cv2.putText(
                disp_bgr,
                hud,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            try:
                cs = predictor.condition_state
                thr = float(cs.get("reid_thr", getattr(predictor, "reid_thr", float("nan"))))
                rl = cs.get("reid_last", {}) if isinstance(cs, dict) else {}

                obj_list = cs.get("obj_ids", None) if isinstance(cs, dict) else None

                if isinstance(obj_list, list) and len(obj_list) > 0:
                    show_ids = [int(x) for x in obj_list]
                else:
                    show_ids = [int(x) for x in state.get("added_obj_ids", [])]

                logits_list = frame_dbg.get("object_score_logits", None)
                probs_list = frame_dbg.get("object_score_prob", None)
                reid_ok_list = frame_dbg.get("reid_ok", None)
                reacq_map = frame_dbg.get("reacquire_mode_per_id", {}) or {}

                y0 = 60
                dy = 60

                for i, oid in enumerate(show_ids):
                    info = rl.get(int(oid), None) if isinstance(rl, dict) else None

                    sim_txt = "--"
                    acc_txt = "NOINFO"
                    gallery_txt = "--"

                    if isinstance(info, dict):
                        sim = info.get("sim", None)
                        acc = info.get("accepted", None)
                        gsz = info.get("gallery_size", None)

                        if sim is not None:
                            sim_txt = f"{float(sim):.3f}"

                        if acc is True:
                            acc_txt = "ACCEPT"
                        elif acc is False:
                            acc_txt = "REJECT"
                        else:
                            acc_txt = "UNKNOWN"

                        if gsz is not None:
                            gallery_txt = str(int(gsz))

                    obj_logit_txt = "--"
                    obj_prob_txt = "--"
                    reid_ok_txt = "--"
                    reacq_txt = str(bool(reacq_map.get(int(oid), False)))

                    if isinstance(logits_list, list) and i < len(logits_list) and logits_list[i] is not None:
                        obj_logit_txt = f"{float(logits_list[i]):.3f}"

                    if isinstance(probs_list, list) and i < len(probs_list) and probs_list[i] is not None:
                        obj_prob_txt = f"{float(probs_list[i]):.3f}"

                    if isinstance(reid_ok_list, list) and i < len(reid_ok_list):
                        reid_ok_txt = str(int(reid_ok_list[i]))

                    line1 = (
                        f"id={oid} sim={sim_txt} thr={thr:.2f} "
                        f"reid={acc_txt} reacq={reacq_txt} gallery={gallery_txt}"
                    )

                    line2 = (
                        f"id={oid} obj_logit={obj_logit_txt} "
                        f"obj_prob={obj_prob_txt} mem_reid_ok={reid_ok_txt}"
                    )

                    cv2.putText(
                        disp_bgr,
                        line1,
                        (10, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        disp_bgr,
                        line2,
                        (10, y0 + i * dy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            except Exception:
                pass

            debug_writer.write(disp_bgr)
            clean_writer.write(clean_bgr)

            cv2.imshow(win, disp_bgr)
            key = cv2.waitKeyEx(1)

            if key in (27, ord("q"), ord("Q")):
                break

            elif key == 2424832:  # left arrow
                state["selected_idx"] = max(0, state["selected_idx"] - 1)

            elif key == 2555904:  # right arrow
                state["selected_idx"] = state["selected_idx"] + 1

            elif key in (ord("a"), ord("A")):
                state["paused"] = True
                add_prompt_from_selected()

            elif key in (ord("t"), ord("T")):
                start_tracking()

            elif key == ord(" "):
                if state["tracking"]:
                    state["paused"] = not state["paused"]
                    print(f"[frames] paused={state['paused']}", flush=True)

            elif key in (ord("p"), ord("P")):
                state["paused"] = True
                print("[frames] Paused for prompting.", flush=True)

            elif key in (ord("d"), ord("D")):
                dump_simple_case()

            elif key in (ord("y"), ord("Y")):
                state["yolo_enabled"] = not state["yolo_enabled"]
                print(f"[yolo] overlay: {'ON' if state['yolo_enabled'] else 'OFF'}", flush=True)

            elif key in (ord("+"), ord("=")):
                state["yolo_conf"] = min(0.95, state["yolo_conf"] + 0.05)
                print(f"[yolo] conf -> {state['yolo_conf']:.2f}", flush=True)

            elif key in (ord("-"), ord("_")):
                state["yolo_conf"] = max(0.01, state["yolo_conf"] - 0.05)
                print(f"[yolo] conf -> {state['yolo_conf']:.2f}", flush=True)

            elif key in (ord("r"), ord("R")):
                reset_all()

    finally:
        try:
            cap.release()
        except Exception:
            pass

        try:
            debug_writer.release()
        except Exception:
            pass

        try:
            clean_writer.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("\n[done] Exited cleanly.", flush=True)
        print(f"[saved] Debug video: {debug_video_path}", flush=True)
        print(f"[saved] Clean video: {clean_video_path}", flush=True)
        print(f"[saved] Debug folder: {run_dir}", flush=True)


if __name__ == "__main__":
    main()