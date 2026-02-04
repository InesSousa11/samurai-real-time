# Terminal commands (type + Enter anytime):
#   help                 -> show commands
#   a <idx>              -> add person using YOLO candidate index (e.g., "a 0")
#   s <idx>              -> set/highlight current candidate index
#   t                    -> start tracking
#   y                    -> toggle YOLO overlay on/off
#   c <thr>              -> set YOLO conf threshold (e.g., "c 0.35")
#   r                    -> reset predictor/state (rebuild SAMURAI predictor)
#   q                    -> quit
#
# Keys in the OpenCV window:
#   ESC or q             -> quit

import sys
import time
import cv2
import threading
from queue import Queue, Empty
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# add repo root (parent of /demo) to python path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

from sam2.build_sam import build_sam2_camera_predictor


# ---------------- HARD-CODED PATHS (as in your demo) ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "demo") else Path.cwd()

CKPT_PATH = (REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt").resolve()
CFG_PATH  = (REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()


# ---------------- Utilities ----------------
def yolo_person_bboxes(bgr_frame, model, conf_thres=0.25):
    """Returns list of (x1,y1,x2,y2,conf) for class=person, sorted by conf desc."""
    if bgr_frame is None:
        return []
    res = model(bgr_frame, verbose=False, conf=conf_thres)[0]
    out = []
    if res.boxes is None:
        return out
    for det in res.boxes:
        if int(det.cls) == 0:  # person
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


def _id_to_hue(obj_id: int) -> int:
    return int((37 * int(obj_id) + 61) % 180)


def draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits, alpha=0.5):
    """Overlay segmentation masks on rgb_frame with deterministic per-ID colors."""
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
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = 0

    for i in range(n):
        logits_i = get_logits(i)
        if logits_i is None or not torch.is_tensor(logits_i):
            continue

        if logits_i.ndim == 3:
            m = (logits_i[0] > 0)
        elif logits_i.ndim == 2:
            m = (logits_i > 0)
        else:
            continue

        m = m.detach().cpu().numpy().astype(bool)
        hue = _id_to_hue(ids[i])
        hsv[m, 0] = hue
        hsv[m, 2] = 255

    overlay_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, overlay_rgb, float(alpha), 0.0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------- Terminal input thread ----------------
def stdin_reader(cmd_queue: Queue, stop_flag: threading.Event):
    while not stop_flag.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.05)
                continue
            line = line.strip()
            if line:
                cmd_queue.put(line)
        except Exception:
            time.sleep(0.05)


def print_help():
    print(
        "\nCommands:\n"
        "  help                 -> show this help\n"
        "  a <idx>              -> add person using YOLO candidate index (e.g., a 0)\n"
        "  s <idx>              -> select/highlight candidate index\n"
        "  t                    -> start tracking\n"
        "  y                    -> toggle YOLO overlay\n"
        "  c <thr>              -> set YOLO conf threshold (e.g., c 0.35)\n"
        "  r                    -> reset predictor/state\n"
        "  q                    -> quit\n"
    )


# ---------------- Main app ----------------
@torch.inference_mode()
def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {CKPT_PATH}")
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found:\n  {CFG_PATH}")

    print("[paths]")
    print("  REPO_ROOT:", REPO_ROOT)
    print("  CKPT     :", CKPT_PATH)
    print("  CFG      :", CFG_PATH)

    # Performance knobs (safe)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else None

    # Build predictor + YOLO
    print("[init] Building SAM2 camera predictor...")
    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))

    print("[init] Loading YOLO (yolov8s.pt)...")
    yolo_model = YOLO("yolov8s.pt")

    # State
    state = {
        "first_frame_loaded": False,
        "tracking": False,
        "injecting": False,

        "yolo_enabled": True,
        "yolo_conf": 0.25,

        "cands": [],
        "selected_idx": 0,
        "last_rgb": None,

        "next_obj_id": 1,
        "added_obj_ids": [],

        "out_obj_ids": None,
        "out_mask_logits": None,
    }

    # Terminal command queue + thread
    cmd_queue = Queue()
    stop_flag = threading.Event()
    th = threading.Thread(target=stdin_reader, args=(cmd_queue, stop_flag), daemon=True)
    th.start()
    print_help()
    print("[info] Video is running. Type commands in the terminal any time.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0. Try changing cv2.VideoCapture(1), etc.")

    win = "SAMURAI minimal demo (ESC/q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def reset_all():
        nonlocal predictor
        print("[reset] Rebuilding predictor and clearing state...")
        predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))
        state.update({
            "first_frame_loaded": False,
            "tracking": False,
            "injecting": False,
            "cands": [],
            "selected_idx": 0,
            "last_rgb": None,
            "next_obj_id": 1,
            "added_obj_ids": [],
            "out_obj_ids": None,
            "out_mask_logits": None,
        })

    def add_prompt_from_selected():
        if not state["cands"]:
            print("[add] No YOLO candidates available.")
            return
        if state["last_rgb"] is None:
            print("[add] No frame yet.")
            return

        idx = clamp(state["selected_idx"], 0, len(state["cands"]) - 1)
        x1, y1, x2, y2, conf = state["cands"][idx]
        bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

        obj_id = state["next_obj_id"]

        # Pre-tracking seeding
        if not state["tracking"]:
            if not state["first_frame_loaded"]:
                predictor.load_first_frame(state["last_rgb"])
                state["first_frame_loaded"] = True

            try:
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=0, obj_id=obj_id, bbox=bbox
                )
                state["out_obj_ids"] = out_obj_ids
                state["out_mask_logits"] = out_mask_logits
                state["added_obj_ids"].append(obj_id)
                state["next_obj_id"] += 1
                print(f"[add] Added object #{obj_id} (conf={conf:.2f}). Added so far: {state['added_obj_ids']}")
            except Exception as e:
                print(f"[add] add_new_prompt failed: {repr(e)}")
            return

        # Late-join during tracking
        try:
            state["injecting"] = True
            predictor.add_conditioning_frame(state["last_rgb"])
            frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_prompt_during_track(
                bbox=bbox,
                if_new_target=True,
                obj_id=obj_id,
                labels=None,
                clear_old_points=True,
            )
            state["out_obj_ids"] = out_obj_ids
            state["out_mask_logits"] = out_mask_logits
            state["added_obj_ids"].append(obj_id)
            state["next_obj_id"] += 1
            print(f"[add] Late-joined object #{obj_id} at predictor frame_idx={frame_idx} (conf={conf:.2f}).")
        except NotImplementedError:
            print("[add] predictor.add_new_prompt_during_track is not implemented in your predictor.")
        except Exception as e:
            print(f"[add] Late-join failed: {repr(e)}")
        finally:
            state["injecting"] = False

    def start_tracking():
        reminder = "Add at least one person first (use 'a <idx>')."
        if not state["added_obj_ids"]:
            print("[track]", reminder)
            return
        state["tracking"] = True
        print(f"[track] Tracking started. Objects: {state['added_obj_ids']}")

    last_time = time.time()
    fps = 0.0
    mask_alpha = 0.5

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("[cam] Frame grab failed.")
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state["last_rgb"] = rgb

            # ---- process terminal commands (non-blocking) ----
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except Empty:
                    break

                parts = cmd.split()
                if not parts:
                    continue
                k = parts[0].lower()

                if k in ("help", "h", "?"):
                    print_help()

                elif k in ("q", "quit", "exit"):
                    raise KeyboardInterrupt

                elif k in ("y", "yolo"):
                    state["yolo_enabled"] = not state["yolo_enabled"]
                    print(f"[yolo] overlay: {'ON' if state['yolo_enabled'] else 'OFF'}")

                elif k in ("c", "conf") and len(parts) >= 2:
                    try:
                        state["yolo_conf"] = float(parts[1])
                        print(f"[yolo] conf threshold set to {state['yolo_conf']:.3f}")
                    except Exception:
                        print("[yolo] usage: c <thr>  (e.g., c 0.35)")

                elif k in ("s", "sel", "select") and len(parts) >= 2:
                    try:
                        state["selected_idx"] = int(parts[1])
                        print(f"[sel] selected_idx = {state['selected_idx']}")
                    except Exception:
                        print("[sel] usage: s <idx>")

                elif k in ("a", "add"):
                    if len(parts) >= 2:
                        try:
                            state["selected_idx"] = int(parts[1])
                        except Exception:
                            pass
                    add_prompt_from_selected()

                elif k in ("t", "track", "start"):
                    start_tracking()

                elif k in ("r", "reset"):
                    reset_all()

                else:
                    print(f"[cmd] Unknown: {cmd}  (type 'help')")

            # ---- tracking ----
            out_rgb = rgb
            if state["tracking"] and (not state["injecting"]):
                try:
                    if autocast_ctx is not None:
                        with autocast_ctx:
                            out_obj_ids, out_mask_logits = predictor.track(rgb)
                    else:
                        out_obj_ids, out_mask_logits = predictor.track(rgb)

                    state["out_obj_ids"] = out_obj_ids
                    state["out_mask_logits"] = out_mask_logits
                    out_rgb = draw_mask_overlay(out_rgb, out_obj_ids, out_mask_logits, alpha=mask_alpha)
                except Exception as e:
                    print(f"[track] predictor.track failed: {repr(e)}")
                    out_rgb = rgb

            # ---- YOLO proposals overlay ----
            disp_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            if state["yolo_enabled"]:
                cands = yolo_person_bboxes(disp_bgr, yolo_model, conf_thres=state["yolo_conf"])
                state["cands"] = cands

                if cands:
                    state["selected_idx"] = clamp(state["selected_idx"], 0, len(cands) - 1)
                    for j, (x1, y1, x2, y2, conf) in enumerate(cands):
                        is_sel = (j == state["selected_idx"])
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

            # ---- HUD ----
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            hud = (
                f"FPS:{fps:4.1f}  "
                f"YOLO:{'ON' if state['yolo_enabled'] else 'OFF'}(conf={state['yolo_conf']:.2f})  "
                f"tracking:{'ON' if state['tracking'] else 'OFF'}  "
                f"objs:{state['added_obj_ids']}  "
                f"sel:{state['selected_idx']}  "
                f"cands:{len(state['cands'])}"
            )
            cv2.putText(disp_bgr, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                disp_bgr,
                "Terminal: a <idx> add | t start | y toggle | r reset | q quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, disp_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q in window
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("\n[done] Exited cleanly.")


if __name__ == "__main__":
    main()