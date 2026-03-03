# Terminal commands (type + Enter anytime):
#   help                 -> show commands
#   a <idx>              -> add person using YOLO candidate index (e.g., "a 0")
#   s <idx>              -> set/highlight current candidate index
#   t                    -> start tracking
#   y                    -> toggle YOLO overlay on/off
#   c <thr>              -> set YOLO conf threshold (e.g., "c 0.35")
#   f <fps>              -> set processing FPS (e.g., "f 15"). "f 0" = uncapped
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

# ---------------- Path setup ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

from sam2.build_sam import build_sam2_camera_predictor


# ---------------- HARD-CODED PATHS ----------------
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


def ema(prev, x, alpha=0.2):
    return x if prev is None else (1.0 - alpha) * prev + alpha * x


def cuda_sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
        "  f <fps>              -> set processing FPS (e.g., f 15). f 0 = uncapped\n"
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

    print("[init] Building SAM2 camera predictor...")
    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))

    print("[init] Loading YOLO (yolov8s.pt)...")
    yolo_model = YOLO("yolov8s.pt")

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

        # Processing FPS control
        "target_fps": 15.0,   # default; set to 0 for uncapped
    }

    cmd_queue = Queue()
    stop_flag = threading.Event()
    th = threading.Thread(target=stdin_reader, args=(cmd_queue, stop_flag), daemon=True)
    th.start()
    print_help()
    print("[info] Video is running. Type commands in the terminal any time.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0. Try changing cv2.VideoCapture(1), etc.")

    win = "SAMURAI fixed-FPS demo (ESC/q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # ---------------- Capture thread: keep ONLY the latest frame ----------------
    latest = {"bgr": None, "t": 0.0}
    latest_lock = threading.Lock()
    cap_stop = threading.Event()

    # Capture metrics
    cap_count = 0
    cap_fps_ema = None
    cap_last_t = time.perf_counter()
    cap_last_count = 0

    def capture_loop():
        nonlocal cap_count, cap_fps_ema, cap_last_t, cap_last_count
        while not cap_stop.is_set():
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            t_now = time.perf_counter()
            with latest_lock:
                latest["bgr"] = bgr
                latest["t"] = t_now

            cap_count += 1

            # update camFPS about ~every 1s
            dt = t_now - cap_last_t
            if dt >= 1.0:
                inst = (cap_count - cap_last_count) / max(dt, 1e-6)
                cap_fps_ema = ema(cap_fps_ema, inst, alpha=0.25)
                cap_last_t = t_now
                cap_last_count = cap_count

    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    cap_thread.start()

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
        if not state["added_obj_ids"]:
            print("[track] Add at least one person first (use 'a <idx>').")
            return
        state["tracking"] = True
        print(f"[track] Tracking started. Objects: {state['added_obj_ids']}")

    # Processing metrics
    proc_count = 0
    proc_fps_ema = 0.0
    proc_dt_ema = None  # avg processing time per processed frame
    yolo_dt_ema = None
    track_dt_ema = None
    overlay_dt_ema = None

    last_wall = time.time()
    next_tick = time.perf_counter()
    mask_alpha = 0.5

    try:
        while True:
            # ---- commands ----
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
                elif k in ("f", "fps") and len(parts) >= 2:
                    try:
                        v = float(parts[1])
                        state["target_fps"] = max(0.0, v)
                        if state["target_fps"] > 0:
                            print(f"[fps] target processing FPS = {state['target_fps']:.2f}")
                        else:
                            print("[fps] uncapped (process as fast as possible)")
                    except Exception:
                        print("[fps] usage: f <fps>  (e.g., f 15) or f 0")
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

            # ---- fixed-rate tick ----
            target_fps = float(state.get("target_fps", 0.0))
            now = time.perf_counter()

            if target_fps > 0:
                period = 1.0 / target_fps
                if now < next_tick:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    time.sleep(min(0.002, next_tick - now))
                    continue

                while next_tick <= now:
                    next_tick += period
            else:
                next_tick = now  # uncapped

            # ---- get latest frame ----
            with latest_lock:
                if latest["bgr"] is None:
                    bgr = None
                    frame_age_ms = 0.0
                else:
                    bgr = latest["bgr"].copy()
                    frame_age_ms = (time.perf_counter() - latest["t"]) * 1000.0

            if bgr is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state["last_rgb"] = rgb

            # ---- timed processing step ----
            t0 = time.perf_counter()

            out_rgb = rgb

            # tracking time (if enabled)
            t_track = 0.0
            if state["tracking"] and (not state["injecting"]):
                try:
                    cuda_sync_if_needed()
                    tt0 = time.perf_counter()
                    if autocast_ctx is not None:
                        with autocast_ctx:
                            out_obj_ids, out_mask_logits = predictor.track(rgb)
                    else:
                        out_obj_ids, out_mask_logits = predictor.track(rgb)
                    cuda_sync_if_needed()
                    t_track = time.perf_counter() - tt0

                    state["out_obj_ids"] = out_obj_ids
                    state["out_mask_logits"] = out_mask_logits

                    # overlay time
                    cuda_sync_if_needed()
                    to0 = time.perf_counter()
                    out_rgb = draw_mask_overlay(out_rgb, out_obj_ids, out_mask_logits, alpha=mask_alpha)
                    cuda_sync_if_needed()
                    t_overlay = time.perf_counter() - to0
                    overlay_dt_ema = ema(overlay_dt_ema, t_overlay, alpha=0.2)

                except Exception as e:
                    print(f"[track] predictor.track failed: {repr(e)}")
                    out_rgb = rgb

            if t_track > 0:
                track_dt_ema = ema(track_dt_ema, t_track, alpha=0.2)

            # YOLO overlay time (if enabled)
            disp_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            t_yolo = 0.0
            if state["yolo_enabled"]:
                cuda_sync_if_needed()
                ty0 = time.perf_counter()
                cands = yolo_person_bboxes(disp_bgr, yolo_model, conf_thres=state["yolo_conf"])
                cuda_sync_if_needed()
                t_yolo = time.perf_counter() - ty0
                yolo_dt_ema = ema(yolo_dt_ema, t_yolo, alpha=0.2)

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
            else:
                state["cands"] = []

            # end processing step timing
            proc_dt = time.perf_counter() - t0
            proc_dt_ema = ema(proc_dt_ema, proc_dt, alpha=0.2)

            proc_count += 1

            # update procFPS (wall time between displayed frames)
            now_wall = time.time()
            dt_wall = now_wall - last_wall
            last_wall = now_wall
            if dt_wall > 0:
                proc_fps_ema = 0.9 * proc_fps_ema + 0.1 * (1.0 / dt_wall)

            # derived metrics
            cam_fps = float(cap_fps_ema) if cap_fps_ema is not None else 0.0
            max_fps_est = (1.0 / proc_dt_ema) if (proc_dt_ema is not None and proc_dt_ema > 1e-6) else 0.0
            drop_fps_est = max(0.0, cam_fps - proc_fps_ema)
            dropped_total_est = max(0, cap_count - proc_count)

            # ---- HUD ----
            hud1 = (
                f"camFPS:{cam_fps:4.1f}  "
                f"procFPS:{proc_fps_ema:4.1f}  "
                f"maxFPS:{max_fps_est:4.1f}  "
                f"dropFPS:{drop_fps_est:4.1f}  "
                f"age:{frame_age_ms:4.0f}ms"
            )
            hud2 = (
                f"target:{state['target_fps']:g}  "
                f"YOLO:{'ON' if state['yolo_enabled'] else 'OFF'}(conf={state['yolo_conf']:.2f})  "
                f"tracking:{'ON' if state['tracking'] else 'OFF'}  "
                f"droppedTot:{dropped_total_est}"
            )
            hud3 = (
                f"objs:{state['added_obj_ids']}  sel:{state['selected_idx']}  cands:{len(state['cands'])}  "
                f"t(proc/yolo/track/ovl): "
                f"{(proc_dt_ema or 0)*1000:4.0f}/"
                f"{(yolo_dt_ema or 0)*1000:4.0f}/"
                f"{(track_dt_ema or 0)*1000:4.0f}/"
                f"{(overlay_dt_ema or 0)*1000:4.0f} ms"
            )

            cv2.putText(disp_bgr, hud1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp_bgr, hud2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp_bgr, hud3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(
                disp_bgr,
                "Terminal: a <idx> add | t start | f <fps> | y toggle | r reset | q quit",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, disp_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        cap_stop.set()
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