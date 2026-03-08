import sys
import time
import cv2
from pathlib import Path
from collections import deque

import numpy as np
import torch
from ultralytics import YOLO

# ---- plotting (matplotlib) ----
import matplotlib
matplotlib.use("TkAgg")  # Windows-friendly
import matplotlib.pyplot as plt

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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "demo") else Path.cwd()

CKPT_PATH = (REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt").resolve()
CFG_PATH  = (REPO_ROOT / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()


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


class LiveScorePlotter:
    """
    Plots per-object:
      - best_iou_score (s_mask)      -> max over candidates
      - object score prob           -> sigmoid(object_score_logits)
      - kf_score                    -> max over candidates if available
      - combined                    -> max over candidates (α_kf*kf + (1-α_kf)*s_mask)
    """
    def __init__(self, window=300, update_every=2, alpha_kf=0.25):
        self.window = int(window)
        self.update_every = int(update_every)
        self.alpha_kf = float(alpha_kf)

        self.t = deque(maxlen=self.window)
        self.hist = {}   # obj_id -> dict of deques
        self.lines = {}  # obj_id -> 4 lines
        self.frame_counter = 0
        self.enabled = True

        plt.ion()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        self.fig.canvas.manager.set_window_title("SAMURAI realtime scores")

        self.axs[0].set_title("best_iou_score (s_mask) [max over candidates]")
        self.axs[1].set_title("object score prob = sigmoid(object_score_logits)")
        self.axs[2].set_title("kf_score [max over candidates if available]")
        self.axs[3].set_title(f"combined = α_kf*kf_score + (1-α_kf)*best_iou_score  (α_kf={self.alpha_kf:.2f})")

        for ax in self.axs:
            ax.grid(True, alpha=0.3)

    def set_alpha(self, alpha_kf: float):
        self.alpha_kf = float(alpha_kf)
        self.axs[3].set_title(
            f"combined = α_kf*kf_score + (1-α_kf)*best_iou_score  (α_kf={self.alpha_kf:.2f})"
        )

    def toggle(self):
        self.enabled = not self.enabled
        print(f"[plot] {'ON' if self.enabled else 'OFF'}")

    def _ensure_obj(self, obj_id: int):
        if obj_id in self.hist:
            return
        self.hist[obj_id] = {
            "iou": deque(maxlen=self.window),
            "objp": deque(maxlen=self.window),
            "kf": deque(maxlen=self.window),
            "comb": deque(maxlen=self.window),
        }
        ls = []
        for ax in self.axs:
            (ln,) = ax.plot([], [], label=f"id={obj_id}")
            ls.append(ln)
        self.lines[obj_id] = ls
        for ax in self.axs:
            ax.legend(loc="upper right", fontsize=9)

    @staticmethod
    def _to_float_or_nan(x):
        if x is None:
            return float("nan")
        if torch.is_tensor(x):
            x = x.detach().float().reshape(-1)
            if x.numel() == 0:
                return float("nan")
            # use max as "best"
            return float(torch.max(x).item())
        try:
            return float(x)
        except Exception:
            return float("nan")

    def update(self, frame_idx: int, scores_by_obj: dict):
        if not self.enabled:
            return

        self.frame_counter += 1
        if self.frame_counter % self.update_every != 0:
            return

        self.t.append(int(frame_idx))
        x = list(self.t)

        for obj_id, sc in scores_by_obj.items():
            obj_id = int(obj_id)
            self._ensure_obj(obj_id)

            iou = self._to_float_or_nan(sc.get("iou", None))
            obj_logit = self._to_float_or_nan(sc.get("obj_logit", None))
            kf = self._to_float_or_nan(sc.get("kf", None))
            comb_in = sc.get("comb", None)
            comb = self._to_float_or_nan(comb_in)

            obj_prob = 1.0 / (1.0 + np.exp(-obj_logit)) if np.isfinite(obj_logit) else float("nan")

            # If comb not provided by caller, compute from iou/kf
            if not np.isfinite(comb):
                a = self.alpha_kf
                iou_f = iou if np.isfinite(iou) else float("nan")
                kf_f = kf if np.isfinite(kf) else float("nan")
                if np.isfinite(kf_f) and np.isfinite(iou_f):
                    comb = a * kf_f + (1.0 - a) * iou_f
                elif np.isfinite(iou_f):
                    comb = iou_f
                elif np.isfinite(kf_f):
                    comb = kf_f
                else:
                    comb = float("nan")

            self.hist[obj_id]["iou"].append(iou if np.isfinite(iou) else float("nan"))
            self.hist[obj_id]["objp"].append(obj_prob)
            self.hist[obj_id]["kf"].append(kf if np.isfinite(kf) else float("nan"))
            self.hist[obj_id]["comb"].append(comb if np.isfinite(comb) else float("nan"))

        # update lines
        for obj_id, lines in self.lines.items():
            h = self.hist[obj_id]
            lines[0].set_data(x, list(h["iou"]))
            lines[1].set_data(x, list(h["objp"]))
            lines[2].set_data(x, list(h["kf"]))
            lines[3].set_data(x, list(h["comb"]))

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # helps on some Windows setups
        plt.pause(0.001)


def get_latest_scores_from_debug_last(predictor):
    """
    NEW (recommended): read per-frame, per-object debug from predictor.condition_state["debug_last"].

    Requires your patched SAM2Base:
      - track_step() sets condition_state["debug_last"] = {..., "per_obj": []}
      - _forward_sam_heads() appends dicts to debug_last["per_obj"] with tensors:
          ious [B,M], kf_ious [1,M] or None, combined [B,M], object_score_logits [B,1], etc.
    """
    cs = getattr(predictor, "condition_state", None)
    if not isinstance(cs, dict):
        return None, {}

    fidx = getattr(predictor, "frame_idx", None)
    if fidx is None:
        return None, {}

    dbg = cs.get("debug_last", None)
    if not isinstance(dbg, dict):
        return int(fidx), {}

    per_obj = dbg.get("per_obj", [])
    if not isinstance(per_obj, list) or len(per_obj) == 0:
        return int(fidx), {}

    # Map per_obj slot index -> real obj_id (same order as your per-object loop)
    obj_ids = cs.get("obj_ids", [])
    if not isinstance(obj_ids, list):
        obj_ids = []

    scores = {}
    for slot_i, d in enumerate(per_obj):
        if not isinstance(d, dict):
            continue

        oid = int(obj_ids[slot_i]) if slot_i < len(obj_ids) else int(slot_i)

        ious = d.get("ious", None)         # [B,M]
        kf_ious = d.get("kf_ious", None)   # [1,M] or None
        comb = d.get("combined", None)     # [B,M]
        obj_logit = d.get("object_score_logits", None)  # [B,1] (usually)

        # Reduce to scalar “best” values (max over candidates)
        def best_over_candidates(x):
            if x is None:
                return None
            if not torch.is_tensor(x):
                return x
            x = x.detach().float()
            if x.numel() == 0:
                return None
            # if [B,M] take max over M then max over B (B should be 1 here)
            return torch.max(x).cpu()

        best_iou = best_over_candidates(ious)
        best_kf  = best_over_candidates(kf_ious)  # may be None
        best_comb = best_over_candidates(comb)

        # object logit: [B,1] -> max
        best_obj_logit = best_over_candidates(obj_logit)

        scores[oid] = {
            "iou": best_iou,
            "kf": best_kf,
            "comb": best_comb,
            "obj_logit": best_obj_logit,
        }

    return int(fidx), scores


@torch.inference_mode()
def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {CKPT_PATH}")
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found:\n  {CFG_PATH}")

    print("[init] Building SAM2 camera predictor...")
    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))

    # Use alpha from model if available
    alpha_kf = float(getattr(predictor, "kf_score_weight", 0.25))

    print("[init] Loading YOLO (yolov8s.pt)...")
    yolo_model = YOLO("yolov8s.pt")

    plotter = LiveScorePlotter(window=300, update_every=2, alpha_kf=alpha_kf)

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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0. Try changing cv2.VideoCapture(1), etc.")

    win = "SAMURAI demo (keys: A add, T track, arrows select, P plot, Y yolo, R reset, Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def reset_all():
        nonlocal predictor, alpha_kf
        print("[reset] Rebuilding predictor and clearing state...")
        predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))
        alpha_kf = float(getattr(predictor, "kf_score_weight", 0.25))
        plotter.set_alpha(alpha_kf)
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

        # late join
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
        except Exception as e:
            print(f"[add] Late-join failed: {repr(e)}")
        finally:
            state["injecting"] = False

    def start_tracking():
        if not state["added_obj_ids"]:
            print("[track] Add at least one person first (press A).")
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

            # ---- tracking ----
            out_rgb = rgb
            if state["tracking"] and (not state["injecting"]):
                try:
                    out_obj_ids, out_mask_logits = predictor.track(rgb)
                    state["out_obj_ids"] = out_obj_ids
                    state["out_mask_logits"] = out_mask_logits
                    out_rgb = draw_mask_overlay(out_rgb, out_obj_ids, out_mask_logits, alpha=mask_alpha)

                    # >>> THIS is the only new thing you needed:
                    # Read scores from condition_state["debug_last"] (populated by your patched model)
                    fidx, scores_by_obj = get_latest_scores_from_debug_last(predictor)
                    if fidx is not None and scores_by_obj:
                        plotter.update(fidx, scores_by_obj)

                except Exception as e:
                    print(f"[track] predictor.track failed: {repr(e)}")
                    out_rgb = rgb

            disp_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            # ---- YOLO overlay ----
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
                f"plot:{'ON' if plotter.enabled else 'OFF'}  "
                f"alpha_kf:{plotter.alpha_kf:.2f}  "
                f"objs:{state['added_obj_ids']}  "
                f"sel:{state['selected_idx']}  "
                f"cands:{len(state['cands'])}"
            )
            cv2.putText(disp_bgr, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                disp_bgr,
                "Keys: <-/-> select | A add | T track | P plot | Y yolo | R reset | +/- conf | Q quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, disp_bgr)
            key = cv2.waitKey(1) & 0xFF

            # quit
            if key in (27, ord("q"), ord("Q")):
                break

            # arrows: left=81, right=83 on Windows OpenCV
            if key == 81:  # left
                state["selected_idx"] = max(0, state["selected_idx"] - 1)
            elif key == 83:  # right
                state["selected_idx"] = state["selected_idx"] + 1

            # commands
            elif key in (ord("a"), ord("A")):
                add_prompt_from_selected()
            elif key in (ord("t"), ord("T")):
                start_tracking()
            elif key in (ord("y"), ord("Y")):
                state["yolo_enabled"] = not state["yolo_enabled"]
                print(f"[yolo] overlay: {'ON' if state['yolo_enabled'] else 'OFF'}")
            elif key in (ord("p"), ord("P")):
                plotter.toggle()
            elif key in (ord("r"), ord("R")):
                reset_all()
            elif key in (ord("+"), ord("=")):  # = is same key as + without shift sometimes
                state["yolo_conf"] = min(0.95, state["yolo_conf"] + 0.05)
                print(f"[yolo] conf -> {state['yolo_conf']:.2f}")
            elif key in (ord("-"), ord("_")):
                state["yolo_conf"] = max(0.01, state["yolo_conf"] - 0.05)
                print(f"[yolo] conf -> {state['yolo_conf']:.2f}")

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            plt.ioff()
            plt.close("all")
        except Exception:
            pass
        print("\n[done] Exited cleanly.")


if __name__ == "__main__":
    main()