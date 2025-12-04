# demo_colab_with_vid.py
import os
import cv2
import time
import numpy as np
import torch
import gradio as gr
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ultralytics import YOLO

import warnings
warnings.filterwarnings(
    "ignore",
    message="cannot import name '_C' from 'sam2'",
    category=UserWarning,
)

# -------- Performance knobs --------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------- Build predictor --------
from sam2.build_sam import build_sam2_camera_predictor

REPO = "/content/samurai-real-time"
CKPT = f"{REPO}/checkpoints/sam2.1_hiera_small.pt"
CFG  = "configs/samurai/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(CFG, CKPT)

# --- runtime score logger ---
from rt_scores import ScoresLogger

# --- check SAMURAI-mode ---
def _read_attr(obj, name):
    for host in (obj, getattr(obj, "model", None), getattr(obj, "module", None)):
        if host is not None and hasattr(host, name):
            return getattr(host, name)
    return None

_val = _read_attr(predictor, "samurai_mode")
if _val is not None:
    print(f"SAMURAI mode (from config): {'ON' if _val else 'OFF'}")

# ---------------- YOLO models ----------------
# Body/person detector (COCO)
yolo_body_model = YOLO("yolov8s.pt")  # auto-downloads if missing

# Face detector (Ultralytics hub, 1-class 'face' id=0). Auto-download if available.
try:
    yolo_face_model = YOLO("yolov8n-face.pt")  # try 'yolov8s-face.pt' for stronger model
    print("[face] Loaded YOLOv8 face model from Ultralytics hub.")
except Exception as e:
    print("[face] Could not load YOLO face model from hub:", repr(e))
    yolo_face_model = None

# ---------- small utils ----------
def _writable_dir():
    return "/tmp"

def _resolve_video_path(video_input):
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict) and "name" in video_input:
        return video_input["name"]
    return None

def _try_open_writer(base_path, size, fps):
    w, h = size
    attempts = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]
    base, _ = os.path.splitext(base_path)
    for fourcc_str, ext in attempts:
        test_path = base + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(test_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, test_path
        writer.release()
    return None, None

# -------- Helpers (vision) --------
def yolo_person_bboxes(rgb_frame, model, conf_thres=0.25):
    """
    Returns list of (x1, y1, x2, y2, conf) for class 'person' from COCO model.
    """
    if rgb_frame is None:
        return []
    res = model(rgb_frame, verbose=False, conf=conf_thres)[0]
    out = []
    for det in res.boxes:
        if int(det.cls) == 0:  # person class in COCO
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item()) if det.conf is not None else 0.0
            out.append((x1, y1, x2, y2, conf))
    out.sort(key=lambda t: t[4], reverse=True)
    return out

def yolo_face_bboxes(rgb_frame, model, conf_thres=0.25):
    """
    Returns list of (x1, y1, x2, y2, conf) for face detections.
    Assumes the face model has a single 'face' class (id 0).
    """
    if rgb_frame is None or model is None:
        return []
    res = model(rgb_frame, verbose=False, conf=conf_thres)[0]
    out = []
    for det in res.boxes:
        if int(det.cls) == 0:  # face model is typically 1-class: 0='face'
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item()) if det.conf is not None else 0.0
            out.append((x1, y1, x2, y2, conf))
    out.sort(key=lambda t: t[4], reverse=True)
    return out

def _count_objs(out_obj_ids):
    if out_obj_ids is None:
        return 0
    if isinstance(out_obj_ids, (list, tuple)):
        return len(out_obj_ids)
    if torch.is_tensor(out_obj_ids):
        return int(out_obj_ids.shape[0]) if out_obj_ids.ndim >= 1 else int(out_obj_ids.numel())
    return 0

# ----- id-stable color helpers -----
def _to_id_list(out_obj_ids):
    """Normalize ids to a Python list[int]."""
    if out_obj_ids is None:
        return []
    if isinstance(out_obj_ids, (list, tuple)):
        return [int(x) for x in out_obj_ids]
    if torch.is_tensor(out_obj_ids):
        return [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
    return [int(out_obj_ids)]

def _id_to_hue(obj_id: int) -> int:
    """Deterministic hue in [0,179] for OpenCV HSV (H channel)."""
    return int((37 * int(obj_id) + 61) % 180)

# ----- stable per-ID overlay -----
def draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits):
    if rgb_frame is None:
        return None
    ids = _to_id_list(out_obj_ids)

    # How many masks do we actually have?
    if isinstance(out_mask_logits, (list, tuple)):
        M = len(out_mask_logits)
        get_logits = lambda i: out_mask_logits[i]
    elif torch.is_tensor(out_mask_logits):
        M = int(out_mask_logits.shape[0]) if out_mask_logits.ndim >= 1 else 0
        get_logits = lambda i: out_mask_logits[i]
    else:
        M = 0
        get_logits = lambda i: None

    n = max(0, min(len(ids), M))
    if n == 0:
        return rgb_frame

    h, w = rgb_frame.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # saturation
    hsv[..., 2] = 0    # value; set to 255 where mask present

    for i in range(n):
        logits_i = get_logits(i)
        if logits_i is None:
            continue

        if isinstance(logits_i, torch.Tensor):
            if logits_i.ndim == 3:
                m = (logits_i > 0).permute(1, 2, 0)
            elif logits_i.ndim == 2:
                m = (logits_i > 0).unsqueeze(-1)
            else:
                continue
            m = m.detach().cpu().numpy().astype(np.uint8) * 255
        else:
            continue

        sel = m[..., 0] == 255
        hue = _id_to_hue(ids[i])
        hsv[sel, 0] = hue
        hsv[sel, 2] = 255

    overlay_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, overlay_rgb, 0.5, 0.0)

# -------- App state --------
state = {
    "first_frame_loaded": False,
    "seeded_any": False,
    "tracking": False,

    # proposals
    "proposal_type": "Body",   # "Body" or "Face"
    "proposals_on": True,      # ON/OFF for both modes
    "selected_idx": 0,
    "cands": [],
    "last_frame": None,

    "next_obj_id": 1,
    "added_obj_ids": [],

    "out_obj_ids": None,
    "out_mask_logits": None,

    "video_path": None,
    "video_fps": 30.0,
    "saving_enabled": False,
    "save_name": "segmented_output",
    "save_fps": 30.0,
    "writer": None,
    "writer_size": None,
    "save_path": None,

    "frame_idx": 0,
    "scores": ScoresLogger(),
    "selected_obj_for_plot": 1,
    "last_scores_row": {},

    "injecting": False,   # pause tracking safely during late-join
}

# ---- writer helpers ----
def _maybe_open_writer_on_first_segmented(frame_rgb):
    if not state["saving_enabled"] or state["writer"] is not None or frame_rgb is None:
        return
    h, w = frame_rgb.shape[:2]
    base_dir = _writable_dir()
    base_path = os.path.join(base_dir, state["save_name"])
    writer, final_path = _try_open_writer(base_path, (w, h), state["save_fps"])
    if writer is None:
        print("[save] Failed to open writer.")
        state["saving_enabled"] = False
        return
    state["writer"] = writer
    state["writer_size"] = (w, h)
    state["save_path"] = final_path
    print(f"[save] Writer opened: {final_path} @ {state['save_fps']:.2f} FPS")

def _write_segmented_frame(frame_rgb):
    if not state["saving_enabled"] or state["writer"] is None or frame_rgb is None:
        return
    w, h = state["writer_size"]
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if (bgr.shape[1], bgr.shape[0]) != (w, h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    state["writer"].write(bgr)

def _finalize_writer():
    if state["writer"] is not None:
        try:
            state["writer"].release()
        except Exception:
            pass
    path = state["save_path"]
    state["writer"] = None
    state["writer_size"] = None
    state["saving_enabled"] = False
    return path if path and os.path.exists(path) else None

# -------- Scores / plot helpers --------
def _refresh_plot(obj_id:int):
    state["selected_obj_for_plot"] = int(obj_id)
    return state["scores"].make_plot(int(obj_id))

def _refresh_latest_scores(obj_id:int):
    state["selected_obj_for_plot"] = int(obj_id)
    row = state["scores"].latest_row(int(obj_id))
    if not row:
        return f"Object #{obj_id}: no scores yet."
    cells = "".join(f"<tr><td><b>{k}</b></td><td>{v:.4f}</td></tr>" for k,v in row.items())
    return f"<table>{cells}</table>"

def _frames_query(obj_id:int, key:str, mode:str, t1, t2):
    try:
        t1 = float(t1) if t1 is not None else 0.0
    except Exception:
        t1 = 0.0
    try:
        t2 = float(t2) if t2 is not None else None
    except Exception:
        t2 = None
    frames = state["scores"].frames_where(int(obj_id), key, mode, t1, t2)
    if not frames:
        return "(no matches)"
    show = frames[:400]
    return ", ".join(map(str, show)) + (" …" if len(frames) > len(show) else "")

def _export_csv(obj_id:int):
    path = f"/tmp/scores_obj_{int(obj_id)}.csv"
    state["scores"].export_csv(int(obj_id), path)
    return path

def _choices_refresh():
    if state["added_obj_ids"]:
        ch = [int(x) for x in state["added_obj_ids"]]
        default = ch[0]
    else:
        ch, default = [1], 1
    return gr.update(choices=ch, value=default)

# -------- New: multi-ID diagnostics --------
def _plot_all_ids_small_multiples():
    """Four panels (combined/affinity/object/motion) with all IDs overlaid."""
    keys = ["combined", "affinity", "object", "motion"]
    ids = sorted(state["scores"].per_obj.keys())
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=tuple(k for k in keys),
                        vertical_spacing=0.05)
    colors = {}
    for i, oid in enumerate(ids):
        # deterministic color per id
        hue = (37 * int(oid) + 61) % 360
        colors[oid] = f"hsl({hue},80%,45%)"
    for r, key in enumerate(keys, start=1):
        for oid in ids:
            ss = state["scores"].per_obj[oid]
            x = list(ss.frames)
            y = list(ss.values[key])
            if any(isinstance(v, float) and not np.isnan(v) for v in y):
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode="lines", name=f"#{oid}",
                               legendgroup=f"id{oid}", line=dict(color=colors[oid])),
                    row=r, col=1
                )
        fig.update_yaxes(title_text=key, row=r, col=1, range=[-0.05, 1.05])
    fig.update_layout(height=800, title="All IDs — small multiples", showlegend=True)
    return fig

def _plot_compare_ids(id_a:int, id_b:int):
    """Overlay A & B for the four scores."""
    keys = ["combined", "affinity", "object", "motion"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=tuple(k for k in keys),
                        vertical_spacing=0.05)
    ids = [id_a, id_b]
    palette = {id_a: "royalblue", id_b: "orangered"}
    for r, key in enumerate(keys, start=1):
        for oid in ids:
            ss = state["scores"].per_obj.get(int(oid))
            if ss is None:
                continue
            x = list(ss.frames)
            y = list(ss.values[key])
            if any(isinstance(v, float) and not np.isnan(v) for v in y):
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode="lines", name=f"{key} #{oid}",
                               line=dict(color=palette[oid])),
                    row=r, col=1
                )
        fig.update_yaxes(title_text=key, row=r, col=1, range=[-0.05, 1.05])
    fig.update_layout(height=800, title=f"Compare IDs #{id_a} vs #{id_b}", showlegend=True)
    return fig

def _detect_events(
    thr_drop=0.10, min_drop_len=5,
    thr_reappear=0.60, min_gap=10,
    swap_window=8, swap_delta=0.20
):
    """
    Heuristics using ONLY the logged scores:
      - drop: combined < thr_drop for >= min_drop_len
      - reappear: combined crosses above thr_reappear after at least min_gap NaN/low frames
      - swap candidate: for any pair, within a window there is a crossing where one ↓ and the other ↑
                        by at least swap_delta in combined.
    Returns HTML table.
    """
    def _segments_below(y, thr, min_len):
        segs = []
        start = None
        for i, v in enumerate(y):
            valok = isinstance(v, float) and not np.isnan(v)
            below = valok and v < thr
            if below and start is None:
                start = i
            if (not below or i == len(y)-1) and start is not None:
                end = i if not below else i
                if end - start + 1 >= min_len:
                    segs.append((start, end))
                start = None
        return segs

    rows = []
    # Drops & reappears per id
    for oid, ss in state["scores"].per_obj.items():
        x = list(ss.frames)
        y = list(ss.values["combined"])
        if not x:
            continue
        # drops
        for i0, i1 in _segments_below(y, thr_drop, min_drop_len):
            rows.append(("drop", int(oid), int(x[i0]), int(x[i1]), f"combined<{thr_drop:.2f}"))
        # reappears
        # gap = consecutive frames with NaN or low
        gap = 0
        for i in range(1, len(y)):
            v_prev = y[i-1]
            v = y[i]
            prev_low = not (isinstance(v_prev, float) and not np.isnan(v_prev)) or v_prev < thr_reappear
            cur_high = isinstance(v, float) and not np.isnan(v) and v >= thr_reappear
            gap = gap + 1 if prev_low else 0
            if cur_high and gap >= min_gap:
                rows.append(("reappear", int(oid), int(x[i]), int(x[i]), f"combined>{thr_reappear:.2f}"))
    # Swap candidates (pairwise)
    ids = sorted(state["scores"].per_obj.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            oi, oj = ids[i], ids[j]
            si, sj = state["scores"].per_obj[oi], state["scores"].per_obj[oj]
            xi, yi = list(si.frames), list(si.values["combined"])
            xj, yj = list(sj.frames), list(sj.values["combined"])
            # two-pointer over frames
            p = q = 0
            window = swap_window
            while p < len(xi) and q < len(xj):
                fi, fj = xi[p], xj[q]
                if fi == fj:
                    vi, vj = yi[p], yj[q]
                    ok_i = isinstance(vi, float) and not np.isnan(vi)
                    ok_j = isinstance(vj, float) and not np.isnan(vj)
                    if ok_i and ok_j:
                        # look ahead within window for opposite trends
                        p2, q2 = min(p+window, len(xi)-1), min(q+window, len(xj)-1)
                        vi2, vj2 = yi[p2], yj[q2]
                        ok_i2 = isinstance(vi2, float) and not np.isnan(vi2)
                        ok_j2 = isinstance(vj2, float) and not np.isnan(vj2)
                        if ok_i2 and ok_j2:
                            di = vi2 - vi
                            dj = vj2 - vj
                            if (di <= -swap_delta and dj >= swap_delta) or (dj <= -swap_delta and di >= swap_delta):
                                rows.append(("swap?", f"{oi}↔{oj}", int(fi), int(xi[p2]), f"Δi={di:+.2f}, Δj={dj:+.2f}"))
                    p += 1; q += 1
                elif fi < fj:
                    p += 1
                else:
                    q += 1

    if not rows:
        return "<i>No events detected yet.</i>"

    # Build HTML table
    header = "<tr><th>type</th><th>id(s)</th><th>frame_start</th><th>frame_end</th><th>note</th></tr>"
    body = "\n".join(
        f"<tr><td>{t}</td><td>{ids_}</td><td>{fs}</td><td>{fe}</td><td>{note}</td></tr>"
        for (t, ids_, fs, fe, note) in rows
    )
    return f"<table>{header}{body}</table>"

# -------- Core (webcam & video) --------
@torch.inference_mode()
def process_frame(rgb_frame):
    if rgb_frame is None:
        return None
    state["last_frame"] = rgb_frame

    base = rgb_frame

    # tracking step (unless we are injecting)
    if state["tracking"] and not state.get("injecting", False):
        try:
            out_obj_ids, out_mask_logits = predictor.track(rgb_frame)
            state["out_obj_ids"] = out_obj_ids
            state["out_mask_logits"] = out_mask_logits
            base = draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits)

            state["scores"].log_from_predictor(
                predictor=predictor,
                obj_ids=out_obj_ids,
                frame_idx=state["frame_idx"]
            )
            state["frame_idx"] += 1

        except Exception as e:
            print("[error] track() failed:", repr(e))
            print(traceback.format_exc())
            base = rgb_frame
        _maybe_open_writer_on_first_segmented(base)
        _write_segmented_frame(base)

    # proposals (body or face), drawn on top of current base image
    if state["proposals_on"]:
        if state["proposal_type"] == "Body":
            cands = yolo_person_bboxes(rgb_frame, yolo_body_model, conf_thres=0.25)
            label = "BODY"
            color_sel = (0, 255, 0)     # green for selected
            color_oth = (0, 200, 255)   # teal for others
        else:
            cands = yolo_face_bboxes(rgb_frame, yolo_face_model, conf_thres=0.30)
            label = "FACE"
            color_sel = (255, 0, 255)   # magenta for selected
            color_oth = (200, 100, 255) # light magenta

        state["cands"] = cands
        bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR).copy()
        if cands:
            state["selected_idx"] = max(0, min(state["selected_idx"], len(cands) - 1))
            for j, (x1, y1, x2, y2, conf) in enumerate(cands):
                color = color_sel if j == state["selected_idx"] else color_oth
                thick = 3 if j == state["selected_idx"] else 1
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thick)
                cv2.putText(
                    bgr, f"{label}:{conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                )
            hint = "[Accept]=add  [Next]/[Prev]=cycle  [Toggle Proposals]=hide/show"
        else:
            if state["proposal_type"] == "Face" and yolo_face_model is None:
                hint = "Face proposals OFF (face model unavailable)."
            else:
                hint = f"No {label.lower()} found."
        cv2.putText(bgr, hint, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        base = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return base

# -------- Controls --------
def on_next():
    if state["proposals_on"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] + 1) % len(state["cands"])
    return None

def on_prev():
    if state["proposals_on"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] - 1) % len(state["cands"])
    return None

def on_toggle_proposals():
    state["proposals_on"] = not state["proposals_on"]
    return f"Proposals: {'ON' if state['proposals_on'] else 'OFF'}"

def on_proposal_mode(choice:str):
    state["proposal_type"] = choice
    state["selected_idx"] = 0
    return f"Proposal mode: {choice}"

def on_accept():
    # Must have a candidate and a current frame
    if not state["cands"] or state["last_frame"] is None:
        return "No candidate available."

    # Clamp index to avoid race-y out-of-range
    n = len(state["cands"])
    state["selected_idx"] = max(0, min(state["selected_idx"], n - 1))

    x1, y1, x2, y2, conf = state["cands"][state["selected_idx"]]
    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

    # ----- CASE A: pre-tracking seeding -----
    if not state["tracking"]:
        if not state["first_frame_loaded"]:
            predictor.load_first_frame(state["last_frame"])
            state["first_frame_loaded"] = True

        obj_id = state["next_obj_id"]
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=0, obj_id=obj_id, bbox=bbox
        )

        state["seeded_any"] = True
        state["next_obj_id"] += 1
        state["added_obj_ids"].append(obj_id)
        state["scores"].register_ids([obj_id])  # align x-axes from the start
        state["out_obj_ids"] = out_obj_ids
        state["out_mask_logits"] = out_mask_logits

        if len(state["added_obj_ids"]) == 1:
            state["selected_obj_for_plot"] = obj_id

        return f"Added object #{obj_id} from {state['proposal_type']} (conf={conf:.2f}). You can add more or press 'Start Tracking'."

    # ----- CASE B: late-join during tracking -----
    obj_id = state["next_obj_id"]
    try:
        state["injecting"] = True   # pause tracking loop safely
        predictor.add_conditioning_frame(state["last_frame"])
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_prompt_during_track(
            bbox=bbox,
            if_new_target=True,
            obj_id=obj_id,
            labels=None,
            clear_old_points=True,
        )
    except NotImplementedError:
        return "Late-join path not implemented in predictor yet. We’ll add it next."
    except Exception as e:
        return f"Failed to add during tracking: {repr(e)}"
    finally:
        state["injecting"] = False  # resume tracking

    # Register & update UI state just like pre-seed case:
    state["next_obj_id"] += 1
    state["added_obj_ids"].append(obj_id)
    state["scores"].register_ids([obj_id])
    state["out_obj_ids"] = out_obj_ids
    state["out_mask_logits"] = out_mask_logits

    return f"Added NEW object during tracking from {state['proposal_type']}: #{obj_id} (conf={conf:.2f})."

def on_start_tracking():
    if not state["seeded_any"]:
        return "No objects added yet. Accept at least one candidate first."

    num_objs = len(state["added_obj_ids"])
    state["scores"].register_ids(state["added_obj_ids"])

    state["tracking"] = True
    state["frame_idx"] = 0
    state["last_scores_row"] = {}
    return f"Tracking started. (objects={num_objs}, samurai_mode=ON)"

def on_reset():
    global predictor
    predictor = build_sam2_camera_predictor(CFG, CKPT)

    _finalize_writer()
    state.update({
        "first_frame_loaded": False,
        "seeded_any": False,
        "tracking": False,
        "proposal_type": "Body",
        "proposals_on": True,
        "selected_idx": 0,
        "cands": [],
        "last_frame": None,
        "next_obj_id": 1,
        "added_obj_ids": [],
        "out_obj_ids": None,
        "out_mask_logits": None,
        "video_path": None,
        "video_fps": 30.0,
        "saving_enabled": False,
        "save_name": "segmented_output",
        "save_fps": 30.0,
        "writer": None,
        "writer_size": None,
        "save_path": None,
        "frame_idx": 0,
        "scores": ScoresLogger(),
        "selected_obj_for_plot": 1,
        "last_scores_row": {},
        "injecting": False,
    })
    return "Reset done."

# UI wrappers
def on_accept_ui():
    status = on_accept()
    choices = _choices_refresh()
    fig = state["scores"].make_plot(state["selected_obj_for_plot"])
    info = _refresh_latest_scores(state["selected_obj_for_plot"])
    # also refresh the new diagnostics
    all_ids_fig = _plot_all_ids_small_multiples()
    events_html = _detect_events()
    return status, choices, fig, info, all_ids_fig, events_html

def on_reset_ui():
    status = on_reset()
    empty_fig = go.Figure()
    all_ids_empty = go.Figure()
    return status, gr.update(choices=[1], value=1), empty_fig, "—", all_ids_empty, "<i>—</i>"

# -------- Video --------
def start_video(video_input, save_basename):
    on_reset()
    state["save_name"] = (save_basename or "").strip() or "segmented_output"
    path = _resolve_video_path(video_input)
    state["video_path"] = path
    if not path or not os.path.exists(path):
        yield None, None
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield None, None
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    state["video_fps"] = float(fps)
    state["save_fps"]  = float(fps)
    state["saving_enabled"] = True

    delay = 1.0 / state["video_fps"]

    ok, bgr = cap.read()
    if not ok:
        cap.release()
        yield None, None
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    state["last_frame"] = rgb

    frame0 = process_frame(rgb)
    yield frame0, None

    while not state["tracking"]:
        time.sleep(0.05)
        yield process_frame(state["last_frame"]), None

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        state["last_frame"] = rgb
        out = process_frame(rgb)
        yield out, None
        time.sleep(delay)

    cap.release()
    file_path = _finalize_writer()
    yield None, file_path

# -------- UI --------
with gr.Blocks() as demo:
    gr.Markdown("## SAMURAI real-time — Multi-person seeding **before & during** tracking (Webcam or Video)")

    src = gr.Radio(["Webcam", "Video"], value="Webcam", label="Source")
    cam = gr.Image(sources=["webcam"], streaming=True, visible=True, label="Webcam", type="numpy")
    vid = gr.File(label="Video file", visible=False, type="filepath", file_types=["video"])
    save_name = gr.Textbox(label="Output base filename (no extension)", value="segmented_output", visible=False)

    out = gr.Image(label="Output", type="numpy")
    download = gr.File(label="Download (appears after video ends)")

    with gr.Row():
        proposal_mode = gr.Radio(["Body", "Face"], value="Body", label="Proposal type")
        btn_prev   = gr.Button("Prev")
        btn_accept = gr.Button("Accept (add)")
        btn_next   = gr.Button("Next")
        btn_toggle = gr.Button("Toggle Proposals")
        btn_start  = gr.Button("Start Tracking")
        btn_reset  = gr.Button("Reset")
        btn_start_vid = gr.Button("Start video")

    status = gr.Markdown("Status: waiting…")

    # --- Scores & Diagnostics ---
    with gr.Accordion("Scores & Diagnostics", open=False):
        with gr.Row():
            obj_select = gr.Dropdown(label="Object to plot", choices=[1], value=1, interactive=True)
            score_info = gr.HTML(label="Latest scores")

        plot = gr.Plot(label="Scores over time (selected object)")

        # --- New: All-IDs small-multiples ---
        all_ids_plot = gr.Plot(label="All IDs — small multiples")

        # --- New: Compare two IDs ---
        with gr.Row():
            cmp_a = gr.Dropdown(label="Compare: ID A", choices=[1], value=1)
            cmp_b = gr.Dropdown(label="Compare: ID B", choices=[1], value=1)
        cmp_plot = gr.Plot(label="Compare IDs")

        def _cmp_refresh(a, b):
            try:
                a = int(a); b = int(b)
            except Exception:
                return go.Figure()
            return _plot_compare_ids(a, b)

        cmp_a.change(fn=_cmp_refresh, inputs=[cmp_a, cmp_b], outputs=cmp_plot)
        cmp_b.change(fn=_cmp_refresh, inputs=[cmp_a, cmp_b], outputs=cmp_plot)

        # --- Frames query ---
        gr.Mardown = gr.Markdown  # defensive alias if old Gradio
        gr.Markdown("**Find frames by score**")
        with gr.Row():
            score_key = gr.Dropdown(choices=["object","iou","motion","affinity","combined"], value="object", label="Score")
            cmp_mode  = gr.Dropdown(choices=["<", ">", "<=", ">=", "between", "nan", "notnan"], value="<", label="Condition")
            t1 = gr.Number(value=0.0, label="T1")
            t2 = gr.Number(value=1.0, label="T2 (used for 'between')")

        frames_btn = gr.Button("Show frames")
        frames_box = gr.Textbox(label="Frames", lines=2)

        def _toggle_t2(mode):
            return gr.update(visible=(mode=="between"))

        cmp_mode.change(fn=_toggle_t2, inputs=cmp_mode, outputs=t2)
        frames_btn.click(fn=_frames_query, inputs=[obj_select, score_key, cmp_mode, t1, t2], outputs=frames_box)

        # --- CSV export ---
        with gr.Row():
            btn_csv = gr.Button("Export CSV (selected object)")
            download_csv = gr.File(label="Download CSV")
        btn_csv.click(fn=_export_csv, inputs=obj_select, outputs=download_csv)

        # --- New: Event detector ---
        gr.Markdown("**Events (drops, reappears, swap candidates)**")
        events_box = gr.HTML("<i>—</i>")

    def toggle_src(choice):
        on_reset()
        return (
            gr.update(visible=(choice=="Webcam")),
            gr.update(visible=(choice=="Video")),
            gr.update(visible=(choice=="Video")),
        )

    src.change(fn=toggle_src, inputs=src, outputs=[cam, vid, save_name])

    # Webcam stream + occasional plot refresh
    def _webcam_step(frame):
        img = process_frame(frame)
        # periodic refresh of diagnostics
        all_ids_fig = _plot_all_ids_small_multiples()
        events_html = _detect_events()
        if state["tracking"] and state["frame_idx"] % 5 == 0:
            p = state["scores"].make_plot(state["selected_obj_for_plot"])
            info = _refresh_latest_scores(state["selected_obj_for_plot"])
            return img, p, info, _choices_refresh(), all_ids_fig, events_html, _choices_refresh(), _choices_refresh(), _plot_compare_ids(state["selected_obj_for_plot"], state["selected_obj_for_plot"])
        return img, gr.update(), gr.update(), _choices_refresh(), all_ids_fig, events_html, _choices_refresh(), _choices_refresh(), gr.update()

    cam.stream(fn=_webcam_step, inputs=cam,
               outputs=[out, plot, score_info, obj_select, all_ids_plot, events_box, cmp_a, cmp_b, cmp_plot])

    # Buttons / controls
    proposal_mode.change(fn=on_proposal_mode, inputs=proposal_mode, outputs=status)
    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)
    btn_accept.click(fn=on_accept_ui, inputs=None,
                     outputs=[status, obj_select, plot, score_info, all_ids_plot, events_box])
    btn_toggle.click(fn=on_toggle_proposals, inputs=None, outputs=status)
    btn_start.click(fn=on_start_tracking, inputs=None, outputs=status)
    btn_reset.click(fn=on_reset_ui, inputs=None,
                    outputs=[status, obj_select, plot, score_info, all_ids_plot, events_box])

    btn_start_vid.click(fn=start_video, inputs=[vid, save_name], outputs=[out, download])

    # Periodic refresh with Timer (older-Gradio safe)
    timer = gr.Timer(0.7)
    timer.tick(fn=_refresh_plot, inputs=obj_select, outputs=plot)
    timer.tick(fn=_refresh_latest_scores, inputs=obj_select, outputs=score_info)
    timer.tick(fn=lambda: _plot_all_ids_small_multiples(), inputs=None, outputs=all_ids_plot)
    timer.tick(fn=lambda: _detect_events(), inputs=None, outputs=events_box)

    gr.Markdown("""
**How to use:**
- Pick **Proposal type** = **Body** (COCO person) or **Face** (YOLO face).
- Press **Accept** for each target you want (e.g., same person twice: once BODY, once FACE).
- Then **Start Tracking**.
- Use **All IDs — small multiples** to see everyone at once; use **Compare IDs** to overlay two IDs;
  check **Events** to jump to drops / reappears / swap candidates.

**Notes:**
- Face proposals auto-download the 'yolov8n-face.pt' model when available; if it fails, Face proposals are disabled.
- Seeding BODY and FACE for the same person creates two separate IDs so you can compare cues.
""")

demo.launch(share=True)