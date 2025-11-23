# demo_colab_with_vid.py
import os
import cv2
import time
import numpy as np
import torch
import gradio as gr
import traceback
import plotly.graph_objects as go

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

# YOLO for proposals
yolo_model = YOLO("yolov8s.pt")

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
    if rgb_frame is None:
        return []
    res = model(rgb_frame, verbose=False, conf=conf_thres)[0]
    out = []
    for det in res.boxes:
        if int(det.cls) == 0:  # person
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

# ----- NEW: id-stable color helpers -----
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
    """
    Deterministic hue in [0, 179] for OpenCV HSV (H channel).
    Using a golden-ratio-ish step to spread colors nicely.
    """
    return int((37 * int(obj_id) + 61) % 180)

# ----- UPDATED: stable per-ID overlay -----
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
    hsv[..., 1] = 255  # full saturation
    hsv[..., 2] = 0    # value; set to 255 only where mask present

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

    "yolo_enabled": True,
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

    "injecting": False,   # NEW: pause tracking safely during late-join
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

# -------- Core (webcam & video) --------
@torch.inference_mode()
def process_frame(rgb_frame):
    if rgb_frame is None:
        return None
    state["last_frame"] = rgb_frame

    base = rgb_frame

    # skip tracking while injecting a late-join prompt
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

    if state["yolo_enabled"]:
        cands = yolo_person_bboxes(rgb_frame, yolo_model, conf_thres=0.25)
        state["cands"] = cands
        bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR).copy()
        if cands:
            state["selected_idx"] = max(0, min(state["selected_idx"], len(cands)-1))
            for j, (x1,y1,x2,y2,conf) in enumerate(cands):
                color = (0,255,0) if j == state["selected_idx"] else (0,200,255)
                thick = 3 if j == state["selected_idx"] else 1
                cv2.rectangle(bgr, (x1,y1), (x2,y2), color, thick)
                cv2.putText(bgr, f"{conf:.2f}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            hint = "[Accept]=add person  [Next]/[Prev]=cycle  [Toggle YOLO]=hide/show"
        else:
            hint = "No person found."
        cv2.putText(bgr, hint, (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        base = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return base

# -------- Controls --------
def on_next():
    if state["yolo_enabled"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] + 1) % len(state["cands"])
    return None

def on_prev():
    if state["yolo_enabled"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] - 1) % len(state["cands"])
    return None

def on_toggle_yolo():
    state["yolo_enabled"] = not state["yolo_enabled"]
    return f"YOLO proposals: {'ON' if state['yolo_enabled'] else 'OFF'}"

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

        # (Removed) do NOT disable SAMURAI when adding more than 1 object

        if len(state["added_obj_ids"]) == 1:
            state["selected_obj_for_plot"] = obj_id

        return f"Added object #{obj_id} (conf={conf:.2f}). You can add more or press 'Start Tracking'."

    # ----- CASE B: late-join during tracking -----
    obj_id = state["next_obj_id"]
    try:
        state["injecting"] = True   # pause tracking loop safely

        # Make sure the current frame exists in predictor's conditioning buffer
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

    # (Removed) do NOT disable SAMURAI when adding more than 1 object

    return f"Added NEW object during tracking: #{obj_id} (conf={conf:.2f})."

def on_start_tracking():
    if not state["seeded_any"]:
        return "No objects added yet. Accept at least one person first."

    num_objs = len(state["added_obj_ids"])
    # Keep SAMURAI/KF on for multi-object as well
    set_samurai_mode(predictor, enable=(num_objs >= 1))

    # ensure all seeded ids are registered before first tracked frame
    state["scores"].register_ids(state["added_obj_ids"])

    state["tracking"] = True
    state["frame_idx"] = 0
    state["last_scores_row"] = {}
    return f"Tracking started. (objects={num_objs}, samurai_mode={'ON' if num_objs>=1 else 'OFF'})"

def on_reset():
    global predictor
    predictor = build_sam2_camera_predictor(CFG, CKPT)
    set_samurai_mode(predictor, True)

    _finalize_writer()
    state.update({
        "first_frame_loaded": False,
        "seeded_any": False,
        "tracking": False,
        "yolo_enabled": True,
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
    return status, choices, fig, info

def on_reset_ui():
    status = on_reset()
    empty_fig = go.Figure()
    return status, gr.update(choices=[1], value=1), empty_fig, "—"

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
        btn_prev   = gr.Button("Prev")
        btn_accept = gr.Button("Accept (add person)")
        btn_next   = gr.Button("Next")
        btn_toggle = gr.Button("Toggle YOLO")
        btn_start  = gr.Button("Start Tracking")
        btn_reset  = gr.Button("Reset")
        btn_start_vid = gr.Button("Start video")

    status = gr.Markdown("Status: waiting…")

    # --- Scores & Diagnostics ---
    with gr.Accordion("Scores & Diagnostics", open=False):
        with gr.Row():
            obj_select = gr.Dropdown(label="Object to plot", choices=[1], value=1, interactive=True)
            score_info = gr.HTML(label="Latest scores")

        plot = gr.Plot(label="Scores over time")

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
        if state["tracking"] and state["frame_idx"] % 5 == 0:
            p = state["scores"].make_plot(state["selected_obj_for_plot"])
            info = _refresh_latest_scores(state["selected_obj_for_plot"])
            return img, p, info, _choices_refresh()
        return img, gr.update(), gr.update(), _choices_refresh()

    cam.stream(fn=_webcam_step, inputs=cam, outputs=[out, plot, score_info, obj_select])

    # Buttons / controls
    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)
    btn_accept.click(fn=on_accept_ui, inputs=None, outputs=[status, obj_select, plot, score_info])
    btn_toggle.click(fn=on_toggle_yolo, inputs=None, outputs=status)
    btn_start.click(fn=on_start_tracking, inputs=None, outputs=status)
    btn_reset.click(fn=on_reset_ui, inputs=None, outputs=[status, obj_select, plot, score_info])

    btn_start_vid.click(fn=start_video, inputs=[vid, save_name], outputs=[out, download])

    # Periodic refresh with Timer (older-Gradio safe)
    timer = gr.Timer(0.5)
    timer.tick(fn=_refresh_plot, inputs=obj_select, outputs=plot)
    timer.tick(fn=_refresh_latest_scores, inputs=obj_select, outputs=score_info)

    gr.Markdown("""
**How to use:**
- **Webcam:** YOLO ON, press **Accept** for each person (you can add many). Then **Start Tracking**.
- You can also press **Accept** *after* you pressed **Start Tracking** to add **new people during tracking** (late-join).
- **Video:** Upload file → **Start video**. On the first frame Accept several, then **Start Tracking**.
  When it finishes, a download appears.

**Score analysis:**
- Hover the Plotly chart to read exact frame/score at any point.
- Use **Find frames by score** to list frames matching conditions (e.g., object < 0).
- Use **Export CSV** to download all scores per frame for the selected object.
""")

demo.launch(share=True)