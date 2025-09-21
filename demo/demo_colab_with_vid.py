import os
import cv2
import time
import numpy as np
import torch
import gradio as gr
from ultralytics import YOLO
import tempfile

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

# YOLO for proposals
yolo_model = YOLO("yolov8s.pt")

def _writable_dir():
    if os.path.isdir("/content"):
        return "/content"
    td = tempfile.gettempdir()
    return td if os.path.isdir(td) else os.getcwd()

# -------- Helpers --------
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

def draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits):
    if rgb_frame is None:
        return None
    h, w = rgb_frame.shape[:2]
    if not out_obj_ids:
        return rgb_frame
    all_mask = np.zeros((h, w, 3), dtype=np.uint8)
    all_mask[..., 1] = 255
    for i in range(len(out_obj_ids)):
        m = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        hue = int((i + 3) / (len(out_obj_ids) + 3) * 255)
        sel = m[..., 0] == 255
        all_mask[sel, 0] = hue
        all_mask[sel, 2] = 255
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, all_mask, 0.5, 0.0)

def _resolve_video_path(video_input):
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict) and "name" in video_input:
        return video_input["name"]
    return None

def _safe_fps_from_file(path, fallback=30.0):
    try:
        cap = cv2.VideoCapture(path)
        vfps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if vfps and vfps > 0:
            return float(vfps)
    except Exception:
        pass
    return float(fallback)

def _try_open_writer(base_path, size, fps):
    """Try multiple codecs; return (writer, final_path)."""
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

# -------- App state --------
state = {
    "seeded": False,
    "selected_idx": 0,
    "cands": [],
    "last_frame": None,
    "out_obj_ids": None,
    "out_mask_logits": None,

    # source
    "source_mode": "Webcam",
    "video_path": None,
    "first_video_frame": None,

    # live saving (works for webcam & video; writes what you see)
    "save_on": False,
    "save_name": "segmented_output",
    "save_fps": 30.0,         # set from file in Video mode, else default
    "save_writer": None,
    "save_writer_size": None,
    "save_path": None,
}

# ---- live-saver: open lazily on first segmented frame, then write each frame you see
def _maybe_open_writer_if_needed(frame_rgb):
    if not state["save_on"] or state["save_writer"] is not None:
        return
    # open on first segmented frame (size lock)
    h, w = frame_rgb.shape[:2]
    base_dir = _writable_dir()
    base_path = os.path.join(base_dir, state["save_name"])
    writer, final_path = _try_open_writer(base_path, (w, h), state["save_fps"])
    if writer is None:
        print("[save] Failed to open writer.")
        state["save_on"] = False
        return
    state["save_writer"] = writer
    state["save_writer_size"] = (w, h)
    state["save_path"] = final_path
    print(f"[save] Writer opened: {final_path} @ {state['save_fps']:.2f} FPS")

def _maybe_write_stream(frame_rgb):
    if not state["save_on"] or state["save_writer"] is None:
        return
    w, h = state["save_writer_size"]
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if (bgr.shape[1], bgr.shape[0]) != (w, h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    state["save_writer"].write(bgr)

def start_saving(name, src_mode, video_input):
    # set desired fps before we open the writer (writer opens lazily on first segmented frame)
    state["save_name"] = (name or "").strip() or "segmented_output"
    state["save_on"] = True
    if src_mode == "Video":
        path = _resolve_video_path(video_input)
        state["video_path"] = path
        state["save_fps"] = _safe_fps_from_file(path, 30.0) if path else 30.0
    else:
        state["save_fps"] = 30.0  # webcam default; change if you want
    return f"Saving turned ON (writer will open on first segmented frame) @ {state['save_fps']:.2f} FPS."

def stop_saving():
    if state["save_writer"] is not None:
        try:
            state["save_writer"].release()
        except Exception:
            pass
    path = state["save_path"]
    # reset saving state
    state["save_on"] = False
    state["save_writer"] = None
    state["save_writer_size"] = None
    state["save_path"] = None
    if path and os.path.exists(path):
        return path, f"Saved: {path}"
    return None, "Nothing saved (no segmented frames were written)."

# -------- Core frame processor (preview & live) --------
@torch.inference_mode()
def process_frame(rgb_frame):
    if rgb_frame is None:
        return None

    state["last_frame"] = rgb_frame

    if not state["seeded"]:
        cands = yolo_person_bboxes(rgb_frame, yolo_model, conf_thres=0.25)
        state["cands"] = cands
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR).copy()
        if cands:
            state["selected_idx"] = max(0, min(state["selected_idx"], len(cands)-1))
            for j, (x1,y1,x2,y2,conf) in enumerate(cands):
                color = (0,255,0) if j == state["selected_idx"] else (0,200,255)
                thick = 3 if j == state["selected_idx"] else 1
                cv2.rectangle(bgr, (x1,y1), (x2,y2), color, thick)
                cv2.putText(bgr, f"{conf:.2f}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(bgr, f"[Accept]=seed  [Next]/[Prev]=cycle  people={len(cands)}",
                        (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(bgr, "No person found.", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Not writing yet (not segmented).
        return out

    # Tracking → segmented frame
    try:
        out_obj_ids, out_mask_logits = predictor.track(rgb_frame)
        state["out_obj_ids"] = out_obj_ids
        state["out_mask_logits"] = out_mask_logits
        out = draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits)
    except Exception as e:
        print("[error] track() failed:", repr(e))
        out = rgb_frame

    # Open writer lazily on the first segmented frame, then write every frame you see
    _maybe_open_writer_if_needed(out)
    _maybe_write_stream(out)
    return out

# -------- Buttons: navigation & seeding --------
def on_next():
    if not state["seeded"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] + 1) % len(state["cands"])
    return None

def on_prev():
    if not state["seeded"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] - 1) % len(state["cands"])
    return None

def on_accept():
    if state["seeded"]:
        return "Already seeded."
    if not state["cands"] or state["last_frame"] is None:
        return "No candidate available."
    x1, y1, x2, y2, conf = state["cands"][state["selected_idx"]]
    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    predictor.load_first_frame(state["last_frame"])
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox)
    state["seeded"] = True
    state["out_obj_ids"] = out_obj_ids
    state["out_mask_logits"] = out_mask_logits
    return f"Seeded. Tracking…"

def on_reset():
    global predictor
    predictor = build_sam2_camera_predictor(CFG, CKPT)
    # stop saving if active
    if state["save_writer"] is not None:
        try: state["save_writer"].release()
        except Exception: pass
    state.update({
        "seeded": False, "selected_idx": 0, "cands": [],
        "out_obj_ids": None, "out_mask_logits": None,
        "save_on": False, "save_writer": None, "save_writer_size": None, "save_path": None,
    })
    return "Reset done."

# -------- Video streaming (preview & play) --------
def stream_video(video_input):
    path = _resolve_video_path(video_input)
    state["video_path"] = path
    if not path or not os.path.exists(path):
        yield None
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield None
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1.0 / fps
    first_frame = True

    while True:
        if first_frame:
            ok, bgr = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state["last_frame"] = rgb
            out = process_frame(rgb)
            yield out
            # pause here until Accept
            while not state["seeded"]:
                time.sleep(0.1)
                yield process_frame(state["last_frame"])
            first_frame = False
            continue

        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = process_frame(rgb)  # this writes if saving is ON
        yield out
        time.sleep(delay)

    cap.release()

# -------- Gradio UI --------
with gr.Blocks() as demo:
    gr.Markdown("## SAMURAI real-time — Webcam or Video input (Save what you see)")

    src = gr.Radio(["Webcam", "Video"], value="Webcam", label="Source")
    cam = gr.Image(sources=["webcam"], streaming=True, visible=True, label="Webcam", type="numpy")
    vid = gr.File(label="Video file", visible=False, type="filepath", file_types=["video"])
    out = gr.Image(label="Output", type="numpy")

    with gr.Row():
        btn_prev = gr.Button("Prev")
        btn_accept = gr.Button("Accept")
        btn_next = gr.Button("Next")
        btn_reset = gr.Button("Reset")
        btn_start_vid = gr.Button("Start video")

    status = gr.Markdown("Status: waiting…")

    with gr.Row():
        save_name = gr.Textbox(label="Save filename (no extension)", value="segmented_output")
        btn_start_save = gr.Button("Start Saving")
        btn_stop_save = gr.Button("Stop & Download")
    download = gr.File(label="Saved video")

    def toggle_src(choice):
        state["source_mode"] = choice
        on_reset()
        return (
            gr.update(visible=(choice=="Webcam")),
            gr.update(visible=(choice=="Video")),
        )
    src.change(fn=toggle_src, inputs=src, outputs=[cam, vid])

    # webcam stream
    cam.stream(fn=process_frame, inputs=cam, outputs=out)

    # nav/seed/reset
    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)
    btn_accept.click(fn=on_accept, inputs=None, outputs=status)
    btn_reset.click(fn=on_reset, inputs=None, outputs=status)

    # video preview/play
    btn_start_vid.click(fn=stream_video, inputs=vid, outputs=out)

    # saving controls (works for both webcam & video, writes what you see)
    btn_start_save.click(fn=start_saving, inputs=[save_name, src, vid], outputs=status)
    btn_stop_save.click(fn=stop_saving, inputs=None, outputs=[download, status])

demo.launch(share=True)