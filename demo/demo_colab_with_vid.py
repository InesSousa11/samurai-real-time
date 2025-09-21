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

# ---------- small utils ----------
def _writable_dir():
    if os.path.isdir("/content"):
        return "/content"
    td = tempfile.gettempdir()
    return td if os.path.isdir(td) else os.getcwd()

def _resolve_video_path(video_input):
    # gr.File(type="filepath") returns a string path
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
    """
    Try multiple codecs; return (writer, final_path).
    We prefer MP4, fallback to AVI if needed.
    """
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

# -------- App state --------
state = {
    "seeded": False,
    "selected_idx": 0,
    "cands": [],
    "last_frame": None,
    "out_obj_ids": None,
    "out_mask_logits": None,

    # video session
    "source_mode": "Webcam",
    "video_path": None,
    "video_fps": 30.0,

    # saving (auto for video mode)
    "saving_enabled": False,     # set True when Start video is pressed
    "save_name": "segmented_output",
    "save_fps": 30.0,            # equals video_fps for video mode
    "writer": None,
    "writer_size": None,         # (w, h)
    "save_path": None,
}

# ---- writer helpers (auto open on first segmented frame, close at the end) ----
def _maybe_open_writer_on_first_segmented(frame_rgb):
    if not state["saving_enabled"] or state["writer"] is not None:
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
    if not state["saving_enabled"] or state["writer"] is None:
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
    # keep save_path to present for download; caller may clear it afterwards
    return path if path and os.path.exists(path) else None

# -------- Core frame processor (used by webcam & video playback) --------
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
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Tracking → segmented frame
    try:
        out_obj_ids, out_mask_logits = predictor.track(rgb_frame)
        state["out_obj_ids"] = out_obj_ids
        state["out_mask_logits"] = out_mask_logits
        out = draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits)
    except Exception as e:
        print("[error] track() failed:", repr(e))
        out = rgb_frame

    # open writer lazily on first segmented frame; then write each segmented frame
    _maybe_open_writer_on_first_segmented(out)
    _write_segmented_frame(out)
    return out

# -------- Seeding & navigation --------
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
    # finalize writer if any
    _finalize_writer()
    state.update({
        "seeded": False, "selected_idx": 0, "cands": [],
        "out_obj_ids": None, "out_mask_logits": None,
        "save_path": None,
    })
    return "Reset done."

# -------- Start video: set up save params and stream --------
def start_video(video_input, save_basename):
    """
    Generator that yields (frame, downloadable_file_or_None).
    - Pauses on first frame until user presses Accept (to pick bbox)
    - Automatically opens writer on first segmented frame
    - Automatically closes writer at end and yields the file for download
    """
    # setup save name & path/fps
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
    state["saving_enabled"] = True      # auto-save enabled for video mode

    delay = 1.0 / state["video_fps"]
    first = True

    while True:
        if first:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state["last_frame"] = rgb
            out = process_frame(rgb)
            yield out, None
            # pause here until Accept (so user picks which person)
            while not state["seeded"]:
                time.sleep(0.1)
                yield process_frame(state["last_frame"]), None
            first = False
            continue

        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = process_frame(rgb)  # writes segmented frame if seeded
        yield out, None
        time.sleep(delay)

    cap.release()
    # finalize & present file
    file_path = _finalize_writer()
    yield None, file_path

# -------- Gradio UI --------
with gr.Blocks() as demo:
    gr.Markdown("## SAMURAI real-time — Webcam or Video input (auto-save for Video)")

    src = gr.Radio(["Webcam", "Video"], value="Webcam", label="Source")
    cam = gr.Image(sources=["webcam"], streaming=True, visible=True, label="Webcam", type="numpy")
    vid = gr.File(label="Video file", visible=False, type="filepath", file_types=["video"])
    save_name = gr.Textbox(label="Output base filename (no extension)", value="segmented_output", visible=False)

    out = gr.Image(label="Output", type="numpy")
    download = gr.File(label="Download (appears after video ends)")

    with gr.Row():
        btn_prev = gr.Button("Prev")
        btn_accept = gr.Button("Accept")
        btn_next = gr.Button("Next")
        btn_reset = gr.Button("Reset")
        btn_start_vid = gr.Button("Start video")

    status = gr.Markdown("Status: waiting…")

    def toggle_src(choice):
        state["source_mode"] = choice
        on_reset()
        return (
            gr.update(visible=(choice=="Webcam")),
            gr.update(visible=(choice=="Video")),
            gr.update(visible=(choice=="Video")),
        )

    src.change(fn=toggle_src, inputs=src, outputs=[cam, vid, save_name])

    # Webcam live preview (no auto-save here)
    cam.stream(fn=process_frame, inputs=cam, outputs=out)

    # nav/seed/reset
    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)
    btn_accept.click(fn=on_accept, inputs=None, outputs=status)
    btn_reset.click(fn=on_reset, inputs=None, outputs=status)

    # Video playback (+auto save)
    btn_start_vid.click(fn=start_video, inputs=[vid, save_name], outputs=[out, download])

demo.launch(share=True)