import os
import cv2
import time
import numpy as np
import torch
import gradio as gr
from ultralytics import YOLO

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
    # Gradio allows /tmp by default; safest choice
    return "/tmp"

def _resolve_video_path(video_input):
    # gr.File(type="filepath") returns a string path
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict) and "name" in video_input:
        return video_input["name"]
    return None

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
    # seeding / tracking
    "first_frame_loaded": False,   # predictor.load_first_frame done?
    "tracking_started": False,     # only True after user clicks "Start tracking"
    "selected_idx": 0,
    "cands": [],
    "last_frame": None,

    # multi-object
    "next_obj_id": 1,              # assigns 1,2,3,...
    "added_obj_ids": [],           # list of obj ids added so far (for preview)
    "out_obj_ids": None,           # last preview obj_ids (after add)
    "out_mask_logits": None,       # last preview masks (after add)

    # source
    "source_mode": "Webcam",
    "video_path": None,
    "video_fps": 30.0,

    # auto-save for video mode
    "saving_enabled": False,       # enabled when video starts; writes after tracking starts
    "save_name": "segmented_output",
    "save_fps": 30.0,
    "writer": None,
    "writer_size": None,
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
    # keep save_path so UI can still show the file
    return path if path and os.path.exists(path) else None

# -------- Core frame processor --------
@torch.inference_mode()
def process_frame(rgb_frame):
    """
    - Before tracking starts: show YOLO boxes + preview of any added objects on frame 0.
    - After tracking starts: call predictor.track() and optionally write segmented frames.
    """
    if rgb_frame is None:
        return None
    state["last_frame"] = rgb_frame

    # PRE-TRACKING (choose multiple people)
    if not state["tracking_started"]:
        # run YOLO proposals
        cands = yolo_person_bboxes(rgb_frame, yolo_model, conf_thres=0.25)
        state["cands"] = cands
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR).copy()

        # draw proposals
        if cands:
            state["selected_idx"] = max(0, min(state["selected_idx"], len(cands)-1))
            for j, (x1,y1,x2,y2,conf) in enumerate(cands):
                color = (0,255,0) if j == state["selected_idx"] else (0,200,255)
                thick = 3 if j == state["selected_idx"] else 1
                cv2.rectangle(bgr, (x1,y1), (x2,y2), color, thick)
                cv2.putText(bgr, f"{conf:.2f}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(bgr, "No person found.", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # if we already added some objects, show mask preview from last add
        if state["out_obj_ids"] and state["out_mask_logits"] is not None:
            preview = draw_mask_overlay(rgb_frame, state["out_obj_ids"], state["out_mask_logits"])
            # blend preview lightly over current boxes for clarity
            bgr_preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            bgr = cv2.addWeighted(bgr, 0.5, bgr_preview, 0.5, 0)

        # user guidance
        msg = "[Add person]=add current box  [Next]/[Prev]=cycle  [Start tracking]=begin"
        cv2.putText(bgr, msg, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # TRACKING (real-time)
    try:
        out_obj_ids, out_mask_logits = predictor.track(rgb_frame)
        state["out_obj_ids"] = out_obj_ids
        state["out_mask_logits"] = out_mask_logits
        out = draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits)
    except Exception as e:
        print("[error] track() failed:", repr(e))
        out = rgb_frame

    # video-mode auto-save
    _maybe_open_writer_on_first_segmented(out)
    _write_segmented_frame(out)
    return out

# -------- Seeding (multi-object) & navigation --------
def on_next():
    if not state["tracking_started"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] + 1) % len(state["cands"])
    return None

def on_prev():
    if not state["tracking_started"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] - 1) % len(state["cands"])
    return None

def on_add_person():
    """
    Add current selected detection as a new object (multiple times allowed before tracking).
    """
    if state["tracking_started"]:
        return "Tracking already started."
    if not state["cands"] or state["last_frame"] is None:
        return "No candidate available."

    # load first frame once
    if not state["first_frame_loaded"]:
        predictor.load_first_frame(state["last_frame"])
        state["first_frame_loaded"] = True

    # add the selected bbox as a new object id
    x1, y1, x2, y2, conf = state["cands"][state["selected_idx"]]
    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    obj_id = state["next_obj_id"]
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=obj_id, bbox=bbox)

    state["added_obj_ids"].append(obj_id)
    state["next_obj_id"] += 1
    state["out_obj_ids"] = out_obj_ids          # preview masks include all added objs
    state["out_mask_logits"] = out_mask_logits

    return f"Added obj #{obj_id} (conf={conf:.2f}). Add more or click Start tracking."

def on_start_tracking():
    """
    Lock prompts and start real-time tracking on subsequent frames.
    """
    if state["tracking_started"]:
        return "Already tracking."
    if not state["added_obj_ids"]:
        return "Add at least one person first."
    state["tracking_started"] = True
    return f"Tracking {len(state['added_obj_ids'])} object(s)…"

def on_reset():
    global predictor
    predictor = build_sam2_camera_predictor(CFG, CKPT)
    _finalize_writer()
    state.update({
        "first_frame_loaded": False,
        "tracking_started": False,
        "selected_idx": 0,
        "cands": [],
        "last_frame": None,
        "next_obj_id": 1,
        "added_obj_ids": [],
        "out_obj_ids": None,
        "out_mask_logits": None,
        "save_path": None,
    })
    return "Reset done."

# -------- Start video (multi-obj pause; auto-save) --------
def start_video(video_input, save_basename):
    """
    Generator that yields (frame, downloadable_file_or_None).
    - Shows frame 0 with proposals
    - You can click Add person multiple times
    - Click Start tracking to begin; auto-saves segmented output
    - On end, returns the file path to download
    """
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
    first = True

    while True:
        if first:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state["last_frame"] = rgb
            # ensure predictor binds frame 0 once you add the first object
            frame0 = process_frame(rgb)
            yield frame0, None

            # pause here until user starts tracking (can add multiple persons meanwhile)
            while not state["tracking_started"]:
                time.sleep(0.1)
                yield process_frame(state["last_frame"]), None
            first = False
            continue

        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = process_frame(rgb)
        yield out, None
        time.sleep(delay)

    cap.release()
    file_path = _finalize_writer()
    yield None, file_path

# -------- Gradio UI --------
with gr.Blocks() as demo:
    gr.Markdown("## SAMURAI real-time — Multi-object seeding (Webcam or Video) + Auto-save for Video")

    src = gr.Radio(["Webcam", "Video"], value="Webcam", label="Source")
    cam = gr.Image(sources=["webcam"], streaming=True, visible=True, label="Webcam", type="numpy")
    vid = gr.File(label="Video file", visible=False, type="filepath", file_types=["video"])
    save_name = gr.Textbox(label="Output base filename (no extension)", value="segmented_output", visible=False)

    out = gr.Image(label="Output", type="numpy")
    download = gr.File(label="Download (appears after video ends)")

    with gr.Row():
        btn_prev = gr.Button("Prev")
        btn_add  = gr.Button("Add person")
        btn_next = gr.Button("Next")
        btn_start = gr.Button("Start tracking")
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

    # Webcam live preview (no auto-save)
    cam.stream(fn=process_frame, inputs=cam, outputs=out)

    # selection & control buttons
    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)
    btn_add.click(fn=on_add_person, inputs=None, outputs=status)
    btn_start.click(fn=on_start_tracking, inputs=None, outputs=status)
    btn_reset.click(fn=on_reset, inputs=None, outputs=status)

    # Video playback + auto-save (download appears when finished)
    btn_start_vid.click(fn=start_video, inputs=[vid, save_name], outputs=[out, download])

    gr.Markdown("**Tip**: In Video mode, click **Add person** multiple times to seed several people, then click **Start tracking**. The segmented result is saved automatically and becomes downloadable when the video ends.")

demo.launch(share=True)