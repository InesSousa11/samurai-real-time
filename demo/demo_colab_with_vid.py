import os
import cv2
import time
import numpy as np
import torch
import gradio as gr
import traceback
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

# YOLO para propostas
yolo_model = YOLO("yolov8s.pt")

# ---------- utils pequenos ----------
def _writable_dir():
    return "/tmp"            # compatível com Gradio

def _resolve_video_path(video_input):
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict) and "name" in video_input:
        return video_input["name"]
    return None

def _try_open_writer(base_path, size, fps):
    """Tenta vários codecs; devolve (writer, caminho_final)."""
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

# -------- Helpers (visão) --------
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

def draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits):
    if rgb_frame is None:
        return None

    n = _count_objs(out_obj_ids)
    if n == 0:
        return rgb_frame

    h, w = rgb_frame.shape[:2]
    all_mask = np.zeros((h, w, 3), dtype=np.uint8)
    all_mask[..., 1] = 255  # saturação

    for i in range(n):
        if isinstance(out_mask_logits, (list, tuple)):
            logits_i = out_mask_logits[i]
        elif torch.is_tensor(out_mask_logits):
            logits_i = out_mask_logits[i]
        else:
            continue

        # garante shape (H,W,1)
        if logits_i.ndim == 3:
            m = (logits_i > 0).permute(1, 2, 0)
        elif logits_i.ndim == 2:
            m = (logits_i > 0).unsqueeze(-1)
        else:
            continue
        m = m.detach().cpu().numpy().astype(np.uint8) * 255

        hue = int((i + 3) / (n + 3) * 255)
        sel = m[..., 0] == 255
        all_mask[sel, 0] = hue
        all_mask[sel, 2] = 255

    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, all_mask, 0.5, 0.0)

# -------- Estado --------
state = {
    # sessão & seeding
    "first_frame_loaded": False,
    "seeded_any": False,
    "tracking": False,

    # propostas
    "yolo_enabled": True,
    "selected_idx": 0,
    "cands": [],
    "last_frame": None,

    # multi-obj
    "next_obj_id": 1,
    "added_obj_ids": [],

    # último output
    "out_obj_ids": None,
    "out_mask_logits": None,

    # vídeo & auto-save
    "video_path": None,
    "video_fps": 30.0,
    "saving_enabled": False,
    "save_name": "segmented_output",
    "save_fps": 30.0,
    "writer": None,
    "writer_size": None,
    "save_path": None,
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

# -------- Core (webcam & vídeo) --------
@torch.inference_mode()
def process_frame(rgb_frame):
    """
    Webcam:
      - Enquanto tracking=False podes dar Accept várias vezes (todos em frame 0).
      - Carrega Start Tracking para começar a seguir (não podes adicionar depois).
    Vídeo:
      - Pausa no 1º frame para aceitar vários; Start Tracking inicia playback.
    """
    if rgb_frame is None:
        return None
    state["last_frame"] = rgb_frame

    base = rgb_frame

    if state["tracking"]:
        try:
            out_obj_ids, out_mask_logits = predictor.track(rgb_frame)
            state["out_obj_ids"] = out_obj_ids
            state["out_mask_logits"] = out_mask_logits
            base = draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits)
        except Exception as e:
            print("[error] track() failed:", repr(e))
            print(traceback.format_exc())  # <<< mostra a linha exacta
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
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return base

# -------- Controlo --------
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
    if state["tracking"]:
        return "Tracking already started; cannot add new objects. Reset to re-seed."
    if not state["cands"] or state["last_frame"] is None:
        return "No candidate available."

    x1, y1, x2, y2, conf = state["cands"][state["selected_idx"]]
    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

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
    state["out_obj_ids"] = out_obj_ids
    state["out_mask_logits"] = out_mask_logits

    return f"Added object #{obj_id} (conf={conf:.2f}). You can add more or press 'Start Tracking'."

def on_start_tracking():
    if not state["seeded_any"]:
        return "No objects added yet. Accept at least one person first."
    state["tracking"] = True
    return "Tracking started."

def on_reset():
    global predictor
    predictor = build_sam2_camera_predictor(CFG, CKPT)
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
    })
    return "Reset done."

# -------- Vídeo (pausa para seed; depois tracking + auto-save) --------
def start_video(video_input, save_basename):
    on_reset()  # sessão nova

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
    gr.Markdown("## SAMURAI real-time — Multi-person seeding **before** tracking (Webcam or Video)")

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

    def toggle_src(choice):
        on_reset()
        return (
            gr.update(visible=(choice=="Webcam")),
            gr.update(visible=(choice=="Video")),
            gr.update(visible=(choice=="Video")),
        )

    src.change(fn=toggle_src, inputs=src, outputs=[cam, vid, save_name])

    cam.stream(fn=process_frame, inputs=cam, outputs=out)

    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)
    btn_accept.click(fn=on_accept, inputs=None, outputs=status)
    btn_toggle.click(fn=on_toggle_yolo, inputs=None, outputs=status)
    btn_start.click(fn=on_start_tracking, inputs=None, outputs=status)
    btn_reset.click(fn=on_reset, inputs=None, outputs=status)

    btn_start_vid.click(fn=start_video, inputs=[vid, save_name], outputs=[out, download])

    gr.Markdown("""
**How to use:**
- **Webcam:** YOLO ON, press **Accept** for each person (can add many). Then **Start Tracking**.
- **Video:** Upload file → **Start video**. On the first frame Accept several, then **Start Tracking**. When it finishes, a download appears.
""")

demo.launch(share=True)