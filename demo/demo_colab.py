import os
import cv2
import numpy as np
import torch
import gradio as gr
from ultralytics import YOLO

# -------- Performance knobs (same as your script) --------
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

# YOLO for proposals (use 'yolov8n.pt' for snappier UI)
yolo_model = YOLO("yolov8s.pt")

# -------- Helpers --------
def yolo_person_bboxes(rgb_frame, model, conf_thres=0.25):
    """Return person detections [(x1,y1,x2,y2,conf), ...], sorted by conf desc."""
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
    """Blend colored masks over rgb_frame and return RGB image."""
    if rgb_frame is None:
        return None
    h, w = rgb_frame.shape[:2]
    if not out_obj_ids:
        return rgb_frame
    all_mask = np.zeros((h, w, 3), dtype=np.uint8)
    all_mask[..., 1] = 255  # saturation
    for i in range(len(out_obj_ids)):
        m = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        hue = int((i + 3) / (len(out_obj_ids) + 3) * 255)
        sel = m[..., 0] == 255
        all_mask[sel, 0] = hue
        all_mask[sel, 2] = 255
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, all_mask, 0.5, 0.0)

# -------- App state (lives across frames) --------
state = {
    "seeded": False,          # once user accepts a person
    "selected_idx": 0,        # which YOLO candidate is highlighted
    "cands": [],              # last YOLO candidates [(x1,y1,x2,y2,conf), ...]
    "last_frame": None,       # last RGB frame seen (numpy)
    "out_obj_ids": None,
    "out_mask_logits": None,
}

# -------- Streaming callback --------
@torch.inference_mode()
def process_frame(rgb_frame):
    """
    Called on every incoming webcam frame (RGB numpy).
    - If not seeded: run YOLO, draw boxes, highlight selected.
    - If seeded: track and draw mask overlay.
    """
    if rgb_frame is None:
        return None

    state["last_frame"] = rgb_frame

    if not state["seeded"]:
        # proposals every frame
        cands = yolo_person_bboxes(rgb_frame, yolo_model, conf_thres=0.25)
        state["cands"] = cands

        # draw proposals
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
            cv2.putText(bgr, "No person found. Move into frame…",
                        (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Seeded → tracking
    try:
        out_obj_ids, out_mask_logits = predictor.track(rgb_frame)
        state["out_obj_ids"] = out_obj_ids
        state["out_mask_logits"] = out_mask_logits
        out = draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits)
        return out
    except Exception as e:
        print("[error] track() failed:", repr(e))
        return rgb_frame

# -------- Button handlers --------
def on_next():
    if not state["seeded"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] + 1) % len(state["cands"])
    return None

def on_prev():
    if not state["seeded"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] - 1) % len(state["cands"])
    return None

def on_accept():
    """Seed SAMURAI using the current frame + selected YOLO bbox."""
    if state["seeded"]:
        return "Already seeded."
    if not state["cands"] or state["last_frame"] is None:
        return "No candidate available."

    x1, y1, x2, y2, conf = state["cands"][state["selected_idx"]]
    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

    # bind the current frame as frame_idx=0 and add prompt once
    predictor.load_first_frame(state["last_frame"])
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=0, obj_id=1, bbox=bbox
    )
    state["seeded"] = True
    state["out_obj_ids"] = out_obj_ids
    state["out_mask_logits"] = out_mask_logits
    return f"Seeded with conf={conf:.2f}. Tracking…"

def on_reset():
    """Reset the whole session to choose again."""
    global predictor
    # rebuild predictor to fully clear memory
    predictor = build_sam2_camera_predictor(CFG, CKPT)
    state.update({
        "seeded": False,
        "selected_idx": 0,
        "cands": [],
        "out_obj_ids": None,
        "out_mask_logits": None,
    })
    return "Reset done."

# -------- Gradio UI --------
with gr.Blocks() as demo:
    gr.Markdown("## SAMURAI real-time (Colab) — YOLO-assisted seeding ➜ tracking")
    gr.Markdown("**Controls:** Use the buttons to cycle detections and seed tracking. When seeded, the mask overlay appears.")
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", type="numpy")
        out = gr.Image(label="Output", type="numpy")

    with gr.Row():
        btn_prev = gr.Button("Prev")
        btn_accept = gr.Button("Accept")
        btn_next = gr.Button("Next")
        btn_reset = gr.Button("Reset")

    status = gr.Markdown("Status: waiting…")

    # stream: webcam → process_frame → output
    cam.stream(fn=process_frame, inputs=cam, outputs=out)

    # buttons
    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)
    btn_accept.click(fn=on_accept, inputs=None, outputs=status)
    btn_reset.click(fn=on_reset, inputs=None, outputs=status)

demo.launch(share=True)