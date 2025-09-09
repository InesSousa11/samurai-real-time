import numpy as np
import torch
import cv2
import gradio as gr
import traceback
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --- Config ---
CKPT = "checkpoints/sam2.1_hiera_small.pt"
CFG  = "configs/samurai/sam2.1_hiera_s.yaml"
MAX_W = 480  # lower if needed (e.g., 384)

# --- Build models ---
def make_predictor():
    return build_sam2_camera_predictor(CFG, CKPT)

predictor = make_predictor()
yolo = YOLO("yolov8n.pt")  # nano is fastest

# --- Stream state ---
frame_idx = 0           # logical frame index for SAM2
tracker_ready = False   # only True after a YOLO person is added
obj_counter = 1
last_output_rgb = None

def downscale(img, max_w=MAX_W):
    H, W = img.shape[:2]
    scale = min(1.0, max_w / max(1, W))
    if scale < 1.0:
        img_s = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_s = img
    return img_s, scale

def seed_person_with_yolo(rgb_small):
    """Try to detect a person and add as a prompt for the *current* frame_idx."""
    global obj_counter
    results = yolo(rgb_small, verbose=False)[0]
    for det in results.boxes:
        if int(det.cls) == 0:  # person
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            predictor.add_new_prompt(frame_idx=frame_idx, obj_id=obj_counter, bbox=bbox)
            obj_counter += 1
            return True
    return False

@torch.inference_mode()
def process_frame(rgb_frame):
    global predictor, frame_idx, tracker_ready, obj_counter, last_output_rgb

    # Gradio sometimes passes None; keep last image and DO NOT advance frame_idx
    if rgb_frame is None:
        return last_output_rgb

    try:
        # Preprocess & keep resolution consistent
        rgb_small, scale = downscale(rgb_frame)
        h, w = rgb_small.shape[:2]

        if not tracker_ready:
            # Initialize SAM2 on THIS frame
            predictor.load_first_frame(rgb_small)

            # Keep trying to find a person; if none, do not mark ready nor advance index
            found = seed_person_with_yolo(rgb_small)
            out_small = rgb_small
            if found:
                tracker_ready = True
                frame_idx += 1  # advance only after a successful seed
            # else: stay not ready; frame_idx unchanged

        else:
            try:
                # Normal tracking step
                out_obj_ids, out_mask_logits = predictor.track(rgb_small)

                if len(out_obj_ids) == 0:
                    # Lost target → try to re-seed on current frame (without reset)
                    if seed_person_with_yolo(rgb_small):
                        # After re-seed, draw nothing this frame; tracker will use it next call
                        out_small = rgb_small
                    else:
                        out_small = rgb_small
                else:
                    # Build overlay
                    all_mask = np.zeros((h, w, 3), dtype=np.uint8)
                    all_mask[..., 1] = 255
                    for i in range(len(out_obj_ids)):
                        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                        hue = int((i + 3) / (len(out_obj_ids) + 3) * 255)
                        sel = out_mask[..., 0] == 255
                        all_mask[sel, 0] = hue
                        all_mask[sel, 2] = 255
                    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
                    out_small = cv2.addWeighted(rgb_small, 1.0, all_mask, 0.5, 0.0)

                # Advance index only after a successful track() call
                frame_idx += 1

            except AssertionError:
                # This is your crash: SAM2 has no conditioned prompts in memory.
                # Recover by re-initializing on current frame and re-seeding.
                print(f"[recover] AssertionError at frame_idx={frame_idx}; re-seeding…")
                predictor = make_predictor()
                predictor.load_first_frame(rgb_small)
                tracker_ready = seed_person_with_yolo(rgb_small)
                out_small = rgb_small
                # frame_idx becomes 1 only if we seeded; otherwise stay at 0
                frame_idx = 1 if tracker_ready else 0

        # Upscale back for display
        if scale < 1.0:
            out_img = cv2.resize(out_small, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            out_img = out_small

        out_img = np.ascontiguousarray(out_img.astype(np.uint8))
        if out_img.ndim == 2:
            out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2RGB)

        last_output_rgb = out_img

        # Heartbeat to verify progress
        if frame_idx % 30 == 0 and tracker_ready:
            print(f"[heartbeat] processed frame {frame_idx}")

        return out_img

    except Exception as e:
        print("process_frame error:", repr(e))
        traceback.print_exc()
        # Do not advance frame_idx on error; show last good frame if available
        return last_output_rgb if last_output_rgb is not None else rgb_frame


# ---------- Gradio 5.x UI ----------
webcam = gr.Image(
    sources=["webcam"],
    streaming=True,
    label="Webcam Input",
    type="numpy",
)

demo = gr.Interface(
    fn=process_frame,
    inputs=webcam,
    outputs=gr.Image(label="Processed Output", type="numpy"),
    live=True,
    flagging_mode="never",
)

demo.launch(share=True)
