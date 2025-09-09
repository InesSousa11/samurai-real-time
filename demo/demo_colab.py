import numpy as np
import torch
import cv2
import gradio as gr
import traceback
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --- Models ---
sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/samurai/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Use the **nano** YOLO for minimal overhead
yolo_model = YOLO("yolov8n.pt")

# --- State ---
frame_idx = 0
obj_counter = 1
if_init = False
last_output_rgb = None

# Lower res to keep realtime
MAX_W = 480  # try 384 if it still freezes

@torch.inference_mode()
def process_frame(rgb_frame):
    global frame_idx, obj_counter, if_init, last_output_rgb

    # Gradio sometimes sends None; keep showing the last good frame
    if rgb_frame is None:
        return last_output_rgb

    try:
        H, W = rgb_frame.shape[:2]

        # Downscale for speed
        scale = min(1.0, MAX_W / max(1, W))
        if scale < 1.0:
            rgb_small = cv2.resize(rgb_frame, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb_frame
        h, w = rgb_small.shape[:2]

        if not if_init:
            # Warmup & seed tracker with first person box (YOLO only once)
            predictor.load_first_frame(rgb_small)
            results = yolo_model(rgb_small, verbose=False)[0]
            for det in results.boxes:
                if int(det.cls) == 0:  # 'person'
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    predictor.add_new_prompt(frame_idx=frame_idx, obj_id=obj_counter, bbox=bbox)
                    obj_counter += 1
                    break
            if_init = True
            out_img_small = rgb_small  # show something immediately

        else:
            # Track every frame (cheap compared to re-running YOLO)
            out_obj_ids, out_mask_logits = predictor.track(rgb_small)
            if len(out_obj_ids) == 0:
                out_img_small = rgb_small
            else:
                all_mask = np.zeros((h, w, 3), dtype=np.uint8)
                all_mask[..., 1] = 255
                for i in range(len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                    hue = int((i + 3) / (len(out_obj_ids) + 3) * 255)
                    sel = out_mask[..., 0] == 255
                    all_mask[sel, 0] = hue
                    all_mask[sel, 2] = 255
                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
                out_img_small = cv2.addWeighted(rgb_small, 1.0, all_mask, 0.5, 0.0)

        # Upscale back for display
        out_img = out_img_small
        if scale < 1.0:
            out_img = cv2.resize(out_img_small, (W, H), interpolation=cv2.INTER_LINEAR)

        # Ensure proper dtype/contiguity
        out_img = np.ascontiguousarray(out_img.astype(np.uint8))
        if out_img.ndim == 2:
            out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2RGB)

        last_output_rgb = out_img
        frame_idx += 1

        # Heartbeat (every ~30 frames)
        if frame_idx % 30 == 0:
            print(f"[heartbeat] processed frame {frame_idx}")

        return out_img

    except Exception as e:
        print("process_frame error:", repr(e))
        traceback.print_exc()
        return last_output_rgb if last_output_rgb is not None else rgb_frame

# ---- Gradio 5.x UI ----
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

# No explicit queue args (v5 defaults)
demo.launch(share=True)
