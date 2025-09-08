import numpy as np
import torch
import cv2
import gradio as gr
import traceback
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load SAMURAI model
sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/samurai/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Load YOLOv8 (auto-downloads)
yolo_model = YOLO("yolov8s.pt")

frame_idx = 0
obj_counter = 1
if_init = False

MAX_W = 640  # reduce compute; try 480/640/720

@torch.inference_mode()
def process_frame(rgb_frame):
    if rgb_frame is None:
        return None

    global frame_idx, obj_counter, if_init
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
            predictor.load_first_frame(rgb_small)
            if_init = True

            # Detect first person on the small image
            results = yolo_model(rgb_small, verbose=False)[0]
            for det in results.boxes:
                if int(det.cls) == 0:  # 'person'
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    predictor.add_new_prompt(frame_idx=frame_idx, obj_id=obj_counter, bbox=bbox)
                    obj_counter += 1
                    break
            out_img = rgb_small  # show the first frame while initializing

        else:
            out_obj_ids, out_mask_logits = predictor.track(rgb_small)
            if len(out_obj_ids) == 0:
                out_img = rgb_small  # nothing tracked; keep stream alive
            else:
                all_mask = np.zeros((h, w, 3), dtype=np.uint8)
                all_mask[..., 1] = 255  # saturation
                for i in range(len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                    hue = int((i + 3) / (len(out_obj_ids) + 3) * 255)
                    sel = out_mask[..., 0] == 255
                    all_mask[sel, 0] = hue
                    all_mask[sel, 2] = 255

                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
                out_img = cv2.addWeighted(rgb_small, 1.0, all_mask, 0.5, 0.0)

        frame_idx += 1

        # Upscale back to original size for display
        if scale < 1.0:
            out_img = cv2.resize(out_img, (W, H), interpolation=cv2.INTER_LINEAR)
        return out_img

    except Exception as e:
        print("process_frame error:", repr(e))
        traceback.print_exc()
        return rgb_frame  # keep stream alive even if something breaks

# Gradio UI
webcam = gr.Camera(streaming=True, label="Webcam Input", type="numpy", mirror_webcam=True)
demo = gr.Interface(
    fn=process_frame,
    inputs=webcam,
    outputs=gr.Image(label="Processed Output", type="numpy"),
    live=True,
    concurrency_count=1,
    allow_flagging="never",
)
demo.queue(concurrency_count=1, max_size=2)
demo.launch(share=True)
