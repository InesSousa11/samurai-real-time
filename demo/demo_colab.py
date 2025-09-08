import numpy as np
import torch
import cv2
import gradio as gr
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load SAMURAI model
sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/samurai/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Load YOLOv8
yolo_model = YOLO("yolov8s.pt")

frame_idx = 0
obj_counter = 1
if_init = False

@torch.inference_mode()
def process_frame(rgb_frame):
    global frame_idx, obj_counter, if_init

    height, width = rgb_frame.shape[:2]

    if not if_init:
        predictor.load_first_frame(rgb_frame)
        if_init = True

        results = yolo_model(rgb_frame, verbose=False)[0]
        for det in results.boxes:
            if int(det.cls) == 0:  # class = person
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                predictor.add_new_prompt(frame_idx=frame_idx, obj_id=obj_counter, bbox=bbox)
                obj_counter += 1
                break
    else:
        out_obj_ids, out_mask_logits = predictor.track(rgb_frame)

        all_mask = np.zeros((height, width, 3), dtype=np.uint8)
        all_mask[..., 1] = 255
        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1,2,0).cpu().numpy().astype(np.uint8) * 255
            hue = (i+3)/(len(out_obj_ids)+3)*255
            all_mask[out_mask[...,0]==255, 0] = hue
            all_mask[out_mask[...,0]==255, 2] = 255

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
        overlayed = cv2.addWeighted(rgb_frame, 1, all_mask, 0.5, 0)
        rgb_frame = overlayed

    frame_idx += 1
    return rgb_frame

demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam Input", type="numpy"),
    outputs=gr.Image(label="Processed Output", type="numpy"),
    live=True,
)

demo.launch(share=True)
