import cv2
import numpy as np
import torch
from ultralytics import YOLO


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor

# Set up model
sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/samurai/sam2.1_hiera_s.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


# Load YOLOv8 model
yolo_model = YOLO("yolov8s.pt")  # You can use 'yolov8n.pt' or any variant

# Set up webcam
cap = cv2.VideoCapture(0)  # Use webcam
assert cap.isOpened(), "Could not open webcam."

if_init = False
frame_idx = 0
obj_counter = 1  # Unique ID for each detected person

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    if not if_init:
        predictor.load_first_frame(rgb_frame)
        if_init = True

        # Detect people in the first frame
        results = yolo_model(rgb_frame, verbose=False)[0]
        for det in results.boxes:
            cls_id = int(det.cls)
            if cls_id == 0:  # class 0 = 'person'
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                predictor.add_new_prompt(
                    frame_idx=frame_idx, obj_id=obj_counter, bbox=bbox
                )
                obj_counter += 1
                break  # ← Only add the **first** detected person

    else:
        out_obj_ids, out_mask_logits = predictor.track(rgb_frame)

        # Prepare segmentation mask overlay
        all_mask = np.zeros((height, width, 3), dtype=np.uint8)
        all_mask[..., 1] = 255  # Saturation

        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            hue = (i + 3) / (len(out_obj_ids) + 3) * 255
            all_mask[out_mask[..., 0] == 255, 0] = hue
            all_mask[out_mask[..., 0] == 255, 2] = 255

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
        overlayed = cv2.addWeighted(rgb_frame, 1, all_mask, 0.5, 0)
        bgr_output = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
        cv2.imshow("SAMURAI Tracking (Webcam)", bgr_output)

    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
