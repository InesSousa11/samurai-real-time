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

# Build the camera predictor; this loads the network and the checkpoint
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Load YOLOv8 model
yolo_model = YOLO("yolov8s.pt")  # You can use 'yolov8n.pt' or any variant

# -----------------------------
# Open the webcam
# -----------------------------
cap = cv2.VideoCapture(2)  # AUTOMATIZAR ISTO PARA O CASO TER DE CONECTAR A UMA CAMERA ESPECIFICA
assert cap.isOpened(), "Could not open webcam."

# -----------------------------
# Runtime state variables
# -----------------------------
if_init = False  # whether we loaded the first frame into the predictor
frame_idx = 0    # logical frame counter you pass to SAMURAI when adding prompts
obj_counter = 1  # Unique ID for each detected person

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read() # Grab a BGR frame from the camera
    if not ret:
        break

    # Convert BGR (OpenCV) -> RGB (models expect RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    if not if_init:
        # 1) Initialize SAMURAI with the first frame.
        predictor.load_first_frame(rgb_frame)
        if_init = True

        # 2) Run YOLO on the *same* first frame to find a person
        results = yolo_model(rgb_frame, verbose=False)[0]

        # 3) Loop over detections and take the *first* 'person' class (class 0)
        for det in results.boxes:
            cls_id = int(det.cls)
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

                # 4) Add a *prompt* (one-time) telling SAMURAI which object to follow.
                predictor.add_new_prompt(
                    frame_idx=frame_idx, obj_id=obj_counter, bbox=bbox
                )
                obj_counter += 1
                break  # ← Only add the **first** detected person

    else:
        # From now on, we track the person that was prompted.
        # This returns:
        #   - out_obj_ids: list of object IDs that have masks this frame
        #   - out_mask_logits: list of (C,H,W) tensors with logits per object
        out_obj_ids, out_mask_logits = predictor.track(rgb_frame)

        # Prepare segmentation mask overlay
        all_mask = np.zeros((height, width, 3), dtype=np.uint8)
        all_mask[..., 1] = 255  # Saturation

        for i in range(len(out_obj_ids)):
            # Convert logits to a binary mask: (C,H,W)→(H,W,1) then to 0/255 uint8
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            hue = (i + 3) / (len(out_obj_ids) + 3) * 255
            all_mask[out_mask[..., 0] == 255, 0] = hue
            all_mask[out_mask[..., 0] == 255, 2] = 255

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
        overlayed = cv2.addWeighted(rgb_frame, 1, all_mask, 0.5, 0)
        bgr_output = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
        cv2.imshow("SAMURAI Tracking (Webcam)", bgr_output)

    # Advance our external frame counter (used only when adding the prompt above)
    frame_idx += 1

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
