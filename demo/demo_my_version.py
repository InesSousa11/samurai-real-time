import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# -------- Performance knobs --------

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -------- Build predictor --------

from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt" # NOTE: this path assumes you run from the demo/ folder
model_cfg = "configs/samurai/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# YOLO for proposals (use 'yolov8n.pt' if you want faster proposals)
yolo_model = YOLO("yolov8s.pt")


# -------- Input: video or webcam --------

# cap = cv2.VideoCapture("../notebooks/videos/aquarium/aquarium.mp4")
cap = cv2.VideoCapture(2)  # change index if using an external camera (e.g., 2 or 4 for RealSense)


# -------- Helpers --------

def yolo_person_bboxes(rgb_frame, model, conf_thres=0.25):
    """Return person detections [(x1,y1,x2,y2,conf), ...], sorted by conf desc."""
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


# -------- Main loop state --------

seeded = False            # becomes True after user accepts a bbox and we seed SAMURAI
selected_idx = 0          # which YOLO candidate is highlighted
out_obj_ids = None
out_mask_logits = None

print("[info] Starting stream. Press 'q' to quit.")
print("[info] While unseeded: YOLO boxes are shown. Keys: [y]=accept, [n]=next, [p]=prev, [q]=quit")

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        print("[info] End of stream.")
        break

    # Convert to RGB for models
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if not seeded: # User hasn't accepted a bbox yet
        # 1) Propose persons every frame
        cands = yolo_person_bboxes(frame, yolo_model, conf_thres=0.25)

        # 2) Draw proposals (all in thin lines) and highlight current selection (thick/green)
        disp = frame.copy()
        bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

        if cands:
            # Clamp index in case number of cands changes frame-to-frame
            selected_idx = max(0, min(selected_idx, len(cands) - 1))

            for j, (x1, y1, x2, y2, conf) in enumerate(cands):
                color = (0, 255, 0) if j == selected_idx else (0, 200, 255)
                thick = 3 if j == selected_idx else 1
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thick)
                cv2.putText(bgr, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            cv2.putText(bgr, f"[y]=accept  [n]=next  [p]=prev  [q]=quit  persons={len(cands)}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(bgr, "No person found. Move into frame…",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            

        # 3) Handle keys (non-blocking)
        cv2.imshow("SAMURAI chooser/preview", bgr)
        k = cv2.waitKey(1) & 0xFF 

        if k == ord('q'):
            print("[info] Quit requested.")
            break
        elif k == ord('n') and cands:
            selected_idx = (selected_idx + 1) % len(cands)
        elif k == ord('p') and cands:
            selected_idx = (selected_idx - 1) % len(cands)
        elif k == ord('y') and cands:
            # 4) User accepted current candidate → seed SAMURAI now
            x1, y1, x2, y2, conf = cands[selected_idx]
            bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

            print(f"[seed] Seeding with person idx={selected_idx+1}/{len(cands)} conf={conf:.2f}")
            predictor.load_first_frame(frame)  # bind the *current* frame as frame_idx=0
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=1, bbox=bbox
            )
            # Show mask on this same frame
            masked = draw_mask_overlay(frame, out_obj_ids, out_mask_logits)
            cv2.imshow("SAMURAI chooser/preview", cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
            seeded = True
            print("[seed] Done. Tracking starts.")

        # continue loop until seeded or quit
        continue


    # ---------- Seeded: tracking mode ----------

    try:
        out_obj_ids, out_mask_logits = predictor.track(frame) # NEED TO PUT SOME PRINTS INSIDE THE TRACK FUNTION !!
        frame_masked = draw_mask_overlay(frame, out_obj_ids, out_mask_logits)
        cv2.imshow("SAMURAI tracking", cv2.cvtColor(frame_masked, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[error] track() failed: {repr(e)}")
        # Show raw frame so UI keeps alive
        cv2.imshow("SAMURAI tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Common key handling
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[info] Quit requested.")
        break

cap.release()
cv2.destroyAllWindows()