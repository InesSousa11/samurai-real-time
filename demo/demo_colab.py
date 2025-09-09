import numpy as np
import torch
import cv2
import gradio as gr
import traceback
from ultralytics import YOLO
from sam2.build_sam import build_sam2_camera_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --- Build models ---
CKPT = "checkpoints/sam2.1_hiera_small.pt"
CFG  = "configs/samurai/sam2.1_hiera_s.yaml"

def make_predictor():
    return build_sam2_camera_predictor(CFG, CKPT)

predictor = make_predictor()
yolo_model = YOLO("yolov8n.pt")  # nano for lower overhead

# --- Stream state ---
frame_idx = 0              # SAM2's logical frame counter
obj_counter = 1
tracker_ready = False
last_output_rgb = None

MAX_W = 480  # lower if still choppy (e.g. 384)

@torch.inference_mode()
def process_frame(rgb_frame):
    """Gradio 5.x webcam callback: RGB HxWx3 uint8 numpy array."""
    global predictor, frame_idx, obj_counter, tracker_ready, last_output_rgb

    # If Gradio sends None, keep showing the last good image and do NOT advance.
    if rgb_frame is None:
        return last_output_rgb

    H, W = rgb_frame.shape[:2]

    # Downscale for speed (fixed size so SAM2 sees consistent resolution)
    scale = min(1.0, MAX_W / max(1, W))
    if scale < 1.0:
        rgb_small = cv2.resize(rgb_frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        rgb_small = rgb_frame
    h, w = rgb_small.shape[:2]

    try:
        if not tracker_ready:
            # Initialize tracker on THIS frame index
            predictor.load_first_frame(rgb_small)

            # Run YOLO ONCE to seed the first person
            results = yolo_model(rgb_small, verbose=False)[0]
            for det in results.boxes:
                if int(det.cls) == 0:  # person
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    # Important: tie the prompt to the CURRENT frame_idx (which is 0 at init)
                    predictor.add_new_prompt(frame_idx=frame_idx, obj_id=obj_counter, bbox=bbox)
                    obj_counter += 1
                    break

            out_img_small = rgb_small  # show something right away
            tracker_ready = True       # tracker is now ready
            frame_idx += 1             # advance ONLY after successful init step

        else:
            # Regular tracking step
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

            frame_idx += 1  # advance ONLY after a successful track()

        # Upscale back for display
        out_img = out_img_small if scale == 1.0 else cv2.resize(out_img_small, (W, H), interpolation=cv2.INTER_LINEAR)
        out_img = np.ascontiguousarray(out_img.astype(np.uint8))
        if out_img.ndim == 2:
            out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2RGB)

        last_output_rgb = out_img
        # Heartbeat to see progress in logs
        if frame_idx % 30 == 0:
            print(f"[heartbeat] processed frame {frame_idx}")
        return out_img

    except KeyError as e:
        # This is the freeze you saw: SAM2 wanted a frame id that wasn't in memory.
        print(f"[recover] SAM2 KeyError {e} at frame_idx={frame_idx}. Reinitializing tracker on current frame…")
        traceback.print_exc()
        try:
            # Rebuild predictor and reinitialize on CURRENT frame
            predictor = make_predictor()
            frame_idx = 0
            tracker_ready = False
        except Exception as ee:
            print("[recover] failed to rebuild predictor:", repr(ee))
            traceback.print_exc()
        # Show the last good image (or current raw frame) so UI doesn't blank
        return last_output_rgb if last_output_rgb is not None else rgb_frame

    except Exception as e:
        print("process_frame error:", repr(e))
        traceback.print_exc()
        # Do NOT advance frame_idx; keep stream alive with last good image
        return last_output_rgb if last_output_rgb is not None else rgb_frame


# --------- Gradio 5.x UI ----------
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
