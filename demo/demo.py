import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import time

from sam2.build_sam import build_sam2_camera_predictor


sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/samurai/sam2.1_hiera_s.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


cap = cv2.VideoCapture("../notebooks/videos/aquarium/aquarium.mp4") # video input
# cap = cv2.VideoCapture(0)  # 0 = default laptop webcam

if_init = False
tracking_i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width, height = frame.shape[:2][::-1]
    if not if_init:

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with

        # First annotation
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        ##! add points, `1` means positive click and `0` means negative click
        points = np.array([[600, 255]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        )

        # Second annotation
        ann_obj_id = 2
        ## ! add bbox
        bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        )

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)
        tracking_i += 1

        if tracking_i == 100:
            predictor.add_conditioning_frame(frame)

            ## ! add new bbox
            bbox = np.array([[450, 280], [520, 340]], dtype=np.float32)
            ann_obj_id = 2
            predictor.add_new_prompt_during_track(
                bbox=bbox,
                obj_id=ann_obj_id,
                if_new_target=False,
                clear_old_points=False,
            )

        if tracking_i == 160:
            predictor.add_conditioning_frame(frame)

            # ! add new point
            points = np.array([[460, 270]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            ann_obj_id = 1
            predictor.add_new_prompt_during_track(
                point=points,
                labels=labels,
                obj_id=ann_obj_id,
                if_new_target=False,
                clear_old_points=False,
            )

        all_mask = np.zeros((height, width, 3), dtype=np.uint8)
        all_mask[..., 1] = 255
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            hue = (i + 3) / (len(out_obj_ids) + 3) * 255
            all_mask[out_mask[..., 0] == 255, 0] = hue
            all_mask[out_mask[..., 0] == 255, 2] = 255

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
