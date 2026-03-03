# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import cv2
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores

# torch._dynamo.config.capture_dynamic_output_shape_ops = True


class SAM2CameraPredictor(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.condition_state = {}
        self.frame_idx = 0
        self.dedupe_iou_thr = 0.6
        self.dedupe_min_area = 0

    ###
    def perpare_data(
        self,
        img,
        image_size=1024,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
    ):
        if isinstance(img, np.ndarray):
            img_np = img
            img_np = cv2.resize(img_np, (image_size, image_size)) / 255.0
            height, width = img.shape[:2]
        else:
            img_np = (
                np.array(img.convert("RGB").resize((image_size, image_size))) / 255.0
            )
            width, height = img.size
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()

        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        img -= img_mean
        img /= img_std
        return img, width, height

    ###
    @torch.inference_mode()
    def load_first_frame(self, img):

        self.condition_state = self._init_state(
            offload_video_to_cpu=False, offload_state_to_cpu=False
        )
        img, width, height = self.perpare_data(img, image_size=self.image_size)
        self.condition_state["images"] = [img]
        self.condition_state["num_frames"] = len(self.condition_state["images"])
        self.condition_state["video_height"] = height
        self.condition_state["video_width"] = width
        self._get_image_feature(frame_idx=0, batch_size=1)

    @torch.inference_mode()
    def add_conditioning_frame(self, img):
        img, width, height = self.perpare_data(img, image_size=self.image_size)

        # Append the new conditioning frame
        self.condition_state["images"].append(img)

        # CRITICAL: num_frames must match the actual stored images
        self.condition_state["num_frames"] = len(self.condition_state["images"])

        # Use the true last index (safe even if num_frames was corrupted before)
        cond_frame_idx = len(self.condition_state["images"]) - 1

        # Extract features for that frame
        self._get_image_feature(frame_idx=cond_frame_idx, batch_size=1)

    ###
    def _init_state(
        self,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        self.condition_state = {}

        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        self.condition_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        self.condition_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores

        self.condition_state["device"] = torch.device("cuda")
        if offload_state_to_cpu:
            self.condition_state["storage_device"] = torch.device("cpu")
        else:
            self.condition_state["storage_device"] = torch.device("cuda")
        # inputs on each frame
        self.condition_state["point_inputs_per_obj"] = {}
        self.condition_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        self.condition_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        self.condition_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        self.condition_state["obj_id_to_idx"] = OrderedDict()
        self.condition_state["obj_idx_to_id"] = OrderedDict()
        self.condition_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        self.condition_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        self.condition_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        self.condition_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        self.condition_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        self.condition_state["tracking_has_started"] = False
        self.condition_state["frames_already_tracked"] = {}
        return self.condition_state

    ###
    def _obj_id_to_idx(self, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = self.condition_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not self.condition_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(self.condition_state["obj_id_to_idx"])
            self.condition_state["obj_id_to_idx"][obj_id] = obj_idx
            self.condition_state["obj_idx_to_id"][obj_idx] = obj_id
            self.condition_state["obj_ids"] = list(
                self.condition_state["obj_id_to_idx"]
            )
            # set up input and output structures for this object
            self.condition_state["point_inputs_per_obj"][obj_idx] = {}
            self.condition_state["mask_inputs_per_obj"][obj_idx] = {}
            self.condition_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            self.condition_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {self.condition_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, obj_idx):
        """Map model-side object index to client-side object id."""
        return self.condition_state["obj_idx_to_id"][obj_idx]

    ###
    def _get_obj_num(self):
        """Get the total number of unique object ids received so far in this session."""
        return len(self.condition_state["obj_idx_to_id"])

    ###
    @torch.inference_mode()
    def add_new_prompt(
        self,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        bbox=None,
        clear_old_points=True,
        normalize_coords=True,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.condition_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.condition_state["mask_inputs_per_obj"][obj_idx]

        assert (
            bbox is not None or points is not None
        ), "Either bbox or points is required"

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension
        if bbox is not None:
            if not isinstance(bbox, torch.Tensor):
                bbox = torch.tensor(bbox, dtype=torch.float32, device=points.device)
                box_coords = bbox.reshape(1, 2, 2)
                box_labels = torch.tensor(
                    [2, 3], dtype=torch.int32, device=labels.device
                )
                box_labels = box_labels.reshape(1, 2)
                points = torch.cat([box_coords, points], dim=1)
                labels = torch.cat([box_labels, labels], dim=1)
        if normalize_coords:
            video_H = self.condition_state["video_height"]
            video_W = self.condition_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(self.condition_state["device"])
        labels = labels.to(self.condition_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = (
            frame_idx not in self.condition_state["frames_already_tracked"]
        )
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = self.condition_state["frames_already_tracked"][frame_idx][
                "reverse"
            ]
        obj_output_dict = self.condition_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.condition_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.condition_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    ###
    @torch.inference_mode()
    def add_new_points(
        self,
        frame_idx,
        obj_id,
        points,
        labels,
        clear_old_points=True,
        normalize_coords=True,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.condition_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.condition_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension
        if normalize_coords:
            video_H = self.condition_state["video_height"]
            video_W = self.condition_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(self.condition_state["device"])
        labels = labels.to(self.condition_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = (
            frame_idx not in self.condition_state["frames_already_tracked"]
        )
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = self.condition_state["frames_already_tracked"][frame_idx][
                "reverse"
            ]
        obj_output_dict = self.condition_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.condition_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.condition_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    ###
    @torch.inference_mode()
    def add_new_mask(
        self,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.condition_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.condition_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(self.condition_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = (
            frame_idx not in self.condition_state["frames_already_tracked"]
        )
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = self.condition_state["frames_already_tracked"][frame_idx][
                "reverse"
            ]
        obj_output_dict = self.condition_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.condition_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.condition_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    ###
    def _get_orig_video_res_output(self, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = self.condition_state["device"]
        video_H = self.condition_state["video_height"]
        video_W = self.condition_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """

        # print(f"[DBG consolidate] frame_idx={frame_idx} is_cond={is_cond} run_mem_encoder={run_mem_encoder} "
        #       f"consolidate_at_video_res={consolidate_at_video_res}")
        self._dbg_state("consolidate:ENTER")

        batch_size = self._get_obj_num()
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = self.condition_state["video_height"]
            consolidated_W = self.condition_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=self.condition_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=self.condition_state["device"],
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # default to 10.0 for object_score_logits, i.e. assuming the object is
                # present as sigmoid(10)=1, same as in `predict_masks` of `MaskDecoder`
                fill_value=10.0,
                dtype=torch.float32,
                device=self.condition_state["device"],
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = self.condition_state["temp_output_dict_per_obj"][
                obj_idx
            ]
            obj_output_dict = self.condition_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(frame_idx)
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            device = self.condition_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,  # these frames are what the user interacted with
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

            self._dbg_state("consolidate:EXIT")

        return consolidated_out

    """
    def _get_empty_mask_ptr(self, frame_idx):
        # Get a dummy object pointer based on an empty mask on the current frame.
        # A dummy (empty) mask with a single object
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=torch.float32,
            device=self.condition_state["device"],
        )

        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(frame_idx, batch_size)

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={},
            num_frames=self.condition_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]
    """

    def _get_empty_mask_ptr(self, frame_idx):
        """
        Return a dummy object pointer without touching condition_state['images'].
        Used only for padding / 'no-object' rows.
        """
        # Try to match device/dtype/width of any existing obj_ptr we’ve already stored.
        out_dict = self.condition_state.get("output_dict", {})
        for bucket in ("non_cond_frame_outputs", "cond_frame_outputs"):
            bucket_dict = out_dict.get(bucket, {})
            if bucket_dict:
                any_out = next(iter(bucket_dict.values()))
                if "obj_ptr" in any_out and isinstance(any_out["obj_ptr"], torch.Tensor):
                    like = any_out["obj_ptr"]
                    return torch.zeros(1, like.shape[1], device=like.device, dtype=like.dtype)

        # Fallback if nothing stored yet: use model defaults.
        dev = self.condition_state.get("device", None)
        if dev is None:
            try:
                dev = next(self.parameters()).device
            except Exception:
                dev = "cuda" if torch.cuda.is_available() else "cpu"

        C = getattr(self, "hidden_dim", getattr(self, "mem_dim", 256))
        return torch.zeros(1, C, device=dev, dtype=torch.float32)

    ###
    @torch.inference_mode()
    def propagate_in_video_preflight(self):
        """Prepare self.condition_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        self.condition_state["tracking_has_started"] = True
        batch_size = self._get_obj_num()

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = self.condition_state["temp_output_dict_per_obj"]
        output_dict = self.condition_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = self.condition_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outptus
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temprary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(frame_idx, consolidated_out, storage_key)
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in self.condition_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in self.condition_state[
            "point_inputs_per_obj"
        ].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in self.condition_state[
            "mask_inputs_per_obj"
        ].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    def _register_new_object_if_needed(self, obj_id: int):
        cs = self.condition_state
        if obj_id in cs["obj_id_to_idx"]:
            return cs["obj_id_to_idx"][obj_id]

        obj_idx = len(cs["obj_id_to_idx"])
        cs["obj_id_to_idx"][obj_id] = obj_idx
        cs["obj_idx_to_id"][obj_idx] = obj_id
        cs["obj_ids"].append(obj_id)

        cs["point_inputs_per_obj"][obj_idx] = {}
        cs["mask_inputs_per_obj"][obj_idx]  = {}

        cs["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        cs["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        return obj_idx
    
    def _pad_outputs_to_batch_size(self, out: dict, frame_idx: int, new_B: int):
        """Pad a single frame's stored outputs to batch size new_B."""
        pm = out["pred_masks"]; B = pm.shape[0]
        if B == new_B:
            return

        pad = new_B - B
        device = pm.device
        dtype  = pm.dtype

        # 1) pred_masks -> fill with NO_OBJ_SCORE (means 'no object')
        pm_pad = torch.full((pad, *pm.shape[1:]), NO_OBJ_SCORE, dtype=dtype, device=device)
        out["pred_masks"] = torch.cat([pm, pm_pad], dim=0)

        # 2) obj_ptr -> pad with zeros (dummy pointers); **do not** fetch images
        ptr = out["obj_ptr"]
        ptr_pad = torch.zeros((pad, ptr.shape[1]), dtype=ptr.dtype, device=ptr.device)
        out["obj_ptr"] = torch.cat([ptr, ptr_pad], dim=0)

        # 3) object_score_logits -> push towards 'no obj' (negative logit)
        osl = out["object_score_logits"]
        osl_pad = torch.full((pad, *osl.shape[1:]), -10.0, dtype=osl.dtype, device=osl.device)
        out["object_score_logits"] = torch.cat([osl, osl_pad], dim=0)

        # 4) maskmem_features -> zeros are fine as inert memory
        mmf = out.get("maskmem_features", None)
        if mmf is not None:
            mmf_pad = torch.zeros((pad, *mmf.shape[1:]), dtype=mmf.dtype, device=mmf.device)
            out["maskmem_features"] = torch.cat([mmf, mmf_pad], dim=0)

        # 5) maskmem_pos_enc -> expand by repeating one slice (pos-enc is identical across objs)
        mmpe = out.get("maskmem_pos_enc", None)
        if mmpe is not None:
            out["maskmem_pos_enc"] = [torch.cat([x, x[:1].expand(pad, -1, -1, -1)], dim=0) for x in mmpe]

        # 6) best_iou_score -> pad with very negative (so it never passes threshold)
        bis = out.get("best_iou_score", None)
        if torch.is_tensor(bis):
            bis_pad = torch.full((pad, *bis.shape[1:]), -1e9, dtype=bis.dtype, device=bis.device)
            out["best_iou_score"] = torch.cat([bis, bis_pad], dim=0)

        # 7) kf_score -> pad with very negative (so it never passes threshold)
        kfs = out.get("kf_score", None)
        if torch.is_tensor(kfs):
            kfs_pad = torch.full((pad, *kfs.shape[1:]), -1e9, dtype=kfs.dtype, device=kfs.device)
            out["kf_score"] = torch.cat([kfs, kfs_pad], dim=0)

    def _expand_all_stored_outputs_to_current_batch(self):

        #try:
        #    print(f"\n[DBG EXPAND] BEFORE new_B? obj_ids={self.condition_state.get('obj_ids')} "
        #        f"obj_id_to_idx={self.condition_state.get('obj_id_to_idx')}")
        #except Exception:
        #    pass

        self._dbg_state("expand_all:ENTER")
        new_B = self._get_obj_num()
        #print(f"[DBG expand_all] new_B={new_B}")

        """Ensure every stored frame matches current number of objects (batch size)."""
        new_B = self._get_obj_num()
        if new_B <= 0:
            return

        output_dict = self.condition_state["output_dict"]

        od = self.condition_state["output_dict"]
        #print(f"[DBG expand_all] cond_frame_outputs keys sample={list(od['cond_frame_outputs'].keys())[:20]}")
        #print(f"[DBG expand_all] non_cond_frame_outputs keys sample={list(od['non_cond_frame_outputs'].keys())[:20]}")

        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            for frame_idx, out in list(output_dict[storage_key].items()):
                # out["obj_ptr"] always exists; use it to check current B
                if out["obj_ptr"].shape[0] != new_B:
                    self._pad_outputs_to_batch_size(out, frame_idx, new_B)
                    # also refresh per-object views for this frame
                    self._add_output_per_object(frame_idx, out, storage_key)

                    # DEBUG: log after padding (safe quoting)
                    pm_shape  = tuple(out["pred_masks"].shape)
                    ptr_shape = tuple(out["obj_ptr"].shape)
                    print(f"[pad] global_frame={self.frame_idx} {storage_key}[{frame_idx}] -> new_B={new_B} "
                          f"pred_masks={pm_shape} obj_ptr={ptr_shape} obj_ids={self.condition_state.get('obj_ids')}")
                    
        #try:
        #    print(f"[DBG EXPAND] AFTER  obj_ids={self.condition_state.get('obj_ids')} "
        #        f"obj_id_to_idx={self.condition_state.get('obj_id_to_idx')}\n")
        #except Exception:
        #    pass

        self._dbg_state("expand_all:EXIT")

    @torch.inference_mode()
    def add_new_prompt_during_track(
        self,
        point=None,
        bbox=None,
        mask=None,
        if_new_target=True,
        obj_id=None,
        labels=None,
        clear_old_points=True,
    ):
        self._dbg_state("add_new_prompt_during_track:ENTER")
        #print(f"[DBG add_new_prompt_during_track] if_new_target={if_new_target} obj_id={obj_id}")

        assert self.condition_state["tracking_has_started"] is True, \
            "Cannot add new points or mask during tracking without calling track()"

        # Pause tracking while we inject
        self.condition_state["tracking_has_started"] = False

        # The demo already did add_conditioning_frame(rgb_frame)
        # So the frame we should work on is the LAST conditioning frame index.
        cond_frame_idx = int(len(self.condition_state["images"]) - 1)

        # We ALSO want the already-tracked output of the SAME image in the non_cond timeline.
        # After track() this should exist at the current global frame index self.frame_idx.
        src_global_fidx = int(self.frame_idx)

        output_dict = self.condition_state["output_dict"]

        # --- capture "old world" mapping BEFORE registering new object ---
        old_obj_ids = list(self.condition_state["obj_ids"])
        old_obj_id_to_idx = self.condition_state.get("obj_id_to_idx", None)

        #print(f"[DBG INJECT] cond_frame_idx={cond_frame_idx} src_global_fidx={src_global_fidx}")
        #print(f"[DBG INJECT] old_obj_ids={old_obj_ids} old_obj_id_to_idx={old_obj_id_to_idx}")

        # --- fetch the latest non_cond output to preserve old objects ---
        src_out = None
        try:
            src_out = output_dict.get("non_cond_frame_outputs", {}).get(src_global_fidx, None)
            if src_out is None:
                # fallback: pick the closest previous key if exact isn't present
                keys = sorted(list(output_dict.get("non_cond_frame_outputs", {}).keys()))
                prev_keys = [k for k in keys if int(k) <= src_global_fidx]
                if prev_keys:
                    src_k = prev_keys[-1]
                    src_out = output_dict["non_cond_frame_outputs"][src_k]
                    #print(f"[DBG INJECT] src_out fallback used: {src_k} (instead of {src_global_fidx})")
        except Exception as e:
            #print(f"[DBG INJECT] src_out fetch failed: {e}")
            src_out = None

        if src_out is None:
            print("[DBG INJECT] WARNING: no src_out found in non_cond_frame_outputs; "
                "old objects may be killed on consolidation.")
        else:
            try:
                pm = src_out.get("pred_masks", None)
                if torch.is_tensor(pm):
                    B = int(pm.shape[0])
                    mx = [float(pm[i].max().item()) for i in range(B)]
                    #print(f"[DBG INJECT] src_out pred_masks B={B} max={['%.1f'%m for m in mx]}")
            except Exception as e:
                print(f"[DBG INJECT] src_out stats failed: {e}")

        # (1) Register/choose id and create the temp output on this conditioning frame
        if if_new_target:
            if obj_id is None:
                obj_id = (max(self.condition_state["obj_ids"]) + 1) if self.condition_state["obj_ids"] else 0
            _ = self._register_new_object_if_needed(obj_id)

            #print(f"[DBG add_new_prompt_during_track] after register: obj_ids={self.condition_state['obj_ids']} "
            #    f"obj_id_to_idx={self.condition_state.get('obj_id_to_idx', None)}")

            if (point is not None) or (bbox is not None):
                frame_idx, obj_ids, video_res_masks = self.add_new_prompt(
                    frame_idx=cond_frame_idx,
                    obj_id=obj_id,
                    points=point,
                    bbox=bbox,
                    labels=labels,
                    clear_old_points=clear_old_points,
                    normalize_coords=True,
                )
            else:
                frame_idx, obj_ids, video_res_masks = self.add_new_mask(
                    frame_idx=cond_frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )
        else:
            if obj_id is None:
                obj_id = self.condition_state["obj_ids"][-1]
            _ = self._register_new_object_if_needed(obj_id)

            #print(f"[DBG add_new_prompt_during_track] after register: obj_ids={self.condition_state['obj_ids']} "
            #    f"obj_id_to_idx={self.condition_state.get('obj_id_to_idx', None)}")

            if (point is not None) or (bbox is not None):
                frame_idx, obj_ids, video_res_masks = self.add_new_prompt(
                    frame_idx=cond_frame_idx,
                    obj_id=obj_id,
                    points=point,
                    bbox=bbox,
                    labels=labels,
                    clear_old_points=clear_old_points,
                    normalize_coords=True,
                )
            else:
                frame_idx, obj_ids, video_res_masks = self.add_new_mask(
                    frame_idx=cond_frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )

        #print(f"[DBG add_new_prompt_during_track] chosen frame_idx={frame_idx} num_frames={self.condition_state.get('num_frames')}")

        # (2) Consolidate temp outputs and commit them to main state
        is_init_cond_frame = (frame_idx not in self.condition_state["frames_already_tracked"])

        #print(f"[DBG add_new_prompt_during_track] is_init_cond_frame={is_init_cond_frame} "
        #    f"frames_already_tracked_has={frame_idx in self.condition_state['frames_already_tracked']}")

        is_cond = is_init_cond_frame or getattr(self, "add_all_frames_to_correct_as_cond", False)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        other_key = "non_cond_frame_outputs" if storage_key == "cond_frame_outputs" else "cond_frame_outputs"

        #print(f"[DBG add_new_prompt_during_track] is_cond={is_cond} storage_key={storage_key}")

        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx=frame_idx,
            is_cond=is_cond,
            run_mem_encoder=True,
            consolidate_at_video_res=False,
        )

        # ------------------------------------------------------------------
        # IMPORTANT FIX:
        # The consolidation for a *new* conditioning frame often sets old objects
        # to "absent" (pred_masks = -1024). That kills them (e.g., oid=4).
        #
        # We patch consolidated_out so that for all OLD objects we keep their
        # already-tracked masks/memory from src_out (non_cond at current global frame).
        # ------------------------------------------------------------------
        try:
            new_obj_id_to_idx = self.condition_state.get("obj_id_to_idx", None)
            if src_out is not None and torch.is_tensor(consolidated_out.get("pred_masks", None)) and torch.is_tensor(src_out.get("pred_masks", None)) \
            and isinstance(old_obj_id_to_idx, dict) and isinstance(new_obj_id_to_idx, dict):

                dst_pm = consolidated_out["pred_masks"]
                src_pm = src_out["pred_masks"]
                dst_B = int(dst_pm.shape[0])
                src_B = int(src_pm.shape[0])

                #print(f"[DBG MERGE] dst_B={dst_B} src_B={src_B}")
                # print dst per-slot max BEFORE merge
                try:
                    mx_before = [float(dst_pm[i].max().item()) for i in range(dst_B)]
                    #print(f"[DBG MERGE] dst pred_masks max BEFORE={['%.1f'%m for m in mx_before]}")
                except Exception:
                    pass

                # keys we attempt to merge slot-wise if present and tensor
                slot_keys = ["pred_masks", "obj_ptr", "maskmem_features"]
                # maskmem_pos_enc can be list/tuple; we handle separately

                for oid in old_obj_ids:
                    oid = int(oid)
                    #if oid not in old_obj_id_to_idx or oid not in new_obj_id_to_idx:
                    #    print(f"[DBG MERGE] oid={oid} missing in mapping (old or new).")
                    #    continue

                    si = int(old_obj_id_to_idx[oid])
                    di = int(new_obj_id_to_idx[oid])

                    #if si < 0 or si >= src_B or di < 0 or di >= dst_B:
                    #    print(f"[DBG MERGE] oid={oid} index OOB: src_i={si}/{src_B} dst_i={di}/{dst_B}")
                    #    continue

                    #print(f"[DBG MERGE] oid={oid} src_i={si} -> dst_i={di}")

                    for k in slot_keys:
                        s = src_out.get(k, None)
                        d = consolidated_out.get(k, None)
                        if torch.is_tensor(s) and torch.is_tensor(d) and s.ndim >= 1 and d.ndim >= 1:
                            # move src slice to dst device/dtype if needed
                            ss = s[si].to(device=d.device, dtype=d.dtype, non_blocking=True)
                            d[di].copy_(ss)

                    # maskmem_pos_enc: often list/tuple of tensors shaped [B,...]
                    sp = src_out.get("maskmem_pos_enc", None)
                    dp = consolidated_out.get("maskmem_pos_enc", None)
                    if isinstance(sp, (list, tuple)) and isinstance(dp, (list, tuple)) and len(sp) == len(dp):
                        new_dp = []
                        for sp_i, dp_i in zip(sp, dp):
                            if torch.is_tensor(sp_i) and torch.is_tensor(dp_i) and sp_i.shape[0] > si and dp_i.shape[0] > di:
                                tmp = dp_i.clone()
                                tmp[di].copy_(sp_i[si].to(device=dp_i.device, dtype=dp_i.dtype, non_blocking=True))
                                new_dp.append(tmp)
                            else:
                                new_dp.append(dp_i)
                        consolidated_out["maskmem_pos_enc"] = type(dp)(new_dp)

                # print dst per-slot max AFTER merge
                try:
                    mx_after = [float(consolidated_out["pred_masks"][i].max().item()) for i in range(dst_B)]
                    #print(f"[DBG MERGE] dst pred_masks max AFTER ={['%.1f'%m for m in mx_after]}")
                except Exception:
                    pass

            #else:
            #    print("[DBG MERGE] skipped (missing src_out or mappings or tensors)")
        except Exception as e:
            print(f"[DBG MERGE] failed: {e}")

        self._dbg_state("add_new_prompt_during_track:BEFORE_EXPAND")

        # Make all previously stored frames compatible with the new batch size
        self._expand_all_stored_outputs_to_current_batch()

        self._dbg_state("add_new_prompt_during_track:AFTER_EXPAND")

        # Commit like preflight does
        temp_per_obj = self.condition_state["temp_output_dict_per_obj"]
        consolidated_inds = self.condition_state["consolidated_frame_inds"]

        # Add consolidated frame to the right bucket and remove from the other
        consolidated_inds[storage_key].add(frame_idx)
        consolidated_inds[other_key].discard(frame_idx)

        # Write the consolidated output to the main dict
        output_dict[storage_key][frame_idx] = consolidated_out

        # Create per-object slices for this frame
        self._add_output_per_object(frame_idx, consolidated_out, storage_key)

        # Clear temp outputs (mirrors preflight)
        for obj_temp in temp_per_obj.values():
            obj_temp[storage_key].pop(frame_idx, None)
            obj_temp[other_key].pop(frame_idx, None)

        # (3) Resume tracking
        self.condition_state["tracking_has_started"] = True

        print("shape ", len(self.condition_state["images"]), " frame index ", frame_idx)
        return frame_idx, obj_ids, video_res_masks

    def _dbg_state(self, tag: str):
        cs = self.condition_state
        od = cs.get("output_dict", {})
        cfo = od.get("cond_frame_outputs", {})
        nfo = od.get("non_cond_frame_outputs", {})
        cinds = cs.get("consolidated_frame_inds", {})
        fa = cs.get("frames_already_tracked", {})

        #print(
        #    f"[DBG {tag}] "
        #    f"num_frames={cs.get('num_frames')} len(images)={len(cs.get('images', []))} "
        #    f"tracking_has_started={cs.get('tracking_has_started')} "
        #    f"obj_ids={cs.get('obj_ids')} "
        #    f"obj_id_to_idx={cs.get('obj_id_to_idx')} "
        #    f"add_all_frames_to_correct_as_cond={getattr(self, 'add_all_frames_to_correct_as_cond', None)} "
        #    f"cond_out={len(cfo)} noncond_out={len(nfo)} "
        #    f"cond_inds={len(cinds.get('cond_frame_outputs', set())) if isinstance(cinds.get('cond_frame_outputs', None), set) else cinds.get('cond_frame_outputs')} "
        #    f"noncond_inds={len(cinds.get('non_cond_frame_outputs', set())) if isinstance(cinds.get('non_cond_frame_outputs', None), set) else cinds.get('non_cond_frame_outputs')} "
        #    f"frames_already_tracked={len(fa)}"
        #)


    def _mask_iou(self, a: torch.Tensor, b: torch.Tensor) -> float:
        # a,b: bool tensors [H,W]
        inter = torch.logical_and(a, b).sum().item()
        if inter == 0:
            return 0.0
        union = torch.logical_or(a, b).sum().item()
        return float(inter) / float(union) if union > 0 else 0.0


    def _dedupe_by_mask_iou(
        self,
        obj_ids,
        video_res_masks_raw: torch.Tensor,
        object_score_logits: torch.Tensor = None,
        iou_thr: float = 0.6,
        min_area: int = 50,
        ):
        """
        Keep at most one mask per physical person by suppressing highly-overlapping masks (mask-NMS).
        Returns filtered (obj_ids, masks_raw).
        """
        if video_res_masks_raw is None:
            return obj_ids, video_res_masks_raw

        # Normalize shapes to [N,H,W]
        m = video_res_masks_raw
        if torch.is_tensor(m):
            if m.ndim == 4:        # [N,1,H,W]
                m = m[:, 0]
            elif m.ndim == 3:      # [N,H,W]
                pass
            else:
                return obj_ids, video_res_masks_raw
        else:
            return obj_ids, video_res_masks_raw

        N = m.shape[0]
        if N <= 1:
            return obj_ids, video_res_masks_raw

        # Scores: prefer object_score_logits if present, else compute from logits
        if object_score_logits is not None and torch.is_tensor(object_score_logits):
            scores = object_score_logits.detach().float()
            scores = scores.reshape(-1)
            if scores.numel() != N:
                scores = None
        else:
            scores = None

        if scores is None:
            probs = torch.sigmoid(m.detach().float())
            bin_masks = probs > 0.5
            scores_list = []
            for i in range(N):
                area = int(bin_masks[i].sum().item())
                if area < min_area:
                    scores_list.append(-1e9)
                else:
                    # mean prob inside mask
                    scores_list.append(float(probs[i][bin_masks[i]].mean().item()))
            scores = torch.tensor(scores_list)

        # Build binary masks once (on CPU for easy IoU)
        probs = torch.sigmoid(m.detach().float()).cpu()
        bin_masks = (probs > 0.5)

        order = torch.argsort(scores, descending=True).tolist()
        keep = []

        for idx in order:
            if scores[idx].item() < -1e8:
                continue
            cand = bin_masks[idx]
            # suppress if overlaps too much with something kept
            duplicate = False
            for j in keep:
                if self._mask_iou(cand, bin_masks[j]) >= iou_thr:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(idx)

        keep = sorted(keep)  # preserve stable ordering
        if len(keep) == N:
            return obj_ids, video_res_masks_raw

        # Filter obj_ids consistently
        if isinstance(obj_ids, (list, tuple)):
            new_obj_ids = [obj_ids[i] for i in keep]
        elif torch.is_tensor(obj_ids):
            new_obj_ids = obj_ids[keep]
        else:
            # obj_ids may be something else; best-effort
            new_obj_ids = obj_ids

        # Filter masks back to original returned shape
        kept = m[keep]  # [K,H,W]
        if video_res_masks_raw.ndim == 4:
            kept = kept[:, None, ...]  # [K,1,H,W]

        return new_obj_ids, kept.to(video_res_masks_raw.device)


    ###
    @torch.inference_mode()
    def track(self, img):
        """
        Streaming tracking step (RAW OUTPUT ONLY).

        - Always returns raw video-resolution mask logits (no gating).
        - Ensures num_frames stays in the same timeline as frame_idx.
        """
        # ---- advance global timeline ----
        self.frame_idx += 1

        # num_frames reflects global video timeline length (NOT len(images))
        if "num_frames" not in self.condition_state:
            self.condition_state["num_frames"] = 0
        self.condition_state["num_frames"] = max(int(self.condition_state["num_frames"]), int(self.frame_idx + 1))

        # preflight once
        if not self.condition_state.get("tracking_has_started", False):
            self.propagate_in_video_preflight()

        # keep stored outputs consistent with current #objects
        self._expand_all_stored_outputs_to_current_batch()

        # prepare input
        img, _, _ = self.perpare_data(img, image_size=self.image_size)

        output_dict = self.condition_state["output_dict"]
        obj_ids = self.condition_state["obj_ids"]
        batch_size = self._get_obj_num()

        # get features
        (_, _, current_vision_feats, current_vision_pos_embeds, feat_sizes) = self._get_feature(img, batch_size)

        # ---- track step ----
        current_out = self.track_step(
            frame_idx=self.frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=self.condition_state["num_frames"],  # IMPORTANT
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )

        # ---- offload / memory write (unchanged) ----
        storage_device = self.condition_state["storage_device"]

        maskmem_features = current_out.get("maskmem_features", None)
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16).to(storage_device, non_blocking=True)

        pred_masks_gpu = current_out["pred_masks"]
        if getattr(self, "fill_hole_area", 0) > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)

        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(current_out)
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out.get("object_score_logits", None)
        best_iou_score = current_out.get("best_iou_score", None)
        kf_ious = current_out.get("kf_ious", None)  # you call it kf_ious, not kf_score

        mem_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            # NEW (needed for SAMURAI memory selection)
            "best_iou_score": best_iou_score,
            "kf_score": kf_ious,  # store under the key your selector expects
        }

        # Write to memory (never gate)
        self._manage_memory_obj(self.frame_idx, mem_out)

        # ---- output at video resolution (RAW) ----
        _, video_res_masks_raw = self._get_orig_video_res_output(pred_masks_gpu)

        # store for debugging
        self._last_video_res_masks_raw = video_res_masks_raw

        # suppress duplicate masks that overlap the same person
        video_obj_ids = obj_ids
        video_obj_ids, video_res_masks_raw = self._dedupe_by_mask_iou(
            obj_ids=video_obj_ids,
            video_res_masks_raw=video_res_masks_raw,
            object_score_logits=object_score_logits,
            iou_thr=getattr(self, "dedupe_iou_thr", 0.6),
            min_area=getattr(self, "dedupe_min_area", 200),
        )

        return video_obj_ids, video_res_masks_raw


    def _manage_memory_obj(self, frame_idx, current_out):
        """
        Keep the non_cond_frame_outputs map bounded by self.num_maskmem while storing
        the provided current_out. This function is intentionally minimal: it assumes
        any filtering/suppression has already been applied to current_out before calling.
        """
        output_dict = self.condition_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        non_cond_frame_outputs[frame_idx] = current_out

        key_list = [key for key in output_dict["non_cond_frame_outputs"]]
        #! TODO: better way to manage memory
        if len(non_cond_frame_outputs) > self.num_maskmem:
            # pop the oldest entries
            for t in range(0, len(non_cond_frame_outputs) - self.num_maskmem):
                _ = non_cond_frame_outputs.pop(key_list[t], None)


    @torch.inference_mode()
    def propagate_in_video(
        self,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""

        self.propagate_in_video_preflight(self.condition_state)

        output_dict = self.condition_state["output_dict"]
        consolidated_frame_inds = self.condition_state["consolidated_frame_inds"]
        obj_ids = self.condition_state["obj_ids"]
        num_frames = self.condition_state["num_frames"]
        batch_size = self._get_obj_num()
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(frame_idx)

            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out

            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(frame_idx, current_out, storage_key)
            self.condition_state["frames_already_tracked"][frame_idx] = {
                "reverse": reverse
            }

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(pred_masks)
            yield frame_idx, obj_ids, video_res_masks

    def _add_output_per_object(self, frame_idx, current_out, storage_key):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = self.condition_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    @torch.inference_mode()
    def reset_state(self):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results()
        # Remove all object ids
        self.condition_state["obj_id_to_idx"].clear()
        self.condition_state["obj_idx_to_id"].clear()
        self.condition_state["obj_ids"].clear()
        self.condition_state["point_inputs_per_obj"].clear()
        self.condition_state["mask_inputs_per_obj"].clear()
        self.condition_state["output_dict_per_obj"].clear()
        self.condition_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self):
        """Reset all tracking inputs and results across the videos."""
        for v in self.condition_state["point_inputs_per_obj"].values():
            v.clear()
        for v in self.condition_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in self.condition_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in self.condition_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        self.condition_state["output_dict"]["cond_frame_outputs"].clear()
        self.condition_state["output_dict"]["non_cond_frame_outputs"].clear()
        self.condition_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        self.condition_state["consolidated_frame_inds"][
            "non_cond_frame_outputs"
        ].clear()
        self.condition_state["tracking_has_started"] = False
        self.condition_state["frames_already_tracked"].clear()

    def _get_image_feature(self, frame_idx, batch_size):
        """Compute the image features on a given frame."""

        # --- SAFETY: clamp conditioning-frame index ---
        # During late seeding, some call paths can pass frame_idx == len(images)
        # which is out of range (valid: 0..len(images)-1).
        images = self.condition_state.get("images", [])
        if len(images) == 0:
            raise RuntimeError("No conditioning images available in condition_state['images']")
        if frame_idx >= len(images):
            frame_idx = len(images) - 1
        if frame_idx < 0:
            frame_idx = 0

        # Look up in the cache first
        image, backbone_out = self.condition_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            image = (
                self.condition_state["images"][frame_idx].cuda().float().unsqueeze(0)
            )
            backbone_out = self.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            self.condition_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    ###
    def _get_feature(self, img, batch_size):
        image = img.cuda().float().unsqueeze(0)
        backbone_out = self.forward_image(image)
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=self.condition_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = self.condition_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = self.condition_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = self.condition_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    def _clear_non_cond_mem_around_input(self, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = self.condition_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in self.condition_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)


class SAM2CameraPredictorVOS(SAM2CameraPredictor):
    """Optimized for the VOS setting"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_memory_encoder = kwargs.get("compile_memory_encoder", False)
        self.compile_memory_attention = kwargs.get("compile_memory_attention", False)
        self.compile_prompt_encoder = kwargs.get("compile_prompt_encoder", False)
        self.compile_mask_decoder = kwargs.get("compile_mask_decoder", False)
        self._compile_all_components()

    def _compile_all_components(self):
        print("Compiling all components for VOS setting. First time may be very slow.")
        if self.compile_memory_encoder:
            print("Compiling memory encoder...")
            self.memory_encoder.forward = torch.compile(
                self.memory_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
        if self.compile_memory_attention:
            print("Compiling memory attention...")
            self.memory_attention.forward = torch.compile(
                self.memory_attention.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=True,
            )
        if self.compile_prompt_encoder:
            self.sam_prompt_encoder.forward = torch.compile(
                self.sam_prompt_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,  # Accuracy regression on True
            )
        if self.compile_mask_decoder:
            self.sam_mask_decoder.forward = torch.compile(
                self.sam_mask_decoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,  # Accuracy regression on True
            )

    def forward_image(self, img_batch: torch.Tensor):
        """
        Identical to the corresponding method in the parent (SAM2VideoPredictor), but
        cloning the backbone features and pos encoding to enable compilation.
        """
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        # Clone to help torch.compile
        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
            backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][
                i
            ].clone()
        return backbone_out

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Identical to the corresponding method in the parent (SAM2VideoPredictor), but
        cloning the outputs of prompt_encoder and mask_decoder to enable compilation.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        # Clone image_pe and the outputs of sam_prompt_encoder
        # to enable compilation
        sparse_embeddings = sparse_embeddings.clone()
        dense_embeddings = dense_embeddings.clone()
        image_pe = self.sam_prompt_encoder.get_dense_pe().clone()
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        # Clone the output of sam_mask_decoder
        # to enable compilation
        low_res_multimasks = low_res_multimasks.clone()
        ious = ious.clone()
        sam_output_tokens = sam_output_tokens.clone()
        object_score_logits = object_score_logits.clone()

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Identical to the corresponding method in the parent (SAM2VideoPredictor), but
        cloning the memories and their pos enc to enable compilation.
        """
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        # Clone the feats and pos_enc to enable compilation
        maskmem_features = maskmem_out["vision_features"].clone()
        maskmem_pos_enc = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc
