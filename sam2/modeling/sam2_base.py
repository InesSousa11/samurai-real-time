# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from loguru import logger

import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
import cv2

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

from sam2.utils.kalman_filter import KalmanFilter

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # Whether to use SAMURAI or original SAM 2
        samurai_mode: bool = True,
        # Hyperparameters for SAMURAI
        stable_frames_threshold: int = 15,
        stable_ious_threshold: float = 0.3,
        min_obj_score_logits: float = -1,
        kf_score_weight: float = 0.15,
        memory_bank_iou_threshold: float = 0.5,
        memory_bank_obj_score_threshold: float = 0.0,
        memory_bank_kf_score_threshold: float = 0.0,
        memory_bank_reid_threshold: float = 0.55,
        reid_gallery_max_size=6,
        reid_gallery_add_sim_threshold=0.85,
        reid_gallery_add_cooldown=15,
        reid_gallery_min_bbox_area_ratio=0.02,
        reid_gallery_border_margin_ratio=0.02,
        reid_gallery_max_border_touches=1,
        reid_gallery_min_fill_ratio=0.18,
        reid_thr = 0.85,
        reid_gallery_random_replace_prob: float = 0.15,
        reid_gallery_random_replace_if_diverse_prob: float = 0.30,
        reid_gallery_anchor_protect: bool = True,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Whether to use SAMURAI or original SAM 2
        self.samurai_mode = samurai_mode

        # Init Kalman Filter
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0

        # A per-object bank: obj_id -> {"kf": KalmanFilter, "mean": ..., "cov": ..., "stable": int}
        self._kf_bank = dict()

        # Debug purpose
        self.history = {} # debug
        self.frame_cnt = 0 # debug

        # Hyperparameters for SAMURAI
        self.stable_frames_threshold = stable_frames_threshold
        self.stable_ious_threshold = stable_ious_threshold
        self.min_obj_score_logits = min_obj_score_logits
        self.kf_score_weight = kf_score_weight
        self.memory_bank_iou_threshold = memory_bank_iou_threshold
        self.memory_bank_obj_score_threshold = memory_bank_obj_score_threshold
        self.memory_bank_kf_score_threshold = memory_bank_kf_score_threshold
        self.memory_bank_reid_threshold = memory_bank_reid_threshold
        self.reid_gallery_max_size = int(reid_gallery_max_size)
        self.reid_gallery_add_sim_threshold = float(reid_gallery_add_sim_threshold)
        self.reid_gallery_add_cooldown = int(reid_gallery_add_cooldown)
        self.reid_gallery_min_bbox_area_ratio = float(reid_gallery_min_bbox_area_ratio)
        self.reid_gallery_border_margin_ratio = float(reid_gallery_border_margin_ratio)
        self.reid_gallery_max_border_touches = int(reid_gallery_max_border_touches)
        self.reid_gallery_min_fill_ratio = float(reid_gallery_min_fill_ratio)
        self.reid_thr = float(reid_thr)
        self.reid_gallery_random_replace_prob = reid_gallery_random_replace_prob
        self.reid_gallery_random_replace_if_diverse_prob = reid_gallery_random_replace_if_diverse_prob
        self.reid_gallery_anchor_protect = reid_gallery_anchor_protect

        print(f"\033[93mSAMURAI mode: {self.samurai_mode}\033[0m")

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.
        (same docstring as your current version)
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # ---------------- helpers ----------------
        def _tensor_stats(t):
            """Small numeric summary (NO huge dumps)."""
            if not torch.is_tensor(t):
                return None
            x = t.detach().float()
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return {
                "shape": list(x.shape),
                "mean": float(x.mean().item()) if x.numel() else float("nan"),
                "std": float(x.std().item()) if x.numel() else float("nan"),
                "min": float(x.min().item()) if x.numel() else float("nan"),
                "max": float(x.max().item()) if x.numel() else float("nan"),
                "norm": float(x.norm().item()) if x.numel() else float("nan"),
            }

        def _mask_bbox_xyxy_from_binary(mask_hw: torch.Tensor):
            # mask_hw: bool/0-1 tensor [H,W] on CPU
            nz = torch.nonzero(mask_hw, as_tuple=False)
            if nz.numel() == 0:
                return [0, 0, 0, 0]
            y_min = int(nz[:, 0].min().item())
            y_max = int(nz[:, 0].max().item())
            x_min = int(nz[:, 1].min().item())
            x_max = int(nz[:, 1].max().item())
            return [x_min, y_min, x_max, y_max]

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        # ---- prompt encoder ----
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )

        # ---- prompt debug ----
        prompt_debug = {}
        try:
            prompt_debug = {
                "point_coords": sam_point_coords.detach().cpu().tolist() if torch.is_tensor(sam_point_coords) else None,
                "point_labels": sam_point_labels.detach().cpu().tolist() if torch.is_tensor(sam_point_labels) else None,
                "sparse_embeddings_stats": _tensor_stats(sparse_embeddings),
                "dense_embeddings_stats": _tensor_stats(dense_embeddings),
            }
        except Exception:
            prompt_debug = {}

        # ---- mask decoder ----
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # object presence gate
        is_obj_appearing = None
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.min_obj_score_logits
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks to fp32
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        kf_ious = None
        best_iou_inds = None  # Tensor[B] in multimask, else int 0

        # NEW debug variables
        selection_mode = "single_mask_no_multimask"
        kf_influence_active = False

        if multimask_output and self.samurai_mode:
            if (self.kf_mean is None and self.kf_covariance is None) or (self.stable_frames == 0):
                selection_mode = "plain_iou_kf_uninitialized"
                kf_influence_active = False

                best_iou_inds = torch.argmax(ious, dim=-1)  # [B]
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)

                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]

                self.kf_mean, self.kf_covariance = self.kf.initiate(
                    self.kf.xyxy_to_xyah(high_res_bbox)
                )

                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]

                self.frame_cnt += 1
                self.stable_frames += 1

            elif self.stable_frames < self.stable_frames_threshold:
                selection_mode = "plain_iou_kf_stabilizing"
                kf_influence_active = False

                self.kf_mean, self.kf_covariance = self.kf.predict(
                    self.kf_mean, self.kf_covariance
                )

                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)

                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]

                if B == 1:
                    sel = int(best_iou_inds.detach().cpu().reshape(-1)[0].item())
                    if float(ious[0, sel].item()) > self.stable_ious_threshold:
                        self.kf_mean, self.kf_covariance = self.kf.update(
                            self.kf_mean,
                            self.kf_covariance,
                            self.kf.xyxy_to_xyah(high_res_bbox),
                        )
                        self.stable_frames += 1
                    else:
                        self.stable_frames = 0

                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]

                self.frame_cnt += 1

            else:
                selection_mode = "combined_kf_iou"
                kf_influence_active = True

                self.kf_mean, self.kf_covariance = self.kf.predict(
                    self.kf_mean, self.kf_covariance
                )

                high_res_multibboxes = []
                batch_inds = torch.arange(B, device=device)

                # high_res_multimasks is [B, M, H, W]
                for mi in range(ious.shape[1]):  # M candidates
                    non_zero_indices = torch.argwhere(
                        high_res_multimasks[batch_inds, mi].unsqueeze(1)[0][0] > 0.0
                    )
                    if len(non_zero_indices) == 0:
                        high_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_multibboxes.append([
                            x_min.item(), y_min.item(), x_max.item(), y_max.item()
                        ])

                # [M]
                kf_ious = torch.tensor(
                    self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes),
                    device=device,
                    dtype=ious.dtype,
                )

                weighted_ious = self.kf_score_weight * kf_ious + (1.0 - self.kf_score_weight) * ious
                best_iou_inds = torch.argmax(weighted_ious, dim=-1)  # [B]

                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)

                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]

                self.frame_cnt += 1

                if B == 1:
                    sel = int(best_iou_inds.detach().cpu().reshape(-1)[0].item())
                    if float(ious[0, sel].item()) < self.stable_ious_threshold:
                        self.stable_frames = 0
                    else:
                        self.kf_mean, self.kf_covariance = self.kf.update(
                            self.kf_mean,
                            self.kf_covariance,
                            self.kf.xyxy_to_xyah(high_res_multibboxes[sel]),
                        )

        elif multimask_output and (not self.samurai_mode):
            selection_mode = "plain_iou_no_samurai"
            kf_influence_active = False

            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)

            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]

        else:
            selection_mode = "single_mask_no_multimask"
            kf_influence_active = False

            best_iou_inds = 0
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = (
                    is_obj_appearing.float() if is_obj_appearing is not None
                    else torch.ones_like(object_score_logits)
                )

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        # ------------------------ DEBUG (per-object append) ------------------------
        try:
            cs = getattr(self, "condition_state", None)
            if isinstance(cs, dict):
                dbg = cs.setdefault("debug_last", {})
                per_obj = dbg.setdefault("per_obj", [])

                ious_cpu = ious.detach().float().cpu()          # [B,M]
                obj_logit_cpu = object_score_logits.detach().float().cpu()
                obj_prob_cpu = obj_logit_cpu.sigmoid()

                if torch.is_tensor(best_iou_inds):
                    sel_idx = int(best_iou_inds.detach().cpu().reshape(-1)[0].item())
                    sel_idx_vec = best_iou_inds.detach().cpu().reshape(-1).to(torch.int64).tolist()
                else:
                    sel_idx = int(best_iou_inds)
                    sel_idx_vec = [sel_idx]

                cand_bboxes = []
                cand_areas = []
                try:
                    if torch.is_tensor(high_res_multimasks) and high_res_multimasks.ndim == 4:
                        hr_cpu = high_res_multimasks.detach().float().cpu()  # [B,M,H,W] or [B,1,H,W]
                        M_here = int(hr_cpu.shape[1])
                        for m in range(M_here):
                            mask_bin = (hr_cpu[0, m] > 0)
                            cand_areas.append(int(mask_bin.sum().item()))
                            cand_bboxes.append(_mask_bbox_xyxy_from_binary(mask_bin))
                except Exception:
                    pass

                kf_pred_xyxy = None
                try:
                    if getattr(self, "kf_mean", None) is not None:
                        kf_pred_xyxy = [float(x) for x in self.kf.xyah_to_xyxy(self.kf_mean[:4])]
                except Exception:
                    kf_pred_xyxy = None

                kf_cpu = None
                combined_cpu = None
                alpha = float(getattr(self, "kf_score_weight", 0.0))
                if kf_ious is not None and torch.is_tensor(kf_ious):
                    kf_cpu = kf_ious.detach().float().cpu().reshape(1, -1)  # [1,M]
                    combined_cpu = alpha * kf_cpu + (1.0 - alpha) * ious_cpu
                else:
                    combined_cpu = ious_cpu

                sel_iou = float(ious_cpu[0, sel_idx].item()) if ious_cpu.numel() else float("nan")
                sel_kf = float(kf_cpu.reshape(-1)[sel_idx].item()) if (kf_cpu is not None and kf_cpu.numel()) else None
                sel_comb = float(combined_cpu[0, sel_idx].item()) if combined_cpu.numel() else float("nan")

                per_obj.append({
                    "prompt_debug": prompt_debug,

                    "multimask_output": bool(multimask_output),
                    "samurai_mode": bool(getattr(self, "samurai_mode", False)),
                    "selection_mode": selection_mode,
                    "kf_influence_active": bool(kf_influence_active),
                    "stable_frames": int(getattr(self, "stable_frames", 0)),
                    "stable_frames_threshold": int(getattr(self, "stable_frames_threshold", 0)),
                    "stable_ious_threshold": float(getattr(self, "stable_ious_threshold", 0.0)),
                    "B": int(B),
                    "M": int(ious_cpu.shape[1]) if ious_cpu.ndim == 2 else None,

                    "selected_mask_index": sel_idx_vec,
                    "ious": ious_cpu,
                    "kf_ious": kf_cpu,
                    "combined": combined_cpu,
                    "object_score_logits": obj_logit_cpu,
                    "object_score_prob": obj_prob_cpu,
                    "is_obj_appearing": (
                        is_obj_appearing.detach().cpu() if torch.is_tensor(is_obj_appearing) else None
                    ),

                    "kf_pred_bbox_xyxy": kf_pred_xyxy,
                    "cand_bboxes_xyxy": cand_bboxes,
                    "cand_mask_areas": cand_areas,
                    "selected_bbox_xyxy": (
                        cand_bboxes[sel_idx] if (cand_bboxes and sel_idx < len(cand_bboxes)) else None
                    ),

                    "low_res_multimasks": low_res_multimasks.detach().cpu(),

                    "selected_iou": sel_iou,
                    "selected_kf_iou": sel_kf,
                    "selected_combined": sel_comb,
                    "kf_score_weight": alpha,
                })
        except Exception:
            pass
        # ---------------------------------------------------------------------------

        # score of selected mask
        if torch.is_tensor(best_iou_inds):
            best_iou_score = ious[torch.arange(B, device=device), best_iou_inds]  # [B]
        else:
            best_iou_score = ious[:, 0] if ious.ndim == 2 else ious

        # selected KF value
        kf_selected = None
        if kf_ious is not None and torch.is_tensor(kf_ious):
            if torch.is_tensor(best_iou_inds):
                sel0 = int(best_iou_inds.detach().cpu().reshape(-1)[0].item()) if B == 1 else None
                kf_selected = kf_ious[sel0] if sel0 is not None else None
            else:
                kf_selected = kf_ious[int(best_iou_inds)]

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            best_iou_score if best_iou_score.ndim > 0 else best_iou_score.unsqueeze(0),
            kf_selected,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _, _, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            1,
            1
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
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
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        debug_obj_id=None,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        def _reduce_max_or_none(x):
            if not torch.is_tensor(x):
                return None
            x = x.detach().float().reshape(-1)
            if x.numel() == 0:
                return None
            x = x[torch.isfinite(x)]
            if x.numel() == 0:
                return None
            return float(x.max().item())

        def _frame_passes_memory_filters(prev_out):
            """
            prev_out is already per-object sliced before this function is called.
            So each tensor here should correspond only to the current object.
            """
            if prev_out is None or (not isinstance(prev_out, dict)):
                return False

            iou_score = prev_out.get("best_iou_score", None)
            obj_score = prev_out.get("object_score_logits", None)
            kf_score = prev_out.get("kf_score", None)
            reid_sim = prev_out.get("reid_sim", None)
            reacquire = prev_out.get("reacquire", None)

            # Never use memory frames that were captured during reacquisition
            if torch.is_tensor(reacquire):
                rv = reacquire.detach().reshape(-1)
                if rv.numel() > 0 and bool(rv[0].item()):
                    return False

            iou_v = _reduce_max_or_none(iou_score)
            obj_v = _reduce_max_or_none(obj_score)
            kf_v = _reduce_max_or_none(kf_score)
            sim_v = _reduce_max_or_none(reid_sim)

            if iou_v is None or obj_v is None:
                return False

            if iou_v <= float(self.memory_bank_iou_threshold):
                return False

            if obj_v <= float(self.memory_bank_obj_score_threshold):
                return False

            if kf_v is not None and kf_v <= float(self.memory_bank_kf_score_threshold):
                return False

            reid_thr_mem = float(getattr(self, "memory_bank_reid_threshold", float("-inf")))
            if sim_v is not None and sim_v <= reid_thr_mem:
                return False

            return True

        if not is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed = [], []

            assert len(output_dict["cond_frame_outputs"]) > 0

            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )

            # Cond frames are allowed to stay as-is.
            # They are already per-object sliced before arriving here.
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]

            stride = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.samurai_mode:
                # With num_maskmem=7 there are 6 temporal non-cond slots.
                max_noncond_frames = max(1, int(self.num_maskmem) - 1)

                valid_indices = []

                # Prefer the per-object history built by _manage_memory_obj
                cs = getattr(self, "condition_state", None)
                good_memory_frames_per_id = {}
                if isinstance(cs, dict):
                    good_memory_frames_per_id = cs.get("good_memory_frames_per_id", {})

                if debug_obj_id is not None and isinstance(good_memory_frames_per_id, dict):
                    obj_hist = good_memory_frames_per_id.get(int(debug_obj_id), [])
                    candidate_frames = [
                        int(f) for f in obj_hist
                        if int(f) < int(frame_idx)
                    ]

                    for f in candidate_frames:
                        prev_out = output_dict["non_cond_frame_outputs"].get(int(f), None)
                        if _frame_passes_memory_filters(prev_out):
                            valid_indices.append(int(f))
                else:
                    # Fallback to old scan if per-object history is unavailable
                    if frame_idx > 1:
                        for i in range(frame_idx - 1, 1, -1):
                            prev_out = output_dict["non_cond_frame_outputs"].get(i, None)
                            if _frame_passes_memory_filters(prev_out):
                                valid_indices.insert(0, i)
                            if len(valid_indices) >= max_noncond_frames:
                                break

                # Keep only the most recent max_noncond_frames
                if len(valid_indices) > max_noncond_frames:
                    valid_indices = valid_indices[-max_noncond_frames:]

                # Force previous frame only if it passes and is not already there
                forced = frame_idx - 1
                prev_forced = output_dict["non_cond_frame_outputs"].get(forced, None)
                if _frame_passes_memory_filters(prev_forced) and forced not in valid_indices:
                    valid_indices.append(int(forced))

                # again keep only the most recent max_noncond_frames
                valid_indices = sorted(set(int(x) for x in valid_indices))
                if len(valid_indices) > max_noncond_frames:
                    valid_indices = valid_indices[-max_noncond_frames:]

                selected_noncond_for_attn = []

                # Assign non-cond temporal positions 1..max_noncond_frames
                # oldest -> smallest t_pos, newest -> largest t_pos
                recent_frames = valid_indices[-max_noncond_frames:]
                for j, sel_frame in enumerate(recent_frames, start=1):
                    out = output_dict["non_cond_frame_outputs"].get(int(sel_frame), None)
                    if out is None:
                        out = unselected_cond_outputs.get(int(sel_frame), None)

                    if out is not None:
                        t_pos_and_prevs.append((j, out))
                        selected_noncond_for_attn.append(int(sel_frame))

                # Persist per-object debug
                try:
                    if isinstance(cs, dict):
                        od_global = cs.setdefault("output_dict", {})
                        dbg = od_global.setdefault("debug_memory_attn", {})
                    else:
                        dbg = output_dict.setdefault("debug_memory_attn", {})
                except Exception:
                    dbg = output_dict.setdefault("debug_memory_attn", {})

                frame_dbg = dbg.setdefault(int(frame_idx), {})
                obj_dbg_key = int(debug_obj_id) if debug_obj_id is not None else -1

                frame_dbg[obj_dbg_key] = {
                    "frame_idx": int(frame_idx),
                    "obj_id": obj_dbg_key,
                    "selected_cond_frames": [int(k) for k in selected_cond_outputs.keys()],
                    "selected_noncond_frames": [int(x) for x in selected_noncond_for_attn],
                    "valid_indices_pool": [int(x) for x in valid_indices],
                    "num_maskmem": int(self.num_maskmem),
                    "max_cond_frames_in_attn": int(self.max_cond_frames_in_attn),
                    "max_noncond_frames_in_attn": int(max_noncond_frames),
                    "memory_temporal_stride_for_eval": int(self.memory_temporal_stride_for_eval)
                    if hasattr(self, "memory_temporal_stride_for_eval")
                    else None,
                    "thresholds": {
                        "iou": float(self.memory_bank_iou_threshold)
                        if hasattr(self, "memory_bank_iou_threshold")
                        else None,
                        "obj": float(self.memory_bank_obj_score_threshold)
                        if hasattr(self, "memory_bank_obj_score_threshold")
                        else None,
                        "kf": float(self.memory_bank_kf_score_threshold)
                        if hasattr(self, "memory_bank_kf_score_threshold")
                        else None,
                        "reid": float(getattr(self, "memory_bank_reid_threshold", float("-inf"))),
                    },
                }

            else:
                for t_pos in range(1, self.num_maskmem):
                    t_rel = self.num_maskmem - t_pos
                    if t_rel == 1:
                        if not track_in_reverse:
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        if not track_in_reverse:
                            prev_frame_idx = ((frame_idx - 2) // stride) * stride
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                        else:
                            prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride

                    out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    if out is None:
                        out = unselected_cond_outputs.get(prev_frame_idx, None)
                    t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue

                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))

                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)

            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)

                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs

                pos_and_ptrs = [
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]

                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))

                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)

                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)

                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0

        else:
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
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
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
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
    
    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features_full = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features_full = None

        # If mask passthrough mode
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features_full, mask_inputs)
            return current_out, sam_outputs, high_res_features_full, pix_feat

        # If prev logits present, use them as mask_inputs
        if prev_sam_mask_logits is not None:
            assert point_inputs is not None and mask_inputs is None
            mask_inputs = prev_sam_mask_logits

        # ---- Authoritative number of objects: batch size on the top-level feature ----
        # current_vision_feats[-1] shape is (HW, B, C)
        n_obj = int(current_vision_feats[-1].size(1))

        # Real object ids in the same order as the batch dimension
        obj_ids_list = list(self.condition_state.get("obj_ids", []))
        if len(obj_ids_list) != n_obj:
            # safe fallback
            obj_ids_list = list(range(n_obj))

        # Helper: slice per-object prompt tensors
        def _slice_prompt(i, point_inputs, mask_inputs, n_obj):
            if isinstance(point_inputs, dict):
                p_i = {}
                for k, v in point_inputs.items():
                    if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == n_obj:
                        p_i[k] = v[i:i+1]
                    else:
                        p_i[k] = v
            else:
                p_i = point_inputs

            if mask_inputs is not None and torch.is_tensor(mask_inputs) and mask_inputs.dim() >= 3:
                m_i = mask_inputs[i:i+1] if mask_inputs.size(0) == n_obj else mask_inputs
            else:
                m_i = mask_inputs
            return p_i, m_i

        # Helpers: per-object KF state
        def _load_kf_for(real_obj_id: int):
            st = self._kf_bank.get(real_obj_id, None)
            if st is None:
                st = {"kf": KalmanFilter(), "mean": None, "cov": None, "stable": 0}
                self._kf_bank[real_obj_id] = st
            self.kf = st["kf"]
            self.kf_mean = st["mean"]
            self.kf_covariance = st["cov"]
            self.stable_frames = st["stable"]

        def _save_kf_for(real_obj_id: int):
            self._kf_bank[real_obj_id] = {
                "kf": self.kf,
                "mean": self.kf_mean,
                "cov": self.kf_covariance,
                "stable": self.stable_frames,
            }

        # Utility: slice (HW,B,C) -> (HW,1,C)
        def _slice_hwbc(t, i):
            if t is None:
                return None
            return t[:, i:i+1, :]

        # Utility: slice [B, ...] -> [1, ...]
        def _slice_b_leading(t, i):
            if t is None:
                return None
            return t[i:i+1]

        # Build a tailored output_dict that only carries memories for this object.
        # IMPORTANT: if that frame’s memory doesn’t have this object (B <= i), we drop that frame.
        def _output_dict_for_obj(orig, i):
            new = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

            def _slice0(t):
                if torch.is_tensor(t) and t.dim() >= 1 and t.size(0) > i:
                    return t[i:i+1]
                return t

            for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for t, out in orig[key].items():
                    if out is None:
                        new[key][t] = None
                        continue
                    if not isinstance(out, dict):
                        new[key][t] = None
                        continue

                    out_new = dict(out)

                    # maskmem_features: [B, C, H, W]
                    mf = out_new.get("maskmem_features", None)
                    if torch.is_tensor(mf) and mf.dim() >= 4:
                        if mf.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["maskmem_features"] = mf[i:i+1]

                    # maskmem_pos_enc: list/tuple of levels, each [B, C, H, W]
                    mpe = out_new.get("maskmem_pos_enc", None)
                    if isinstance(mpe, (list, tuple)) and len(mpe) > 0:
                        first_tensor = None
                        for lvl in mpe:
                            if torch.is_tensor(lvl):
                                first_tensor = lvl
                                break

                        if first_tensor is not None and first_tensor.dim() >= 4 and first_tensor.size(0) <= i:
                            new[key][t] = None
                            continue

                        mpe_sliced = []
                        for lvl in mpe:
                            if torch.is_tensor(lvl) and lvl.dim() >= 4:
                                mpe_sliced.append(lvl[i:i+1])
                            else:
                                mpe_sliced.append(lvl)
                        out_new["maskmem_pos_enc"] = type(mpe)(mpe_sliced)

                    # obj_ptr: [B, C]
                    op = out_new.get("obj_ptr", None)
                    if torch.is_tensor(op) and op.dim() >= 2:
                        if op.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["obj_ptr"] = op[i:i+1]

                    # pred_masks: [B, 1, H, W] or [B, H, W]
                    pm = out_new.get("pred_masks", None)
                    if torch.is_tensor(pm) and pm.dim() >= 3:
                        if pm.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["pred_masks"] = pm[i:i+1]

                    # optional pred_masks_high_res: [B, 1, H, W] or [B, H, W]
                    pmhr = out_new.get("pred_masks_high_res", None)
                    if torch.is_tensor(pmhr) and pmhr.dim() >= 3:
                        if pmhr.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["pred_masks_high_res"] = pmhr[i:i+1]

                    # best_iou_score: [B] or [B, ...]
                    bi = out_new.get("best_iou_score", None)
                    if torch.is_tensor(bi):
                        if bi.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["best_iou_score"] = _slice0(bi)

                    # object_score_logits: [B] or [B, ...]
                    osl = out_new.get("object_score_logits", None)
                    if torch.is_tensor(osl):
                        if osl.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["object_score_logits"] = _slice0(osl)

                    # kf_score: [B] or [B, ...]
                    kfs = out_new.get("kf_score", None)
                    if torch.is_tensor(kfs):
                        if kfs.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["kf_score"] = _slice0(kfs)

                    # reid_ok: [B] or [B, ...]
                    rok = out_new.get("reid_ok", None)
                    if torch.is_tensor(rok):
                        if rok.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["reid_ok"] = _slice0(rok)

                    # memory_accept_mask: [B] or [B, ...]
                    mam = out_new.get("memory_accept_mask", None)
                    if torch.is_tensor(mam):
                        if mam.size(0) <= i:
                            new[key][t] = None
                            continue
                        out_new["memory_accept_mask"] = _slice0(mam)

                    new[key][t] = out_new

            return new

        # Collectors
        lows, highs, obj_ptrs, obj_scores, best_ious, kf_ious_list = [], [], [], [], [], []

        for obj_slot in range(n_obj):
            real_obj_id = int(obj_ids_list[obj_slot])

            # Slice prompts
            p_i, m_i = _slice_prompt(obj_slot, point_inputs, mask_inputs, n_obj)

            # Load KF for REAL object id
            _load_kf_for(real_obj_id)

            # Slice current vision feats/pos to B==1 for this object
            curr_feats_i = [_slice_hwbc(current_vision_feats[-1], obj_slot)]
            curr_pos_i = [_slice_hwbc(current_vision_pos_embeds[-1], obj_slot)]

            # Tailor previous memories to this object
            od_i = _output_dict_for_obj(output_dict, obj_slot)

            # Build memory-conditioned features for this object only
            pix_feat_i = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=curr_feats_i,
                current_vision_pos_embeds=curr_pos_i,
                feat_sizes=feat_sizes[-1:],
                output_dict=od_i,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                debug_obj_id=real_obj_id,   # <-- added
            )

            # Slice high-res features to the same B==1
            if high_res_features_full is not None:
                high_res_features_i = [_slice_b_leading(h, obj_slot) for h in high_res_features_full]
            else:
                high_res_features_i = None

            multimask_output = self._use_multimask(is_init_cond_frame, p_i)
            # TODO
            # print("FORWARD_SAM_HEADS FROM:", self._forward_sam_heads.__func__.__qualname__, flush=True)
            sam_out = self._forward_sam_heads(
                backbone_features=pix_feat_i,
                point_inputs=p_i,
                mask_inputs=m_i,
                high_res_features=high_res_features_i,
                multimask_output=multimask_output,
            )

            (
                _a, _b, _c,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                best_iou_score,
                kf_ious,
            ) = sam_out

            lows.append(low_res_masks)
            highs.append(high_res_masks)
            obj_ptrs.append(obj_ptr)
            obj_scores.append(object_score_logits)

            # Normalize best_iou_score -> 1D tensor
            if best_iou_score is None:
                best_iou_score = torch.full(
                    (1,), float("nan"),
                    device=low_res_masks.device,
                    dtype=low_res_masks.dtype
                )
            elif not torch.is_tensor(best_iou_score):
                best_iou_score = torch.as_tensor(
                    [best_iou_score],
                    device=low_res_masks.device,
                    dtype=low_res_masks.dtype
                )
            elif best_iou_score.ndim == 0:
                best_iou_score = best_iou_score.unsqueeze(0)
            best_ious.append(best_iou_score)

            # Normalize kf_ious -> 1D tensor
            if kf_ious is None:
                kf_ious = torch.full(
                    (1,), float("nan"),
                    device=low_res_masks.device,
                    dtype=low_res_masks.dtype
                )
            elif not torch.is_tensor(kf_ious):
                kf_ious = torch.as_tensor(
                    [kf_ious],
                    device=low_res_masks.device,
                    dtype=low_res_masks.dtype
                )
            elif kf_ious.ndim == 0:
                kf_ious = kf_ious.unsqueeze(0)
            kf_ious_list.append(kf_ious)

            # Save KF for REAL object id
            _save_kf_for(real_obj_id)

        # Concatenate per-object outputs
        def _cat_safe(ts):
            return ts[0] if len(ts) == 1 else torch.cat(ts, dim=0)

        low_res_masks = _cat_safe(lows)
        high_res_masks = _cat_safe(highs)
        obj_ptr = _cat_safe(obj_ptrs)
        object_score_logits = _cat_safe(obj_scores)
        best_iou_score = _cat_safe(best_ious)
        kf_ious = _cat_safe(kf_ious_list)

        sam_outputs = (
            None, None, None,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            best_iou_score,
            kf_ious,
        )
        return current_out, sam_outputs, high_res_features_full, None

    """
    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    """

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        # ---------------- DEBUG: initialize per-frame debug container ----------------
        try:
            cs = getattr(self, "condition_state", None)
            if isinstance(cs, dict):
                cs["debug_last"] = {
                    "frame_idx": int(frame_idx),
                    "is_init_cond_frame": bool(is_init_cond_frame),
                    "num_frames": int(num_frames) if num_frames is not None else None,
                    # store per-object dicts appended from _forward_sam_heads()
                    "per_obj": [],
                }
        except Exception:
            pass
        # ---------------------------------------------------------------------------

        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            best_iou_score,
            kf_ious
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["best_iou_score"] = best_iou_score
        current_out["kf_ious"] = kf_ious

        if not self.training:
            current_out["object_score_logits"] = object_score_logits

        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
    

    def _reid_bbox_from_mask_bool(self, mask_bool_np):
        # mask_bool_np: HxW bool
        ys, xs = np.where(mask_bool_np)
        if xs.size == 0 or ys.size == 0:
            return None
        x1 = int(xs.min()); x2 = int(xs.max())
        y1 = int(ys.min()); y2 = int(ys.max())
        return (x1, y1, x2, y2)


    def _reid_crop_bgr_from_bbox(self, frame_bgr, bb_xyxy, pad=0.10):
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bb_xyxy
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        px = int(round(bw * pad))
        py = int(round(bh * pad))
        x1p = max(0, x1 - px); y1p = max(0, y1 - py)
        x2p = min(W - 1, x2 + px); y2p = min(H - 1, y2 + py)
        return frame_bgr[y1p:y2p + 1, x1p:x2p + 1].copy()


    def _reid_cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if a is None or b is None:
            return float("nan")
        a = a.detach().float().reshape(-1)
        b = b.detach().float().reshape(-1)
        if a.numel() == 0 or b.numel() == 0:
            return float("nan")
        a = torch.nn.functional.normalize(a, p=2, dim=0)
        b = torch.nn.functional.normalize(b, p=2, dim=0)
        return float(torch.dot(a, b).item())