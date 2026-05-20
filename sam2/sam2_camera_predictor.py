# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from collections import deque

import cv2
import numpy as np
import math

import torch
import torch.nn.functional as F

from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores
#from sam2.reid_embedder import OSNetReIDEmbedder
from sam2.reid_backends.factory import build_reid_backend

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

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
        #reid_backend_name="osnet_x1_0",
        reid_backend_name="transreid",
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
        
        self.reid_backend_name = reid_backend_name
        reid_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reid = build_reid_backend(
            name=self.reid_backend_name,
            device=reid_device,
        )

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
        # Keep original RGB for internal ReID reference creation
        orig_rgb = img.copy() if isinstance(img, np.ndarray) else np.array(img.convert("RGB"))

        self.condition_state = self._init_state(
            offload_video_to_cpu=False, offload_state_to_cpu=False
        )

        img, width, height = self.perpare_data(img, image_size=self.image_size)
        self.condition_state["images"] = [img]
        self.condition_state["images_orig_rgb"] = [orig_rgb]
        self.condition_state["num_frames"] = len(self.condition_state["images"])
        self.condition_state["video_height"] = height
        self.condition_state["video_width"] = width
        self._get_image_feature(frame_idx=0, batch_size=1)

    @torch.inference_mode()
    def add_conditioning_frame(self, img):
        # Keep original RGB for internal ReID reference creation / debug
        if isinstance(img, np.ndarray):
            orig_rgb = img.copy()
        else:
            orig_rgb = np.array(img.convert("RGB"))

        # Keep last_rgb in sync with the frame being injected
        self.condition_state["last_rgb"] = orig_rgb.copy()

        img, width, height = self.perpare_data(img, image_size=self.image_size)

        # Append the new conditioning frame
        self.condition_state["images"].append(img)
        self.condition_state.setdefault("images_orig_rgb", []).append(orig_rgb)

        # CRITICAL: num_frames must match the actual stored images
        self.condition_state["num_frames"] = len(self.condition_state["images"])

        # Use the true last index
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

        # ---------------- INTERNAL ReID STATE ----------------
        self.condition_state["reid"] = self.reid
        self.condition_state["reid_ref"] = {}
        self.condition_state["reid_thr"] = float(self.reid_thr)
        self.condition_state["reid_last"] = {}
        self.condition_state["reid_gallery_meta"] = {}
        self.condition_state["reid_gallery_last_add_frame"] = {}
        self.condition_state["reacquire_mode_per_id"] = {}
        self.condition_state["good_memory_frames"] = []
        self.condition_state["good_memory_frames_per_id"] = {}
        # ----------------------------------------------------

        return self.condition_state
    
    def _get_yolo_reacq_model(self):
        """
        Lazy-load YOLO for internal reacquisition.

        Expected runtime attributes:
            self.yolo_reacq_enabled: bool
            self.yolo_reacq_model_path: str
            self.yolo_reacq_model: optional YOLO instance
        """
        if not bool(getattr(self, "yolo_reacq_enabled", False)):
            return None

        model = getattr(self, "yolo_reacq_model", None)
        if model is not None:
            return model

        if YOLO is None:
            print("[yolo_reacq] ultralytics is not available; YOLO reacquisition disabled.")
            return None

        model_path = getattr(self, "yolo_reacq_model_path", "yolov8n.pt")
        try:
            model = YOLO(model_path)
            self.yolo_reacq_model = model
            print(f"[yolo_reacq] loaded YOLO model: {model_path}")
            return model
        except Exception as e:
            print(f"[yolo_reacq] failed to load YOLO model {model_path}: {repr(e)}")
            return None


    def _run_yolo_person_boxes_for_reacq(self, rgb_frame):
        """
        Runs YOLO on the full RGB frame and returns person boxes:
            [(x1, y1, x2, y2, conf), ...]
        """
        model = self._get_yolo_reacq_model()
        if model is None or rgb_frame is None:
            return []

        try:
            H, W = rgb_frame.shape[:2]

            conf = float(getattr(self, "yolo_reacq_conf", 0.25))
            imgsz = int(getattr(self, "yolo_reacq_imgsz", 320))
            top_k = int(getattr(self, "yolo_reacq_top_k", 5))
            device = getattr(self, "yolo_reacq_device", None)

            predict_kwargs = dict(
                source=rgb_frame,
                conf=conf,
                imgsz=imgsz,
                classes=[0],      # COCO person
                verbose=False,
            )

            if device is not None:
                predict_kwargs["device"] = device

            results = model.predict(**predict_kwargs)
            if not results:
                return []

            boxes = getattr(results[0], "boxes", None)
            if boxes is None or boxes.xyxy is None:
                return []

            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = (
                boxes.conf.detach().cpu().numpy()
                if boxes.conf is not None
                else np.ones((xyxy.shape[0],), dtype=np.float32)
            )

            dets = []
            min_area_frac = float(getattr(self, "yolo_reacq_min_box_area_frac", 0.001))
            frame_area = float(W * H + 1e-9)

            for bb, cf in zip(xyxy, confs):
                x1, y1, x2, y2 = [int(round(v)) for v in bb[:4]]

                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                x2 = max(0, min(W, x2))
                y2 = max(0, min(H, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                area_frac = ((x2 - x1) * (y2 - y1)) / frame_area
                if area_frac < min_area_frac:
                    continue

                dets.append((x1, y1, x2, y2, float(cf)))

            dets.sort(key=lambda d: d[4], reverse=True)

            if top_k > 0:
                dets = dets[:top_k]

            return dets

        except Exception as e:
            print(f"[yolo_reacq] YOLO detection failed: {repr(e)}")
            return []


    def _select_yolo_reid_candidate_for_obj(self, oid, dets, rgb_frame):
        """
        Baseline-like YOLO+TransReID candidate selection.

        Given YOLO detections and one object id, choose the detection whose
        TransReID embedding best matches this object's gallery.

        This version reproduces the TransReID baseline more closely:
        - every YOLO detection is considered;
        - the best detection is selected by cosine similarity;
        - the candidate is accepted if sim >= threshold;
        - margin checking is optional and disabled when margin_thr <= 0.
        """
        cs = self.condition_state
        reid_model = cs.get("reid", None)

        if reid_model is None or rgb_frame is None:
            return None

        gallery = self._reid_gallery_get(int(oid))
        if len(gallery) == 0:
            return None

        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        scored = []

        for x1, y1, x2, y2, det_conf in dets:
            crop = frame_bgr[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            try:
                emb = reid_model.embed_crop_bgr(crop)
            except Exception:
                continue

            if emb is None or (torch.is_tensor(emb) and emb.numel() == 0):
                continue

            try:
                sim, best_ref_idx, all_sims = self._reid_gallery_best_sim(int(oid), emb)
            except Exception:
                continue

            if sim is None or not np.isfinite(sim):
                continue

            scored.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "sim": float(sim),
                "det_conf": float(det_conf),
                "emb": emb,
                "best_ref_idx": best_ref_idx,
                "all_sims": all_sims,
            })

        if not scored:
            return None

        scored.sort(key=lambda x: x["sim"], reverse=True)
        best = scored[0]

        second_sim = float(scored[1]["sim"]) if len(scored) > 1 else float("-inf")
        margin = float(best["sim"] - second_sim) if np.isfinite(second_sim) else float("inf")

        reid_thr = float(
            getattr(
                self,
                "yolo_reacq_thr",
                self.condition_state.get("reid_thr", 0.80),
            )
        )

        margin_thr = float(getattr(self, "yolo_reacq_margin", 0.0))

        # Baseline rule: accept if similarity passes threshold.
        if best["sim"] < reid_thr:
            return None

        # Optional extra safety. Disabled when margin_thr <= 0.
        if margin_thr > 0 and len(scored) > 1 and margin < margin_thr:
            return None

        best["second_sim"] = second_sim
        best["margin"] = margin
        return best


    def _try_internal_yolo_reacquisition(self, current_out=None):
        """
        Internal YOLO + TransReID reacquisition.

        This version uses the TransReID-baseline-style global assignment, but it
        no longer blindly trusts the SAMURAI mask generated from the selected YOLO
        bbox.

        Behavior:
        1. Run only when at least one object is in reacquisition mode.
        2. Run YOLO once on the full frame.
        3. Embed all YOLO person detections once with TransReID.
        4. Compare every reacquiring identity against every detection.
        5. Solve a global one-to-one assignment between lost IDs and detections.
        6. For each accepted ID-detection match:
                - store the YOLO+TransReID bbox as fallback localization;
                - ask SAMURAI for a temporary mask proposal;
                - accept/commit the mask only if it passes validation;
                - otherwise keep the object in reacquisition and keep only bbox fallback.
        7. Does NOT permanently add a conditioning frame.
        8. Does NOT call add_new_prompt_during_track().

        Returns:
            (obj_ids, video_res_masks) if at least one SAMURAI mask was accepted
            and committed into current_out.

            None if no trusted SAMURAI mask was recovered. In that case, fallback
            bboxes are still stored in:
                condition_state["reacq_bbox_fallback_by_id"]
            for the KTP evaluation script to use.
        """
        if not bool(getattr(self, "yolo_reacq_enabled", False)):
            return None

        if current_out is None:
            return None

        cs = self.condition_state
        reacquire_map = cs.setdefault("reacquire_mode_per_id", {})
        obj_ids = list(cs.get("obj_ids", []))
        rgb_frame = cs.get("last_rgb", None)

        if rgb_frame is None:
            return None

        frame_idx = int(self.frame_idx)
        verbose = bool(getattr(self, "yolo_reacq_verbose", False))

        # Store bbox fallback information here. The KTP evaluation script can later
        # use entries whose frame_idx matches the current frame.
        fallback_by_id = cs.setdefault("reacq_bbox_fallback_by_id", {})

        # Remove stale fallback boxes from previous frames. This avoids accidentally
        # using an old bbox as if it belonged to the current frame.
        for old_oid in list(fallback_by_id.keys()):
            try:
                old_frame = int(fallback_by_id[old_oid].get("frame_idx", -10**9))
                if old_frame != frame_idx:
                    fallback_by_id.pop(old_oid, None)
            except Exception:
                fallback_by_id.pop(old_oid, None)

        # IDs currently marked as lost/reacquiring.
        reacq_ids = [
            int(oid)
            for oid in obj_ids
            if bool(reacquire_map.get(int(oid), False))
        ]

        if not reacq_ids:
            return None

        # Apply cooldown per ID, but do not otherwise change the reacquisition state.
        cooldown = int(getattr(self, "yolo_reacq_cooldown", 10))
        last_try = cs.setdefault("yolo_reacq_last_try_by_id", {})

        eligible_reacq_ids = []
        for oid in reacq_ids:
            oid = int(oid)

            # Cannot reacquire an identity with no gallery.
            if len(self._reid_gallery_get(oid)) == 0:
                if verbose:
                    print(
                        f"[yolo_reacq] frame={frame_idx} "
                        f"oid={oid} skipped: empty gallery"
                    )
                continue

            last = int(last_try.get(oid, -10**9))
            if frame_idx - last >= cooldown:
                eligible_reacq_ids.append(oid)
                last_try[oid] = frame_idx

        if not eligible_reacq_ids:
            return None

        # ------------------------------------------------------------
        # 1. Run YOLO once on the full frame.
        # ------------------------------------------------------------
        dets = self._run_yolo_person_boxes_for_reacq(rgb_frame)
        if not dets:
            if verbose:
                print(f"[yolo_reacq] frame={frame_idx} no YOLO detections")
            return None

        reid_model = cs.get("reid", None)
        if reid_model is None:
            return None

        # ------------------------------------------------------------
        # 2. Embed all YOLO detections once.
        # ------------------------------------------------------------
        try:
            frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[yolo_reacq] RGB->BGR conversion failed: {repr(e)}")
            return None

        det_candidates = []

        for det_idx, det in enumerate(dets):
            try:
                x1, y1, x2, y2, det_conf = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                det_conf = float(det_conf)

                crop = frame_bgr[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue

                emb = reid_model.embed_crop_bgr(crop)

                if emb is None or (torch.is_tensor(emb) and emb.numel() == 0):
                    continue

                det_candidates.append({
                    "det_idx": int(det_idx),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "det_conf": float(det_conf),
                    "emb": emb,
                })

            except Exception as e:
                if verbose:
                    print(
                        f"[yolo_reacq] frame={frame_idx} "
                        f"det_idx={det_idx} embed failed: {repr(e)}"
                    )
                continue

        if not det_candidates:
            if verbose:
                print(f"[yolo_reacq] frame={frame_idx} no valid ReID detections")
            return None

        # ------------------------------------------------------------
        # 3. Build similarity matrix:
        #       rows    = reacquiring identities
        #       columns = YOLO detections
        # ------------------------------------------------------------
        n_ids = len(eligible_reacq_ids)
        n_dets = len(det_candidates)

        sim_matrix = np.full((n_ids, n_dets), -np.inf, dtype=np.float32)
        best_ref_idx_matrix = [[None for _ in range(n_dets)] for _ in range(n_ids)]
        all_sims_matrix = [[None for _ in range(n_dets)] for _ in range(n_ids)]

        for r, oid in enumerate(eligible_reacq_ids):
            for c, cand in enumerate(det_candidates):
                try:
                    sim, best_ref_idx, all_sims = self._reid_gallery_best_sim(
                        int(oid),
                        cand["emb"],
                    )

                    if sim is None or not np.isfinite(sim):
                        continue

                    sim_matrix[r, c] = float(sim)
                    best_ref_idx_matrix[r][c] = best_ref_idx
                    all_sims_matrix[r][c] = all_sims

                except Exception:
                    continue

        if not np.isfinite(sim_matrix).any():
            if verbose:
                print(
                    f"[yolo_reacq] frame={frame_idx} "
                    f"no finite similarities for global assignment"
                )
            return None

        # ------------------------------------------------------------
        # 4. Global one-to-one assignment.
        # ------------------------------------------------------------
        assignments = []

        try:
            from scipy.optimize import linear_sum_assignment

            cost = np.where(np.isfinite(sim_matrix), -sim_matrix, 1e6)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if r < n_ids and c < n_dets and np.isfinite(sim_matrix[r, c]):
                    assignments.append((int(r), int(c), float(sim_matrix[r, c])))

        except Exception as e:
            if verbose:
                print(
                    f"[yolo_reacq] Hungarian assignment unavailable, "
                    f"using greedy fallback: {repr(e)}"
                )

            pairs = []
            for r in range(n_ids):
                for c in range(n_dets):
                    if np.isfinite(sim_matrix[r, c]):
                        pairs.append((r, c, float(sim_matrix[r, c])))

            pairs.sort(key=lambda x: x[2], reverse=True)

            used_rows = set()
            used_cols = set()

            for r, c, sim_val in pairs:
                if r in used_rows or c in used_cols:
                    continue

                assignments.append((int(r), int(c), float(sim_val)))
                used_rows.add(r)
                used_cols.add(c)

        if not assignments:
            if verbose:
                print(f"[yolo_reacq] frame={frame_idx} no global assignments")
            return None

        reid_thr = float(
            getattr(
                self,
                "yolo_reacq_thr",
                cs.get("reid_thr", 0.80),
            )
        )

        # Margin is optional. To reproduce the baseline exactly, use margin 0.
        margin_thr = float(getattr(self, "yolo_reacq_margin", 0.0))

        accepted_matches = []

        for r, c, sim_val in assignments:
            oid = int(eligible_reacq_ids[r])
            cand = det_candidates[c]

            if sim_val < reid_thr:
                if verbose:
                    print(
                        f"[yolo_reacq] REJECT frame={frame_idx} oid={oid} "
                        f"bbox={cand['bbox']} sim={sim_val:.3f} < thr={reid_thr:.3f}"
                    )
                continue

            row = sim_matrix[r, :]
            finite_row = row[np.isfinite(row)]

            if margin_thr > 0 and finite_row.size > 1:
                sorted_sims = np.sort(finite_row)[::-1]
                best_row_sim = float(sorted_sims[0])
                second_row_sim = float(sorted_sims[1])

                if abs(sim_val - best_row_sim) < 1e-6:
                    margin = float(best_row_sim - second_row_sim)
                else:
                    margin = float(sim_val - best_row_sim)
            else:
                margin = float("inf")

            if margin_thr > 0 and margin < margin_thr:
                if verbose:
                    print(
                        f"[yolo_reacq] REJECT frame={frame_idx} oid={oid} "
                        f"bbox={cand['bbox']} sim={sim_val:.3f} "
                        f"margin={margin:.3f} < margin_thr={margin_thr:.3f}"
                    )
                continue

            accepted_matches.append({
                "oid": int(oid),
                "det_idx": int(cand["det_idx"]),
                "bbox": list(cand["bbox"]),
                "det_conf": float(cand["det_conf"]),
                "emb": cand["emb"],
                "sim": float(sim_val),
                "margin": float(margin),
                "best_ref_idx": best_ref_idx_matrix[r][c],
                "all_sims": all_sims_matrix[r][c],
            })

        if not accepted_matches:
            if verbose:
                print(
                    f"[yolo_reacq] frame={frame_idx} "
                    f"no accepted matches after thresholding"
                )
            return None

        accepted_matches.sort(key=lambda m: m["sim"], reverse=True)

        if verbose:
            print(
                f"[yolo_reacq] GLOBAL frame={frame_idx} "
                f"lost_ids={eligible_reacq_ids} "
                f"dets={len(det_candidates)} "
                f"accepted={[(m['oid'], m['det_idx'], round(m['sim'], 3)) for m in accepted_matches]}",
                flush=True,
            )

        # ------------------------------------------------------------
        # 5. Store YOLO+TransReID fallback bboxes and optionally let
        #    SAMURAI take over only if its mask proposal is valid.
        # ------------------------------------------------------------
        corrected_video_res_masks = None
        accepted_mask_any = False

        for match in accepted_matches:
            oid = int(match["oid"])
            bbox_xyxy = [int(v) for v in match["bbox"]]

            # First: store the fallback bbox. Even if SAMURAI fails to produce a
            # trusted mask, the KTP evaluation can still use this bbox for this frame.
            fallback_by_id[int(oid)] = {
                "frame_idx": int(frame_idx),
                "bbox": list(bbox_xyxy),
                "sim": float(match["sim"]),
                "det_conf": float(match["det_conf"]),
                "det_idx": int(match["det_idx"]),
                "source": "yolo_transreid_reacq",
                "mask_accepted": False,
                "mask_info": None,
            }

            if verbose:
                print(
                    f"[yolo_reacq] FALLBACK BBOX frame={frame_idx} oid={oid} "
                    f"det_idx={match['det_idx']} bbox={bbox_xyxy} "
                    f"sim={match['sim']:.3f} det_conf={match['det_conf']:.3f}",
                    flush=True,
                )

            # Optionally try to get a SAMURAI mask from this bbox.
            # If disabled, we keep bbox fallback only.
            use_temp_prompt = bool(getattr(self, "yolo_reacq_try_sam_mask", True))

            if not use_temp_prompt:
                cs.setdefault("reid_last", {}).setdefault(int(oid), {})
                cs["reid_last"][int(oid)].update({
                    "sim": float(match["sim"]),
                    "bbox": bbox_xyxy,
                    "accepted": True,
                    "frame_idx": int(frame_idx),
                    "reason": "yolo_reid_bbox_fallback_only",
                    "reacquire": True,
                    "yolo_det_conf": float(match["det_conf"]),
                    "yolo_margin": float(match.get("margin", float("nan"))),
                    "yolo_det_idx": int(match["det_idx"]),
                    "best_ref_idx": match.get("best_ref_idx", None),
                    "gallery_size": len(self._reid_gallery_get(int(oid))),
                    "mask_accepted": False,
                })
                continue

            if verbose:
                print(
                    f"[yolo_reacq] TEMP TRY frame={frame_idx} oid={oid} "
                    f"det_idx={match['det_idx']} bbox={bbox_xyxy} "
                    f"sim={match['sim']:.3f} "
                    f"margin={match.get('margin', float('nan')):.3f} "
                    f"det_conf={match['det_conf']:.3f}",
                    flush=True,
                )

            try:
                mask_accepted, new_video_res_masks, mask_info = self.add_temporary_reacquisition_prompt(
                    rgb_frame=rgb_frame,
                    bbox=bbox_xyxy,
                    obj_id=int(oid),
                    current_out=current_out,
                    clear_old_points=True,
                )

                fallback_by_id[int(oid)]["mask_accepted"] = bool(mask_accepted)
                fallback_by_id[int(oid)]["mask_info"] = mask_info

                cs.setdefault("reid_last", {}).setdefault(int(oid), {})

                if not mask_accepted:
                    # Keep the object in reacquisition mode. The bbox fallback is valid
                    # for this frame, but SAMURAI should not update memory from the
                    # rejected mask.
                    reacquire_map[int(oid)] = True

                    cs["reid_last"][int(oid)].update({
                        "sim": float(match["sim"]),
                        "bbox": bbox_xyxy,
                        "accepted": True,
                        "frame_idx": int(frame_idx),
                        "reason": "yolo_reid_bbox_fallback_mask_rejected",
                        "reacquire": True,
                        "yolo_det_conf": float(match["det_conf"]),
                        "yolo_margin": float(match.get("margin", float("nan"))),
                        "yolo_det_idx": int(match["det_idx"]),
                        "best_ref_idx": match.get("best_ref_idx", None),
                        "gallery_size": len(self._reid_gallery_get(int(oid))),
                        "mask_accepted": False,
                        "mask_info": mask_info,
                    })

                    if verbose:
                        print(
                            f"[yolo_reacq] MASK REJECTED -> KEEP BBOX frame={frame_idx} "
                            f"oid={oid} bbox={bbox_xyxy} sim={match['sim']:.3f} "
                            f"reason={mask_info.get('reason', None) if isinstance(mask_info, dict) else None}",
                            flush=True,
                        )
                    continue

                # SAMURAI mask was validated and committed into current_out.
                accepted_mask_any = True
                corrected_video_res_masks = new_video_res_masks

                reacquire_map[int(oid)] = False
                cs.setdefault("yolo_reacq_last_success_by_id", {})[int(oid)] = int(frame_idx)

                # If the mask was accepted, the fallback is no longer needed for this ID
                # unless the evaluation script chooses to inspect it for debugging.
                fallback_by_id[int(oid)]["source"] = "yolo_transreid_reacq_mask_accepted"

                obj_id_to_idx = cs.get("obj_id_to_idx", {})
                obj_idx = obj_id_to_idx.get(int(oid), None)

                if obj_idx is not None:
                    obj_idx = int(obj_idx)

                    reid_ok = current_out.get("reid_ok", None)
                    if torch.is_tensor(reid_ok) and obj_idx < reid_ok.numel():
                        reid_ok.reshape(-1)[obj_idx] = 1

                    object_score_logits = current_out.get("object_score_logits", None)
                    if torch.is_tensor(object_score_logits):
                        if object_score_logits.ndim >= 2 and obj_idx < object_score_logits.shape[0]:
                            object_score_logits[obj_idx, 0] = 10.0
                        elif object_score_logits.ndim == 1 and obj_idx < object_score_logits.shape[0]:
                            object_score_logits[obj_idx] = 10.0

                cs["reid_last"][int(oid)].update({
                    "sim": float(match["sim"]),
                    "bbox": bbox_xyxy,
                    "accepted": True,
                    "frame_idx": int(frame_idx),
                    "reason": "yolo_reid_global_temp_reacquired_mask_accepted",
                    "reacquire": False,
                    "yolo_det_conf": float(match["det_conf"]),
                    "yolo_margin": float(match.get("margin", float("nan"))),
                    "yolo_det_idx": int(match["det_idx"]),
                    "best_ref_idx": match.get("best_ref_idx", None),
                    "gallery_size": len(self._reid_gallery_get(int(oid))),
                    "mask_accepted": True,
                    "mask_info": mask_info,
                })

                if verbose:
                    print(
                        f"[yolo_reacq] MASK ACCEPT frame={frame_idx} oid={oid} "
                        f"det_idx={match['det_idx']} bbox={bbox_xyxy} "
                        f"sim={match['sim']:.3f} "
                        f"det_conf={match['det_conf']:.3f}",
                        flush=True,
                    )

            except Exception as e:
                # If SAM prompt fails, keep the bbox fallback and keep the object in
                # reacquisition. Do not corrupt SAMURAI memory.
                reacquire_map[int(oid)] = True

                fallback_by_id[int(oid)]["mask_accepted"] = False
                fallback_by_id[int(oid)]["mask_info"] = {
                    "accepted": False,
                    "reason": f"temporary_prompt_exception:{repr(e)}",
                }

                cs.setdefault("reid_last", {}).setdefault(int(oid), {})
                cs["reid_last"][int(oid)].update({
                    "sim": float(match["sim"]),
                    "bbox": bbox_xyxy,
                    "accepted": True,
                    "frame_idx": int(frame_idx),
                    "reason": "yolo_reid_bbox_fallback_temp_prompt_exception",
                    "reacquire": True,
                    "yolo_det_conf": float(match["det_conf"]),
                    "yolo_margin": float(match.get("margin", float("nan"))),
                    "yolo_det_idx": int(match["det_idx"]),
                    "best_ref_idx": match.get("best_ref_idx", None),
                    "gallery_size": len(self._reid_gallery_get(int(oid))),
                    "mask_accepted": False,
                    "temp_prompt_error": repr(e),
                })

                print(
                    f"[yolo_reacq] temporary prompt failed for "
                    f"frame={frame_idx} oid={oid}: {repr(e)}"
                )
                continue

        # If at least one SAMURAI mask was accepted, return the corrected masks.
        # track() will then update memory using current_out.
        if accepted_mask_any and corrected_video_res_masks is not None:
            return list(cs.get("obj_ids", [])), corrected_video_res_masks

        # If no SAMURAI mask was accepted, return None. This prevents track() from
        # treating a rejected SAM mask as a valid mask recovery. The fallback bbox is
        # still stored in condition_state["reacq_bbox_fallback_by_id"].
        return None

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

    def _extract_mask_bool_from_video_masks(self, video_res_masks, obj_ids, obj_id):
        """
        video_res_masks can be [N,1,H,W] or [N,H,W].
        obj_ids is the returned list/tensor of object ids aligned with masks.
        Returns HxW bool numpy mask for obj_id, or None.
        """
        if video_res_masks is None or obj_ids is None:
            return None

        if torch.is_tensor(obj_ids):
            ids = [int(x) for x in obj_ids.detach().reshape(-1).tolist()]
        elif isinstance(obj_ids, (list, tuple)):
            ids = [int(x) for x in obj_ids]
        else:
            ids = [int(obj_ids)]

        if int(obj_id) not in ids:
            return None

        k = ids.index(int(obj_id))

        if not torch.is_tensor(video_res_masks):
            return None

        if video_res_masks.ndim == 4:
            if k >= video_res_masks.shape[0]:
                return None
            return (video_res_masks[k, 0] > 0).detach().cpu().numpy().astype(bool)

        if video_res_masks.ndim == 3:
            if k >= video_res_masks.shape[0]:
                return None
            return (video_res_masks[k] > 0).detach().cpu().numpy().astype(bool)

        return None


    def _reid_gallery_get(self, obj_id: int):
        cs = self.condition_state
        ref_map = cs.setdefault("reid_ref", {})
        gallery = ref_map.get(int(obj_id), None)

        if gallery is None:
            gallery = []
            ref_map[int(obj_id)] = gallery
        elif torch.is_tensor(gallery):
            # backward compatibility with old single-embedding format
            gallery = [gallery.detach().cpu()]
            ref_map[int(obj_id)] = gallery
        elif not isinstance(gallery, list):
            gallery = [gallery]
            ref_map[int(obj_id)] = gallery

        return gallery


    def _reid_gallery_best_sim(self, obj_id: int, emb: torch.Tensor):
        """
        Returns:
            best_sim: float or None
            best_idx: int or None
            sims: list[float]
        """
        gallery = self._reid_gallery_get(obj_id)
        if emb is None or len(gallery) == 0:
            return None, None, []

        sims = []
        reid_model = self.condition_state.get("reid", None)

        for ref_emb in gallery:
            try:
                if reid_model is not None and hasattr(reid_model, "cosine"):
                    sim = reid_model.cosine(ref_emb, emb)
                    sims.append(float(sim))
                else:
                    a = ref_emb.detach().float().reshape(-1)
                    b = emb.detach().float().reshape(-1)
                    a = F.normalize(a, p=2, dim=0)
                    b = F.normalize(b, p=2, dim=0)
                    sims.append(float(torch.dot(a, b).item()))
            except Exception:
                sims.append(float("nan"))

        finite = [(i, s) for i, s in enumerate(sims) if np.isfinite(s)]
        if len(finite) == 0:
            return None, None, sims

        best_idx, best_sim = max(finite, key=lambda x: x[1])
        return float(best_sim), int(best_idx), sims
    

    def _reid_gallery_promote_best_match_to_anchor(self, obj_id: int, best_ref_idx: int):
        """
        Mark an existing gallery entry as anchor and record that it was used
        successfully for reacquisition.
        """
        try:
            cs = self.condition_state
            meta_map = cs.setdefault("reid_gallery_meta", {})
            meta = meta_map.get(int(obj_id), None)

            if not isinstance(meta, list):
                return False
            if best_ref_idx is None:
                return False
            if not (0 <= int(best_ref_idx) < len(meta)):
                return False
            if not isinstance(meta[int(best_ref_idx)], dict):
                return False

            entry = meta[int(best_ref_idx)]

            entry["is_anchor"] = True
            entry["promoted_by_reacquire"] = True
            entry["promoted_at_frame"] = int(self.frame_idx)

            entry["used_for_reacquire"] = True
            entry["reacquire_use_count"] = int(entry.get("reacquire_use_count", 0)) + 1
            entry.setdefault("reacquire_used_frames", [])
            entry["reacquire_used_frames"].append(int(self.frame_idx))

            meta[int(best_ref_idx)] = entry
            meta_map[int(obj_id)] = meta
            return True
        except Exception:
            return False


    def _reid_gallery_add(
        self,
        obj_id: int,
        emb: torch.Tensor,
        frame_idx: int,
        bbox=None,
        source: str = "track",
        force: bool = False,
        is_anchor: bool = False,
        quality_score: float = None,
    ):
        """
        Add one embedding to the gallery using a curated + exploratory replacement policy.

        Policy:
        - prompt entries are anchors
        - anchors are protected by default
        - if gallery has room, add directly
        - if gallery is full:
            (a) replace worst non-anchor if new candidate is better
            (b) otherwise, sometimes replace a random weak non-anchor to increase diversity
        - if all entries are anchors, reject the new one

        IMPORTANT:
        - reacquired current frames are NOT auto-anchors
        - if you want to protect the matched reference used for reacquisition,
          do that separately by promoting that existing entry's metadata
        """
        if emb is None:
            return False

        import random

        cs = self.condition_state
        gallery = self._reid_gallery_get(obj_id)

        max_size = int(getattr(self, "reid_gallery_max_size", 6))

        meta_map = cs.setdefault("reid_gallery_meta", {})
        meta = meta_map.setdefault(int(obj_id), [])

        emb_cpu = emb.detach().cpu() if torch.is_tensor(emb) else emb

        source = str(source)

        # Only prompt frames are auto-anchors
        if source == "prompt":
            is_anchor = True

        if quality_score is None:
            quality_score = 0.0

        new_meta = {
            "frame_idx": int(frame_idx),
            "bbox": bbox,
            "source": source,
            "is_anchor": bool(is_anchor),
            "quality_score": float(quality_score),
        }

        # helpful debug similarity
        try:
            best_sim, _, _ = self._reid_gallery_best_sim(obj_id, emb)
        except Exception:
            best_sim = None

        # -------------------------------------------------
        # Case 1: gallery still has room
        # -------------------------------------------------
        if len(gallery) < max_size:
            gallery.append(emb_cpu)
            meta.append(new_meta)

            cs["reid_ref"][int(obj_id)] = gallery
            meta_map[int(obj_id)] = meta
            #TODO
            # print(
            #     f"[gallery] oid={obj_id} action=append "
            #     f"frame={int(frame_idx)} source={source} "
            #     f"new_q={float(quality_score):.3f} "
            #     f"best_sim={best_sim if best_sim is not None else 'nan'} "
            #     f"size={len(gallery)}/{max_size}",
            #     flush=True,
            # )
            return True

        # -------------------------------------------------
        # Case 2: gallery full
        # -------------------------------------------------
        anchor_protect = bool(getattr(self, "reid_gallery_anchor_protect", True))
        random_replace_prob = float(getattr(self, "reid_gallery_random_replace_prob", 0.15))
        random_replace_if_diverse_prob = float(
            getattr(self, "reid_gallery_random_replace_if_diverse_prob", 0.30)
        )
        add_sim_thr = float(getattr(self, "reid_gallery_add_sim_threshold", 0.85))

        # collect replaceable entries
        replace_candidates = []
        for i, m in enumerate(meta):
            if not isinstance(m, dict):
                continue

            old_anchor = bool(m.get("is_anchor", False))
            if anchor_protect and old_anchor:
                continue

            replace_candidates.append((i, m))

        # no replaceable entries -> reject
        if len(replace_candidates) == 0:
            #TODO
            # print(
            #     f"[gallery] oid={obj_id} action=reject_full_all_anchors "
            #     f"frame={int(frame_idx)} source={source} "
            #     f"new_q={float(quality_score):.3f} "
            #     f"best_sim={best_sim if best_sim is not None else 'nan'}",
            #     flush=True,
            # )
            return False

        # weaker = lower quality, then older
        def _rank_key(item):
            i, m = item
            q = float(m.get("quality_score", 0.0))
            f = int(m.get("frame_idx", -1))
            return (q, f)

        # sorted weakest -> strongest
        replace_candidates_sorted = sorted(replace_candidates, key=_rank_key)

        worst_idx, worst_meta = replace_candidates_sorted[0]
        worst_quality = float(worst_meta.get("quality_score", 0.0))
        worst_anchor = bool(worst_meta.get("is_anchor", False))

        # -------------------------------------------------
        # Path A: normal curated replacement
        # -------------------------------------------------
        if float(quality_score) > worst_quality or force:
            old_q = worst_quality
            gallery[worst_idx] = emb_cpu
            meta[worst_idx] = new_meta

            cs["reid_ref"][int(obj_id)] = gallery
            meta_map[int(obj_id)] = meta
            #TODO
            # print(
            #     f"[gallery] oid={obj_id} action=replace_worst "
            #     f"idx={worst_idx} old_anchor={worst_anchor} "
            #     f"old_q={old_q:.3f} new_q={float(quality_score):.3f} "
            #     f"best_sim={best_sim if best_sim is not None else 'nan'} "
            #     f"frame={int(frame_idx)} source={source}",
            #     flush=True,
            # )
            return True

        # -------------------------------------------------
        # Path B: exploratory replacement
        # Even if not strictly better, sometimes replace one weak non-anchor
        # to avoid gallery collapse / local minima.
        # -------------------------------------------------
        if best_sim is None:
            p = random_replace_if_diverse_prob
        else:
            if float(best_sim) < add_sim_thr:
                p = random_replace_if_diverse_prob
            else:
                p = random_replace_prob

        if (not force) and (random.random() >= p):
            #TODO
            # print(
            #     f"[gallery] oid={obj_id} action=reject_not_better "
            #     f"worst_idx={worst_idx} old_q={worst_quality:.3f} "
            #     f"new_q={float(quality_score):.3f} "
            #     f"best_sim={best_sim if best_sim is not None else 'nan'} "
            #     f"p={p:.2f} frame={int(frame_idx)} source={source}",
            #     flush=True,
            # )
            return False

        # choose randomly among weaker half of replaceable entries
        weak_count = max(1, len(replace_candidates_sorted) // 2)
        weak_pool = replace_candidates_sorted[:weak_count]
        chosen_idx, chosen_meta = random.choice(weak_pool)

        old_q = float(chosen_meta.get("quality_score", 0.0))
        old_anchor = bool(chosen_meta.get("is_anchor", False))

        gallery[chosen_idx] = emb_cpu
        meta[chosen_idx] = new_meta

        cs["reid_ref"][int(obj_id)] = gallery
        meta_map[int(obj_id)] = meta
        # TODO
        # print(
        #     f"[gallery] oid={obj_id} action=replace_random_weak "
        #     f"idx={chosen_idx} old_anchor={old_anchor} "
        #     f"old_q={old_q:.3f} new_q={float(quality_score):.3f} "
        #     f"best_sim={best_sim if best_sim is not None else 'nan'} "
        #     f"p={p:.2f} weak_pool={len(weak_pool)} "
        #     f"frame={int(frame_idx)} source={source}",
        #     flush=True,
        # )
        return True
    

    def _reid_gallery_candidate_score(
        self,
        bbox_xyxy,
        mask_bool,
        frame_shape,
        sim_to_gallery=None,
        accepted_by_reid=None,
        best_iou_val=None,
        obj_logit_val=None,
    ):
        """
        Simple score for deciding whether a candidate deserves a gallery slot.

        Higher is better.
        """
        try:
            if bbox_xyxy is None or mask_bool is None or frame_shape is None:
                return -1e9

            H, W = int(frame_shape[0]), int(frame_shape[1])
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))

            bw = max(1, x2 - x1 + 1)
            bh = max(1, y2 - y1 + 1)

            bbox_area_ratio = float(bw * bh) / max(float(H * W), 1.0)

            crop_mask = mask_bool[y1:y2 + 1, x1:x2 + 1]
            if crop_mask.size == 0:
                return -1e9

            mask_area = float((crop_mask > 0).sum())
            fill_ratio = mask_area / max(float(bw * bh), 1.0)

            score = 0.0

            # bigger + better-filled crops are better references
            score += 2.0 * bbox_area_ratio
            score += 2.0 * fill_ratio

            # being somewhat different from current gallery is good
            if sim_to_gallery is not None and np.isfinite(sim_to_gallery):
                score += 1.0 - float(sim_to_gallery)

            if accepted_by_reid is True:
                score += 0.5

            if best_iou_val is not None and np.isfinite(best_iou_val):
                score += 0.5 * float(best_iou_val)

            if obj_logit_val is not None and np.isfinite(obj_logit_val):
                score += 0.02 * float(obj_logit_val)

            return float(score)

        except Exception:
            return -1e9


    def _reid_gallery_should_add(
        self,
        obj_id: int,
        emb: torch.Tensor,
        frame_idx: int,
        bbox_xyxy=None,
        mask_bool=None,
        frame_shape=None,
    ):
        """
        Decide whether a new embedding is worth trying to insert into the gallery.

        Conditions:
        - crop quality must be acceptable
        - cooldown must have passed
        - candidate must be sufficiently different from the current gallery
        - OR be admitted through a small exploration probability
        """
        if emb is None:
            return False

        # 1) quality gate
        if not self._reid_gallery_crop_is_good(
            bbox_xyxy=bbox_xyxy,
            mask_bool=mask_bool,
            frame_shape=frame_shape,
        ):
            return False

        cs = self.condition_state

        # 2) cooldown
        last_add_map = cs.setdefault("reid_gallery_last_add_frame", {})
        last_add_f = last_add_map.get(int(obj_id), None)

        cooldown = int(getattr(self, "reid_gallery_add_cooldown", 15))
        if last_add_f is not None and (int(frame_idx) - int(last_add_f) < cooldown):
            return False

        # 3) if empty, accept
        gallery = self._reid_gallery_get(obj_id)
        if len(gallery) == 0:
            return True

        # 4) similarity test
        best_sim, _, _ = self._reid_gallery_best_sim(obj_id, emb)
        if best_sim is None:
            return True

        add_sim_thr = float(getattr(self, "reid_gallery_add_sim_threshold", 0.85))
        if float(best_sim) < add_sim_thr:
            return True

        # 5) small exploration path:
        # if candidate is not different enough, still sometimes try to add it
        # to avoid gallery collapse / local minima
        import random

        base_prob = float(getattr(self, "reid_gallery_random_replace_prob", 0.15))
        diverse_prob = float(getattr(self, "reid_gallery_random_replace_if_diverse_prob", 0.30))

        # "near-threshold but still somewhat different" gets a slightly higher chance
        margin = float(add_sim_thr) - float(best_sim)
        # margin < 0 means too similar; closer to threshold is better than extremely similar
        if margin > -0.05:
            p = diverse_prob
        else:
            p = base_prob

        return bool(random.random() < p)


    def _reid_gallery_mark_added(self, obj_id: int, frame_idx: int):
        cs = self.condition_state
        last_add_map = cs.setdefault("reid_gallery_last_add_frame", {})
        last_add_map[int(obj_id)] = int(frame_idx)


    def _reid_gallery_crop_is_good(
        self,
        bbox_xyxy,
        mask_bool,
        frame_shape,
    ):
        """
        Quality filter for candidate gallery crops.

        Rejects crops that are:
        - too small
        - too thin / narrow
        - too close to image borders
        - too truncated at the sides
        - too poorly filled by the mask
        - too low in visible mask area
        """
        if bbox_xyxy is None or mask_bool is None or frame_shape is None:
            return False

        try:
            H, W = int(frame_shape[0]), int(frame_shape[1])
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))

            bw = x2 - x1 + 1
            bh = y2 - y1 + 1
            if bw <= 1 or bh <= 1:
                return False

            bbox_area = float(bw * bh)
            frame_area = float(H * W)

            # -------------------------------------------------
            # 1) Reject tiny crops
            # -------------------------------------------------
            min_bbox_area_ratio = float(getattr(self, "reid_gallery_min_bbox_area_ratio", 0.02))
            if bbox_area / max(frame_area, 1.0) < min_bbox_area_ratio:
                return False

            # -------------------------------------------------
            # 2) Reject crops that are too narrow or too short
            # -------------------------------------------------
            min_bbox_width_ratio = float(getattr(self, "reid_gallery_min_bbox_width_ratio", 0.12))
            min_bbox_height_ratio = float(getattr(self, "reid_gallery_min_bbox_height_ratio", 0.25))

            if (bw / max(float(W), 1.0)) < min_bbox_width_ratio:
                return False
            if (bh / max(float(H), 1.0)) < min_bbox_height_ratio:
                return False

            # -------------------------------------------------
            # 3) Reject overly thin aspect ratios
            #    (very narrow crops are often bad side fragments)
            # -------------------------------------------------
            min_aspect_ratio = float(getattr(self, "reid_gallery_min_aspect_ratio", 0.18))
            aspect_ratio = bw / max(float(bh), 1.0)
            if aspect_ratio < min_aspect_ratio:
                return False

            # -------------------------------------------------
            # 4) Border-touch checks
            # -------------------------------------------------
            border_margin_ratio = float(getattr(self, "reid_gallery_border_margin_ratio", 0.03))
            mx = int(round(border_margin_ratio * W))
            my = int(round(border_margin_ratio * H))

            touches_left = (x1 <= mx)
            touches_right = (x2 >= W - 1 - mx)
            touches_top = (y1 <= my)
            touches_bottom = (y2 >= H - 1 - my)

            # Stronger rule: touching left/right is especially bad for re-id gallery
            max_horizontal_border_touches = int(getattr(self, "reid_gallery_max_horizontal_border_touches", 0))
            horizontal_touches = int(touches_left) + int(touches_right)
            if horizontal_touches > max_horizontal_border_touches:
                return False

            max_border_touches = int(getattr(self, "reid_gallery_max_border_touches", 1))
            num_touches = int(touches_left) + int(touches_right) + int(touches_top) + int(touches_bottom)
            if num_touches > max_border_touches:
                return False

            # -------------------------------------------------
            # 5) Reject if bbox center is too close to image side borders
            #    (helps remove heavily truncated side entries)
            # -------------------------------------------------
            cx = 0.5 * (x1 + x2)
            min_center_x_ratio = float(getattr(self, "reid_gallery_min_center_x_ratio", 0.18))
            max_center_x_ratio = float(getattr(self, "reid_gallery_max_center_x_ratio", 0.82))
            cx_ratio = cx / max(float(W), 1.0)

            if cx_ratio < min_center_x_ratio or cx_ratio > max_center_x_ratio:
                return False

            # -------------------------------------------------
            # 6) Mask crop inside bbox
            # -------------------------------------------------
            crop_mask = mask_bool[y1:y2 + 1, x1:x2 + 1]
            if crop_mask.size == 0:
                return False

            mask_area = float((crop_mask > 0).sum())

            # Reject if visible mask itself is too small
            min_mask_area_ratio = float(getattr(self, "reid_gallery_min_mask_area_ratio", 0.01))
            if mask_area / max(frame_area, 1.0) < min_mask_area_ratio:
                return False

            # Reject masks that fill too little of bbox
            fill_ratio = mask_area / max(bbox_area, 1.0)
            min_fill_ratio = float(getattr(self, "reid_gallery_min_fill_ratio", 0.28))
            if fill_ratio < min_fill_ratio:
                return False

            return True

        except Exception:
            return False


    def _safe_sigmoid_float(self, x):
        try:
            if x is None:
                return None
            x = float(x)
            return 1.0 / (1.0 + math.exp(-x))
        except Exception:
            return None


    def _clamp01(self, x):
        try:
            if x is None or not np.isfinite(float(x)):
                return None
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return None


    def _soft_reid_score(self, sim, thr=None, temperature=None):
        """
        Turns raw ReID similarity into a soft [0,1] score centered on the ReID threshold.
        sim == thr -> around 0.5
        """
        try:
            if sim is None or not np.isfinite(float(sim)):
                return None

            if thr is None:
                thr = float(self.condition_state.get("reid_thr", 0.80))
            else:
                thr = float(thr)

            if temperature is None:
                temperature = float(getattr(self, "reacquire_reid_temperature", 0.05))
            else:
                temperature = float(temperature)

            temperature = max(1e-6, temperature)
            z = (float(sim) - thr) / temperature
            return 1.0 / (1.0 + math.exp(-z))
        except Exception:
            return None


    def _combined_reacquire_score(
        self,
        sim,
        obj_logit_val,
        kf_score_val=None,
        iou_val=None,
    ):
        """
        Weighted fusion score for reacquisition.

        Returns:
            final_score: float or None
            parts: dict with each term for debug
        """
        s_reid = self._soft_reid_score(sim)
        s_obj = self._safe_sigmoid_float(obj_logit_val)
        s_kf = self._clamp01(kf_score_val)
        s_iou = self._clamp01(iou_val)

        # neutral fallback if missing
        if s_kf is None:
            s_kf = 0.5
        if s_iou is None:
            s_iou = 0.5

        # weights
        w_reid = float(getattr(self, "reacquire_w_reid", 0.50))
        w_obj  = float(getattr(self, "reacquire_w_obj",  0.35))
        w_kf   = float(getattr(self, "reacquire_w_kf",   0.10))
        w_iou  = float(getattr(self, "reacquire_w_iou",  0.05))

        denom = w_reid + w_obj + w_kf + w_iou
        if denom <= 0:
            return None, {
                "s_reid": s_reid,
                "s_obj": s_obj,
                "s_kf": s_kf,
                "s_iou": s_iou,
            }

        # if a required term is missing, still proceed if enough is present
        if s_reid is None and s_obj is None:
            return None, {
                "s_reid": s_reid,
                "s_obj": s_obj,
                "s_kf": s_kf,
                "s_iou": s_iou,
            }

        s_reid_use = 0.0 if s_reid is None else s_reid
        s_obj_use  = 0.0 if s_obj  is None else s_obj

        final_score = (
            w_reid * s_reid_use +
            w_obj  * s_obj_use  +
            w_kf   * s_kf +
            w_iou  * s_iou
        ) / denom

        return float(final_score), {
            "s_reid": s_reid,
            "s_obj": s_obj,
            "s_kf": s_kf,
            "s_iou": s_iou,
        }


    def _reacquire_accept(
        self,
        sim,
        obj_logit_val,
        kf_score_val=None,
        iou_val=None,
    ):
        """
        Final reacquisition decision based on fused score.
        """
        score, parts = self._combined_reacquire_score(
            sim=sim,
            obj_logit_val=obj_logit_val,
            kf_score_val=kf_score_val,
            iou_val=iou_val,
        )

        thr = float(getattr(self, "reacquire_score_threshold", 0.70))

        accepted = (score is not None) and (score >= thr)
        return accepted, score, parts


    def _maybe_store_reid_reference(
        self,
        frame_idx,
        obj_id,
        obj_ids,
        video_res_masks,
        prompt_bbox=None,
    ):
        """
        Create initial ReID gallery entries for obj_id.

        Important:
        The TransReID baseline compares YOLO/person-bbox crops against a gallery
        built from person-bbox crops. Therefore, for compatibility with the
        YOLO+TransReID reacquisition branch, the first anchor should preferably be
        a bbox-style crop from the original prompt bbox.

        We still optionally store a mask-style reference too, because it can be
        useful during normal SAMURAI mask tracking, but the bbox-style prompt
        reference is added first and forced into the gallery.
        """

        if bool(self.condition_state.get("suppress_reid_gallery_update", False)):
            return False

        try:
            cs = self.condition_state
            reid = cs.get("reid", None)
            if reid is None:
                return False

            if "images_orig_rgb" not in cs:
                print("[reid] images_orig_rgb missing in condition_state", flush=True)
                return False

            if frame_idx < 0 or frame_idx >= len(cs["images_orig_rgb"]):
                print(f"[reid] invalid frame_idx for reference: {frame_idx}", flush=True)
                return False

            rgb = cs["images_orig_rgb"][frame_idx]
            if rgb is None:
                print(f"[reid] original RGB missing for frame {frame_idx}", flush=True)
                return False

            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            H, W = frame_bgr.shape[:2]

            added_any = False
            added_entries = []

            # ------------------------------------------------------------
            # 1. Add bbox-style reference from the original prompt bbox.
            #    This is the most important one for YOLO+TransReID reacq,
            #    because YOLO reacq queries are also bbox crops.
            # ------------------------------------------------------------
            prompt_bb = None

            if prompt_bbox is not None:
                try:
                    if torch.is_tensor(prompt_bbox):
                        bb_np = prompt_bbox.detach().cpu().numpy()
                    else:
                        bb_np = np.asarray(prompt_bbox)

                    bb_np = bb_np.reshape(-1, 2)

                    x1 = int(round(float(np.min(bb_np[:, 0]))))
                    y1 = int(round(float(np.min(bb_np[:, 1]))))
                    x2 = int(round(float(np.max(bb_np[:, 0]))))
                    y2 = int(round(float(np.max(bb_np[:, 1]))))

                    x1 = max(0, min(W - 1, x1))
                    y1 = max(0, min(H - 1, y1))
                    x2 = max(0, min(W, x2))
                    y2 = max(0, min(H, y2))

                    if x2 > x1 and y2 > y1:
                        prompt_bb = [x1, y1, x2, y2]

                except Exception as e:
                    print(f"[reid] failed to parse prompt bbox for obj_id={obj_id}: {e}", flush=True)
                    prompt_bb = None

            if prompt_bb is not None:
                try:
                    x1, y1, x2, y2 = prompt_bb
                    crop = frame_bgr[y1:y2, x1:x2].copy()

                    if crop.size > 0 and hasattr(reid, "embed_crop_bgr"):
                        emb_bbox = reid.embed_crop_bgr(crop)

                        if emb_bbox is not None and not (
                            torch.is_tensor(emb_bbox) and emb_bbox.numel() == 0
                        ):
                            added = self._reid_gallery_add(
                                obj_id=int(obj_id),
                                emb=emb_bbox,
                                frame_idx=int(frame_idx),
                                bbox=prompt_bb,
                                source="prompt_bbox",
                                force=True,
                            )

                            added_any = bool(added_any or added)
                            added_entries.append(("prompt_bbox", prompt_bb, bool(added)))

                except Exception as e:
                    print(f"[reid] prompt bbox embedding failed for obj_id={obj_id}: {e}", flush=True)

            # ------------------------------------------------------------
            # 2. Add mask-style reference from the prompt-time SAMURAI mask.
            #    This keeps the previous behavior, but it should not replace
            #    the bbox-style anchor as the first reference.
            # ------------------------------------------------------------
            mask_bool = self._extract_mask_bool_from_video_masks(video_res_masks, obj_ids, obj_id)

            mask_bb = None
            emb_mask = None

            if mask_bool is None:
                print(f"[reid] could not extract prompt mask for obj_id={obj_id}", flush=True)
            else:
                if hasattr(reid, "embed_from_mask"):
                    try:
                        emb_mask, mask_bb = reid.embed_from_mask(frame_bgr, mask_bool)
                    except Exception as e:
                        print(f"[reid] embed_from_mask failed for obj_id={obj_id}: {e}", flush=True)

                if emb_mask is None:
                    try:
                        ys, xs = np.where(mask_bool)
                        if xs.size > 0 and ys.size > 0:
                            x1, x2 = int(xs.min()), int(xs.max()) + 1
                            y1, y2 = int(ys.min()), int(ys.max()) + 1

                            x1 = max(0, min(W - 1, x1))
                            y1 = max(0, min(H - 1, y1))
                            x2 = max(0, min(W, x2))
                            y2 = max(0, min(H, y2))

                            if x2 > x1 and y2 > y1:
                                mask_bb = [x1, y1, x2, y2]
                                crop = frame_bgr[y1:y2, x1:x2].copy()

                                if crop.size > 0 and hasattr(reid, "embed_crop_bgr"):
                                    emb_mask = reid.embed_crop_bgr(crop)

                    except Exception as e:
                        print(f"[reid] mask bbox embedding failed for obj_id={obj_id}: {e}", flush=True)

                if emb_mask is not None and not (
                    torch.is_tensor(emb_mask) and emb_mask.numel() == 0
                ):
                    try:
                        # If bbox-style prompt was not available, force the mask reference.
                        # If bbox-style prompt was already added, still add the mask reference,
                        # but do not need to force it as the only anchor.
                        added = self._reid_gallery_add(
                            obj_id=int(obj_id),
                            emb=emb_mask,
                            frame_idx=int(frame_idx),
                            bbox=mask_bb,
                            source="prompt_mask",
                            force=True,
                        )

                        added_any = bool(added_any or added)
                        added_entries.append(("prompt_mask", mask_bb, bool(added)))

                    except Exception as e:
                        print(f"[reid] failed to add mask reference for obj_id={obj_id}: {e}", flush=True)

            if not added_any:
                print(f"[reid] failed to create any reference embedding for obj_id={obj_id}", flush=True)
                return False

            gallery = self._reid_gallery_get(int(obj_id))
            gallery_size = len(gallery)

            # Prefer reporting the bbox-style prompt if it exists.
            main_bb = prompt_bb if prompt_bb is not None else mask_bb

            cs.setdefault("reid_last", {})[int(obj_id)] = {
                "sim": 1.0,
                "bbox": main_bb,
                "accepted": True,
                "ref_set": True,
                "frame_idx": int(frame_idx),
                "gallery_size": gallery_size,
                "best_ref_idx": 0 if gallery_size > 0 else None,
                "gallery_added": bool(added_any),
                "reason": "prompt_ref_bbox_and_mask" if prompt_bb is not None and mask_bb is not None else "prompt_ref",
                "added_entries": added_entries,
            }

            self._reid_gallery_mark_added(int(obj_id), int(frame_idx))

            print(
                f"[reid] saved gallery reference for obj_id={obj_id} "
                f"frame_idx={frame_idx} main_bb={main_bb} "
                f"gallery_size={gallery_size} entries={added_entries}",
                flush=True,
            )

            return True

        except Exception as e:
            print(f"[reid] _maybe_store_reid_reference failed for obj_id={obj_id}: {e}", flush=True)
            return False
    

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
        # ---------------- per-ID reacquisition init ----------------
        self.condition_state.setdefault("reacquire_mode_per_id", {})
        self.condition_state["reacquire_mode_per_id"].setdefault(int(obj_id), False)
        # ----------------------------------------------------------
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

        # If this frame hasn't been tracked before, we treat it as an initial conditioning frame
        is_init_cond_frame = (
            frame_idx not in self.condition_state["frames_already_tracked"]
        )

        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = self.condition_state["frames_already_tracked"][frame_idx]["reverse"]

        obj_output_dict = self.condition_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.condition_state["temp_output_dict_per_obj"][obj_idx]

        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
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

        # ---------------- INTERNAL ReID reference creation ----------------
        if is_init_cond_frame:
            self._maybe_store_reid_reference(
                frame_idx=frame_idx,
                obj_id=obj_id,
                obj_ids=obj_ids,
                video_res_masks=video_res_masks,
                prompt_bbox=bbox,
            )
        # ---------------------------------------------------------------

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

        assert self.condition_state["tracking_has_started"] is True, \
            "Cannot add new points or mask during tracking without calling track()"

        # Pause tracking while we inject
        self.condition_state["tracking_has_started"] = False

        # The demo already did add_conditioning_frame(rgb_frame)
        cond_frame_idx = int(len(self.condition_state["images"]) - 1)

        # The already-tracked output of the same image in the non_cond timeline
        src_global_fidx = int(self.frame_idx)

        output_dict = self.condition_state["output_dict"]

        # --- capture "old world" mapping BEFORE registering new object ---
        old_obj_ids = list(self.condition_state["obj_ids"])
        old_obj_id_to_idx = self.condition_state.get("obj_id_to_idx", None)

        # --- fetch the latest non_cond output to preserve old objects ---
        src_out = None
        try:
            src_out = output_dict.get("non_cond_frame_outputs", {}).get(src_global_fidx, None)
            if src_out is None:
                keys = sorted(list(output_dict.get("non_cond_frame_outputs", {}).keys()))
                prev_keys = [k for k in keys if int(k) <= src_global_fidx]
                if prev_keys:
                    src_k = prev_keys[-1]
                    src_out = output_dict["non_cond_frame_outputs"][src_k]
        except Exception:
            src_out = None

        if src_out is None:
            print("[DBG INJECT] WARNING: no src_out found in non_cond_frame_outputs; old objects may be killed on consolidation.")
        else:
            try:
                pm = src_out.get("pred_masks", None)
                if torch.is_tensor(pm):
                    B = int(pm.shape[0])
                    mx = [float(pm[i].max().item()) for i in range(B)]
            except Exception as e:
                print(f"[DBG INJECT] src_out stats failed: {e}")

        # (1) Register/choose id and create the temp output on this conditioning frame
        if if_new_target:
            if obj_id is None:
                obj_id = (max(self.condition_state["obj_ids"]) + 1) if self.condition_state["obj_ids"] else 0
            _ = self._register_new_object_if_needed(obj_id)

            self.condition_state.setdefault("reacquire_mode_per_id", {})
            self.condition_state["reacquire_mode_per_id"].setdefault(int(obj_id), False)

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

            self.condition_state.setdefault("reacquire_mode_per_id", {})
            self.condition_state["reacquire_mode_per_id"].setdefault(int(obj_id), False)

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

        # (2) Consolidate temp outputs and commit them to main state
        is_init_cond_frame = (frame_idx not in self.condition_state["frames_already_tracked"])
        is_cond = is_init_cond_frame or getattr(self, "add_all_frames_to_correct_as_cond", False)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        other_key = "non_cond_frame_outputs" if storage_key == "cond_frame_outputs" else "cond_frame_outputs"

        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx=frame_idx,
            is_cond=is_cond,
            run_mem_encoder=True,
            consolidate_at_video_res=False,
        )

        # ------------------------------------------------------------------
        # Preserve old objects from src_out
        # ------------------------------------------------------------------
        try:
            new_obj_id_to_idx = self.condition_state.get("obj_id_to_idx", None)
            if src_out is not None and torch.is_tensor(consolidated_out.get("pred_masks", None)) and torch.is_tensor(src_out.get("pred_masks", None)) \
            and isinstance(old_obj_id_to_idx, dict) and isinstance(new_obj_id_to_idx, dict):

                dst_pm = consolidated_out["pred_masks"]
                src_pm = src_out["pred_masks"]
                dst_B = int(dst_pm.shape[0])
                src_B = int(src_pm.shape[0])

                slot_keys = ["pred_masks", "obj_ptr", "maskmem_features", "best_iou_score", "object_score_logits", "kf_score", "reid_ok"]

                for oid in old_obj_ids:
                    oid = int(oid)
                    if oid not in old_obj_id_to_idx or oid not in new_obj_id_to_idx:
                        continue

                    si = int(old_obj_id_to_idx[oid])
                    di = int(new_obj_id_to_idx[oid])

                    if si < 0 or si >= src_B or di < 0 or di >= dst_B:
                        continue

                    for k in slot_keys:
                        s = src_out.get(k, None)
                        d = consolidated_out.get(k, None)
                        if torch.is_tensor(s) and torch.is_tensor(d) and s.ndim >= 1 and d.ndim >= 1 and s.shape[0] > si and d.shape[0] > di:
                            ss = s[si].to(device=d.device, dtype=d.dtype, non_blocking=True)
                            d[di].copy_(ss)

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

        except Exception as e:
            print(f"[DBG MERGE] failed: {e}")

        self._dbg_state("add_new_prompt_during_track:BEFORE_EXPAND")

        # Make all previously stored frames compatible with the new batch size
        self._expand_all_stored_outputs_to_current_batch()

        self._dbg_state("add_new_prompt_during_track:AFTER_EXPAND")

        # Commit like preflight does
        temp_per_obj = self.condition_state["temp_output_dict_per_obj"]
        consolidated_inds = self.condition_state["consolidated_frame_inds"]

        consolidated_inds[storage_key].add(frame_idx)
        consolidated_inds[other_key].discard(frame_idx)

        output_dict[storage_key][frame_idx] = consolidated_out

        self._add_output_per_object(frame_idx, consolidated_out, storage_key)

        for obj_temp in temp_per_obj.values():
            obj_temp[storage_key].pop(frame_idx, None)
            obj_temp[other_key].pop(frame_idx, None)

        # (3) Resume tracking
        self.condition_state["tracking_has_started"] = True

        print("shape ", len(self.condition_state["images"]), " frame index ", frame_idx)
        return frame_idx, obj_ids, video_res_masks
    
    @torch.inference_mode()
    def add_temporary_reacquisition_prompt(
        self,
        rgb_frame,
        bbox,
        obj_id: int,
        current_out: dict,
        clear_old_points: bool = True,
    ):
        """
        Use a YOLO+TransReID bbox as a temporary SAM prompt for an existing object.

        New safer behavior:
        - the YOLO bbox is used only to ask SAMURAI for a candidate mask;
        - the candidate mask is validated against the YOLO bbox;
        - only if the mask is geometrically consistent with the bbox is it copied
            into current_out and used to update memory;
        - if the mask is bad, current_out is NOT modified and memory is NOT updated;
        - the temporary prompt/frame is always cleaned up afterwards;
        - the ReID gallery is not updated during this temporary prompt.

        Returns:
            accepted_mask: bool
                True if the SAMURAI mask proposal was accepted and committed into current_out.
                False if the mask proposal was rejected.
            video_res_masks: torch.Tensor or None
                If accepted, this is the corrected video-resolution mask tensor after
                committing the mask into current_out.
                If rejected, this may contain the proposed video-resolution masks for
                debugging, but should not be treated as a trusted output mask.
            info: dict
                Debug/validation information.
        """

        def _bbox_iou_xyxy(a, b):
            ax1, ay1, ax2, ay2 = [float(x) for x in a]
            bx1, by1, bx2, by2 = [float(x) for x in b]

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih

            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

            union = area_a + area_b - inter + 1e-9
            return float(inter / union)

        def _mask_bbox_and_area(mask_bool):
            ys, xs = np.where(mask_bool > 0)
            if xs.size == 0 or ys.size == 0:
                return None, 0

            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max())
            y2 = int(ys.max())
            area = int(mask_bool.sum())

            return [x1, y1, x2, y2], area

        cs = self.condition_state
        old_last_rgb = cs.get("last_rgb", None)

        info = {
            "accepted": False,
            "reason": None,
            "obj_id": int(obj_id),
            "frame_idx": int(getattr(self, "frame_idx", -1)),
            "yolo_bbox": None,
            "mask_bbox": None,
            "mask_bbox_iou": None,
            "mask_area": None,
            "yolo_bbox_area": None,
            "mask_area_ratio": None,
        }

        if current_out is None:
            info["reason"] = "current_out_none"
            return False, None, info

        obj_id = int(obj_id)

        obj_id_to_idx = cs.get("obj_id_to_idx", {})
        if obj_id not in obj_id_to_idx:
            info["reason"] = "obj_id_not_registered"
            print(f"[temp_reacq] obj_id={obj_id} is not registered")
            return False, None, info

        obj_idx = int(obj_id_to_idx[obj_id])

        # Save state so the temporary frame can be removed afterwards.
        old_tracking_has_started = bool(cs.get("tracking_has_started", False))
        old_num_frames = int(cs.get("num_frames", len(cs.get("images", []))))
        old_len_images = len(cs.get("images", []))
        old_len_orig = len(cs.get("images_orig_rgb", []))

        # Temporary reacquisition prompts must not update the ReID gallery.
        old_suppress_gallery = bool(cs.get("suppress_reid_gallery_update", False))
        cs["suppress_reid_gallery_update"] = True

        cached_features = cs.setdefault("cached_features", {})
        old_cached_keys = set(cached_features.keys())

        output_dict = cs.get("output_dict", {})
        consolidated_frame_inds = cs.get("consolidated_frame_inds", {})

        tmp_frame_idx = None

        try:
            # Temporarily pause tracking because add_new_prompt expects
            # interaction-style state.
            cs["tracking_has_started"] = False

            # Temporarily append the current frame only so SAM can use the bbox
            # prompt on this image. This is removed in finally.
            self.add_conditioning_frame(rgb_frame)
            tmp_frame_idx = len(cs["images"]) - 1

            # Convert bbox to SAM prompt format and keep xyxy for validation.
            bbox_np = np.asarray(bbox, dtype=np.float32)

            if bbox_np.shape == (4,):
                x1, y1, x2, y2 = bbox_np.tolist()
                bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]
                bbox_prompt = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            elif bbox_np.shape == (2, 2):
                x1, y1 = bbox_np[0].tolist()
                x2, y2 = bbox_np[1].tolist()
                bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]
                bbox_prompt = bbox_np.astype(np.float32)
            else:
                info["reason"] = f"bad_bbox_shape:{bbox_np.shape}"
                print(f"[temp_reacq] bad bbox shape: {bbox_np.shape}")
                return False, None, info

            info["yolo_bbox"] = [int(round(v)) for v in bbox_xyxy]

            yolo_w = max(0.0, bbox_xyxy[2] - bbox_xyxy[0])
            yolo_h = max(0.0, bbox_xyxy[3] - bbox_xyxy[1])
            yolo_bbox_area = float(yolo_w * yolo_h)
            info["yolo_bbox_area"] = yolo_bbox_area

            if yolo_bbox_area <= 1.0:
                info["reason"] = "bad_yolo_bbox_area"
                return False, None, info

            # Create temporary SAM output for obj_id on tmp_frame_idx.
            # Gallery updates are suppressed while this runs.
            self.add_new_prompt(
                frame_idx=tmp_frame_idx,
                obj_id=obj_id,
                points=None,
                bbox=bbox_prompt,
                labels=None,
                clear_old_points=clear_old_points,
                normalize_coords=True,
            )

            # Consolidate temporary outputs for this temporary frame, but do not
            # commit this as a normal conditioning frame.
            consolidated_out = self._consolidate_temp_output_across_obj(
                frame_idx=tmp_frame_idx,
                is_cond=True,
                run_mem_encoder=False,
                consolidate_at_video_res=False,
            )

            if not torch.is_tensor(consolidated_out.get("pred_masks", None)):
                info["reason"] = "no_pred_masks_from_prompt"
                print("[temp_reacq] no pred_masks from temporary prompt")
                return False, None, info

            if obj_idx < 0 or obj_idx >= int(consolidated_out["pred_masks"].shape[0]):
                info["reason"] = "obj_idx_out_of_bounds"
                print(f"[temp_reacq] obj_idx={obj_idx} out of bounds")
                return False, None, info

            # Get video-resolution candidate masks for validation.
            _, proposed_video_res_masks = self._get_orig_video_res_output(
                consolidated_out["pred_masks"]
            )

            if not torch.is_tensor(proposed_video_res_masks):
                info["reason"] = "bad_proposed_video_res_masks"
                return False, None, info

            if proposed_video_res_masks.ndim == 4:
                if obj_idx >= proposed_video_res_masks.shape[0]:
                    info["reason"] = "proposal_mask_oob"
                    return False, proposed_video_res_masks, info
                mask_logits = proposed_video_res_masks[obj_idx, 0]
            elif proposed_video_res_masks.ndim == 3:
                if obj_idx >= proposed_video_res_masks.shape[0]:
                    info["reason"] = "proposal_mask_oob"
                    return False, proposed_video_res_masks, info
                mask_logits = proposed_video_res_masks[obj_idx]
            else:
                info["reason"] = f"bad_proposal_mask_shape:{tuple(proposed_video_res_masks.shape)}"
                return False, proposed_video_res_masks, info

            mask_bool = (mask_logits > 0).detach().cpu().numpy().astype(np.uint8)
            mask_bbox, mask_area = _mask_bbox_and_area(mask_bool)

            info["mask_bbox"] = mask_bbox
            info["mask_area"] = int(mask_area)

            if mask_bbox is None or mask_area <= 0:
                info["reason"] = "empty_proposed_mask"
                return False, proposed_video_res_masks, info

            mask_bbox_iou = _bbox_iou_xyxy(mask_bbox, bbox_xyxy)
            mask_area_ratio = float(mask_area) / float(yolo_bbox_area + 1e-9)

            info["mask_bbox_iou"] = float(mask_bbox_iou)
            info["mask_area_ratio"] = float(mask_area_ratio)

            # Validation thresholds.
            # These are deliberately attributes so we can tune from the evaluation script later.
            bbox_iou_thr = float(getattr(self, "yolo_reacq_mask_bbox_iou_thr", 0.25))
            area_ratio_min = float(getattr(self, "yolo_reacq_mask_area_ratio_min", 0.15))
            area_ratio_max = float(getattr(self, "yolo_reacq_mask_area_ratio_max", 1.80))

            valid_mask = (
                mask_bbox_iou >= bbox_iou_thr
                and mask_area_ratio >= area_ratio_min
                and mask_area_ratio <= area_ratio_max
            )

            if not valid_mask:
                info["reason"] = (
                    f"mask_validation_failed:"
                    f"bbox_iou={mask_bbox_iou:.3f},"
                    f"area_ratio={mask_area_ratio:.3f}"
                )

                print(
                    f"[temp_reacq] REJECT mask frame={int(self.frame_idx)} "
                    f"obj_id={obj_id} yolo_bbox={info['yolo_bbox']} "
                    f"mask_bbox={mask_bbox} bbox_iou={mask_bbox_iou:.3f} "
                    f"area_ratio={mask_area_ratio:.3f}"
                )

                # Important: do NOT copy anything into current_out.
                # Important: do NOT recompute memory.
                return False, proposed_video_res_masks, info

            # ------------------------------------------------------------
            # If we reached here, the prompted SAM mask is geometrically
            # consistent with the YOLO+TransReID bbox. Now it is safe to
            # commit the mask into current_out.
            # ------------------------------------------------------------
            copy_keys = [
                "pred_masks",
                "obj_ptr",
                "object_score_logits",
            ]

            for key in copy_keys:
                src = consolidated_out.get(key, None)
                dst = current_out.get(key, None)

                if torch.is_tensor(src) and torch.is_tensor(dst):
                    if src.ndim >= 1 and dst.ndim >= 1:
                        if obj_idx < src.shape[0] and obj_idx < dst.shape[0]:
                            dst[obj_idx:obj_idx + 1].copy_(
                                src[obj_idx:obj_idx + 1].to(
                                    device=dst.device,
                                    dtype=dst.dtype,
                                    non_blocking=True,
                                )
                            )

            # Mark the reacquired object as confidently present.
            object_score_logits = current_out.get("object_score_logits", None)
            if torch.is_tensor(object_score_logits):
                if object_score_logits.ndim >= 2 and obj_idx < object_score_logits.shape[0]:
                    object_score_logits[obj_idx, 0] = 10.0
                elif object_score_logits.ndim == 1 and obj_idx < object_score_logits.shape[0]:
                    object_score_logits[obj_idx] = 10.0

            # Recompute memory features only for accepted masks.
            try:
                pred_masks_for_mem = current_out["pred_masks"].to(
                    cs["device"], non_blocking=True
                )

                high_res_masks = torch.nn.functional.interpolate(
                    pred_masks_for_mem,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )

                if getattr(self, "non_overlap_masks_for_mem_enc", False):
                    high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)

                maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                    frame_idx=tmp_frame_idx,
                    batch_size=self._get_obj_num(),
                    high_res_masks=high_res_masks,
                    object_score_logits=current_out["object_score_logits"],
                    is_mask_from_pts=True,
                )

                current_out["maskmem_features"] = maskmem_features
                current_out["maskmem_pos_enc"] = maskmem_pos_enc

            except Exception as e:
                print(f"[temp_reacq] memory re-encoding failed: {repr(e)}")

            # Return video-resolution masks after correction.
            _, video_res_masks = self._get_orig_video_res_output(current_out["pred_masks"])

            info["accepted"] = True
            info["reason"] = "accepted"

            print(
                f"[temp_reacq] ACCEPT mask frame={int(self.frame_idx)} "
                f"obj_id={obj_id} yolo_bbox={info['yolo_bbox']} "
                f"mask_bbox={mask_bbox} bbox_iou={mask_bbox_iou:.3f} "
                f"area_ratio={mask_area_ratio:.3f} tmp_frame_idx={tmp_frame_idx}"
            )

            return True, video_res_masks, info

        except Exception as e:
            info["reason"] = f"exception:{repr(e)}"
            print(f"[temp_reacq] failed obj_id={obj_id}: {repr(e)}")
            return False, None, info

        finally:
            # ------------------------------------------------------------
            # Cleanup: remove everything related to the temporary prompt.
            # ------------------------------------------------------------
            try:
                if tmp_frame_idx is not None:
                    # Remove temporary per-object outputs.
                    temp_per_obj = cs.get("temp_output_dict_per_obj", {})
                    for obj_temp in temp_per_obj.values():
                        for bucket in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                            try:
                                obj_temp[bucket].pop(tmp_frame_idx, None)
                            except Exception:
                                pass

                    # Remove temporary frame from committed output dict, defensively.
                    for bucket in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                        try:
                            output_dict.get(bucket, {}).pop(tmp_frame_idx, None)
                        except Exception:
                            pass

                    # Remove temporary frame from consolidated indices.
                    for bucket in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                        try:
                            inds = consolidated_frame_inds.get(bucket, None)
                            if hasattr(inds, "discard"):
                                inds.discard(tmp_frame_idx)
                        except Exception:
                            pass

                    # Remove temporary prompt inputs created by add_new_prompt.
                    for key in ["point_inputs_per_obj", "mask_inputs_per_obj"]:
                        per_obj_inputs = cs.get(key, None)
                        if isinstance(per_obj_inputs, dict):
                            for _, obj_inputs in list(per_obj_inputs.items()):
                                if isinstance(obj_inputs, dict):
                                    obj_inputs.pop(tmp_frame_idx, None)

                    # Some SAM2 variants may also store global prompt inputs.
                    for key in ["point_inputs", "mask_inputs"]:
                        inputs = cs.get(key, None)
                        if isinstance(inputs, dict):
                            inputs.pop(tmp_frame_idx, None)

                    # Remove cached features created for the temporary frame.
                    for key in list(cached_features.keys()):
                        if key not in old_cached_keys:
                            cached_features.pop(key, None)

                # Restore image lists exactly to their previous length.
                if "images" in cs:
                    del cs["images"][old_len_images:]

                if "images_orig_rgb" in cs:
                    del cs["images_orig_rgb"][old_len_orig:]

                cs["num_frames"] = old_num_frames
                cs["last_rgb"] = old_last_rgb
                cs["tracking_has_started"] = old_tracking_has_started

                # Restore gallery-update behavior.
                cs["suppress_reid_gallery_update"] = old_suppress_gallery

            except Exception as e:
                print(f"[temp_reacq] cleanup failed: {repr(e)}")
                cs["num_frames"] = old_num_frames
                cs["last_rgb"] = old_last_rgb
                cs["tracking_has_started"] = old_tracking_has_started
                cs["suppress_reid_gallery_update"] = old_suppress_gallery

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
        """
        IoU between two binary masks a,b of shape [H,W].
        """
        try:
            if (not torch.is_tensor(a)) or (not torch.is_tensor(b)):
                return 0.0

            a = a.bool()
            b = b.bool()

            inter = int(torch.logical_and(a, b).sum().item())
            if inter <= 0:
                return 0.0

            union = int(torch.logical_or(a, b).sum().item())
            if union <= 0:
                return 0.0

            return float(inter) / float(union)
        except Exception:
            return 0.0


    def _dedupe_rank_score(self, obj_id: int, obj_logit_val=None):
        """
        Rank score used ONLY for cross-object dedupe of overlapping final masks.

        Intuition:
        - ReID should dominate here because this is an identity conflict
        - object presence still matters
        - selected-mask quality (best_iou of the selected candidate) helps a bit

        Returns:
            score: float or None
            parts: dict for debug
        """
        try:
            cs = getattr(self, "condition_state", {})
            reid_last = cs.get("reid_last", {}) if isinstance(cs, dict) else {}
            info = reid_last.get(int(obj_id), {}) if isinstance(reid_last, dict) else {}

            sim = info.get("sim", None) if isinstance(info, dict) else None
            accepted = info.get("accepted", None) if isinstance(info, dict) else None
            best_iou = info.get("best_iou", None) if isinstance(info, dict) else None
            obj_prob_from_info = info.get("obj_prob", None) if isinstance(info, dict) else None

            # Identity-consistency term (soft around reid_thr)
            s_reid = self._soft_reid_score(sim)

            # Presence term
            s_obj = self._safe_sigmoid_float(obj_logit_val)
            if s_obj is None and obj_prob_from_info is not None:
                try:
                    s_obj = float(obj_prob_from_info)
                except Exception:
                    s_obj = None

            # Selected-mask quality term
            s_mask = self._clamp01(best_iou)

            # Optional tiny bonus/penalty from explicit accept/reject
            # Keep it small: similarity is still the main ReID signal.
            if s_reid is not None:
                if accepted is True:
                    s_reid = min(1.0, float(s_reid) + 0.05)
                elif accepted is False:
                    s_reid = max(0.0, float(s_reid) - 0.05)

            # Default weights: identity should dominate cross-object dedupe
            w_reid = float(getattr(self, "dedupe_w_reid", 0.60))
            w_obj  = float(getattr(self, "dedupe_w_obj",  0.25))
            w_mask = float(getattr(self, "dedupe_w_mask", 0.15))

            num = 0.0
            den = 0.0

            if s_reid is not None:
                num += w_reid * float(s_reid)
                den += w_reid

            if s_obj is not None:
                num += w_obj * float(s_obj)
                den += w_obj

            if s_mask is not None:
                num += w_mask * float(s_mask)
                den += w_mask

            if den <= 0.0:
                return None, {
                    "s_reid": s_reid,
                    "s_obj": s_obj,
                    "s_mask": s_mask,
                    "sim": sim,
                    "accepted": accepted,
                    "best_iou": best_iou,
                }

            score = num / den
            return float(score), {
                "s_reid": s_reid,
                "s_obj": s_obj,
                "s_mask": s_mask,
                "sim": sim,
                "accepted": accepted,
                "best_iou": best_iou,
            }

        except Exception:
            return None, {
                "s_reid": None,
                "s_obj": None,
                "s_mask": None,
                "sim": None,
                "accepted": None,
                "best_iou": None,
            }


    def _dedupe_by_mask_iou(
        self,
        obj_ids,
        video_res_masks_raw: torch.Tensor,
        object_score_logits: torch.Tensor = None,
        iou_thr: float = 0.6,
        min_area: int = 50,
    ):
        """
        Keep at most one mask per physical person by suppressing highly-overlapping masks.

        Updated logic:
        - still uses greedy mask-NMS by overlap
        - BUT ranking score is now identity-aware:
            dedupe_score ~= ReID consistency + object presence + selected-mask quality
        - KF is intentionally NOT used here, because this stage is mainly resolving
        cross-object identity conflicts among overlapping final masks.
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

        N = int(m.shape[0])
        if N <= 1:
            return obj_ids, video_res_masks_raw

        # Normalize object score logits to [N] if available
        obj_logits = None
        if object_score_logits is not None and torch.is_tensor(object_score_logits):
            obj_logits = object_score_logits.detach().float().reshape(-1)
            if obj_logits.numel() != N:
                obj_logits = None

        # Binary masks for overlap check
        probs_cpu = torch.sigmoid(m.detach().float()).cpu()
        bin_masks = (probs_cpu > 0.5)

        def _obj_id_at(i: int) -> int:
            try:
                if isinstance(obj_ids, (list, tuple)):
                    return int(obj_ids[i])
                if torch.is_tensor(obj_ids):
                    return int(obj_ids[i].item())
            except Exception:
                pass
            return int(i)

        scores_list = []
        debug_entries = []

        for i in range(N):
            mask_i = bin_masks[i]
            area = int(mask_i.sum().item())
            oid = _obj_id_at(i)

            if area < int(min_area):
                scores_list.append(-1e9)
                debug_entries.append({
                    "idx": int(i),
                    "obj_id": int(oid),
                    "area": int(area),
                    "score": -1e9,
                    "reason": "area_below_min",
                })
                continue

            obj_logit_val = None
            if obj_logits is not None and i < int(obj_logits.numel()):
                try:
                    obj_logit_val = float(obj_logits[i].item())
                except Exception:
                    obj_logit_val = None

            rank_score, parts = self._dedupe_rank_score(
                obj_id=int(oid),
                obj_logit_val=obj_logit_val,
            )

            # Fallback if identity-aware score is unavailable
            if rank_score is None:
                try:
                    inside = probs_cpu[i][mask_i]
                    if inside.numel() > 0:
                        rank_score = float(inside.mean().item())
                    else:
                        rank_score = -1e9
                except Exception:
                    rank_score = -1e9

            scores_list.append(float(rank_score))
            debug_entries.append({
                "idx": int(i),
                "obj_id": int(oid),
                "area": int(area),
                "score": float(rank_score),
                "obj_logit": obj_logit_val,
                "parts": parts,
            })

        scores = torch.tensor(scores_list, dtype=torch.float32)
        order = torch.argsort(scores, descending=True).tolist()

        keep = []
        suppressions = []

        for idx in order:
            if float(scores[idx].item()) < -1e8:
                continue

            cand = bin_masks[idx]
            duplicate = False
            duplicate_of = None
            duplicate_iou = 0.0

            for j in keep:
                ov = self._mask_iou(cand, bin_masks[j])
                if ov >= float(iou_thr):
                    duplicate = True
                    duplicate_of = int(j)
                    duplicate_iou = float(ov)
                    break

            if not duplicate:
                keep.append(int(idx))
            else:
                suppressions.append({
                    "suppressed_idx": int(idx),
                    "suppressed_obj_id": _obj_id_at(int(idx)),
                    "kept_idx": int(duplicate_of),
                    "kept_obj_id": _obj_id_at(int(duplicate_of)),
                    "overlap_iou": float(duplicate_iou),
                    "suppressed_score": float(scores[idx].item()),
                    "kept_score": float(scores[duplicate_of].item()),
                })

        keep = sorted(keep)  # preserve stable ordering

        # Optional debug storage
        try:
            cs = getattr(self, "condition_state", None)
            if isinstance(cs, dict):
                cs["last_dedupe_debug"] = {
                    "iou_thr": float(iou_thr),
                    "min_area": int(min_area),
                    "entries": debug_entries,
                    "keep_indices": [int(x) for x in keep],
                    "keep_obj_ids": [_obj_id_at(int(x)) for x in keep],
                    "suppressions": suppressions,
                }
        except Exception:
            pass

        if len(keep) == N:
            return obj_ids, video_res_masks_raw

        # Filter obj_ids consistently
        if isinstance(obj_ids, (list, tuple)):
            new_obj_ids = [obj_ids[i] for i in keep]
        elif torch.is_tensor(obj_ids):
            new_obj_ids = obj_ids[keep]
        else:
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
        Streaming tracking step.

        Behavior:
        - During normal tracking, SAMURAI outputs masks and ReID verifies them.
        - If SAMURAI object score says an object is absent, that object enters
        reacquisition mode.
        - While in reacquisition mode:
            * YOLO+TransReID provides a bbox fallback/localization constraint.
            * SAMURAI's own predicted mask is checked with ReID.
            * The SAMURAI mask is accepted only if:
                1) ReID similarity >= reid_thr
                2) the SAMURAI mask bbox is spatially consistent with the
                    YOLO+TransReID fallback bbox for the same identity.
            * If either condition fails, the SAMURAI mask is suppressed and does
            not update memory/gallery.
        - YOLO bbox prompts should be disabled externally with:
            --no-yolo_reacq_try_sam_mask
        """

        def _bbox_intersection_area(a, b):
            ax1, ay1, ax2, ay2 = [float(x) for x in a]
            bx1, by1, bx2, by2 = [float(x) for x in b]

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            return float(iw * ih)

        def _bbox_area(a):
            x1, y1, x2, y2 = [float(x) for x in a]
            return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

        def _bbox_iou(a, b):
            inter = _bbox_intersection_area(a, b)
            area_a = _bbox_area(a)
            area_b = _bbox_area(b)
            return float(inter / (area_a + area_b - inter + 1e-9))

        def _center_inside(inner_bbox, outer_bbox):
            x1, y1, x2, y2 = [float(x) for x in inner_bbox]
            ox1, oy1, ox2, oy2 = [float(x) for x in outer_bbox]

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            return bool(ox1 <= cx <= ox2 and oy1 <= cy <= oy2)

        def _sam_mask_agrees_with_reid_bbox(mask_bbox, reid_bbox):
            """
            Check whether SAMURAI's mask bbox is spatially compatible with the
            YOLO+TransReID bbox.

            This is deliberately not strict full containment, because SAMURAI masks
            can be tighter than YOLO boxes and YOLO boxes can include background.
            The default rule is:
                - mask center must be inside the ReID bbox
                - enough of the SAM mask bbox must lie inside the ReID bbox
            """
            mask_area = _bbox_area(mask_bbox)
            inter_area = _bbox_intersection_area(mask_bbox, reid_bbox)
            iou = _bbox_iou(mask_bbox, reid_bbox)

            inside_frac = float(inter_area / (mask_area + 1e-9))
            center_inside = _center_inside(mask_bbox, reid_bbox)

            min_inside_frac = float(getattr(self, "yolo_reacq_sam_mask_inside_frac", 0.50))
            min_iou = float(getattr(self, "yolo_reacq_sam_mask_bbox_iou_min", 0.05))

            ok = bool(
                center_inside
                and inside_frac >= min_inside_frac
                and iou >= min_iou
            )

            return ok, {
                "sam_mask_bbox": [int(v) for v in mask_bbox],
                "reid_bbox": [int(v) for v in reid_bbox],
                "bbox_iou": float(iou),
                "inside_frac": float(inside_frac),
                "center_inside": bool(center_inside),
                "min_inside_frac": float(min_inside_frac),
                "min_iou": float(min_iou),
            }

        # ---- store raw RGB for internal ReID/Yolo gating (must be BEFORE perpare_data) ----
        try:
            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                self.condition_state["last_rgb"] = img.copy()
        except Exception:
            pass

        # ---- advance global timeline ----
        self.frame_idx += 1

        if "num_frames" not in self.condition_state:
            self.condition_state["num_frames"] = 0
        self.condition_state["num_frames"] = max(
            int(self.condition_state["num_frames"]),
            int(self.frame_idx + 1),
        )

        # Clear stale bbox fallbacks from previous frames.
        try:
            cs = self.condition_state
            fallback_by_id = cs.setdefault("reacq_bbox_fallback_by_id", {})
            current_frame_idx = int(self.frame_idx)

            for old_oid in list(fallback_by_id.keys()):
                try:
                    old_frame = int(fallback_by_id[old_oid].get("frame_idx", -10**9))
                    if old_frame != current_frame_idx:
                        fallback_by_id.pop(old_oid, None)
                except Exception:
                    fallback_by_id.pop(old_oid, None)
        except Exception:
            pass

        # preflight once
        if not self.condition_state.get("tracking_has_started", False):
            self.propagate_in_video_preflight()

        # keep stored outputs consistent with current #objects
        self._expand_all_stored_outputs_to_current_batch()

        # prepare input for SAMURAI/SAM2
        img, _, _ = self.perpare_data(img, image_size=self.image_size)

        output_dict = self.condition_state["output_dict"]
        obj_ids = self.condition_state["obj_ids"]
        batch_size = self._get_obj_num()

        # get features
        (_, _, current_vision_feats, current_vision_pos_embeds, feat_sizes) = self._get_feature(
            img, batch_size
        )

        # ---- SAMURAI/SAM2 track step ----
        current_out = self.track_step(
            frame_idx=self.frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=self.condition_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )

        # ---- prepare raw masks ----
        storage_device = self.condition_state["storage_device"]

        maskmem_features = current_out.get("maskmem_features", None)
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16).to(
                storage_device, non_blocking=True
            )

        pred_masks_gpu = current_out["pred_masks"]
        if getattr(self, "fill_hole_area", 0) > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)

        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out.get("object_score_logits", None)
        best_iou_score = current_out.get("best_iou_score", None)
        kf_ious = current_out.get("kf_ious", None)

        obj_score_thr = float(getattr(self, "min_obj_score_logits", 0.0))

        # Get video-res masks before ReID checks.
        _, video_res_masks_raw = self._get_orig_video_res_output(pred_masks_gpu)

        # defaults
        reid_ok_list = [-1 for _ in range(len(obj_ids))]
        reid_sim_list = [None for _ in range(len(obj_ids))]
        reacquire_state_list = [False for _ in range(len(obj_ids))]
        live_obj_logits = [None for _ in range(len(obj_ids))]
        live_obj_probs = [None for _ in range(len(obj_ids))]

        suppress_mask_by_oid = {}
        early_yolo_reacq_attempted = False

        # ============================================================
        # INTERNAL ReID CHECK
        # ============================================================
        try:
            cs = self.condition_state
            reid_model = cs.get("reid", None)
            reid_thr = float(cs.get("reid_thr", 0.80))
            reid_last = cs.setdefault("reid_last", {})
            reacquire_map = cs.setdefault("reacquire_mode_per_id", {})
            fallback_by_id = cs.setdefault("reacq_bbox_fallback_by_id", {})
            last_rgb = cs.get("last_rgb", None)

            # -------- read per-object current object score from LIVE current_out --------
            for k in range(len(obj_ids)):
                obj_logit_val = None
                obj_prob_val = None

                try:
                    if torch.is_tensor(object_score_logits):
                        if object_score_logits.ndim >= 2:
                            if k < object_score_logits.shape[0]:
                                obj_logit_val = float(
                                    object_score_logits[k].detach().float().reshape(-1)[0].item()
                                )
                        elif object_score_logits.ndim == 1:
                            if k < object_score_logits.shape[0]:
                                obj_logit_val = float(object_score_logits[k].detach().float().item())

                    if obj_logit_val is not None:
                        obj_prob_val = float(torch.sigmoid(torch.tensor(obj_logit_val)).item())

                except Exception:
                    obj_logit_val = None
                    obj_prob_val = None

                live_obj_logits[k] = obj_logit_val
                live_obj_probs[k] = obj_prob_val

            # First pass: mark objects as reacquiring if SAMURAI says absent.
            for k, oid in enumerate(obj_ids):
                oid = int(oid)
                reacquire_map.setdefault(oid, False)
                suppress_mask_by_oid.setdefault(oid, False)

                obj_logit_val = live_obj_logits[k]
                obj_present_by_score = (
                    obj_logit_val is not None and obj_logit_val > obj_score_thr
                )

                if not obj_present_by_score:
                    reacquire_map[oid] = True

            # ------------------------------------------------------------
            # Important:
            # If any object is in reacquisition mode, run YOLO+TransReID now
            # to populate condition_state["reacq_bbox_fallback_by_id"] for the
            # current frame. We temporarily force bbox-only mode here, because
            # we do NOT want YOLO boxes to be passed as SAMURAI prompts.
            # ------------------------------------------------------------
            any_reacquiring_now = any(
                bool(reacquire_map.get(int(oid), False)) for oid in obj_ids
            )

            if any_reacquiring_now and bool(getattr(self, "yolo_reacq_enabled", False)):
                try:
                    old_try_sam_mask = getattr(self, "yolo_reacq_try_sam_mask", False)
                    self.yolo_reacq_try_sam_mask = False
                    early_yolo_reacq_attempted = True
                    _ = self._try_internal_yolo_reacquisition(current_out=current_out)
                    self.yolo_reacq_try_sam_mask = old_try_sam_mask
                except Exception as e:
                    try:
                        self.yolo_reacq_try_sam_mask = old_try_sam_mask
                    except Exception:
                        pass
                    print(f"[yolo_reacq] early bbox fallback failed in track(): {repr(e)}", flush=True)

            if reid_model is not None and torch.is_tensor(video_res_masks_raw) and last_rgb is not None:
                frame_bgr = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)

                for k, oid in enumerate(obj_ids):
                    oid = int(oid)
                    reacquire_map.setdefault(oid, False)
                    suppress_mask_by_oid.setdefault(oid, False)

                    gallery = self._reid_gallery_get(oid)
                    has_gallery = len(gallery) > 0
                    obj_logit_val = live_obj_logits[k]
                    obj_prob_val = live_obj_probs[k]
                    was_reacquiring = bool(reacquire_map.get(oid, False))

                    # If there is no gallery, we cannot ReID-check the mask.
                    if not has_gallery:
                        if was_reacquiring:
                            suppress_mask_by_oid[oid] = True
                            reacquire_state_list[k] = True
                            reid_ok_list[k] = -1
                            reid_sim_list[k] = None
                            reid_last[oid] = {
                                "sim": None,
                                "bbox": None,
                                "accepted": None,
                                "frame_idx": int(self.frame_idx),
                                "reason": "reacquire_no_ref",
                                "obj_logit": obj_logit_val,
                                "obj_prob": obj_prob_val,
                                "obj_score_thr": obj_score_thr,
                                "reacquire": True,
                                "gallery_size": 0,
                                "best_ref_idx": None,
                                "gallery_added": False,
                            }
                        else:
                            reacquire_state_list[k] = False
                            reid_ok_list[k] = -1
                            reid_sim_list[k] = None
                            reid_last[oid] = {
                                "sim": None,
                                "bbox": None,
                                "accepted": None,
                                "frame_idx": int(self.frame_idx),
                                "reason": "no_ref",
                                "obj_logit": obj_logit_val,
                                "obj_prob": obj_prob_val,
                                "obj_score_thr": obj_score_thr,
                                "reacquire": False,
                                "gallery_size": 0,
                                "best_ref_idx": None,
                                "gallery_added": False,
                            }
                        continue

                    # Extract mask logits for this object.
                    if video_res_masks_raw.ndim == 4:
                        if k >= video_res_masks_raw.shape[0]:
                            if was_reacquiring:
                                suppress_mask_by_oid[oid] = True
                                reacquire_state_list[k] = True
                            reid_ok_list[k] = -1
                            reid_last[oid] = {
                                "sim": None,
                                "bbox": None,
                                "accepted": None,
                                "frame_idx": int(self.frame_idx),
                                "reason": "mask_oob",
                                "obj_logit": obj_logit_val,
                                "obj_prob": obj_prob_val,
                                "obj_score_thr": obj_score_thr,
                                "reacquire": bool(reacquire_map.get(oid, False)),
                                "gallery_size": len(gallery),
                                "gallery_added": False,
                            }
                            continue
                        mask_logits = video_res_masks_raw[k, 0]

                    elif video_res_masks_raw.ndim == 3:
                        if k >= video_res_masks_raw.shape[0]:
                            if was_reacquiring:
                                suppress_mask_by_oid[oid] = True
                                reacquire_state_list[k] = True
                            reid_ok_list[k] = -1
                            reid_last[oid] = {
                                "sim": None,
                                "bbox": None,
                                "accepted": None,
                                "frame_idx": int(self.frame_idx),
                                "reason": "mask_oob",
                                "obj_logit": obj_logit_val,
                                "obj_prob": obj_prob_val,
                                "obj_score_thr": obj_score_thr,
                                "reacquire": bool(reacquire_map.get(oid, False)),
                                "gallery_size": len(gallery),
                                "gallery_added": False,
                            }
                            continue
                        mask_logits = video_res_masks_raw[k]

                    else:
                        if was_reacquiring:
                            suppress_mask_by_oid[oid] = True
                            reacquire_state_list[k] = True
                        reid_ok_list[k] = -1
                        reid_last[oid] = {
                            "sim": None,
                            "bbox": None,
                            "accepted": None,
                            "frame_idx": int(self.frame_idx),
                            "reason": "bad_mask_shape",
                            "obj_logit": obj_logit_val,
                            "obj_prob": obj_prob_val,
                            "obj_score_thr": obj_score_thr,
                            "reacquire": bool(reacquire_map.get(oid, False)),
                            "gallery_size": len(gallery),
                            "gallery_added": False,
                        }
                        continue

                    mask_bool = (mask_logits > 0).detach().cpu().numpy().astype(np.uint8)

                    if mask_bool.sum() == 0:
                        if was_reacquiring:
                            suppress_mask_by_oid[oid] = True
                            reacquire_state_list[k] = True
                        reid_ok_list[k] = 0 if was_reacquiring else -1
                        reid_sim_list[k] = None
                        reid_last[oid] = {
                            "sim": None,
                            "bbox": None,
                            "accepted": False if was_reacquiring else None,
                            "frame_idx": int(self.frame_idx),
                            "reason": "empty_mask_reacquire" if was_reacquiring else "empty_mask",
                            "obj_logit": obj_logit_val,
                            "obj_prob": obj_prob_val,
                            "obj_score_thr": obj_score_thr,
                            "reacquire": bool(reacquire_map.get(oid, False)),
                            "gallery_size": len(gallery),
                            "gallery_added": False,
                        }
                        continue

                    ys, xs = np.where(mask_bool > 0)
                    if xs.size == 0 or ys.size == 0:
                        if was_reacquiring:
                            suppress_mask_by_oid[oid] = True
                            reacquire_state_list[k] = True
                        reid_ok_list[k] = 0 if was_reacquiring else -1
                        reid_sim_list[k] = None
                        reid_last[oid] = {
                            "sim": None,
                            "bbox": None,
                            "accepted": False if was_reacquiring else None,
                            "frame_idx": int(self.frame_idx),
                            "reason": "empty_bbox_reacquire" if was_reacquiring else "empty_bbox",
                            "obj_logit": obj_logit_val,
                            "obj_prob": obj_prob_val,
                            "obj_score_thr": obj_score_thr,
                            "reacquire": bool(reacquire_map.get(oid, False)),
                            "gallery_size": len(gallery),
                            "gallery_added": False,
                        }
                        continue

                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    bbox_xyxy = [x1, y1, x2, y2]

                    crop = frame_bgr[y1:y2 + 1, x1:x2 + 1].copy()

                    if crop.size == 0:
                        if was_reacquiring:
                            suppress_mask_by_oid[oid] = True
                            reacquire_state_list[k] = True
                        reid_ok_list[k] = 0 if was_reacquiring else -1
                        reid_sim_list[k] = None
                        reid_last[oid] = {
                            "sim": None,
                            "bbox": bbox_xyxy,
                            "accepted": False if was_reacquiring else None,
                            "frame_idx": int(self.frame_idx),
                            "reason": "empty_crop_reacquire" if was_reacquiring else "empty_crop",
                            "obj_logit": obj_logit_val,
                            "obj_prob": obj_prob_val,
                            "obj_score_thr": obj_score_thr,
                            "reacquire": bool(reacquire_map.get(oid, False)),
                            "gallery_size": len(gallery),
                            "gallery_added": False,
                        }
                        continue

                    try:
                        cur_emb = reid_model.embed_crop_bgr(crop)
                    except Exception as e:
                        if was_reacquiring:
                            suppress_mask_by_oid[oid] = True
                            reacquire_state_list[k] = True
                        reid_ok_list[k] = 0 if was_reacquiring else -1
                        reid_sim_list[k] = None
                        reid_last[oid] = {
                            "sim": None,
                            "bbox": bbox_xyxy,
                            "accepted": False if was_reacquiring else None,
                            "frame_idx": int(self.frame_idx),
                            "reason": f"embed_fail:{repr(e)}",
                            "obj_logit": obj_logit_val,
                            "obj_prob": obj_prob_val,
                            "obj_score_thr": obj_score_thr,
                            "reacquire": bool(reacquire_map.get(oid, False)),
                            "gallery_size": len(gallery),
                            "gallery_added": False,
                        }
                        continue

                    if cur_emb is None or (torch.is_tensor(cur_emb) and cur_emb.numel() == 0):
                        if was_reacquiring:
                            suppress_mask_by_oid[oid] = True
                            reacquire_state_list[k] = True
                        reid_ok_list[k] = 0 if was_reacquiring else -1
                        reid_sim_list[k] = None
                        reid_last[oid] = {
                            "sim": None,
                            "bbox": bbox_xyxy,
                            "accepted": False if was_reacquiring else None,
                            "frame_idx": int(self.frame_idx),
                            "reason": "embed_none",
                            "obj_logit": obj_logit_val,
                            "obj_prob": obj_prob_val,
                            "obj_score_thr": obj_score_thr,
                            "reacquire": bool(reacquire_map.get(oid, False)),
                            "gallery_size": len(gallery),
                            "gallery_added": False,
                        }
                        continue

                    try:
                        sim, best_ref_idx, all_sims = self._reid_gallery_best_sim(oid, cur_emb)
                    except Exception as e:
                        if was_reacquiring:
                            suppress_mask_by_oid[oid] = True
                            reacquire_state_list[k] = True
                        reid_ok_list[k] = 0 if was_reacquiring else -1
                        reid_sim_list[k] = None
                        reid_last[oid] = {
                            "sim": None,
                            "bbox": bbox_xyxy,
                            "accepted": False if was_reacquiring else None,
                            "frame_idx": int(self.frame_idx),
                            "reason": f"gallery_match_fail:{repr(e)}",
                            "obj_logit": obj_logit_val,
                            "obj_prob": obj_prob_val,
                            "obj_score_thr": obj_score_thr,
                            "reacquire": bool(reacquire_map.get(oid, False)),
                            "gallery_size": len(gallery),
                            "best_ref_idx": None,
                            "gallery_added": False,
                        }
                        continue

                    sim_ok = sim is not None and np.isfinite(sim)
                    reid_accepted = bool(sim_ok and float(sim) >= reid_thr)

                    reid_sim_list[k] = float(sim) if sim_ok else None
                    reid_ok_list[k] = 1 if reid_accepted else 0

                    best_iou_val = None
                    if torch.is_tensor(best_iou_score):
                        if best_iou_score.ndim >= 1 and k < best_iou_score.shape[0]:
                            best_iou_val = float(best_iou_score[k].detach().float().reshape(-1)[0].item())

                    kf_score_val = None
                    if torch.is_tensor(kf_ious):
                        if kf_ious.ndim >= 1 and k < kf_ious.shape[0]:
                            kf_score_val = float(kf_ious[k].detach().float().reshape(-1)[0].item())

                    # ---------------------------------------------------------
                    # Reacquisition decision:
                    # accept SAMURAI's own mask only if:
                    #   1. mask crop passes ReID
                    #   2. mask is inside / consistent with YOLO+TransReID bbox
                    # ---------------------------------------------------------
                    if was_reacquiring:
                        fb_info = fallback_by_id.get(int(oid), None)
                        fb_bbox = None

                        if isinstance(fb_info, dict):
                            try:
                                if int(fb_info.get("frame_idx", -1)) == int(self.frame_idx):
                                    raw_fb_bbox = fb_info.get("bbox", None)
                                    if raw_fb_bbox is not None and len(raw_fb_bbox) == 4:
                                        fb_bbox = [int(round(float(v))) for v in raw_fb_bbox]
                            except Exception:
                                fb_bbox = None

                        if fb_bbox is not None:
                            bbox_agrees, bbox_agreement_info = _sam_mask_agrees_with_reid_bbox(
                                bbox_xyxy,
                                fb_bbox,
                            )
                        else:
                            bbox_agrees = False
                            bbox_agreement_info = {
                                "reason": "no_current_transreid_bbox",
                                "sam_mask_bbox": [int(v) for v in bbox_xyxy],
                                "reid_bbox": None,
                            }

                        accepted = bool(reid_accepted and bbox_agrees)

                        if accepted:
                            reacquire_map[oid] = False
                            reacquire_state_list[k] = False
                            suppress_mask_by_oid[oid] = False

                            try:
                                if torch.is_tensor(object_score_logits):
                                    if object_score_logits.ndim >= 2 and k < object_score_logits.shape[0]:
                                        object_score_logits[k, 0] = 10.0
                                    elif object_score_logits.ndim == 1 and k < object_score_logits.shape[0]:
                                        object_score_logits[k] = 10.0
                            except Exception:
                                pass

                            reason = "reacquire_samurai_mask_reid_and_bbox_accepted"

                        else:
                            reacquire_map[oid] = True
                            reacquire_state_list[k] = True
                            suppress_mask_by_oid[oid] = True

                            if not reid_accepted:
                                reason = "reacquire_samurai_mask_reid_rejected"
                            elif not bbox_agrees:
                                reason = "reacquire_samurai_mask_bbox_rejected"
                            else:
                                reason = "reacquire_samurai_mask_rejected"

                        reid_last[oid] = {
                            "sim": float(sim) if sim_ok else None,
                            "bbox": bbox_xyxy,
                            "accepted": accepted,
                            "frame_idx": int(self.frame_idx),
                            "reason": reason,
                            "obj_logit": obj_logit_val,
                            "obj_prob": obj_prob_val,
                            "obj_score_thr": obj_score_thr,
                            "reacquire": bool(reacquire_map.get(oid, False)),
                            "gallery_size": len(gallery),
                            "best_ref_idx": best_ref_idx if sim_ok else None,
                            "kf_score": kf_score_val,
                            "best_iou": best_iou_val,
                            "reacq_score": None,
                            "reacq_parts": None,
                            "gallery_added": False,
                            "transreid_bbox": fb_bbox,
                            "bbox_agrees": bool(bbox_agrees),
                            "bbox_agreement_info": bbox_agreement_info,
                            "reid_accepted": bool(reid_accepted),
                        }

                        # Never update gallery on reacquisition frames.
                        continue

                    # ---------------------------------------------------------
                    # Normal tracking decision.
                    # ---------------------------------------------------------
                    reacquire_state_list[k] = False
                    suppress_mask_by_oid[oid] = False

                    accepted = bool(reid_accepted)

                    reid_last[oid] = {
                        "sim": float(sim) if sim_ok else None,
                        "bbox": bbox_xyxy,
                        "accepted": accepted,
                        "frame_idx": int(self.frame_idx),
                        "reason": "ok" if sim_ok else "nan_sim",
                        "obj_logit": obj_logit_val,
                        "obj_prob": obj_prob_val,
                        "obj_score_thr": obj_score_thr,
                        "reacquire": False,
                        "gallery_size": len(gallery),
                        "best_ref_idx": best_ref_idx if sim_ok else None,
                        "kf_score": kf_score_val,
                        "best_iou": best_iou_val,
                        "reacq_score": None,
                        "reacq_parts": None,
                    }

                    # ---------------------------------------------------------
                    # ONLINE GALLERY UPDATE during normal tracking only.
                    # Do not add if ReID rejected the current crop.
                    # ---------------------------------------------------------
                    try:
                        iou_add_thr = float(getattr(self, "memory_bank_iou_threshold", 0.0))
                        obj_present_for_add = (
                            obj_logit_val is not None and obj_logit_val > obj_score_thr
                        )
                        iou_good_for_add = (
                            best_iou_val is not None and best_iou_val > iou_add_thr
                        )

                        safe_to_add = (
                            accepted
                            and obj_present_for_add
                            and iou_good_for_add
                        )

                        if safe_to_add and self._reid_gallery_should_add(
                            oid,
                            cur_emb,
                            self.frame_idx,
                            bbox_xyxy=bbox_xyxy,
                            mask_bool=mask_bool,
                            frame_shape=frame_bgr.shape,
                        ):
                            quality_score = self._reid_gallery_candidate_score(
                                bbox_xyxy=bbox_xyxy,
                                mask_bool=mask_bool,
                                frame_shape=frame_bgr.shape,
                                sim_to_gallery=sim,
                                accepted_by_reid=accepted,
                                best_iou_val=best_iou_val,
                                obj_logit_val=obj_logit_val,
                            )

                            added = self._reid_gallery_add(
                                obj_id=oid,
                                emb=cur_emb,
                                frame_idx=int(self.frame_idx),
                                bbox=bbox_xyxy,
                                source="track",
                                is_anchor=False,
                                quality_score=quality_score,
                            )

                            if added:
                                self._reid_gallery_mark_added(oid, int(self.frame_idx))
                                reid_last[oid]["gallery_size"] = len(self._reid_gallery_get(oid))
                                reid_last[oid]["gallery_added"] = True
                            else:
                                reid_last[oid]["gallery_size"] = len(self._reid_gallery_get(oid))
                                reid_last[oid]["gallery_added"] = False
                        else:
                            reid_last[oid]["gallery_added"] = False

                    except Exception as e:
                        reid_last[oid]["gallery_add_error"] = repr(e)

            current_out["reid_ok"] = torch.tensor(
                reid_ok_list,
                dtype=torch.int8,
                device=pred_masks_gpu.device,
            )

        except Exception as e:
            print(f"[reid/internal] failed in track(): {repr(e)}", flush=True)
            current_out["reid_ok"] = torch.full(
                (len(obj_ids),),
                -1,
                dtype=torch.int8,
                device=pred_masks_gpu.device,
            )

        # ------------------------------------------------
        # FINAL VISIBILITY GATE
        # Hide masks still in reacquisition / failed ReID / failed bbox agreement.
        # ------------------------------------------------
        try:
            cs = self.condition_state
            reacquire_map = cs.setdefault("reacquire_mode_per_id", {})

            for i in range(len(obj_ids)):
                oid = int(obj_ids[i])
                obj_reacquire = bool(reacquire_map.get(oid, False))
                must_suppress = bool(suppress_mask_by_oid.get(oid, False)) or obj_reacquire

                if must_suppress:
                    if pred_masks_gpu.ndim == 4 and i < pred_masks_gpu.shape[0]:
                        pred_masks_gpu[i, 0].fill_(-1024.0)
                    elif pred_masks_gpu.ndim == 3 and i < pred_masks_gpu.shape[0]:
                        pred_masks_gpu[i].fill_(-1024.0)

                    if torch.is_tensor(video_res_masks_raw):
                        if video_res_masks_raw.ndim == 4 and i < video_res_masks_raw.shape[0]:
                            video_res_masks_raw[i, 0].fill_(-1024.0)
                        elif video_res_masks_raw.ndim == 3 and i < video_res_masks_raw.shape[0]:
                            video_res_masks_raw[i].fill_(-1024.0)

        except Exception as e:
            print(f"[reid/internal] visibility gate failed: {repr(e)}", flush=True)

        # ---- build storage tensors AFTER visibility gating ----
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(current_out)

        reid_ok = current_out.get("reid_ok", None)
        if reid_ok is not None and torch.is_tensor(reid_ok):
            reid_ok = reid_ok.to(storage_device, non_blocking=True)

        def _to_float_tensor(vals, device):
            out = []
            for v in vals:
                if v is None:
                    out.append(float("nan"))
                else:
                    out.append(float(v))
            return torch.tensor(out, dtype=torch.float32, device=device)

        def _to_bool_tensor(vals, device):
            return torch.tensor([bool(v) for v in vals], dtype=torch.bool, device=device)

        reid_sim_tensor = _to_float_tensor(reid_sim_list, pred_masks_gpu.device)
        reacquire_tensor = _to_bool_tensor(reacquire_state_list, pred_masks_gpu.device)

        mem_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "best_iou_score": best_iou_score,
            "kf_score": kf_ious,
            "reid_ok": reid_ok,
            "reid_sim": reid_sim_tensor,
            "reacquire": reacquire_tensor,
        }

        current_out["kf_score"] = current_out.get("kf_ious", None)

        # ------------------------------------------------
        # YOLO reacquisition fallback
        #
        # Usually already called early in this function so the bbox can be used
        # as a spatial constraint. If it was not called earlier, call it here.
        # ------------------------------------------------
        try:
            if not early_yolo_reacq_attempted:
                old_try_sam_mask = getattr(self, "yolo_reacq_try_sam_mask", False)
                self.yolo_reacq_try_sam_mask = False
                yolo_reacq_result = self._try_internal_yolo_reacquisition(current_out=current_out)
                self.yolo_reacq_try_sam_mask = old_try_sam_mask

                if yolo_reacq_result is not None:
                    yolo_obj_ids, yolo_video_res_masks = yolo_reacq_result

                    self._last_video_res_masks_raw = yolo_video_res_masks

                    self._manage_memory_obj(self.frame_idx, current_out)

                    video_obj_ids = yolo_obj_ids
                    video_obj_ids, yolo_video_res_masks = self._dedupe_by_mask_iou(
                        obj_ids=video_obj_ids,
                        video_res_masks_raw=yolo_video_res_masks,
                        object_score_logits=current_out.get("object_score_logits", None),
                        iou_thr=getattr(self, "dedupe_iou_thr", 0.6),
                        min_area=getattr(self, "dedupe_min_area", 200),
                    )

                    return video_obj_ids, yolo_video_res_masks

        except Exception as e:
            try:
                self.yolo_reacq_try_sam_mask = old_try_sam_mask
            except Exception:
                pass
            print(f"[yolo_reacq] internal reacquisition failed in track(): {repr(e)}", flush=True)

        # Memory receives accepted masks normally and suppressed masks for objects
        # that failed ReID / failed bbox agreement / remain in reacquisition.
        self._manage_memory_obj(self.frame_idx, mem_out)

        # ---- store LIVE debug info for HUD ----
        try:
            cs = self.condition_state
            good_mem_frames = list(cs.get("good_memory_frames", []))
            reacquire_map = cs.setdefault("reacquire_mode_per_id", {})
            fallback_by_id = cs.get("reacq_bbox_fallback_by_id", {})

            reacq_score_list = []
            reacq_parts_list = []
            best_iou_list = []
            kf_score_list = []

            for oid in obj_ids:
                info = cs.get("reid_last", {}).get(int(oid), {})
                reacq_score_list.append(info.get("reacq_score", None))
                reacq_parts_list.append(info.get("reacq_parts", None))
                best_iou_list.append(info.get("best_iou", None))
                kf_score_list.append(info.get("kf_score", None))

            cs["live_debug"] = {
                "frame_idx": int(self.frame_idx),
                "object_score_logits": live_obj_logits,
                "object_score_prob": live_obj_probs,
                "object_score_thr": obj_score_thr,
                "reid_ok": current_out["reid_ok"].detach().cpu().tolist()
                if torch.is_tensor(current_out.get("reid_ok", None)) else None,
                "reacquire_mode_per_id": {
                    int(oid): bool(reacquire_map.get(int(oid), False)) for oid in obj_ids
                },
                "any_reacquire": any(bool(reacquire_map.get(int(oid), False)) for oid in obj_ids),
                "good_mem_count": len(good_mem_frames),
                "good_mem_frames": [int(x) for x in good_mem_frames],
                "current_frame_in_good_mem": int(self.frame_idx) in set(int(x) for x in good_mem_frames),
                "reacq_score": reacq_score_list,
                "reacq_parts": reacq_parts_list,
                "best_iou": best_iou_list,
                "kf_score": kf_score_list,
                "bbox_fallback_by_id": {
                    int(oid): val for oid, val in fallback_by_id.items()
                    if isinstance(val, dict) and int(val.get("frame_idx", -1)) == int(self.frame_idx)
                },
            }
        except Exception:
            pass

        self._last_video_res_masks_raw = video_res_masks_raw

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
        Keep rolling windows of GOOD non-conditioning memory frames, PER OBJECT.

        A frame can be good for some objects and bad for others.
        We store the full batched frame entry once in non_cond_frame_outputs,
        and maintain per-object frame lists telling which frames are valid for each object.

        Later, _prepare_memory_conditioned_features() can pick the correct frames
        for the current object only.

        Notes:
        - non_cond_frame_outputs keeps the UNION of all per-object kept frames
        - good_memory_frames remains as a global union (mainly for debug/backward compatibility)
        - good_memory_frames_per_id stores the actual per-object rolling history
        """
        output_dict = self.condition_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]

        # Global debug/backward-compat list
        good_memory_frames = self.condition_state.setdefault("good_memory_frames", [])

        # Per-object memory history
        good_memory_frames_per_id = self.condition_state.setdefault("good_memory_frames_per_id", {})

        obj_ids = list(self.condition_state.get("obj_ids", []))

        obj_score = current_out.get("object_score_logits", None)
        iou_score = current_out.get("best_iou_score", None)
        kf_score = current_out.get("kf_score", None)
        reid_sim = current_out.get("reid_sim", None)
        reacquire = current_out.get("reacquire", None)

        def _to_1d_list(x):
            if not torch.is_tensor(x):
                return None
            x = x.detach().float().reshape(-1)
            out = []
            for v in x:
                val = float(v.item())
                out.append(val if np.isfinite(val) else None)
            return out

        def _to_1d_bool_list(x):
            if not torch.is_tensor(x):
                return None
            x = x.detach().cpu().reshape(-1)
            out = []
            for v in x:
                try:
                    out.append(bool(v.item()))
                except Exception:
                    out.append(None)
            return out

        obj_list = _to_1d_list(obj_score)
        iou_list = _to_1d_list(iou_score)
        kf_list = _to_1d_list(kf_score)
        reid_sim_list = _to_1d_list(reid_sim)
        reacq_list = _to_1d_bool_list(reacquire)

        # infer batch size from any available tensor
        batch_size = 0
        for arr in (obj_list, iou_list, kf_list, reid_sim_list, reacq_list):
            if arr is not None:
                batch_size = max(batch_size, len(arr))

        if batch_size == 0:
            return

        # safe fallback if obj_ids length is unexpected
        if len(obj_ids) != batch_size:
            obj_ids = list(range(batch_size))

        obj_thr = float(getattr(self, "memory_bank_obj_score_threshold", 0.0))
        iou_thr = float(getattr(self, "memory_bank_iou_threshold", 0.0))
        kf_thr = float(getattr(self, "memory_bank_kf_score_threshold", 0.0))
        reid_thr_mem = float(getattr(self, "memory_bank_reid_threshold", float("-inf")))

        # For SAM2/SAMURAI with num_maskmem=7, there are 6 non-cond temporal slots
        max_noncond_keep = max(1, int(getattr(self, "num_maskmem", 7)) - 1)

        per_obj_accept = []

        for i in range(batch_size):
            obj_v = obj_list[i] if (obj_list is not None and i < len(obj_list)) else None
            iou_v = iou_list[i] if (iou_list is not None and i < len(iou_list)) else None
            kf_v  = kf_list[i] if (kf_list is not None and i < len(kf_list)) else None
            sim_v = reid_sim_list[i] if (reid_sim_list is not None and i < len(reid_sim_list)) else None
            reacq_v = reacq_list[i] if (reacq_list is not None and i < len(reacq_list)) else False

            obj_ok = (obj_v is not None) and (obj_v > obj_thr)
            iou_ok = (iou_v is not None) and (iou_v > iou_thr)
            kf_ok = True if (kf_v is None) else (kf_v > kf_thr)

            # New memory-specific ReID rule:
            # - if sim missing, do not block by itself
            # - if sim present, require sim > memory_bank_reid_threshold
            reid_ok_mem = True if (sim_v is None) else (sim_v > reid_thr_mem)

            # Never write memory during reacquisition mode
            not_reacquiring = not bool(reacq_v)

            accepted_i = obj_ok and iou_ok and kf_ok and reid_ok_mem and not_reacquiring
            per_obj_accept.append(bool(accepted_i))

        # Save debug mask
        current_out["memory_accept_mask"] = torch.tensor(
            per_obj_accept, dtype=torch.bool
        )

        # If nobody accepts this frame, do not store it
        if not any(per_obj_accept):
            return

        # Store full batched frame once
        non_cond_frame_outputs[int(frame_idx)] = current_out

        # Update each object's own rolling history
        for i, accepted_i in enumerate(per_obj_accept):
            if not accepted_i:
                continue

            oid = int(obj_ids[i])
            hist = good_memory_frames_per_id.setdefault(oid, [])

            hist.append(int(frame_idx))

            # dedup while preserving order
            dedup = []
            seen = set()
            for f in hist:
                f = int(f)
                if f not in seen:
                    dedup.append(f)
                    seen.add(f)

            # keep only last max_noncond_keep for this object
            if len(dedup) > max_noncond_keep:
                dedup = dedup[-max_noncond_keep:]

            good_memory_frames_per_id[oid] = dedup

        # Rebuild global union for debug/backward compatibility
        union_keep = set()
        for hist in good_memory_frames_per_id.values():
            for f in hist:
                union_keep.add(int(f))

        union_keep_sorted = sorted(union_keep)
        good_memory_frames.clear()
        good_memory_frames.extend(union_keep_sorted)

        # Remove globally stored frames that are no longer needed by any object
        remove_keys = [
            k for k in list(non_cond_frame_outputs.keys())
            if int(k) not in union_keep
        ]
        for k in remove_keys:
            non_cond_frame_outputs.pop(k, None)


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

    """
    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        
        #Identical to the corresponding method in the parent (SAM2VideoPredictor), but
        #cloning the outputs of prompt_encoder and mask_decoder to enable compilation.
        
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
    """

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
