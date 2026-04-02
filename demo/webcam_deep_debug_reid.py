#!/usr/bin/env python3
"""
webcam_deep_debug_reid.py

Webcam debug harness to "see inside" the SAMURAI/SAM2 pipeline.

This version assumes ReID is built INTO the model/predictor:
- predictor already creates and owns the ReID embedder
- predictor._init_state() already initializes reid-related condition_state keys
- sam2_base.py track_step() calls internal ReID update/gating
- predictor.track() stores current_out["reid_ok"] into memory (mem_out["reid_ok"])
- _prepare_memory_conditioned_features() filters memory frames using prev_out["reid_ok"]

We keep:
- YOLO proposals + prompting
- deep-debug dumps (memory attn frames, multimask candidates, ptr drift, stored memory masks)
- HUD lines for ReID using condition_state["reid_last"]
- HUD lines for object score / memory status / reacquire mode / good memory queue
- export ReID gallery entries per object

Run:
  python .\demo\webcam_deep_debug_reid.py --reid_thr 0.80 --reid_print
  python .\demo\webcam_deep_debug_reid.py --hide_hud
"""

print("=== webcam_deep_debug_reid.py VERSION: INTERNAL-REID-004 (MODEL-OWNED-REID) ===", flush=True)

import sys
import time
import json
import math
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'", category=UserWarning)

# repo root (parent of /demo)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_camera_predictor


# ---------------- paths ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT2 = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "demo") else Path.cwd()
CKPT_PATH = (REPO_ROOT2 / "checkpoints" / "sam2.1_hiera_small.pt").resolve()
CFG_PATH  = (REPO_ROOT2 / "sam2" / "configs" / "samurai" / "sam2.1_hiera_s.yaml").resolve()


# ---------------- utils ----------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def yolo_person_bboxes(bgr_frame, model, conf_thres=0.25):
    """Returns list of (x1,y1,x2,y2,conf) for class=person, sorted by conf desc."""
    if bgr_frame is None:
        return []
    res = model(bgr_frame, verbose=False, conf=conf_thres)[0]
    out = []
    if res.boxes is None:
        return out
    for det in res.boxes:
        if int(det.cls) == 0:  # person
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item()) if det.conf is not None else 0.0
            out.append((x1, y1, x2, y2, conf))
    out.sort(key=lambda t: t[4], reverse=True)
    return out

def _to_id_list(out_obj_ids):
    if out_obj_ids is None:
        return []
    if isinstance(out_obj_ids, (list, tuple)):
        return [int(x) for x in out_obj_ids]
    if torch.is_tensor(out_obj_ids):
        return [int(x) for x in out_obj_ids.detach().reshape(-1).tolist()]
    return [int(out_obj_ids)]

def _id_to_hue(obj_id: int) -> int:
    return int((37 * int(obj_id) + 61) % 180)

def draw_mask_overlay(rgb_frame, out_obj_ids, out_mask_logits, alpha=0.5):
    """Overlay segmentation masks on rgb_frame with deterministic per-ID colors."""
    if rgb_frame is None or out_mask_logits is None:
        return rgb_frame

    ids = _to_id_list(out_obj_ids)

    if isinstance(out_mask_logits, (list, tuple)):
        M = len(out_mask_logits)
        get_logits = lambda i: out_mask_logits[i]
    elif torch.is_tensor(out_mask_logits):
        M = int(out_mask_logits.shape[0]) if out_mask_logits.ndim >= 1 else 0
        get_logits = lambda i: out_mask_logits[i]
    else:
        return rgb_frame

    n = max(0, min(len(ids), M))
    if n == 0:
        return rgb_frame

    h, w = rgb_frame.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = 0

    for i in range(n):
        logits_i = get_logits(i)
        if logits_i is None or not torch.is_tensor(logits_i):
            continue

        if logits_i.ndim == 3:
            m = (logits_i[0] > 0)
        elif logits_i.ndim == 2:
            m = (logits_i > 0)
        else:
            continue

        m = m.detach().cpu().numpy().astype(bool)
        hsv[m, 0] = _id_to_hue(ids[i])
        hsv[m, 2] = 255

    overlay_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, overlay_rgb, float(alpha), 0.0)

def overlay_single_mask(rgb: np.ndarray, mask_hw_bool: np.ndarray, alpha=0.5, hue=90):
    """Overlay one binary mask on an RGB image (mask_hw_bool is HxW bool)."""
    if rgb is None or mask_hw_bool is None:
        return rgb
    H, W = rgb.shape[:2]
    if mask_hw_bool.shape[:2] != (H, W):
        return rgb

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = 0
    hsv[mask_hw_bool, 0] = int(hue) % 180
    hsv[mask_hw_bool, 2] = 255

    overlay_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb, 1.0, overlay_rgb, float(alpha), 0.0)

def get_obj_mask_from_mem_entry(entry: dict, obj_idx: int):
    """
    entry is one output_dict["non_cond_frame_outputs"][t] dict.
    Returns mask tensor [1, h, w] (logits) if present, else None.
    """
    if not isinstance(entry, dict):
        return None
    pm = entry.get("pred_masks", None)
    if not torch.is_tensor(pm):
        return None
    if pm.ndim == 4:
        if obj_idx >= pm.shape[0]:
            return None
        return pm[obj_idx, 0:1]   # [1,h,w]
    if pm.ndim == 3:
        if obj_idx >= pm.shape[0]:
            return None
        return pm[obj_idx:obj_idx+1]  # [1,h,w]
    return None

def save_binary_masks(out_dir: Path, out_obj_ids, out_mask_logits):
    """Save per-object binary masks. Uses threshold logit>0."""
    ids = _to_id_list(out_obj_ids)
    if out_mask_logits is None:
        return []

    saved = []

    if torch.is_tensor(out_mask_logits):
        logits = out_mask_logits
        if logits.ndim == 4:
            logits = logits[:, 0]
        elif logits.ndim != 3:
            return []
        N = int(logits.shape[0])
        n = min(N, len(ids))
        for i in range(n):
            m = (logits[i] > 0).detach().cpu().numpy().astype(np.uint8) * 255
            fp = out_dir / f"mask_id{ids[i]}.png"
            cv2.imwrite(str(fp), m)
            saved.append(fp.name)
        return saved

    if isinstance(out_mask_logits, (list, tuple)):
        n = min(len(out_mask_logits), len(ids))
        for i in range(n):
            t = out_mask_logits[i]
            if not torch.is_tensor(t):
                continue
            if t.ndim == 3:
                m = (t[0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
            elif t.ndim == 2:
                m = (t > 0).detach().cpu().numpy().astype(np.uint8) * 255
            else:
                continue
            fp = out_dir / f"mask_id{ids[i]}.png"
            cv2.imwrite(str(fp), m)
            saved.append(fp.name)
        return saved

    return []

def closest_debug_key(dbg: Dict[int, Any], target: int) -> Optional[int]:
    if not isinstance(dbg, dict) or len(dbg) == 0:
        return None
    keys = sorted([int(k) for k in dbg.keys()])
    if target in dbg:
        return target
    best = None
    best_dist = 10**18
    for k in keys:
        d = abs(k - target)
        if d < best_dist:
            best_dist = d
            best = k
    return best

def _json_safe(x):
    try:
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): _json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_json_safe(v) for v in x]
        json.dumps(x)
        return x
    except Exception:
        return str(x)

def _draw_bbox(bgr, bb_xyxy, color, label=None):
    if bb_xyxy is None:
        return bgr
    try:
        x1, y1, x2, y2 = bb_xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(bgr, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    except Exception:
        pass
    return bgr


# ---------------- OBJ_PTR debug helpers ----------------
def _get_ptr_from_out(out: dict, obj_batch_index: int):
    if not isinstance(out, dict):
        return None
    ptr = out.get("obj_ptr", None)
    if not torch.is_tensor(ptr):
        return None
    if ptr.ndim == 2:
        if obj_batch_index >= ptr.shape[0]:
            return None
        return ptr[obj_batch_index].detach().cpu()
    if ptr.ndim == 1:
        return ptr.detach().cpu()
    return None

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None:
        return float("nan")
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    an = torch.norm(a)
    bn = torch.norm(b)
    if float(an.item()) == 0.0 or float(bn.item()) == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / (an.item() * bn.item()))

def _ptr_to_list(ptr: torch.Tensor, max_len=32):
    if ptr is None or (not torch.is_tensor(ptr)):
        return None
    v = ptr.detach().float().reshape(-1).cpu().tolist()
    return {"dim": int(len(v)), "head": v[:max_len]}

def _torch_to_list_safe(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x

def _float_or_none(x):
    try:
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.detach().cpu().reshape(-1)
            if x.numel() == 0:
                return None
            return float(x[0].item())
        return float(x)
    except Exception:
        return None

def _sigmoid_float(x):
    try:
        v = _float_or_none(x)
        if v is None:
            return None
        return 1.0 / (1.0 + math.exp(-v))
    except Exception:
        return None

def _extract_entry_debug_info(predictor, pf_now: int):
    """
    Read debug info for HUD.

    Priority:
    1) live_debug written by predictor.track() for the CURRENT frame
    2) fallback to stored non-cond entry if present
    """
    out = {
        "object_score_logits": None,
        "object_score_prob": None,
        "object_score_thr": None,
        "reid_ok": None,
        "good_mem_count": 0,
        "good_mem_frames": [],
        "reacquire_mode_per_id": {},
        "any_reacquire": False,
        "current_frame_in_good_mem": False,

        "reacq_score": None,
        "reacq_parts": None,
        "best_iou": None,
        "kf_score": None,
    }

    try:
        cs = getattr(predictor, "condition_state", None)
        if not isinstance(cs, dict):
            return out

        live = cs.get("live_debug", None)
        if isinstance(live, dict) and int(live.get("frame_idx", -999999)) == int(pf_now):
            out["object_score_logits"] = live.get("object_score_logits", None)
            out["object_score_prob"] = live.get("object_score_prob", None)
            out["object_score_thr"] = live.get("object_score_thr", None)
            out["reid_ok"] = live.get("reid_ok", None)
            out["good_mem_count"] = int(live.get("good_mem_count", 0))
            out["good_mem_frames"] = [int(x) for x in live.get("good_mem_frames", [])]
            out["reacquire_mode_per_id"] = {
                int(k): bool(v) for k, v in live.get("reacquire_mode_per_id", {}).items()
            }
            out["any_reacquire"] = bool(live.get("any_reacquire", False))
            out["current_frame_in_good_mem"] = bool(live.get("current_frame_in_good_mem", False))

            out["reacq_score"] = live.get("reacq_score", None)
            out["reacq_parts"] = live.get("reacq_parts", None)
            out["best_iou"] = live.get("best_iou", None)
            out["kf_score"] = live.get("kf_score", None)

            return out

        try:
            thr = float(getattr(predictor, "min_obj_score_logits", None))
            out["object_score_thr"] = thr
        except Exception:
            out["object_score_thr"] = None

        gm = cs.get("good_memory_frames", None)
        if gm is not None:
            try:
                gm_list = [int(x) for x in list(gm)]
            except Exception:
                gm_list = []
            out["good_mem_frames"] = gm_list
            out["good_mem_count"] = len(gm_list)
            out["current_frame_in_good_mem"] = int(pf_now) in set(gm_list)

        reacq_map = cs.get("reacquire_mode_per_id", {})
        if isinstance(reacq_map, dict):
            out["reacquire_mode_per_id"] = {int(k): bool(v) for k, v in reacq_map.items()}
            out["any_reacquire"] = any(out["reacquire_mode_per_id"].values())

        od = cs.get("output_dict", {})
        entry = od.get("non_cond_frame_outputs", {}).get(int(pf_now), None)
        if not isinstance(entry, dict):
            return out

        osl = entry.get("object_score_logits", None)
        if torch.is_tensor(osl):
            out["object_score_logits"] = osl.detach().cpu().reshape(-1).tolist()
            out["object_score_prob"] = [_sigmoid_float(v) for v in out["object_score_logits"]]

        rok = entry.get("reid_ok", None)
        if torch.is_tensor(rok):
            out["reid_ok"] = rok.detach().cpu().reshape(-1).tolist()

    except Exception:
        pass

    return out

def _get_gallery_meta_map(cs: dict):
    if not isinstance(cs, dict):
        return {}
    meta = cs.get("reid_gallery_meta", {})
    if not isinstance(meta, dict):
        return {}
    return meta

def _get_gallery_entries_for_obj(cs: dict, obj_id: int):
    meta_map = _get_gallery_meta_map(cs)
    entries = meta_map.get(int(obj_id), [])
    return entries if isinstance(entries, list) else []

def _resolve_gallery_frame_rgb(frame_idx: int, source: str, condframe_to_rgb: dict, noncond_ring: dict):
    """
    Resolve the RGB image corresponding to a gallery entry.

    source == "prompt" -> usually conditioning timeline
    source == "track"  -> non-cond / global frame timeline
    """
    frame_idx = int(frame_idx)

    if str(source) == "prompt":
        img = condframe_to_rgb.get(frame_idx, None)
        if img is not None:
            return img

    img = noncond_ring.get(frame_idx, None)
    if img is not None:
        return img

    img = condframe_to_rgb.get(frame_idx, None)
    return img

def export_reid_gallery(case_dir: Path, cs: dict, condframe_to_rgb: dict, noncond_ring: dict):
    """
    Export gallery crops and full-frame visualizations for every object's gallery.

    Also annotates entries that were used for reacquisition / promoted because of reacquisition,
    so this is easy to debug from the saved images.
    """
    out = {"available": False, "objects": {}}

    if not isinstance(cs, dict):
        return out

    obj_ids = cs.get("obj_ids", [])
    if not isinstance(obj_ids, list):
        return out

    gallery_root = case_dir / "reid_gallery"
    safe_mkdir(gallery_root)

    meta_map = _get_gallery_meta_map(cs)
    if not isinstance(meta_map, dict):
        return out

    any_saved = False

    for obj_id in [int(x) for x in obj_ids]:
        obj_dir = gallery_root / f"obj_{obj_id}"
        safe_mkdir(obj_dir)

        entries = _get_gallery_entries_for_obj(cs, obj_id)
        obj_summary = {
            "obj_id": int(obj_id),
            "num_entries": len(entries),
            "entries": [],
        }

        for gi, meta in enumerate(entries):
            if not isinstance(meta, dict):
                continue

            frame_idx = int(meta.get("frame_idx", -1))
            bbox = meta.get("bbox", None)
            source = str(meta.get("source", "unknown"))

            is_anchor = bool(meta.get("is_anchor", False))
            quality_score = meta.get("quality_score", None)

            used_for_reacquire = bool(meta.get("used_for_reacquire", False))
            promoted_by_reacquire = bool(meta.get("promoted_by_reacquire", False))
            reacquire_use_count = int(meta.get("reacquire_use_count", 0))
            last_reacquire_frame_idx = meta.get("last_reacquire_frame_idx", None)

            entry_summary = {
                "gallery_index": int(gi),
                "frame_idx": int(frame_idx),
                "bbox": bbox,
                "source": source,
                "is_anchor": is_anchor,
                "quality_score": quality_score,
                "used_for_reacquire": used_for_reacquire,
                "promoted_by_reacquire": promoted_by_reacquire,
                "reacquire_use_count": reacquire_use_count,
                "last_reacquire_frame_idx": last_reacquire_frame_idx,
                "have_frame": False,
                "saved_crop": None,
                "saved_frame": None,
            }

            rgb = _resolve_gallery_frame_rgb(frame_idx, source, condframe_to_rgb, noncond_ring)
            if rgb is None:
                obj_summary["entries"].append(entry_summary)
                continue

            entry_summary["have_frame"] = True

            tag_parts = []

            if is_anchor:
                tag_parts.append("ANCHOR")

            if used_for_reacquire:
                if reacquire_use_count > 0:
                    tag_parts.append(f"USED_FOR_REACQ x{reacquire_use_count}")
                else:
                    tag_parts.append("USED_FOR_REACQ")

            if promoted_by_reacquire:
                tag_parts.append("PROMOTED_BY_REACQ")

            if last_reacquire_frame_idx is not None:
                try:
                    tag_parts.append(f"last_reacq_f={int(last_reacquire_frame_idx)}")
                except Exception:
                    pass

            label = f"gallery#{gi} {source} f={frame_idx}"
            if len(tag_parts) > 0:
                label += " [" + " | ".join(tag_parts) + "]"

            full_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            full_bgr = _draw_bbox(full_bgr, bbox, (0, 255, 255), label)

            full_name = f"gallery_full_g{gi:02d}_f{frame_idx:06d}_{source}.png"
            cv2.imwrite(str(obj_dir / full_name), full_bgr)
            entry_summary["saved_frame"] = full_name

            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    H, W = rgb.shape[:2]
                    x1 = max(0, min(W - 1, x1))
                    x2 = max(0, min(W - 1, x2))
                    y1 = max(0, min(H - 1, y1))
                    y2 = max(0, min(H - 1, y2))

                    if x2 >= x1 and y2 >= y1:
                        crop = rgb[y1:y2 + 1, x1:x2 + 1].copy()
                        if crop.size > 0:
                            crop_name = f"gallery_crop_g{gi:02d}_f{frame_idx:06d}_{source}.png"
                            cv2.imwrite(str(obj_dir / crop_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                            entry_summary["saved_crop"] = crop_name
                            any_saved = True
                except Exception:
                    pass

            obj_summary["entries"].append(entry_summary)

        with (obj_dir / "gallery_summary.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(obj_summary), f, indent=2)

        out["objects"][str(obj_id)] = obj_summary

    out["available"] = any_saved or (len(out["objects"]) > 0)
    return out

# ---------------- candidates dump ----------------
def dump_multimask_candidates(case_dir: Path, rgb: np.ndarray, predictor) -> Dict[str, Any]:
    out = {"available": False}

    cs = getattr(predictor, "condition_state", None)
    if not isinstance(cs, dict):
        return out

    dbg = cs.get("debug_last", None)
    if not isinstance(dbg, dict):
        return out

    cand_root = case_dir / "candidates"
    safe_mkdir(cand_root)

    H, W = rgb.shape[:2]

    def _infer_selected_index(sel):
        if sel is None:
            return 0
        if torch.is_tensor(sel):
            sel = sel.detach().cpu().reshape(-1).tolist()
        if isinstance(sel, list):
            return int(sel[0]) if sel else 0
        try:
            return int(sel)
        except Exception:
            return 0

    def dump_one(obj_dir: Path, entry: Dict[str, Any], obj_index: int) -> Dict[str, Any]:
        safe_mkdir(obj_dir)

        lr = entry.get("low_res_multimasks", None)   # Tensor [B,M,h,w]
        ious = entry.get("ious", None)               # Tensor [B,M]
        if not torch.is_tensor(lr) or not torch.is_tensor(ious):
            return {"available": False}

        lr_f = lr.detach().float().cpu()
        ious_cpu = ious.detach().float().cpu()

        B = int(entry.get("B", lr_f.shape[0]))
        M = int(entry.get("M", lr_f.shape[1] if lr_f.ndim >= 2 else 1))
        sel = entry.get("selected_mask_index", None)

        kf_ious = entry.get("kf_ious", None)
        combined = entry.get("combined", None)
        obj_logit = entry.get("object_score_logits", None)
        obj_prob = entry.get("object_score_prob", None)
        kf_pred_bbox = entry.get("kf_pred_bbox_xyxy", None)
        cand_bboxes = entry.get("cand_bboxes_xyxy", None)
        cand_areas = entry.get("cand_mask_areas", None)
        sel_bbox = entry.get("selected_bbox_xyxy", None)
        prompt_debug = entry.get("prompt_debug", None)

        bm = int(lr_f.shape[0] * lr_f.shape[1])
        up = F.interpolate(
            lr_f.reshape(bm, 1, lr_f.shape[2], lr_f.shape[3]),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).reshape(lr_f.shape[0], lr_f.shape[1], H, W)

        saved_files = []
        for b in range(int(up.shape[0])):
            for m in range(int(up.shape[1])):
                mask = (up[b, m] > 0).numpy().astype(np.uint8) * 255
                fp = obj_dir / f"cand_mask_b{b}_m{m}.png"
                cv2.imwrite(str(fp), mask)

                overlay = rgb.copy()
                hsv = np.zeros((H, W, 3), dtype=np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = 0
                hsv[mask > 0, 0] = int((37 * (m + 1) + 61) % 180)
                hsv[mask > 0, 2] = 255
                overlay_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                vis = cv2.addWeighted(overlay, 1.0, overlay_rgb, 0.5, 0.0)
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

                vis_bgr = _draw_bbox(vis_bgr, kf_pred_bbox, (0, 255, 255), "KF pred")
                if isinstance(cand_bboxes, list) and m < len(cand_bboxes):
                    vis_bgr = _draw_bbox(vis_bgr, cand_bboxes[m], (255, 255, 255), f"cand m{m}")
                if sel_bbox is not None and m == _infer_selected_index(sel):
                    vis_bgr = _draw_bbox(vis_bgr, sel_bbox, (0, 0, 255), "SELECTED")

                txt = []
                try:
                    txt.append(f"IoU={float(ious_cpu[b, m].item()):.3f}")
                except Exception:
                    pass

                if torch.is_tensor(kf_ious):
                    kf_flat = kf_ious.detach().float().cpu().reshape(-1)
                    if m < int(kf_flat.numel()):
                        txt.append(f"KF={float(kf_flat[m].item()):.3f}")

                if torch.is_tensor(combined):
                    comb_cpu = combined.detach().float().cpu()
                    if comb_cpu.ndim == 2 and m < int(comb_cpu.shape[1]):
                        txt.append(f"Comb={float(comb_cpu[b, m].item()):.3f}")

                if isinstance(cand_areas, list) and m < len(cand_areas):
                    txt.append(f"Area={int(cand_areas[m])}")

                cv2.putText(
                    vis_bgr,
                    f"obj{obj_index} b={b} m={m} | " + " | ".join(txt),
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                fp2 = obj_dir / f"cand_overlay_b{b}_m{m}.png"
                cv2.imwrite(str(fp2), vis_bgr)
                saved_files.append([fp.name, fp2.name])

        summary = {
            "available": True,
            "obj_index": int(obj_index),
            "multimask_output": bool(entry.get("multimask_output", False)),
            "samurai_mode": bool(entry.get("samurai_mode", False)),
            "B": B,
            "M": M,

            "selected_mask_index": _torch_to_list_safe(sel),
            "selected_iou": _float_or_none(entry.get("selected_iou", None)),
            "selected_kf_iou": _float_or_none(entry.get("selected_kf_iou", None)),
            "selected_combined": _float_or_none(entry.get("selected_combined", None)),
            "kf_score_weight": _float_or_none(entry.get("kf_score_weight", None)),

            "ious": _torch_to_list_safe(ious_cpu),
            "kf_ious": _torch_to_list_safe(kf_ious),
            "combined": _torch_to_list_safe(combined),

            "object_score_logits": _torch_to_list_safe(obj_logit),
            "object_score_prob": _torch_to_list_safe(obj_prob),
            "is_obj_appearing": _torch_to_list_safe(entry.get("is_obj_appearing", None)),

            "kf_pred_bbox_xyxy": _json_safe(kf_pred_bbox),
            "cand_bboxes_xyxy": _json_safe(cand_bboxes),
            "cand_mask_areas": _json_safe(cand_areas),
            "selected_bbox_xyxy": _json_safe(sel_bbox),

            "prompt_debug": _json_safe(prompt_debug),

            "saved_files": saved_files,
        }

        with (obj_dir / "candidates_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    per_obj_list = dbg.get("per_obj", None)

    summaries = []
    if isinstance(per_obj_list, list) and len(per_obj_list) > 0:
        for k, entry in enumerate(per_obj_list):
            if not isinstance(entry, dict):
                continue
            obj_dir = cand_root / f"obj_{k:02d}"
            summaries.append(dump_one(obj_dir, entry, obj_index=k))

        overview = {
            "available": True,
            "format": "per_obj_list",
            "num_objects_dumped": len(summaries),
            "objects": summaries,
            "note": "Each obj_* folder corresponds to one _forward_sam_heads call (per object in your pipeline).",
        }
        with (cand_root / "debug_last_overview.json").open("w", encoding="utf-8") as f:
            json.dump(overview, f, indent=2)

        out["available"] = True
        out["format"] = "per_obj_list"
        out["num_objects_dumped"] = len(summaries)
        return out

    obj_dir = cand_root / "obj_00"
    s = dump_one(obj_dir, dbg, obj_index=0)
    overview = {
        "available": bool(s.get("available", False)),
        "format": "flat_debug_last",
        "objects": [s],
    }
    with (cand_root / "debug_last_overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    out["available"] = bool(s.get("available", False))
    out["format"] = "flat_debug_last"
    out["num_objects_dumped"] = 1 if out["available"] else 0
    return out


# ---------------- main ----------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--out_root", type=str, default=str(REPO_ROOT2 / "debug_cases_webcam"))
    ap.add_argument("--ring_size", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.5)

    # ReID is always enabled inside the model now; this only overrides threshold for experiments
    ap.add_argument("--reid_thr", type=float, default=None)
    ap.add_argument("--reid_print", action="store_true")

    # NEW: hide on-screen HUD/debug text while keeping masks and controls
    ap.add_argument("--hide_hud", action="store_true")

    args = ap.parse_args()

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n  {CKPT_PATH}")
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found:\n  {CFG_PATH}")

    out_root = Path(args.out_root).resolve()
    safe_mkdir(out_root)

    print("[init] Building SAM2 camera predictor...", flush=True)
    predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))

    def _sync_reid_threshold():
        if args.reid_thr is None:
            return
        try:
            predictor.reid_thr = float(args.reid_thr)
        except Exception:
            pass
        cs = getattr(predictor, "condition_state", None)
        if isinstance(cs, dict):
            cs["reid_thr"] = float(args.reid_thr)

    _sync_reid_threshold()

    cs0 = getattr(predictor, "condition_state", {})
    try:
        thr0 = float(cs0.get("reid_thr", getattr(predictor, "reid_thr", float("nan"))))
    except Exception:
        thr0 = getattr(predictor, "reid_thr", None)

    print(f"[init] Internal ReID ON (thr={thr0})", flush=True)

    print("[init] Loading YOLO (yolov8s.pt)...", flush=True)
    yolo_model = YOLO("yolov8s.pt")

    state = {
        "first_frame_loaded": False,
        "tracking": False,
        "injecting": False,
        "yolo_enabled": True,
        "yolo_conf": float(args.yolo_conf),
        "cands": [],
        "selected_idx": 0,
        "last_rgb": None,
        "next_obj_id": 1,
        "added_obj_ids": [],
        "out_obj_ids": None,
        "out_mask_logits": None,
    }

    condframe_to_rgb: Dict[int, np.ndarray] = {}
    noncond_ring: Dict[int, np.ndarray] = {}
    noncond_keys = deque(maxlen=int(args.ring_size))

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    win = "SAMURAI deep debug (A add | T track | D dump | arrows select | Y yolo | +/- conf | R reset | Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _ring_store(global_fidx: int, rgb_img: np.ndarray):
        noncond_ring[int(global_fidx)] = rgb_img.copy()
        noncond_keys.append(int(global_fidx))
        while len(noncond_keys) > noncond_keys.maxlen:
            old = noncond_keys.popleft()
            noncond_ring.pop(int(old), None)

    def reset_all():
        nonlocal predictor, condframe_to_rgb, noncond_ring, noncond_keys
        print("[reset] Rebuilding predictor and clearing state...", flush=True)
        predictor = build_sam2_camera_predictor(str(CFG_PATH), str(CKPT_PATH))
        _sync_reid_threshold()

        state.update({
            "first_frame_loaded": False,
            "tracking": False,
            "injecting": False,
            "cands": [],
            "selected_idx": 0,
            "last_rgb": None,
            "next_obj_id": 1,
            "added_obj_ids": [],
            "out_obj_ids": None,
            "out_mask_logits": None,
        })
        condframe_to_rgb = {}
        noncond_ring = {}
        noncond_keys = deque(maxlen=int(args.ring_size))

    def add_prompt_from_selected():
        if not state["cands"]:
            print("[add] No YOLO candidates available.", flush=True)
            return
        if state["last_rgb"] is None:
            print("[add] No frame yet.", flush=True)
            return

        idx = clamp(state["selected_idx"], 0, len(state["cands"]) - 1)
        x1, y1, x2, y2, conf = state["cands"][idx]
        bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

        obj_id = int(state["next_obj_id"])

        if not state["tracking"]:
            if not state["first_frame_loaded"]:
                predictor.load_first_frame(state["last_rgb"])
                _sync_reid_threshold()
                state["first_frame_loaded"] = True
                condframe_to_rgb[0] = state["last_rgb"].copy()

            try:
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=obj_id, bbox=bbox)
                cs = predictor.condition_state
                print("reid_ref keys after add:", list(cs.get("reid_ref", {}).keys()))
                print("reid_last after add:", cs.get("reid_last", {}))
                state["out_obj_ids"] = out_obj_ids
                state["out_mask_logits"] = out_mask_logits
                state["added_obj_ids"].append(obj_id)
                state["next_obj_id"] += 1
                cs.setdefault("reacquire_mode_per_id", {})
                cs["reacquire_mode_per_id"][int(obj_id)] = False
                print(f"[add] Added object #{obj_id} (conf={conf:.2f}). Added so far: {state['added_obj_ids']}", flush=True)
            except Exception as e:
                print(f"[add] add_new_prompt failed: {repr(e)}", flush=True)
            return

        try:
            state["injecting"] = True
            predictor.add_conditioning_frame(state["last_rgb"])
            _sync_reid_threshold()
            try:
                cs = predictor.condition_state
                cond_idx = int(len(cs.get("images", [])) - 1)
                if cond_idx >= 0:
                    condframe_to_rgb[cond_idx] = state["last_rgb"].copy()
            except Exception:
                pass

            frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_prompt_during_track(
                bbox=bbox,
                if_new_target=True,
                obj_id=obj_id,
                labels=None,
                clear_old_points=True,
            )
            state["out_obj_ids"] = out_obj_ids
            state["out_mask_logits"] = out_mask_logits
            state["added_obj_ids"].append(obj_id)
            state["next_obj_id"] += 1
            cs.setdefault("reacquire_mode_per_id", {})
            cs["reacquire_mode_per_id"][int(obj_id)] = False
            print(f"[add] Late-joined object #{obj_id} at predictor frame_idx={frame_idx} (conf={conf:.2f}).", flush=True)
        except Exception as e:
            print(f"[add] Late-join failed: {repr(e)}", flush=True)
        finally:
            state["injecting"] = False

    def start_tracking():
        if not state["added_obj_ids"]:
            print("[track] Add at least one person first (press A).", flush=True)
            return
        state["tracking"] = True
        print(f"[track] Tracking started. Objects: {state['added_obj_ids']}", flush=True)

    # ---------------- dump_case ----------------
    def dump_case(tag: str = ""):
        cs = getattr(predictor, "condition_state", {}) if predictor is not None else {}
        pf = getattr(predictor, "frame_idx", None)
        pf = int(pf) if pf is not None else -1

        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"case_{ts}_pf{pf:06d}"
        if tag:
            name += f"_{tag}"
        case_dir = out_root / name
        safe_mkdir(case_dir)
        safe_mkdir(case_dir / "attn_memory_frames")
        safe_mkdir(case_dir / "masks")

        rgb = state["last_rgb"]
        if rgb is not None:
            cv2.imwrite(str(case_dir / "00_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        overlay = rgb.copy() if rgb is not None else None
        if overlay is not None and state["out_mask_logits"] is not None:
            overlay = draw_mask_overlay(overlay, state["out_obj_ids"], state["out_mask_logits"], alpha=args.alpha)
            cv2.imwrite(str(case_dir / "01_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if rgb is not None:
            saved_masks = save_binary_masks(case_dir / "masks", state["out_obj_ids"], state["out_mask_logits"])
        else:
            saved_masks = []

        cand_overview = None
        if rgb is not None:
            cand_overview = dump_multimask_candidates(case_dir, rgb, predictor)

        od = cs.get("output_dict", {}) if isinstance(cs, dict) else {}
        dbg = od.get("debug_memory_attn", {})
        dbg_key = closest_debug_key(dbg, pf) if isinstance(dbg, dict) else None
        attn_info = dbg.get(dbg_key, None) if (dbg_key is not None and isinstance(dbg, dict)) else None

        obj_ids = cs.get("obj_ids", []) if isinstance(cs, dict) else []
        obj_ids = [int(x) for x in obj_ids] if isinstance(obj_ids, list) else []

        per_obj_attn = attn_info if isinstance(attn_info, dict) else {}

        def save_attn_frame(prefix: str, idx: int, img: Optional[np.ndarray], out_dir: Path):
            if img is None:
                return False
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"{prefix} idx={idx}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(out_dir / f"{prefix}_{idx:06d}.png"), bgr)
            return True

        exported_attn = {}

        for obj_id in obj_ids:
            obj_dbg = per_obj_attn.get(int(obj_id), None)
            if not isinstance(obj_dbg, dict):
                continue

            selected_cond = [int(x) for x in obj_dbg.get("selected_cond_frames", [])]
            selected_noncond = [int(x) for x in obj_dbg.get("selected_noncond_frames", [])]

            obj_attn_dir = case_dir / "attn_memory_frames" / f"obj_{obj_id}"
            safe_mkdir(obj_attn_dir)

            for cidx in selected_cond:
                save_attn_frame("cond", cidx, condframe_to_rgb.get(int(cidx), None), obj_attn_dir)

            for nidx in selected_noncond:
                save_attn_frame("noncond", nidx, noncond_ring.get(int(nidx), None), obj_attn_dir)

            exported_attn[int(obj_id)] = {
                "selected_cond_frames": selected_cond,
                "selected_noncond_frames": selected_noncond,
                "valid_indices_pool": [int(x) for x in obj_dbg.get("valid_indices_pool", [])],
            }

        memmask_exported = False
        try:
            obj_id_to_idx = cs.get("obj_id_to_idx", {}) if isinstance(cs, dict) else {}
            ncfo = od.get("non_cond_frame_outputs", {})

            if isinstance(obj_ids, list) and isinstance(obj_id_to_idx, dict) and isinstance(ncfo, dict):
                memmask_dir = case_dir / "memory_masks_noncond"
                safe_mkdir(memmask_dir)

                for obj_id in obj_ids:
                    obj_dbg = per_obj_attn.get(int(obj_id), None)
                    if not isinstance(obj_dbg, dict):
                        continue

                    selected_noncond = [int(x) for x in obj_dbg.get("selected_noncond_frames", [])]

                    obj_idx = int(obj_id_to_idx.get(obj_id, 0))
                    obj_dir = memmask_dir / f"obj_{obj_id}"
                    safe_mkdir(obj_dir)

                    for nidx in selected_noncond:
                        entry = ncfo.get(int(nidx), None)
                        rgb_mem = noncond_ring.get(int(nidx), None)
                        if entry is None or rgb_mem is None:
                            continue

                        m = get_obj_mask_from_mem_entry(entry, obj_idx)
                        if m is None:
                            continue

                        Hm, Wm = rgb_mem.shape[:2]
                        m_up = F.interpolate(
                            m.detach().float().cpu().unsqueeze(0),  # [1,1,h,w]
                            size=(Hm, Wm),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]

                        m_bin = (m_up.numpy() > 0.0)
                        mask_png = (m_bin.astype(np.uint8) * 255)
                        cv2.imwrite(str(obj_dir / f"memmask_f{int(nidx):06d}_id{obj_id}.png"), mask_png)

                        hue = _id_to_hue(obj_id)
                        ov = overlay_single_mask(rgb_mem.copy(), m_bin, alpha=0.5, hue=hue)
                        cv2.imwrite(
                            str(obj_dir / f"memmask_overlay_f{int(nidx):06d}_id{obj_id}.png"),
                            cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                        )
                        memmask_exported = True

        except Exception as e:
            print("[dump] warning: failed to export noncond memory masks:", repr(e), flush=True)

        gallery_export = {"available": False}
        try:
            gallery_export = export_reid_gallery(
                case_dir=case_dir,
                cs=cs,
                condframe_to_rgb=condframe_to_rgb,
                noncond_ring=noncond_ring,
            )
        except Exception as e:
            print("[dump] warning: failed to export ReID gallery:", repr(e), flush=True)
            gallery_export = {"available": False, "error": repr(e)}

        ptr_debug = {"available": False, "objects": []}
        try:
            obj_id_to_idx = cs.get("obj_id_to_idx", {}) if isinstance(cs, dict) else {}

            cfo = od.get("cond_frame_outputs", {}) if isinstance(od, dict) else {}
            ncfo = od.get("non_cond_frame_outputs", {}) if isinstance(od, dict) else {}

            ref_cond_key = None
            if isinstance(cfo, dict) and 0 in cfo:
                ref_cond_key = 0
            elif isinstance(cfo, dict) and len(cfo) > 0:
                ref_cond_key = sorted([int(k) for k in cfo.keys()])[0]

            cur_nc_key = None
            if isinstance(ncfo, dict):
                if int(pf) in ncfo:
                    cur_nc_key = int(pf)
                else:
                    keys_le = [int(k) for k in ncfo.keys() if int(k) <= int(pf)]
                    if keys_le:
                        cur_nc_key = int(max(keys_le))

            for oid in obj_ids:
                bi = int(obj_id_to_idx.get(oid, 0))
                obj_dbg = per_obj_attn.get(int(oid), {}) if isinstance(per_obj_attn, dict) else {}
                selected_cond = [int(x) for x in obj_dbg.get("selected_cond_frames", [])]
                selected_noncond = [int(x) for x in obj_dbg.get("selected_noncond_frames", [])]

                ref_ptr = None
                cur_ptr = None

                if ref_cond_key is not None and isinstance(cfo, dict):
                    ref_ptr = _get_ptr_from_out(cfo.get(int(ref_cond_key), None), bi)

                if cur_nc_key is not None and isinstance(ncfo, dict):
                    cur_ptr = _get_ptr_from_out(ncfo.get(int(cur_nc_key), None), bi)

                obj_entry = {
                    "obj_id": oid,
                    "obj_batch_index": bi,
                    "ref_from": (f"cond_frame_outputs[{ref_cond_key}]" if ref_cond_key is not None else None),
                    "cur_from": (f"non_cond_frame_outputs[{cur_nc_key}]" if cur_nc_key is not None else None),
                    "ref_sim_current": _cosine_sim(ref_ptr, cur_ptr) if (ref_ptr is not None and cur_ptr is not None) else None,
                    "ref_ptr": _ptr_to_list(ref_ptr),
                    "current_ptr": _ptr_to_list(cur_ptr),
                    "memory_ptrs": [],
                }

                if ref_ptr is not None:
                    if isinstance(cfo, dict):
                        for cidx in selected_cond:
                            out_c = cfo.get(int(cidx), None)
                            p = _get_ptr_from_out(out_c, bi)
                            obj_entry["memory_ptrs"].append({
                                "kind": "cond",
                                "frame": int(cidx),
                                "sim_to_ref": _cosine_sim(ref_ptr, p) if p is not None else None,
                                "ptr": _ptr_to_list(p),
                            })

                    if isinstance(ncfo, dict):
                        for nidx in selected_noncond:
                            out_n = ncfo.get(int(nidx), None)
                            p = _get_ptr_from_out(out_n, bi)
                            obj_entry["memory_ptrs"].append({
                                "kind": "noncond",
                                "frame": int(nidx),
                                "sim_to_ref": _cosine_sim(ref_ptr, p) if p is not None else None,
                                "ptr": _ptr_to_list(p),
                            })

                ptr_debug["objects"].append(obj_entry)

            ptr_debug["available"] = True if ptr_debug["objects"] else False

        except Exception as e:
            ptr_debug["error"] = repr(e)

        with (case_dir / "obj_ptr_debug.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(ptr_debug), f, indent=2)

        reid_debug = {"available": False}
        try:
            if isinstance(cs, dict):
                reid_debug = {
                    "available": True,
                    "thr": float(cs.get("reid_thr", getattr(predictor, "reid_thr", float("nan")))),
                    "have_refs": sorted([int(k) for k in cs.get("reid_ref", {}).keys()]) if isinstance(cs.get("reid_ref", None), dict) else [],
                    "last": cs.get("reid_last", None),
                    "reacquire_mode_per_id": {
                        int(k): bool(v) for k, v in cs.get("reacquire_mode_per_id", {}).items()
                    } if isinstance(cs.get("reacquire_mode_per_id", {}), dict) else {},
                    "any_reacquire": any(bool(v) for v in cs.get("reacquire_mode_per_id", {}).values())
                    if isinstance(cs.get("reacquire_mode_per_id", {}), dict) else False,
                    "good_memory_frames": [int(x) for x in list(cs.get("good_memory_frames", []))]
                    if cs.get("good_memory_frames", None) is not None else [],
                    "debug_memory_attn_per_object": exported_attn,
                    "reid_gallery_meta": cs.get("reid_gallery_meta", {}),
                    "gallery_export": gallery_export,
                }
        except Exception as e:
            reid_debug = {"available": False, "error": repr(e)}

        with (case_dir / "reid_debug.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(reid_debug), f, indent=2)

        noncond_keys_now = []
        try:
            ncfo2 = od.get("non_cond_frame_outputs", {})
            if isinstance(ncfo2, dict):
                noncond_keys_now = sorted([int(k) for k in ncfo2.keys()])
        except Exception:
            pass

        cond_keys_now = []
        try:
            cfo2 = od.get("cond_frame_outputs", {})
            if isinstance(cfo2, dict):
                cond_keys_now = sorted([int(k) for k in cfo2.keys()])
        except Exception:
            pass

        summary = {
            "timestamp": ts,
            "predictor_frame_idx": pf,
            "added_obj_ids": list(state["added_obj_ids"]),
            "obj_ids_condition_state": cs.get("obj_ids", None) if isinstance(cs, dict) else None,
            "obj_id_to_idx": cs.get("obj_id_to_idx", None) if isinstance(cs, dict) else None,
            "tracking": bool(state["tracking"]),
            "yolo_conf": float(state["yolo_conf"]),
            "saved_final_masks": saved_masks,

            "debug_last_candidates_available": bool(cand_overview.get("available", False)) if isinstance(cand_overview, dict) else False,
            "debug_last_format": cand_overview.get("format", None) if isinstance(cand_overview, dict) else None,
            "debug_last_num_objects_dumped": cand_overview.get("num_objects_dumped", None) if isinstance(cand_overview, dict) else None,

            "debug_memory_attn_key_used": int(dbg_key) if dbg_key is not None else None,
            "debug_memory_attn_per_object": exported_attn,

            "cond_frame_outputs_keys": cond_keys_now,
            "noncond_frame_outputs_keys": noncond_keys_now,

            "memory_masks_noncond_exported": bool(memmask_exported),
            "obj_ptr_debug_saved": True,

            "reid_enabled": True,
            "reid_thr": float(cs.get("reid_thr", getattr(predictor, "reid_thr", float("nan")))) if isinstance(cs, dict) else getattr(predictor, "reid_thr", None),
            "reacquire_mode_per_id": {
                int(k): bool(v) for k, v in cs.get("reacquire_mode_per_id", {}).items()
            } if isinstance(cs, dict) and isinstance(cs.get("reacquire_mode_per_id", {}), dict) else {},
            "any_reacquire": any(bool(v) for v in cs.get("reacquire_mode_per_id", {}).values())
            if isinstance(cs, dict) and isinstance(cs.get("reacquire_mode_per_id", {}), dict) else False,
            "good_memory_frames": [int(x) for x in list(cs.get("good_memory_frames", []))]
            if isinstance(cs, dict) and cs.get("good_memory_frames", None) is not None else [],

            "gallery_export_available": bool(gallery_export.get("available", False)),
            "reid_gallery_meta": cs.get("reid_gallery_meta", {}) if isinstance(cs, dict) else {},

            "note": "attn_memory_frames is exported per object under attn_memory_frames/obj_<id>/ ; gallery under reid_gallery/obj_<id>/",
        }

        with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[dump] wrote: {case_dir}", flush=True)

    last_time = time.time()
    fps = 0.0

    print("\nControls:", flush=True)
    print("  Left/Right arrows: select YOLO candidate index", flush=True)
    print("  A: add selected candidate as object", flush=True)
    print("  T: start tracking", flush=True)
    print("  D: dump deep debug case folder", flush=True)
    print("  Y: toggle YOLO overlay", flush=True)
    print("  +/-: adjust YOLO conf", flush=True)
    print("  R: reset", flush=True)
    print("  Q or ESC: quit\n", flush=True)

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("[cam] Frame grab failed.", flush=True)
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state["last_rgb"] = rgb

            out_rgb = rgb

            frame_dbg = {
                "object_score_logits": None,
                "object_score_prob": None,
                "object_score_thr": None,
                "reid_ok": None,
                "good_mem_count": 0,
                "good_mem_frames": [],
                "reacquire_mode_per_id": {},
                "any_reacquire": False,
                "current_frame_in_good_mem": False,
            }

            if state["tracking"] and (not state["injecting"]):
                out_obj_ids, out_mask_logits = predictor.track(rgb)

                cs = predictor.condition_state
                state["out_obj_ids"] = out_obj_ids
                state["out_mask_logits"] = out_mask_logits

                pf_now = int(getattr(predictor, "frame_idx", -1))
                if pf_now >= 0:
                    _ring_store(pf_now, rgb)

                frame_dbg = _extract_entry_debug_info(predictor, pf_now)

                if args.reid_print:
                    print("reid keys:", [k for k in cs.keys() if "reid" in str(k)])
                    print("reid_last:", cs.get("reid_last", None))

                    rl = cs.get("reid_last", {})
                    if isinstance(rl, dict) and rl:
                        k0 = sorted([int(k) for k in rl.keys()])[0]
                        print(f"[reid/internal] last[{k0}] = {rl.get(k0)}", flush=True)

                    if frame_dbg.get("reid_ok", None) is not None:
                        print(f"[reid/internal] mem reid_ok @ pf={pf_now}: {frame_dbg['reid_ok']}", flush=True)

                    if frame_dbg.get("object_score_logits", None) is not None:
                        print(f"[objscore] logits @ pf={pf_now}: {frame_dbg['object_score_logits']}", flush=True)
                        print(f"[objscore] probs  @ pf={pf_now}: {frame_dbg['object_score_prob']}", flush=True)

                    print(
                        f"[memdbg] any_reacquire={frame_dbg.get('any_reacquire')} "
                        f"reacquire_per_id={frame_dbg.get('reacquire_mode_per_id')} "
                        f"good_mem_count={frame_dbg.get('good_mem_count')} "
                        f"current_frame_in_good_mem={frame_dbg.get('current_frame_in_good_mem')}",
                        flush=True
                    )

                out_rgb = draw_mask_overlay(out_rgb, out_obj_ids, out_mask_logits, alpha=args.alpha)

            disp_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            if state["yolo_enabled"]:
                cands = yolo_person_bboxes(disp_bgr, yolo_model, conf_thres=state["yolo_conf"])
                state["cands"] = cands
                if cands:
                    state["selected_idx"] = clamp(state["selected_idx"], 0, len(cands) - 1)
                    for j, (x1, y1, x2, y2, conf) in enumerate(cands):
                        is_sel = (j == state["selected_idx"])
                        color = (0, 255, 0) if is_sel else (0, 200, 255)
                        thick = 3 if is_sel else 1
                        cv2.rectangle(disp_bgr, (x1, y1), (x2, y2), color, thick)
                        cv2.putText(
                            disp_bgr,
                            f"#{j} {conf:.2f}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            pf = int(getattr(predictor, "frame_idx", -1))

            if not args.hide_hud:
                hud = (
                    f"FPS:{fps:4.1f}  "
                    f"pf:{pf}  "
                    f"tracking:{'ON' if state['tracking'] else 'OFF'}  "
                    f"objs:{state['added_obj_ids']}  "
                    f"sel:{state['selected_idx']}  "
                    f"cands:{len(state['cands'])}"
                )
                cv2.putText(
                    disp_bgr,
                    hud,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # --- per-object HUD ---
            if not args.hide_hud:
                try:
                    cs = predictor.condition_state
                    thr = float(cs.get("reid_thr", getattr(predictor, "reid_thr", float("nan"))))
                    rl = cs.get("reid_last", {}) if isinstance(cs, dict) else {}

                    obj_list = cs.get("obj_ids", None) if isinstance(cs, dict) else None
                    if isinstance(obj_list, list) and len(obj_list) > 0:
                        show_ids = [int(x) for x in obj_list]
                    else:
                        show_ids = [int(x) for x in state.get("added_obj_ids", [])]

                    y0 = 60
                    dy = 60

                    logits_list = frame_dbg.get("object_score_logits", None)
                    probs_list = frame_dbg.get("object_score_prob", None)
                    reid_ok_list = frame_dbg.get("reid_ok", None)
                    reacq_map = frame_dbg.get("reacquire_mode_per_id", {}) or {}

                    reacq_score_list = frame_dbg.get("reacq_score", None)
                    reacq_parts_list = frame_dbg.get("reacq_parts", None)
                    best_iou_list = frame_dbg.get("best_iou", None)
                    kf_score_list = frame_dbg.get("kf_score", None)

                    for i, oid in enumerate(show_ids):
                        info = rl.get(int(oid), None) if isinstance(rl, dict) else None

                        sim_txt = "--"
                        acc_txt = "NOINFO"
                        reacq_txt = str(bool(reacq_map.get(int(oid), False)))
                        gallery_txt = "--"
                        best_ref_txt = "--"

                        if isinstance(info, dict):
                            sim = info.get("sim", None)
                            acc = info.get("accepted", None)
                            if sim is not None:
                                sim_txt = f"{float(sim):.3f}"
                            if acc is True:
                                acc_txt = "ACCEPT"
                            elif acc is False:
                                acc_txt = "REJECT"
                            else:
                                acc_txt = "UNKNOWN"

                            gsz = info.get("gallery_size", None)
                            bri = info.get("best_ref_idx", None)
                            if gsz is not None:
                                gallery_txt = str(int(gsz))
                            if bri is not None:
                                best_ref_txt = str(int(bri))

                        obj_logit_txt = "--"
                        obj_prob_txt = "--"
                        obj_thr_txt = "--"
                        reid_ok_txt = "--"
                        reacq_score_txt = "--"
                        kf_txt = "--"
                        iou_txt = "--"
                        s_reid_txt = "--"
                        s_obj_txt = "--"
                        s_kf_txt = "--"
                        s_iou_txt = "--"

                        if isinstance(logits_list, list) and i < len(logits_list) and logits_list[i] is not None:
                            obj_logit_txt = f"{float(logits_list[i]):.3f}"
                        if isinstance(probs_list, list) and i < len(probs_list) and probs_list[i] is not None:
                            obj_prob_txt = f"{float(probs_list[i]):.3f}"

                        thr_val = frame_dbg.get("object_score_thr", None)
                        if thr_val is not None:
                            obj_thr_txt = f"{float(thr_val):.3f}"

                        if isinstance(reid_ok_list, list) and i < len(reid_ok_list):
                            reid_ok_txt = str(int(reid_ok_list[i]))

                        if isinstance(reacq_score_list, list) and i < len(reacq_score_list) and reacq_score_list[i] is not None:
                            reacq_score_txt = f"{float(reacq_score_list[i]):.3f}"

                        if isinstance(kf_score_list, list) and i < len(kf_score_list) and kf_score_list[i] is not None:
                            kf_txt = f"{float(kf_score_list[i]):.3f}"

                        if isinstance(best_iou_list, list) and i < len(best_iou_list) and best_iou_list[i] is not None:
                            iou_txt = f"{float(best_iou_list[i]):.3f}"

                        if isinstance(reacq_parts_list, list) and i < len(reacq_parts_list):
                            rp = reacq_parts_list[i]
                            if isinstance(rp, dict):
                                if rp.get("s_reid", None) is not None:
                                    s_reid_txt = f"{float(rp['s_reid']):.3f}"
                                if rp.get("s_obj", None) is not None:
                                    s_obj_txt = f"{float(rp['s_obj']):.3f}"
                                if rp.get("s_kf", None) is not None:
                                    s_kf_txt = f"{float(rp['s_kf']):.3f}"
                                if rp.get("s_iou", None) is not None:
                                    s_iou_txt = f"{float(rp['s_iou']):.3f}"

                        line1 = (
                            f"id={oid}  sim={sim_txt}  thr={thr:.2f}  "
                            f"reid={acc_txt}  reacq={reacq_txt}  reacq_score={reacq_score_txt}"
                        )
                        line2 = (
                            f"id={oid}  obj_logit={obj_logit_txt}  obj_prob={obj_prob_txt}  "
                            f"obj_thr={obj_thr_txt}  kf={kf_txt}  iou={iou_txt}  mem_reid_ok={reid_ok_txt}"
                        )
                        line3 = (
                            f"id={oid}  s_reid={s_reid_txt}  s_obj={s_obj_txt}  "
                            f"s_kf={s_kf_txt}  s_iou={s_iou_txt}  gallery={gallery_txt}  best_ref={best_ref_txt}"
                        )

                        cv2.putText(
                            disp_bgr,
                            line1,
                            (10, y0 + i * dy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            disp_bgr,
                            line2,
                            (10, y0 + i * dy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.54,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            disp_bgr,
                            line3,
                            (10, y0 + i * dy + 36),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.54,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                except Exception:
                    pass

            cv2.imshow(win, disp_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            # arrows: left=81, right=83 on Windows OpenCV
            if key == 81:
                state["selected_idx"] = max(0, state["selected_idx"] - 1)
            elif key == 83:
                state["selected_idx"] = state["selected_idx"] + 1
            elif key in (ord("a"), ord("A")):
                add_prompt_from_selected()
            elif key in (ord("t"), ord("T")):
                start_tracking()
            elif key in (ord("d"), ord("D")):
                dump_case()
            elif key in (ord("y"), ord("Y")):
                state["yolo_enabled"] = not state["yolo_enabled"]
                print(f"[yolo] overlay: {'ON' if state['yolo_enabled'] else 'OFF'}", flush=True)
            elif key in (ord("+"), ord("=")):
                state["yolo_conf"] = min(0.95, state["yolo_conf"] + 0.05)
                print(f"[yolo] conf -> {state['yolo_conf']:.2f}", flush=True)
            elif key in (ord("-"), ord("_")):
                state["yolo_conf"] = max(0.01, state["yolo_conf"] - 0.05)
                print(f"[yolo] conf -> {state['yolo_conf']:.2f}", flush=True)
            elif key in (ord("r"), ord("R")):
                reset_all()

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("\n[done] Exited cleanly.", flush=True)


if __name__ == "__main__":
    main()