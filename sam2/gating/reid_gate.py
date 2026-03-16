# sam2/gating/reid_gate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from sam2.reid_embedder import OSNetReIDEmbedder


@dataclass
class ReIDGateConfig:
    thr: float = 0.55
    pad: float = 0.10
    min_area: int = 200  # ignore tiny masks


def _bbox_from_mask(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min()); x2 = int(xs.max())
    y1 = int(ys.min()); y2 = int(ys.max())
    return (x1, y1, x2, y2)


def _crop_from_bbox(frame_bgr: np.ndarray, bb: Tuple[int, int, int, int], pad: float) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bb
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    px = int(round(bw * pad))
    py = int(round(bh * pad))
    x1p = max(0, x1 - px); y1p = max(0, y1 - py)
    x2p = min(W - 1, x2 + px); y2p = min(H - 1, y2 + py)
    return frame_bgr[y1p:y2p + 1, x1p:x2p + 1].copy()


class ReIDGate:
    """
    Keeps ReID logic separate from the tracker.
    - set_ref(obj_id, frame_bgr, mask_bool)
    - eval(obj_id, frame_bgr, mask_bool) -> (ok, sim, bbox_xyxy)
    """
    def __init__(self, device: str = "cuda", cfg: Optional[ReIDGateConfig] = None):
        self.cfg = cfg or ReIDGateConfig()
        self.embedder = OSNetReIDEmbedder(device=device)
        self.ref: Dict[int, torch.Tensor] = {}
        self.last: Dict[int, Dict[str, Any]] = {}

    def have_ref(self, obj_id: int) -> bool:
        return int(obj_id) in self.ref

    def set_ref(self, obj_id: int, frame_bgr: np.ndarray, mask_bool: np.ndarray) -> bool:
        obj_id = int(obj_id)
        bb = _bbox_from_mask(mask_bool)
        if bb is None:
            self.last[obj_id] = {"sim": None, "accepted": None, "bbox": None, "reason": "no_mask"}
            return False
        area = int(mask_bool.sum())
        if area < self.cfg.min_area:
            self.last[obj_id] = {"sim": None, "accepted": None, "bbox": list(bb), "reason": "too_small"}
            return False
        crop = _crop_from_bbox(frame_bgr, bb, self.cfg.pad)
        emb = self.embedder.embed_crop_bgr(crop)
        if emb is None:
            self.last[obj_id] = {"sim": None, "accepted": None, "bbox": list(bb), "reason": "embed_none"}
            return False
        self.ref[obj_id] = emb
        self.last[obj_id] = {"sim": 1.0, "accepted": True, "bbox": list(bb), "reason": "ref_set"}
        return True

    def eval(self, obj_id: int, frame_bgr: np.ndarray, mask_bool: np.ndarray):
        obj_id = int(obj_id)
        if obj_id not in self.ref:
            self.last[obj_id] = {"sim": None, "accepted": None, "bbox": None, "reason": "no_ref"}
            return False, float("nan"), None

        bb = _bbox_from_mask(mask_bool)
        if bb is None:
            self.last[obj_id] = {"sim": None, "accepted": False, "bbox": None, "reason": "no_mask"}
            return False, float("nan"), None

        area = int(mask_bool.sum())
        if area < self.cfg.min_area:
            self.last[obj_id] = {"sim": None, "accepted": False, "bbox": list(bb), "reason": "too_small"}
            return False, float("nan"), list(bb)

        crop = _crop_from_bbox(frame_bgr, bb, self.cfg.pad)
        cur = self.embedder.embed_crop_bgr(crop)
        if cur is None:
            self.last[obj_id] = {"sim": None, "accepted": False, "bbox": list(bb), "reason": "embed_none"}
            return False, float("nan"), list(bb)

        sim = float(self.embedder.cosine(self.ref[obj_id], cur))
        ok = bool(np.isfinite(sim) and (sim >= float(self.cfg.thr)))
        self.last[obj_id] = {"sim": sim if np.isfinite(sim) else None, "accepted": ok, "bbox": list(bb), "reason": "ok" if ok else "low_sim"}
        return ok, sim, list(bb)