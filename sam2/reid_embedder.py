import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add torchreid submodule to sys.path (repo-relative)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TORCHREID_ROOT = _REPO_ROOT / "external" / "reid" / "deep-person-reid"
if _TORCHREID_ROOT.exists():
    sys.path.insert(0, str(_TORCHREID_ROOT))

from torchreid import models  # noqa: E402


class OSNetReIDEmbedder:
    """
    Minimal OSNet wrapper for embedding extraction (no training).
    Uses ImageNet-pretrained OSNet_x1_0 weights provided by torchreid.

    - Input: BGR crop (numpy HxWx3)
    - Output: embedding tensor [D] on CPU (or on device if you prefer)
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.model = models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True)
        self.model.eval().to(self.device)

        # Torchreid default input size for OSNet is typically 256x128 (H,W)
        self.in_h = 256
        self.in_w = 128

        # ImageNet normalization (common)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def embed_crop_bgr(self, crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        # BGR -> RGB
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # resize to OSNet expected
        rgb = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)

        # to tensor [1,3,H,W], float in [0,1]
        x = torch.from_numpy(rgb).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        x = (x - self.mean) / self.std

        # forward
        feat = self.model(x)  # usually [1, D]
        feat = feat.squeeze(0)

        # return normalized embedding on CPU for easy cosine
        feat = F.normalize(feat, p=2, dim=0).detach().cpu()
        return feat

    def cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if a is None or b is None:
            return float("nan")
        a = a.detach().float().reshape(-1)
        b = b.detach().float().reshape(-1)
        a = F.normalize(a, p=2, dim=0)
        b = F.normalize(b, p=2, dim=0)
        return float(torch.dot(a, b).item())