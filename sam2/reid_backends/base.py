from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class BaseReIDBackend(ABC):
    """
    Common interface for all ReID backends used by SAM2CameraPredictor.

    Required behavior:
    - embed_crop_bgr(crop_bgr) -> normalized embedding tensor or None
    - cosine(a, b) -> similarity float
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def embed_crop_bgr(self, crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        raise NotImplementedError