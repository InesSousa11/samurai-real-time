from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam2.reid_backends.base import BaseReIDBackend


class TransReIDBackend(BaseReIDBackend):
    def __init__(
        self,
        ckpt_path="checkpoints/reid/transreid/vit_transreid_msmt.pth",
        device="cuda",
    ):
        super().__init__(device=device)
        self.device = torch.device(
            device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        )

        repo_root = Path(__file__).resolve().parents[2]
        transreid_root = repo_root / "external" / "reid" / "TransReID"

        if str(transreid_root) not in sys.path:
            sys.path.insert(0, str(transreid_root))

        from model.make_model import make_model

        class Cfg:
            class MODEL:
                NAME = "transformer"
                LAST_STRIDE = 1
                TRANSFORMER_TYPE = "vit_base_patch16_224_TransReID"
                STRIDE_SIZE = [12, 12]
                SIE_CAMERA = False
                SIE_VIEW = False
                SIE_COE = 3.0
                JPM = False
                RE_ARRANGE = False
                PRETRAIN_PATH = ""
                PRETRAIN_CHOICE = "none"
                COS_LAYER = False
                NECK = "bnneck"
                NECK_FEAT = "after"
                ID_LOSS_TYPE = "softmax"
                DROP_PATH = 0.1
                DROP_OUT = 0.0
                ATT_DROP_RATE = 0.0

            class TEST:
                NECK_FEAT = "after"

            class INPUT:
                SIZE_TRAIN = [256, 128]
                SIZE_TEST = [256, 128]

            class SOLVER:
                COSINE_SCALE = 30
                COSINE_MARGIN = 0.5

            class DATALOADER:
                SAMPLER = "softmax"

            class DATASETS:
                NAMES = "msmt17"

        cfg = Cfg()

        self.model = make_model(
            cfg,
            num_class=1,
            camera_num=1,
            view_num=1,
        )

        ckpt_path = str(repo_root / ckpt_path) if not Path(ckpt_path).is_absolute() else ckpt_path
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model_dict = self.model.state_dict()
        filtered_state_dict = {}

        for k, v in state_dict.items():
            k2 = k.replace("module.", "")
            if k2 in model_dict and model_dict[k2].shape == v.shape:
                filtered_state_dict[k2] = v

        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.in_h = 256
        self.in_w = 128
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def embed_crop_bgr(self, crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)

        x = (
            torch.from_numpy(rgb)
            .to(self.device)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )
        x = (x - self.mean) / self.std

        feat = self.model(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        feat = feat.squeeze(0)
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