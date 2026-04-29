from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np


class TransReIDEmbedder:
    def __init__(
        self,
        ckpt_path="checkpoints/reid/transreid/vit_transreid_msmt.pth",
        device="cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        repo_root = Path(__file__).resolve().parents[1]
        transreid_root = repo_root / "external" / "reid" / "TransReID"

        if str(transreid_root) not in sys.path:
            sys.path.insert(0, str(transreid_root))

        from model.make_model import make_model  # noqa: E402

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
        skipped = []

        for k, v in state_dict.items():
            k2 = k.replace("module.", "")

            if k2 not in model_dict:
                skipped.append((k2, "missing_in_model"))
                continue

            if model_dict[k2].shape != v.shape:
                skipped.append((k2, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_dict[k2].shape)}"))
                continue

            filtered_state_dict[k2] = v

        missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)

        print(f"[TransReID] Loaded {len(filtered_state_dict)} keys from checkpoint")
        print(f"[TransReID] Missing keys in checkpoint load: {len(missing)}")
        print(f"[TransReID] Unexpected keys after filtered load: {len(unexpected)}")

        if skipped:
            print("[TransReID] Skipped keys:")
            for name, reason in skipped[:20]:
                print(f"  - {name}: {reason}")
            if len(skipped) > 20:
                print(f"  ... and {len(skipped) - 20} more")

        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return img

    @torch.no_grad()
    def embed_crop_bgr(self, crop_bgr):
        x = self._preprocess(crop_bgr)
        if x is None:
            return None

        feat = self.model(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        feat = F.normalize(feat, dim=1)
        return feat.squeeze(0).detach().cpu().numpy()

    @staticmethod
    def cosine(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return None
        return float(np.dot(a, b) / denom)