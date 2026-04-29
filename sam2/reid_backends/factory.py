from sam2.reid_backends.osnet_backend import OSNetReIDBackend
from sam2.reid_backends.transreid_backend import TransReIDBackend


def build_reid_backend(name: str = "osnet_x1_0", device: str = "cuda", **kwargs):
    """
    Factory for interchangeable ReID backends.

    Supported names:
    - osnet
    - osnet_x1_0
    - osnet_ain
    - osnet_ain_x1_0
    - transreid
    - transreid_msmt
    """
    key = str(name).lower()

    if key in {"osnet", "osnet_x1_0"}:
        return OSNetReIDBackend(device=device, model_name="osnet_x1_0")

    if key in {"osnet_ain", "osnet_ain_x1_0"}:
        return OSNetReIDBackend(device=device, model_name="osnet_ain_x1_0")

    if key in {"transreid", "transreid_msmt"}:
        return TransReIDBackend(device=device, **kwargs)

    raise ValueError(f"Unknown ReID backend: {name}")