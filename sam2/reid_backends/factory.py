from sam2.reid_backends.osnet_backend import OSNetReIDBackend


def build_reid_backend(name: str = "osnet_x1_0", device: str = "cuda", **kwargs):
    """
    Factory for interchangeable ReID backends.

    Supported names for now:
    - osnet
    - osnet_x1_0
    - osnet_ain_x1_0
    """
    key = str(name).lower()

    if key in {"osnet", "osnet_x1_0"}:
        return OSNetReIDBackend(device=device, model_name="osnet_x1_0")

    if key in {"osnet_ain", "osnet_ain_x1_0"}:
        return OSNetReIDBackend(device=device, model_name="osnet_ain_x1_0")

    raise ValueError(f"Unknown ReID backend: {name}")