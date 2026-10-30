"""Complete resident-input benchmark lanes; dependency failures are explicit."""
from __future__ import annotations
from typing import Literal

METHODS = (
    "optimized", "conventional", "optimized-checked", "conventional-checked",
    "ctc-corrected", "ctc-cpp", "ta-cpu", "ta-cuda", "k2-cuda", "nemo-cpu", "nemo-cuda",
)


def make_backend(method: str, *, workers: int = 8, device: str = "cuda",
                 batch_size: int = 64, nemo_batch_order: Literal["input", "length"] = "input"):
    """Construct one lane before timing. Workers affects only corrected C++.

    CPU Torch thread counts are controlled by the runner: eight for NeMo CPU,
    one for every other lane. CUDA lane selection honors the requested device.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown benchmark method {method!r}; choose from {METHODS}")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if method in METHODS[:4]:
        from .package import PackageBackend
        return PackageBackend(method, device)
    if method in ("ctc-cpp", "ctc-corrected"):
        from .cpp import CppBackend
        return CppBackend(corrected=method == "ctc-corrected",
                          workers=workers if method == "ctc-corrected" else 1)
    lane_device = "cpu" if method.endswith("-cpu") else device
    if not method.endswith("-cpu"):
        import torch
        if torch.device(device).type != "cuda":
            raise ValueError(f"{method} requires a CUDA device")
    if method.startswith("ta-"):
        from .torchaudio import TorchAudioBackend
        return TorchAudioBackend(lane_device)
    if method.startswith("nemo-"):
        from .nemo import NemoBackend
        return NemoBackend(lane_device, batch_size=batch_size, batch_order=nemo_batch_order)
    from .k2 import K2Backend
    return K2Backend(lane_device)
