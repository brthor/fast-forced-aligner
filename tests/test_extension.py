"""First-use CUDA requirements fail clearly without affecting import/install."""
import pytest
import torch
from torch.utils import cpp_extension

from fast_forced_aligner import load_extension


@pytest.fixture(autouse=True)
def clear_loader_cache():
    load_extension.cache_clear()
    yield
    load_extension.cache_clear()


def test_cpu_torch_reports_cuda_requirement(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        load_extension()


def test_missing_gpu_reports_driver_requirement(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="accessible NVIDIA GPU"):
        load_extension()


@pytest.mark.parametrize("toolkit", [None, "/nonexistent/cuda-toolkit"])
def test_missing_toolkit_reports_first_use_build_requirements(monkeypatch, toolkit):
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", toolkit)
    with pytest.raises(RuntimeError, match="including nvcc"):
        load_extension()
