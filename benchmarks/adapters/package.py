"""Installed package calls, including per-call topology and output allocation."""
from __future__ import annotations

import numpy as np
import torch

from inputs import Case
from .common import DeviceBackend, ResidentBatch, admit, resident_batch


class PackageBackend(DeviceBackend):
    def __init__(self, method: str, device: str = "cuda") -> None:
        from fast_forced_aligner import align, load_extension, validate_inputs
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("fast_forced_aligner requires a CUDA device")
        self.name = method
        self.mode = "conventional" if method.startswith("conventional") else "optimized"
        self.check_inputs = method.endswith("-checked")
        self.function = align
        self.validate = validate_inputs
        with torch.cuda.device(self.device):
            load_extension()

    def prepare_batch(self, cases: list[Case]) -> ResidentBatch:
        admit(cases)
        resident = resident_batch(cases, self.device)
        self.validate(resident.emissions, resident.targets,
                      blank_id=[int(case.blank_id) for case in cases])
        return resident

    def run_batch(self, resident: ResidentBatch):
        return self.function(resident.emissions, resident.targets,
                             blank_id=[int(case.blank_id) for case in resident.cases],
                             mode=self.mode, check_inputs=self.check_inputs)

    def materialize_batch(self, raw) -> list[tuple[np.ndarray, np.ndarray]]:
        labels, scores = raw.labels.cpu(), raw.scores.cpu()
        return [(a.numpy(), b.numpy()) for a, b in
                zip(labels.split(raw.lengths), scores.split(raw.lengths), strict=True)]

    def status_codes(self, raw) -> list[int]:
        """Read per-record status only during the untimed output audit."""
        return raw.status.cpu().tolist()
