"""Unmodified TorchAudio 2.8 batch-one API, including internal transfers."""
from __future__ import annotations

import numpy as np
import torch

from inputs import Case
from .common import DeviceBackend, ResidentBatch, admit, resident_batch


class TorchAudioBackend(DeviceBackend):
    def __init__(self, device: str) -> None:
        import torchaudio
        if torchaudio.__version__.split("+")[0] != "2.8.0":
            raise RuntimeError(f"This benchmark requires torchaudio 2.8.0, got {torchaudio.__version__}")
        self.device = torch.device(device)
        self.name = f"ta-{self.device.type}"
        self.function = torchaudio.functional.forced_align

    def prepare_batch(self, cases: list[Case]) -> ResidentBatch:
        admit(cases)
        return resident_batch(cases, self.device)

    def run_batch(self, resident: ResidentBatch):
        return [self.function(emissions.unsqueeze(0).contiguous(),
                              targets.unsqueeze(0).contiguous(), blank=int(case.blank_id))
                for case, emissions, targets in zip(resident.cases, resident.emissions,
                                                    resident.targets, strict=True)]

    def materialize_batch(self, raw) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(labels.detach().cpu().numpy().reshape(-1), scores.detach().cpu().numpy().reshape(-1))
                for labels, scores in raw]
