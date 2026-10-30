"""Resident-input contract shared by the independently timed implementations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from inputs import Case


def admit(cases: list[Case], *, cpp: bool = False) -> None:
    """Reject unsafe inputs before native calls; this is excluded admission work.

    The common comparison domain uses nonempty transcripts. The original C++
    traceback needs two additional frames; its corrected lane uses the same
    domain so this adaptation receives no input-domain advantage.
    """
    if not cases:
        raise ValueError("empty batch")
    for case in cases:
        emissions, tokens = np.asarray(case.emissions), np.asarray(case.targets)
        if emissions.ndim != 2 or emissions.shape[1] == 0 or emissions.dtype.kind != "f":
            raise ValueError(f"{case.case_id}: emissions must be floating [T, V]")
        if np.any(np.isnan(emissions)) or np.any(np.isposinf(emissions)):
            raise ValueError(f"{case.case_id}: emissions contain NaN or positive infinity")
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
            raise ValueError(f"{case.case_id}: targets must be an integer vector")
        if (isinstance(case.blank_id, (bool, np.bool_))
                or not isinstance(case.blank_id, (int, np.integer))
                or not 0 <= case.blank_id < emissions.shape[1]):
            raise ValueError(f"{case.case_id}: blank_id is outside the vocabulary")
        if np.any(tokens < 0) or np.any(tokens >= emissions.shape[1]) or np.any(tokens == case.blank_id):
            raise ValueError(f"{case.case_id}: targets contain invalid or blank IDs")
        minimum = len(tokens) + int(np.count_nonzero(tokens[1:] == tokens[:-1]))
        if not len(tokens) or len(emissions) < minimum + (2 if cpp else 0):
            raise ValueError(f"{case.case_id}: unsupported empty transcript or insufficient safe frames")
        if cpp and max(*emissions.shape, len(tokens), 2 * len(tokens) + 1) > np.iinfo(np.int32).max:
            raise ValueError(f"{case.case_id}: C++ signed 32-bit loop limit exceeded")


@dataclass
class ResidentBatch:
    cases: list[Case]
    emissions: list[torch.Tensor]
    targets: list[torch.Tensor]


def resident_batch(cases: list[Case], device: torch.device) -> ResidentBatch:
    """Perform initial FP32/int64 device placement, without topology preparation."""
    return ResidentBatch(
        cases,
        [torch.tensor(c.emissions, dtype=torch.float32, device=device) for c in cases],
        [torch.tensor(c.targets, dtype=torch.int64, device=device) for c in cases],
    )


class DeviceBackend:
    device: torch.device

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def close(self) -> None:
        """Release backend resources after the last completed measurement."""


def audit_scores(cases: list[Case], labels: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Gather audit scores for path-only toolkits, outside measured execution.

    Malformed paths must fail explicitly, rather than indexing with negative IDs
    or accidentally hiding a missing frame. K2 and NeMo do not expose the same
    per-frame score API as TorchAudio or the package.
    """
    outputs = []
    for case, path in zip(cases, labels, strict=True):
        path = np.asarray(path, dtype=np.int64).reshape(-1)
        if len(path) != len(case.emissions) or np.any(path < 0) or np.any(path >= case.emissions.shape[1]):
            raise ValueError(f"{case.case_id}: backend returned an invalid frame-label path")
        outputs.append((path, case.emissions[np.arange(len(path)), path]))
    return outputs
