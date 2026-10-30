"""Pinned NeMo Viterbi function with timed padding, topology and traceback."""
from __future__ import annotations
from typing import Any, Literal
import numpy as np
import torch
from inputs import Case
from upstream import nemo_function
from .common import DeviceBackend, ResidentBatch, admit, resident_batch, audit_scores


class NemoBackend(DeviceBackend):
    """Unmodified NeMo recurrence and traceback, with explicit input/output adapters."""
    def __init__(self, device: str, *, batch_size: int = 64,
                 batch_order: Literal["input", "length"] = "input") -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if batch_order not in ("input", "length"):
            raise ValueError("batch_order must be input or length")
        self.device = torch.device(device)
        self.function = nemo_function()
        self.name = f"nemo-{self.device.type}"
        self.batch_size = batch_size
        self.batch_order = batch_order

    def pack(self, resident: ResidentBatch) -> tuple[Any, ...]:
        cases = resident.cases
        if len({c.emissions.shape[1] for c in cases}) != 1:
            raise ValueError("NeMo requires a common vocabulary size")
        vocab = cases[0].emissions.shape[1]
        lengths = [len(c.emissions) for c in cases]
        sizes = [2 * len(c.targets) + 1 for c in cases]
        emissions = torch.nn.utils.rnn.pad_sequence(resident.emissions, batch_first=True, padding_value=-3.4e38)
        expanded = []
        for c, tokens, size in zip(cases, resident.targets, sizes, strict=True):
            labels = torch.full((size,), c.blank_id, device=self.device, dtype=torch.int64)
            labels[1::2] = tokens
            expanded.append(labels)
        targets = torch.nn.utils.rnn.pad_sequence(expanded, batch_first=True, padding_value=vocab)
        return (emissions, targets, torch.tensor(lengths, device=self.device),
                torch.tensor(sizes, device=self.device), cases)

    def prepare_batch(self, cases: list[Case]) -> Any:
        admit(cases)
        if len({c.emissions.shape[1] for c in cases}) != 1:
            raise ValueError("NeMo requires a common vocabulary size")
        return resident_batch(cases, self.device)

    def run_batch(self, prepared: ResidentBatch) -> tuple[list[Case], list[np.ndarray]]:
        """Schedule an entire resident queue, including sorting and order restoration.

        Length bucketing sorts by decreasing frame count, then transcript size.
        It uses shape metadata only and is repeated inside every timed call.
        The unmodified upstream function still owns padding-dependent DP and
        traceback; only its native batch size and recording order vary.
        """
        order = list(range(len(prepared.cases)))
        if self.batch_order == "length":
            order.sort(key=lambda i: (len(prepared.cases[i].emissions),
                                      len(prepared.cases[i].targets)), reverse=True)
        outputs: list[np.ndarray | None] = [None] * len(order)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            batch = ResidentBatch([prepared.cases[i] for i in indices],
                                  [prepared.emissions[i] for i in indices],
                                  [prepared.targets[i] for i in indices])
            _, labels = self._align_batch(batch)
            if len(labels) != len(indices):
                raise ValueError("NeMo returned the wrong number of alignments")
            for index, path in zip(indices, labels, strict=True):
                outputs[index] = path
        if any(path is None for path in outputs):
            raise ValueError("NeMo did not return every alignment")
        return prepared.cases, [path for path in outputs if path is not None]

    def _align_batch(self, prepared: ResidentBatch) -> tuple[list[Case], list[np.ndarray]]:
        emissions, targets, lengths, sizes, cases = self.pack(prepared)
        states = self.function(emissions, targets, lengths, sizes, viterbi_device=self.device)
        # NeMo returns host state lists. Converting to full labels is production output work.
        labels = []
        for c, state_path in zip(cases, states, strict=True):
            expanded = np.full(2 * len(c.targets) + 1, c.blank_id, dtype=np.int64)
            expanded[1::2] = c.targets
            state_array = np.asarray(state_path, dtype=np.int64)
            if (state_array < 0).any() or (state_array >= len(expanded)).any():
                raise ValueError("NeMo returned an out-of-range transcript state")
            labels.append(expanded[state_array])
        return cases, labels

    def materialize_batch(self, raw: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        cases, labels = raw
        # Scores are derived solely for the audit; NeMo's public output is a complete path.
        return audit_scores(cases, labels)
