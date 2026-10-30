"""Standard k2 CTC graph, unpruned dense intersection and best-path extraction."""
from __future__ import annotations
from importlib.metadata import version
from typing import Any
import numpy as np
import torch
from inputs import Case
from .common import DeviceBackend, ResidentBatch, admit, resident_batch, audit_scores


class K2Backend(DeviceBackend):
    def __init__(self, device: str) -> None:
        import k2
        installed = version("k2")
        if installed.split("+")[0] != "1.24.4.dev20260625":
            raise RuntimeError(f"This benchmark requires k2 1.24.4.dev20260625, got {installed}")
        self.k2 = k2
        self.device = torch.device(device)
        self.name = f"k2-{self.device.type}"

    def pack(self, resident: ResidentBatch) -> tuple[Any, ...]:
        order = sorted(range(len(resident.cases)), key=lambda i: -len(resident.cases[i].emissions))
        cases = [resident.cases[i] for i in order]
        emissions = torch.nn.utils.rnn.pad_sequence(
            [resident.emissions[i] for i in order], batch_first=True)
        # Swapping zero and each record's blank is its own inverse. Construct all
        # indices per call and permute the padded batch with one gather operation.
        if any(case.blank_id != 0 for case in cases):
            batch, frames, vocabulary = emissions.shape
            blanks = torch.tensor([case.blank_id for case in cases], device=self.device)
            columns = torch.arange(vocabulary, device=self.device).expand(batch, -1)
            permutation = torch.where(columns == 0, blanks[:, None],
                                      torch.where(columns == blanks[:, None], 0, columns))
            emissions = emissions.gather(2, permutation[:, None, :].expand(-1, frames, -1))

        # ctc_graph accepts a device-resident RaggedTensor. Build its row splits
        # and remap token IDs here, without a GPU -> Python -> GPU round trip.
        lengths = [len(case.targets) for case in cases]
        tokens = torch.cat([resident.targets[i] for i in order]).to(torch.int32)
        if any(case.blank_id != 0 for case in cases):
            token_blanks = torch.repeat_interleave(
                torch.tensor([case.blank_id for case in cases], device=self.device, dtype=torch.int32),
                torch.tensor(lengths, device=self.device), output_size=sum(lengths))
            tokens = torch.where(tokens == 0, token_blanks,
                                 torch.where(tokens == token_blanks, 0, tokens))
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        shape = self.k2.ragged.create_ragged_shape2(
            row_splits=torch.tensor(offsets, device=self.device, dtype=torch.int32),
            cached_tot_size=offsets[-1])
        transcripts = self.k2.RaggedTensor(shape, tokens)
        supervisions = torch.tensor([[i, 0, len(case.emissions)] for i, case in enumerate(cases)],
                                    dtype=torch.int32)
        graph = self.k2.arc_sort(self.k2.ctc_graph(transcripts, modified=False, device=self.device))
        dense = self.k2.DenseFsaVec(emissions, supervisions)
        return graph, dense, order, resident.cases

    def prepare_batch(self, cases: list[Case]) -> ResidentBatch:
        admit(cases)
        if len({case.emissions.shape[1] for case in cases}) != 1:
            raise ValueError("k2 batch requires common vocabulary size")
        return resident_batch(cases, self.device)

    def run_batch(self, prepared: Any) -> Any:
        graph, dense, order, cases = self.pack(prepared)
        # This is intersect_dense, not search-pruned intersect_dense_pruned. Positive
        # output beam uses exact forward/backward best scores and retains a best path.
        lattice = self.k2.intersect_dense(graph, dense, output_beam=10.0)
        best = self.k2.shortest_path(lattice, use_double_scores=False)
        shape = best.arcs.shape().remove_axis(1)
        labels = self.k2.RaggedTensor(shape, best.labels.contiguous())
        paths: list[Any] = [None] * len(order)
        for sorted_index, original_index in enumerate(order):
            # Every path has T emission arcs plus a terminal arc labelled -1.
            # Slice the terminal; filtering would hide malformed negative emissions.
            paths[original_index] = labels[sorted_index][:-1]
        if any(case.blank_id != 0 for case in cases):
            lengths = [len(path) for path in paths]
            labels = torch.cat(paths)
            blanks = torch.repeat_interleave(
                torch.tensor([case.blank_id for case in cases], device=self.device),
                torch.tensor(lengths, device=self.device), output_size=sum(lengths))
            labels = torch.where(labels == 0, blanks, torch.where(labels == blanks, 0, labels))
            paths = list(labels.split(lengths))
        return cases, paths

    def materialize_batch(self, raw: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        cases, tensors = raw
        paths = [x.detach().cpu().numpy().astype(np.int64, copy=False) for x in tensors]
        return audit_scores(cases, paths)
