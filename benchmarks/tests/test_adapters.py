"""Adapter output contracts, timing boundaries and explicit CPU corrections."""
from __future__ import annotations

from dataclasses import replace
from itertools import product
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapters import make_backend
from adapters.common import admit
from inputs import Case


def fixtures() -> list[tuple[Case, np.ndarray]]:
    result = []
    for index, (labels, targets, blank) in enumerate([
        ([0, 1, 1, 0, 2, 2], [1, 2], 0),
        ([3, 0, 0, 3, 0, 2, 3], [0, 0, 2], 3),
        ([0, 1, 0, 1, 2, 0, 0, 0], [1, 1, 2], 0),
    ]):
        emissions = np.full((len(labels), 4), -8, dtype=np.float32)
        emissions[np.arange(len(labels)), labels] = -0.125
        result.append((Case(str(index), emissions, np.asarray(targets, dtype=np.int64), blank),
                       np.asarray(labels)))
    return result


@pytest.mark.parametrize("method", ["ctc-cpp", "ctc-corrected", "ta-cpu", "nemo-cpu"])
@pytest.mark.external_backend
def test_complete_ragged_paths_and_output_ownership(method: str) -> None:
    torch.set_num_threads(1)
    backend = make_backend(method)
    cases = [case for case, _ in fixtures()]
    try:
        resident = backend.prepare_batch(cases)
        outputs = backend.materialize_batch(backend.run_batch(resident))
        saved = [(a.copy(), b.copy()) for a, b in outputs]
        again = backend.materialize_batch(backend.run_batch(resident))
        for (case, expected), pair, repeated, original in zip(fixtures(), outputs, again, saved, strict=True):
            np.testing.assert_array_equal(pair[0], expected)
            np.testing.assert_array_equal(pair[1], case.emissions[np.arange(len(expected)), expected])
            for value, other, previous in zip(pair, repeated, original, strict=True):
                assert value.ndim == 1
                np.testing.assert_array_equal(value, other)
                np.testing.assert_array_equal(value, previous)
    finally:
        backend.close()


@pytest.mark.external_backend
def test_known_predecessor_tie_correction() -> None:
    case = Case("tie", np.array([[0, 0, -10], [0, 0, -10], [-10, -10, 0],
                                 [0, -10, -10], [0, -10, -10]], dtype=np.float32),
                np.array([1, 2], dtype=np.int64), 0)
    scores = []
    for method in ("ctc-cpp", "ctc-corrected"):
        backend = make_backend(method)
        try:
            pair = backend.materialize_batch(backend.run_batch(backend.prepare_batch([case])))[0]
            scores.append(pair[1].sum())
        finally:
            backend.close()
    assert scores == [-10, 0]


@pytest.mark.external_backend
def test_cpp_contiguous_layout_is_prepared_inside_run(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend("ctc-corrected")
    case = fixtures()[0][0]
    storage = np.zeros((len(case.emissions), 8), dtype=np.float32)
    storage[:, ::2] = case.emissions
    case = replace(case, emissions=storage[:, ::2])
    calls = []
    function = np.ascontiguousarray
    monkeypatch.setattr(np, "ascontiguousarray", lambda value: (calls.append(value), function(value))[1])
    try:
        resident = backend.prepare_batch([case])
        assert not calls
        backend.run_batch(resident)
        assert len(calls) == 2
    finally:
        backend.close()


@pytest.mark.external_backend
def test_torchaudio_batch_axis_is_prepared_inside_run(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend("ta-cpu")
    calls = []

    def function(emissions, targets, *, blank):
        calls.append((tuple(emissions.shape), tuple(targets.shape), blank))
        return torch.zeros_like(emissions[:, :, 0], dtype=torch.int64), emissions[:, :, blank]

    monkeypatch.setattr(backend, "function", function)
    resident = backend.prepare_batch([case for case, _ in fixtures()])
    assert not calls
    backend.run_batch(resident)
    assert calls == [((1, 6, 4), (1, 2), 0), ((1, 7, 4), (1, 3), 3), ((1, 8, 4), (1, 3), 0)]


@pytest.mark.external_backend
def test_thread_failure_drains_work_and_closed_backend_rejects_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = make_backend("ctc-corrected", workers=2)
    completed = threading.Event()
    cases = [case for case, _ in fixtures()][:2]

    def function(emissions, targets, blank):
        if blank == 0:
            raise ValueError("fixture failure")
        time.sleep(0.02)
        completed.set()
        return None

    monkeypatch.setattr(backend, "function", function)
    resident = backend.prepare_batch(cases)
    try:
        with pytest.raises(ValueError, match="fixture failure"):
            backend.run_batch(resident)
        assert completed.is_set()
    finally:
        backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.run_batch(resident)


@pytest.mark.parametrize("kind", ["empty", "blank", "range", "short", "nan"])
def test_invalid_admission_is_explicit(kind: str) -> None:
    case = fixtures()[0][0]
    if kind == "empty":
        case = replace(case, targets=np.array([], dtype=np.int64))
    elif kind == "blank":
        case = replace(case, targets=np.array([0]))
    elif kind == "range":
        case = replace(case, targets=np.array([4]))
    elif kind == "short":
        case = replace(case, emissions=case.emissions[:1])
    else:
        case = replace(case, emissions=np.full_like(case.emissions, np.nan))
    with pytest.raises(ValueError):
        admit([case], cpp=True)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
@pytest.mark.external_backend
def test_k2_arbitrary_blank_and_sorted_order(device, monkeypatch) -> None:
    # CPU execution exercises graph/padding/permutation logic without taking the
    # GPU timing lease. The public comparison lane remains k2-cuda.
    from adapters.k2 import K2Backend
    backend = K2Backend(device)
    graph_inputs = []
    graph_function = backend.k2.ctc_graph

    def graph(symbols, **kwargs):
        assert isinstance(symbols, backend.k2.RaggedTensor)
        assert symbols.values.device.type == device
        assert symbols.values.dtype == torch.int32
        graph_inputs.append(symbols)
        return graph_function(symbols, **kwargs)

    monkeypatch.setattr(backend.k2, "ctc_graph", graph)
    cases = [case for case, _ in fixtures()]
    resident = backend.prepare_batch(cases)
    assert not graph_inputs  # Graph construction remains per-call timed work.
    outputs = backend.materialize_batch(backend.run_batch(resident))
    assert len(graph_inputs) == 1
    assert graph_inputs[0].tolist() == [[1, 1, 2], [3, 3, 2], [1, 2]]
    for (_, expected), (actual, scores) in zip(fixtures(), outputs, strict=True):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(scores, np.full(len(expected), -0.125))


@pytest.mark.parametrize("batch_size,order", [(1, "input"), (2, "input"), (2, "length"), (64, "length")])
@pytest.mark.external_backend
def test_nemo_scheduling_restores_order_and_repeats_preparation(batch_size, order, monkeypatch):
    from adapters.nemo import NemoBackend
    backend = NemoBackend("cpu", batch_size=batch_size, batch_order=order)
    calls = []
    native = backend._align_batch

    def record(batch):
        calls.append([case.case_id for case in batch.cases])
        return native(batch)

    monkeypatch.setattr(backend, "_align_batch", record)
    cases = [case for case, _ in fixtures()]
    resident = backend.prepare_batch(cases)
    assert not calls
    for _ in range(2):
        outputs = backend.materialize_batch(backend.run_batch(resident))
        for (_, expected), (actual, _) in zip(fixtures(), outputs, strict=True):
            np.testing.assert_array_equal(actual, expected)
    expected = [case.case_id for case in (cases if order == "input" else list(reversed(cases)))]
    groups = [expected[i:i + batch_size] for i in range(0, len(cases), batch_size)]
    assert calls == groups * 2


@pytest.mark.external_backend
def test_nemo_scheduler_rejects_missing_outputs(monkeypatch):
    from adapters.nemo import NemoBackend
    backend = NemoBackend("cpu", batch_size=2, batch_order="length")
    monkeypatch.setattr(backend, "_align_batch", lambda batch: (batch.cases, []))
    with pytest.raises(ValueError, match="wrong number"):
        backend.run_batch(backend.prepare_batch([case for case, _ in fixtures()]))


def test_package_audit_preserves_failure_status_and_placeholders() -> None:
    from adapters.package import PackageBackend
    backend = object.__new__(PackageBackend)
    raw = SimpleNamespace(status=torch.tensor([0, 1]), lengths=(2, 3),
                          labels=torch.tensor([0, 1, -1, -1, -1]),
                          scores=torch.tensor([-1.0, -0.5, -torch.inf, -torch.inf, -torch.inf]))
    assert backend.status_codes(raw) == [0, 1]
    outputs = backend.materialize_batch(raw)
    np.testing.assert_array_equal(outputs[1][0], [-1, -1, -1])
    assert np.isneginf(outputs[1][1]).all()


def test_upstream_cached_hash_mismatch_fails_closed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import upstream
    monkeypatch.setattr(upstream, "ROOT", tmp_path)
    spec = upstream.SOURCES["ctc_cpp"]
    path = tmp_path / "ctc_cpp" / spec["revision"] / "forced_align_impl.cpp"
    path.parent.mkdir(parents=True)
    path.write_text("changed source")
    with pytest.raises(ValueError, match="cached upstream hash mismatch"):
        upstream.source_path("ctc_cpp", fetch=False)


@pytest.mark.parametrize("method", ["ctc-corrected", "ta-cpu", "nemo-cpu", "k2-cpu"])
@pytest.mark.external_backend
def test_small_paths_match_exhaustive_ctc_oracle(method: str) -> None:
    """Enumerate legal label paths independently of any DP recurrence."""
    torch.set_num_threads(1)
    random = np.random.default_rng(1729)
    cases, maxima = [], []
    for index in range(12):
        blank = index % 3
        vocabulary = [label for label in range(3) if label != blank]
        tokens = [vocabulary[index % 2], vocabulary[(index // 2) % 2]]
        frames = 5 + index % 2
        emissions = random.integers(-5, 1, size=(frames, 3)).astype(np.float32)
        case = Case(str(index), emissions, np.asarray(tokens, dtype=np.int64), blank)
        legal_scores = []
        for path in product(range(3), repeat=frames):
            collapsed = [label for position, label in enumerate(path)
                         if label != blank and (position == 0 or path[position - 1] != label)]
            if collapsed == tokens:
                legal_scores.append(float(emissions[np.arange(frames), path].sum()))
        cases.append(case)
        maxima.append(max(legal_scores))
    if method == "k2-cpu":
        from adapters.k2 import K2Backend
        backend = K2Backend("cpu")
    else:
        backend = make_backend(method)
    try:
        outputs = backend.materialize_batch(backend.run_batch(backend.prepare_batch(cases)))
        for case, (labels, scores), maximum in zip(cases, outputs, maxima, strict=True):
            collapsed = [label for position, label in enumerate(labels)
                         if label != case.blank_id and (position == 0 or labels[position - 1] != label)]
            assert collapsed == case.targets.tolist()
            assert float(scores.sum()) == maximum
    finally:
        backend.close()


def test_upstream_download_is_verified_before_atomic_publication(tmp_path, monkeypatch) -> None:
    import hashlib
    import io
    import upstream
    monkeypatch.setattr(upstream, "ROOT", tmp_path)
    data = b"verified upstream source"
    monkeypatch.setattr(upstream, "SOURCES", {"fixture": {
        "repository": "fixture/repo", "revision": "abc", "path": "source.py",
        "sha256": hashlib.sha256(data).hexdigest(),
    }})
    monkeypatch.setattr(upstream, "urlopen", lambda *args, **kwargs: io.BytesIO(data))
    expected = tmp_path / "fixture" / "abc" / "source.py"
    original_replace = type(expected).replace

    def replace(source, destination):
        assert not expected.exists()
        assert source.read_bytes() == data
        return original_replace(source, destination)

    monkeypatch.setattr(type(expected), "replace", replace)
    assert upstream.source_path("fixture").read_bytes() == data
    assert list(expected.parent.iterdir()) == [expected]
