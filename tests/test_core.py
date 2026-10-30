"""Public API behavior, independent DP scores, and resident stream ownership."""
import numpy as np
import pytest
import torch

from fast_forced_aligner import align, validate_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def optimum(values, targets, blank):
    states = np.full(2 * len(targets) + 1, blank)
    states[1::2] = targets
    previous = np.full(len(states), -np.inf)
    previous[:2] = values[0, states[:2]]
    for frame in values[1:]:
        current = np.full_like(previous, -np.inf)
        for state, label in enumerate(states):
            incoming = [previous[state]]
            if state:
                incoming.append(previous[state - 1])
            if state > 1 and label != blank and label != states[state - 2]:
                incoming.append(previous[state - 2])
            current[state] = max(incoming) + frame[label]
        previous = current
    return max(previous[-2:])


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
@pytest.mark.parametrize("blank", [0, 5])
def test_ragged_long_repeats_empty_transcript_and_exact_scores(mode, blank):
    rng = np.random.default_rng(32)
    labels = np.array([i for i in range(6) if i != blank])
    values = [rng.integers(-9, 0, (n, 6)).astype(np.float32) for n in (7, 129, 257, 2305)]
    tokens = [rng.choice(labels, n) for n in (0, 11, 44, 120)]
    es = [torch.tensor(v, device="cuda") for v in values]
    ts = [torch.tensor(t, dtype=torch.int64, device="cuda") for t in tokens]
    actual = align(es, ts, blank_id=blank, mode=mode)
    actual.raise_for_status()
    offset = 0
    for v, t, path in zip(values, tokens, actual.paths, strict=True):
        p = path.cpu().numpy()
        collapsed = p[np.r_[True, p[1:] != p[:-1]]]
        np.testing.assert_array_equal(collapsed[collapsed != blank], t)
        selected = v[np.arange(len(p)), p]
        np.testing.assert_array_equal(actual.scores[offset:offset + len(p)].cpu(), selected)
        assert selected.sum(dtype=np.float64) == optimum(v, t, blank)
        offset += len(p)


@pytest.mark.parametrize("tokens", [[-1], [6], [0], [1, 1, 1, 1]])
def test_bad_tokens_are_rejected_before_native_read(tokens):
    result = align([torch.zeros(4, 6, device="cuda")], [torch.tensor(tokens, device="cuda")])
    with pytest.raises(ValueError, match="target IDs|insufficient"):
        result.raise_for_status()


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
def test_infeasible_status_and_input_mutation(mode):
    e = torch.full((257, 4), -torch.inf, device="cuda")
    t = torch.tensor([1], device="cuda")
    failed = align([e], [t], mode=mode)
    with pytest.raises(ValueError, match="no finite"):
        failed.raise_for_status()
    e.zero_()
    good = align([e], [t], mode=mode)
    good.raise_for_status()
    assert good.status.item() == 0 and failed.status.item() == 1
    assert good.labels.data_ptr() != failed.labels.data_ptr()


def test_heterogeneous_blanks_and_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        es = [torch.zeros(256, 6, device="cuda"), torch.zeros(2110, 8, device="cuda")]
        ts = [torch.tensor([1, 1, 2], device="cuda"), torch.tensor([0, 1, 1], device="cuda")]
        validate_inputs(es, ts, blank_id=[0, 7])
        result = align(es, ts, blank_id=[0, 7], check_inputs=False)
        del es, ts
    torch.cuda.current_stream().wait_stream(stream)
    result.raise_for_status()
    assert result.lengths == (256, 2110)


def test_shape_and_option_contracts():
    e = torch.zeros(12, 6, device="cuda")
    t = torch.tensor([1], device="cuda")
    for values, targets, kwargs in [([], [], {}), ([e], [], {}),
                                   ([e[:, ::2]], [t], {}), ([e.double()], [t], {}),
                                   ([e], [t.int()], {}), ([e], [t], {"blank_id": 6}),
                                   ([e], [t], {"mode": "unknown"})]:
        with pytest.raises(ValueError):
            align(values, targets, **kwargs)
