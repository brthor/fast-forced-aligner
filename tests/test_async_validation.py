"""Target validation protects all CUDA alignment routes without a host wait."""

from dataclasses import fields

import pytest
import torch

from fast_forced_aligner import align, load_extension, validate_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
@pytest.mark.parametrize("frames", [7, 257, 2305])
def test_mixed_invalid_targets_preserve_good_outputs_and_cuda_context(mode, frames):
    generator = torch.Generator(device="cuda").manual_seed(12)
    emission = torch.randn(frames, 6, generator=generator, device="cuda").log_softmax(-1)
    token_lists = [[1, 1, 2], [-1], [6], [0], [2**32 + 1], [2**63 - 1], [-2**63], []]
    targets = [torch.tensor(t, dtype=torch.int64, device="cuda") for t in token_lists]
    result = align([emission] * len(targets), targets, mode=mode)
    assert result.status.tolist() == [0, 3, 3, 3, 3, 3, 3, 0]
    for i in range(1, 7):
        assert (result.paths[i] == -1).all()
        assert torch.isneginf(result.scores.split(result.lengths)[i]).all()
        assert torch.isneginf(result.objective[i])
        assert result.stitched[i] == 0
        assert result.repair_passes[i] == 0
        assert result.shortcut_stats[i].count_nonzero() == 0
    with pytest.raises(ValueError, match=r"invalid target IDs.*\[1, 2, 3, 4, 5, 6\]"):
        result.raise_for_status()
    # Ordinary errors must not poison the CUDA context or other batch members.
    expected = align([emission, emission], [targets[0], targets[-1]], mode=mode, check_inputs=False)
    expected.raise_for_status()
    for actual_i, expected_i in ((0, 0), (7, 1)):
        assert torch.equal(result.paths[actual_i], expected.paths[expected_i])
        assert torch.equal(result.scores.split(result.lengths)[actual_i], expected.scores.split(expected.lengths)[expected_i])


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
@pytest.mark.parametrize("tokens", [1, 129, 257, 2047])
def test_repeat_reduction_and_exact_frame_feasibility(mode, tokens):
    target = torch.ones(tokens, dtype=torch.int64, device="cuda")
    enough = torch.zeros(2 * tokens - 1, 3, device="cuda")
    targets, emissions = [target], [enough]
    if tokens > 1:
        targets.append(target)
        emissions.append(enough[:-1])
    result = align(emissions, targets, mode=mode)
    assert result.status.tolist() == ([0, 4] if tokens > 1 else [0])
    path = result.paths[0]
    collapsed = torch.unique_consecutive(path)
    assert torch.equal(collapsed[collapsed != 0], target)
    if tokens > 1:
        assert (result.paths[1] == -1).all()
        with pytest.raises(ValueError, match=r"insufficient CTC frames.*\[1\]"):
            result.raise_for_status()


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
def test_long_target_invalid_tail_and_per_record_blanks(mode):
    good = torch.arange(513, device="cuda", dtype=torch.int64) % 4
    bad = good.clone()
    bad[-1] = 2**32 + 1
    result = align(
        [torch.zeros(600, 6, device="cuda"), torch.zeros(600, 5, device="cuda")],
        [good, bad], blank_id=[5, 4], mode=mode,
    )
    assert result.status.tolist() == [0, 3]


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
def test_error_precedence_and_all_failure_categories(mode):
    emissions = [torch.zeros(3, 4, device="cuda") for _ in range(5)]
    emissions[-1].fill_(-torch.inf)
    targets = [torch.tensor(t, device="cuda") for t in ([9], [1, 1, 1], [1], [0, 0, 0, 0], [1])]
    result = align(emissions, targets, mode=mode,
                   emission_validity=torch.tensor([False, False, False, True, True], device="cuda"))
    assert result.status.tolist() == [3, 4, 2, 3, 1]
    with pytest.raises(ValueError) as error:
        result.raise_for_status()
    for message in ("invalid emissions at batch indices [2]", "no finite complete CTC path for batch indices [4]",
                    "invalid target IDs at batch indices [0, 3]", "insufficient CTC frames at batch indices [1]"):
        assert message in str(error.value)
    assert (result.labels == -1).all()
    assert torch.isneginf(result.scores).all()


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
def test_default_validation_has_no_host_read(mode):
    load_extension()
    e = torch.zeros(257, 4, device="cuda")
    good, bad = torch.tensor([1, 2], device="cuda"), torch.tensor([-1], device="cuda")
    align([e, e], [good, bad], mode=mode)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        result = align([e, e], [good, bad], mode=mode)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert result.status.tolist() == [0, 3]


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
def test_checked_and_unchecked_outputs_and_diagnostics_match(mode):
    generator = torch.Generator(device="cuda").manual_seed(42)
    emissions = [torch.randn(n, 8, device="cuda", generator=generator).log_softmax(-1) for n in (7, 257, 2305)]
    targets = [torch.tensor([1, 1, 2], device="cuda") for _ in emissions]
    checked = align(emissions, targets, mode=mode)
    unchecked = align(emissions, targets, mode=mode, check_inputs=False)
    for field in fields(checked):
        a, b = getattr(checked, field.name), getattr(unchecked, field.name)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    checked.raise_for_status()
    unchecked.raise_for_status()


def test_disabled_validation_and_explicit_synchronous_validator():
    emission = torch.zeros(2, 4, device="cuda")
    target = torch.tensor([1, 1], device="cuda")
    # In-vocabulary but infeasible inputs are safe to execute unchecked: DP
    # reports no finite path rather than a validation rejection.
    assert align([emission], [target]).status.item() == 4
    assert align([emission], [target], check_inputs=False).status.item() == 1
    with pytest.raises(ValueError, match="insufficient CTC frames"):
        validate_inputs([emission], [target])


def test_validation_on_consumer_stream_preserves_input_lifetime():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        emissions = [torch.zeros(2305, 6, device="cuda") for _ in range(2)]
        targets = [torch.tensor(t, device="cuda") for t in ([1, 2], [2**32 + 1])]
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        result = align(emissions, targets)
        del emissions, targets
    torch.cuda.current_stream().wait_stream(consumer)
    assert result.status.tolist() == [0, 3]


def test_unknown_status_still_raises():
    result = align([torch.zeros(7, 4, device="cuda")], [torch.tensor([1], device="cuda")])
    with torch.inference_mode():
        result.status.fill_(99)
    with pytest.raises(RuntimeError, match="unknown alignment status"):
        result.raise_for_status()
