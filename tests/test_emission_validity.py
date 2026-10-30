"""Numeric validation stays device-side and precedes every alignment route."""
import pytest
import torch

from fast_forced_aligner import align, emission_validity, load_extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
@pytest.mark.parametrize("frames", [7, 257, 2305])
def test_invalid_records_are_excluded_and_valid_records_unchanged(mode, frames):
    generator = torch.Generator(device="cuda").manual_seed(11)
    good = torch.randn(frames, 6, generator=generator, device="cuda").log_softmax(-1)
    negative_infinity = good.clone()
    negative_infinity[:, 5] = -torch.inf
    nan, positive_infinity = good.clone(), good.clone()
    # These vocabulary entries need not appear on the target path: the whole
    # recording is rejected, not just nonfinite scores that DP happens to read.
    nan[-1, 5] = torch.nan
    positive_infinity[0, 5] = torch.inf
    impossible = torch.full_like(good, -torch.inf)
    emissions = [good, nan, positive_infinity, negative_infinity, impossible]
    targets = [torch.tensor([1, 1, 2], device="cuda") for _ in emissions]
    mask = emission_validity(emissions)
    assert mask.dtype == torch.bool and mask.device == good.device
    assert mask.tolist() == [True, False, False, True, True]
    actual = align(emissions, targets, mode=mode, emission_validity=mask, check_inputs=False)
    assert actual.status.tolist() == [0, 2, 2, 0, 1]
    for i in (1, 2, 4):
        assert (actual.paths[i] == -1).all()
        assert torch.isneginf(actual.scores.split(actual.lengths)[i]).all()
    assert torch.isneginf(actual.objective[1:3]).all()
    assert actual.stitched[1:3].tolist() == [0, 0]
    assert actual.repair_passes[1:3].tolist() == [0, 0]
    assert actual.shortcut_stats[1:3].count_nonzero() == 0
    expected = align([good, negative_infinity], [targets[0], targets[3]], mode=mode, check_inputs=False)
    for actual_i, expected_i in ((0, 0), (3, 1)):
        assert torch.equal(actual.paths[actual_i], expected.paths[expected_i])
        assert torch.equal(actual.scores.split(actual.lengths)[actual_i], expected.scores.split(expected.lengths)[expected_i])
    assert torch.isnan(nan[-1, 5]) and torch.isposinf(positive_infinity[0, 5])
    with pytest.raises(ValueError, match="invalid emissions.*1, 2.*no finite.*4"):
        actual.raise_for_status()


@pytest.mark.parametrize("mode", ["optimized", "conventional"])
def test_empty_target_and_subset_of_all_record_validity(mode):
    emissions = [torch.zeros(260, 4, device="cuda") for _ in range(4)]
    emissions[0][-1, -1] = torch.nan
    emissions[1][-1, -1] = torch.inf
    emissions[3][-1, -1] = torch.nan
    all_records = emission_validity(emissions)
    # Rows 0 and 2 can lack an alignable transcript, but their numeric validity
    # still controls whether their emissions may be persisted by the caller.
    selection = torch.tensor([1, 3], device="cuda")
    result = align([emissions[1], emissions[3]], [torch.empty(0, dtype=torch.int64, device="cuda") for _ in range(2)],
                   mode=mode, check_inputs=False, emission_validity=all_records.index_select(0, selection))
    assert all_records.tolist() == [False, False, True, False]
    assert result.status.tolist() == [2, 2]
    assert (result.labels == -1).all()


def test_mask_does_not_bypass_target_validation():
    emission = torch.full((7, 4), torch.nan, device="cuda")
    result = align([emission], [torch.tensor([20], device="cuda")], emission_validity=emission_validity([emission]))
    with pytest.raises(ValueError, match="target IDs"):
        result.raise_for_status()


def test_numeric_scan_and_guarded_submission_do_not_read_device_values():
    load_extension()
    emission = torch.zeros(257, 4, device="cuda")
    target = torch.tensor([1, 2], device="cuda")
    # Warm allocations, extension and dispatch before sync instrumentation.
    align([emission], [target], check_inputs=False, emission_validity=emission_validity([emission]))
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        validity = emission_validity([emission])
        result = align([emission], [target], check_inputs=False, emission_validity=validity)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert result.status.item() == 0


def test_numeric_scan_and_mask_have_correct_nondefault_stream_lifetime():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        emissions = [torch.zeros(2305, 6, device="cuda"), torch.full((257, 4), torch.nan, device="cuda")]
        targets = [torch.tensor([1, 2], device="cuda") for _ in emissions]
        validity = emission_validity(emissions)
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        result = align(emissions, targets, check_inputs=False, emission_validity=validity)
        del validity, emissions, targets
    torch.cuda.current_stream().wait_stream(consumer)
    assert result.status.tolist() == [0, 2]


@pytest.mark.parametrize("kind", ["cpu", "dtype", "count", "rank", "strided"])
def test_malformed_mask_is_rejected(kind):
    mask = {
        "cpu": lambda: torch.tensor([True]),
        "dtype": lambda: torch.tensor([1], device="cuda"),
        "count": lambda: torch.empty(2, dtype=torch.bool, device="cuda"),
        "rank": lambda: torch.ones(1, 1, dtype=torch.bool, device="cuda"),
        "strided": lambda: torch.ones(4, dtype=torch.bool, device="cuda")[::2],
    }[kind]()
    with pytest.raises(ValueError, match="emission_validity"):
        align([torch.zeros(7, 4, device="cuda")], [torch.tensor([1], device="cuda")], emission_validity=mask)


@pytest.mark.parametrize("kind", ["empty", "cpu", "dtype", "strided", "zero_frames"])
def test_malformed_scan_inputs_are_rejected(kind):
    emissions = {
        "empty": lambda: [],
        "cpu": lambda: [torch.zeros(7, 4)],
        "dtype": lambda: [torch.zeros(7, 4, dtype=torch.float16, device="cuda")],
        "strided": lambda: [torch.zeros(7, 4, device="cuda").T],
        "zero_frames": lambda: [torch.zeros(0, 4, device="cuda")],
    }[kind]()
    with pytest.raises(ValueError):
        emission_validity(emissions)
