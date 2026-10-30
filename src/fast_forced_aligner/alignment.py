"""Public resident-input interface to the retained CUDA implementation."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

Mode = Literal["optimized", "conventional"]


@lru_cache(maxsize=1)
def load_extension() -> None:
    """Build and load the bundled CUDA extension, reusing PyTorch's build cache.

    Installation and package import do not compile the extension. The first
    call loads a compatible cached build or compiles the shipped CUDA source;
    successful calls are then memoized for the lifetime of this Python process.
    ``align`` calls this function automatically. Call it explicitly before
    warming up and timing alignment to exclude compilation from measurements.

    Returns:
        None. The extension registers its operators with PyTorch as a side effect.

    Raises:
        RuntimeError: PyTorch lacks CUDA support, no compatible NVIDIA GPU is
            accessible, the CUDA development toolkit is missing, or extension
            compilation/loading fails. CUDA runtime libraries alone do not
            provide the required ``nvcc`` compiler.

    Notes:
        Requires a CUDA-enabled PyTorch installation, a compatible GPU driver,
        the CUDA development toolkit, a C++ compiler, and Ninja. Set
        ``CUDA_HOME`` before importing PyTorch's extension utilities if the
        toolkit is outside its usual location. PyTorch honors
        ``TORCH_EXTENSIONS_DIR`` for the persistent build cache and ``MAX_JOBS``
        for build parallelism. These settings belong to the process environment,
        not to the alignment call. Loading can block while building the extension.
    """
    from torch.utils.cpp_extension import CUDA_HOME, load

    if torch.version.cuda is None:
        raise RuntimeError("fast_forced_aligner requires a CUDA-enabled PyTorch installation")
    if not torch.cuda.is_available():
        raise RuntimeError("fast_forced_aligner requires an accessible NVIDIA GPU and compatible driver")
    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        raise RuntimeError(
            "fast_forced_aligner ships CUDA source and compiles it on first use. "
            "Install the matching CUDA development toolkit (including nvcc) and a C++ compiler; "
            "set CUDA_HOME if the toolkit is installed outside its usual location. "
            "PyTorch's CUDA runtime libraries alone are insufficient."
        )

    load(
        name="fast_forced_aligner_cuda",
        sources=[str(Path(__file__).parent / "csrc" / "alignment.cu")],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        is_python_module=False,
        verbose=False,
    )


@dataclass(frozen=True)
class AlignmentBatch:
    """Hold packed frame alignments and per-record diagnostics returned by ``align``.

    Let ``B`` be the batch size, ``T_i`` the frame count of record ``i``, and
    ``N = sum(T_i)``. Every tensor resides on the input CUDA device. Packed
    outputs concatenate records in input order, without padding. Diagnostic
    tensors remain on the GPU; reading their values on the host synchronizes.

    Attributes:
        lengths (tuple[int, ...]): Host-known frame counts ``(T_0, ..., T_{B-1})``.
            These are lengths, not cumulative offsets.
        labels (Tensor): Contiguous int64 tensor of shape ``(N,)`` containing
            vocabulary IDs, including blanks and repeated labels. For a
            successful record, merging consecutive equal labels and then
            removing blanks recovers its target sequence. Failed records
            contain ``-1`` at every frame.
        scores (Tensor): Contiguous float32 tensor of shape ``(N,)``. Each
            successful frame contains its selected input emission value,
            ``emissions[i][t, paths[i][t]]``; these are not probabilities unless
            the supplied scores were probabilities. Failed records contain
            negative infinity.
        status (Tensor): Int32 tensor of shape ``(B,)``. Zero indicates success;
            one indicates that no finite complete path was produced, including
            when FP32 accumulation overflows; two indicates a record rejected
            by the optional emission-validity mask before alignment; three
            indicates invalid target IDs; four indicates insufficient CTC frames.
            Target errors take precedence over the emission mask. Failure does
            not automatically raise; call ``raise_for_status`` before consuming paths.
        objective (Tensor): Float32 tensor of shape ``(B,)`` containing the
            native accumulated path scores. Optimized chunk sums and conventional
            sequential sums can use different accumulation orders. For numerical
            comparisons, rescore both paths from their selected emissions using
            the same accumulation method, for example per-record FP64 sums.
            Objectives for failed records are not valid alignment scores.
        stitched (Tensor): Int32 tensor of shape ``(B,)``. One means the optimized
            chunk paths were successfully joined, possibly after repair. Zero
            means conventional DP was used, either directly or as fallback;
            it does not itself indicate failure. Always zero in conventional mode.
        repair_passes (Tensor): Int32 tensor of shape ``(B,)`` counting repair
            rounds that merged at least one region, from zero to two. Rejected
            rounds are not counted. Always zero in conventional mode.
        repaired_frames (Tensor): Int64 tensor of shape ``(B,)`` summing merged
            region lengths over accepted repair rounds. A frame may be counted
            in multiple rounds, and frontier reuse may skip part of a region;
            this is not a count of actual DP state updates.
        shortcut_stats (Tensor): Int64 tensor of shape ``(B, 3)``. Columns count
            accepted greedy chunks, frames in those chunks, and chunks rejected
            because the greedy path had multiple legal target-state placements.
            The last column does not count every reason for rejecting a shortcut.
            Accepted chunks can subsequently participate in repair or fallback.
            All columns are zero in conventional mode.
        routing_stats (Tensor): Int32 tensor of shape ``(B, 3)``. Columns contain
            the early-fallback state, number of budget-rejected repair rounds,
            and largest proposed connected region in those rejected rounds
            (frames). After completion, the state is zero for the ordinary route
            or two for completed early fallback; one is an internal queued state.
            All columns are zero in conventional mode.
        reuse_stats (Tensor): Int64 tensor of shape ``(B, 4)`` when frontier reuse
            is enabled, otherwise ``(0, 4)``. It is enabled only in optimized mode
            when at least one record has 2048 or more frames. Columns count
            recurrence frames skipped during repair, frames skipped during
            fallback, frontier score values written, and frontier score values
            read. The last two columns count values, not bytes or frames.

    Notes:
        Instances are normally constructed by ``align``. The frozen dataclass
        prevents rebinding fields, but does not make the contained tensors
        immutable. Tensor operations obey PyTorch's CUDA stream semantics;
        returning this object does not imply that GPU work has completed.
    """

    lengths: tuple[int, ...]
    labels: Tensor
    scores: Tensor
    status: Tensor
    objective: Tensor
    stitched: Tensor
    repair_passes: Tensor
    repaired_frames: Tensor
    shortcut_stats: Tensor
    routing_stats: Tensor
    reuse_stats: Tensor

    @property
    def paths(self) -> tuple[Tensor, ...]:
        """Return per-record views into the packed frame labels.

        Returns:
            tuple[Tensor]: ``B`` CUDA int64 tensors, where element ``i`` has
            shape ``(lengths[i],)`` and retains blanks and repeated labels.
            Each tensor shares storage with ``labels``; modifying a view also
            modifies the packed result. This property neither copies tensor
            data nor synchronizes with the host, and does not check ``status``.
        """
        return self.labels.split(self.lengths)

    def raise_for_status(self) -> None:
        """Check completion status on the host and reject failed alignments.

        This is an optional convenience for callers that want any failed record
        to raise a Python exception. ``align`` records target-validation and
        alignment failures in the GPU ``status`` tensor instead of raising
        automatically:
        zero means success, and one means no finite complete path was produced,
        including FP32 accumulation overflow; two means the emission-validity
        mask rejected the input; three means invalid target IDs; four means
        insufficient CTC frames. Metadata and runtime errors are separate and
        may already have raised during ``align``.

        Returns:
            None if every record has status zero. No labels or scores are copied
            to the host, but checking GPU status requires synchronization even
            when every record succeeds.

        Raises:
            ValueError: A record has status one, two, three, or four. The message
                lists the zero-based batch indices and their failure categories.
            RuntimeError: An unknown status or pending asynchronous CUDA execution
                error surfaces while checking status. This is distinct from a
                failed alignment.

        Notes:
            Calling this method after every batch makes the host wait for that
            batch's status, potentially reducing overlap with subsequent work.
            Throughput-oriented callers can defer it until multiple batches have
            been submitted, retaining each result for later checking. Default
            target validation also runs asynchronously on the GPU. Deferring
            this method does not change the need for correct stream ordering.

            Callers that handle failed recordings individually may omit this
            method and inspect ``status`` directly. ``self.status.cpu().tolist()``
            returns one integer per recording and also synchronizes; it permits
            skipping failures while retaining successful records. Forming
            ``self.status == 0`` keeps the validity mask on the GPU without
            reading its values on the host.

            Deferring or omitting this method does not make failed outputs valid.
            Failed records contain ``-1`` labels and negative-infinity scores;
            exclude or otherwise handle them before consuming alignment paths.
            Synchronizing CUDA or copying labels to the CPU does not by itself
            convert a nonzero alignment status into a Python exception. This
            method does not rerun alignment or modify the result tensors.
        """
        statuses = self.status.cpu().tolist()
        if not any(statuses):
            return
        unknown = [i for i, value in enumerate(statuses) if value not in (0, 1, 2, 3, 4)]
        if unknown:
            raise RuntimeError(f"unknown alignment status at batch indices {unknown}")
        invalid = [i for i, value in enumerate(statuses) if value == 2]
        failed = [i for i, value in enumerate(statuses) if value == 1]
        messages = []
        if invalid:
            messages.append(f"invalid emissions at batch indices {invalid}")
        if failed:
            messages.append(f"no finite complete CTC path for batch indices {failed}")
        invalid_targets = [i for i, value in enumerate(statuses) if value == 3]
        insufficient_frames = [i for i, value in enumerate(statuses) if value == 4]
        if invalid_targets:
            messages.append(f"invalid target IDs at batch indices {invalid_targets}")
        if insufficient_frames:
            messages.append(f"insufficient CTC frames at batch indices {insufficient_frames}")
        if messages:
            raise ValueError("; ".join(messages))


def _metadata(
    emissions: Sequence[Tensor], targets: Sequence[Tensor], blank_id: int | Sequence[int],
) -> list[int]:
    count = len(emissions)
    if not 0 < count <= 65535 or len(targets) != count:
        raise ValueError("provide 1..65535 matching emissions and targets")
    blanks = [blank_id] * count if type(blank_id) is int else list(blank_id)
    if len(blanks) != count:
        raise ValueError("provide one blank ID per record")
    device = emissions[0].device
    for e, t, blank in zip(emissions, targets, blanks, strict=True):
        if (e.device.type != "cuda" or e.device != device or t.device != device
                or e.dtype != torch.float32 or t.dtype != torch.int64
                or e.ndim != 2 or t.ndim != 1
                or not e.is_contiguous() or not t.is_contiguous()):
            raise ValueError("inputs must be contiguous CUDA float32 [T,V] and int64 [U] on one device")
        if (not 0 < e.shape[0] <= 2**31 - 1 or not 0 < e.shape[1] <= 8192
                or t.numel() > 2047 or type(blank) is not int or not 0 <= blank < e.shape[1]):
            raise ValueError("unsupported frame count, vocabulary, target length, or blank ID")
    return blanks


@torch.inference_mode()
def validate_inputs(
    emissions: Sequence[Tensor], targets: Sequence[Tensor], *,
    blank_id: int | Sequence[int] = 0,
) -> None:
    """Validate resident tensor metadata, target values, and CTC frame feasibility.

    This performs GPU target checks and copies one Boolean result per record to
    the host, synchronizing that work. It does not compile the alignment extension,
    run alignment, normalize scores, or scan the emission matrices for nonfinite
    values. Topological feasibility does not guarantee a finite-scoring path.

    Args:
        emissions (Sequence[Tensor]): ``B`` contiguous CUDA float32 tensors of
            shape ``(T_i, V_i)``, with ``1 <= B <= 65535``,
            ``1 <= T_i <= 2**31 - 1``, and ``1 <= V_i <= 8192``. Frame counts and
            vocabulary sizes may vary between records. Values must be finite
            scores or negative infinity; NaN and positive infinity are forbidden
            caller-side preconditions, not checked by this function.
        targets (Sequence[Tensor]): ``B`` contiguous CUDA int64 tensors of shape
            ``(U_i,)`` in the same order as ``emissions``, with
            ``0 <= U_i <= 2047``. All emissions and targets must share one CUDA
            device. Each target ID must lie in ``[0, V_i)`` and differ from its
            blank ID. Empty targets are allowed. Feasibility requires at least
            ``U_i + number_of_adjacent_equal_target_pairs`` frames, since each
            repeated pair needs an intervening blank.
        blank_id (int | Sequence[int]): Blank vocabulary ID, default ``0``. A
            single Python int applies to every record; a sequence must contain
            exactly ``B`` Python ints, each in its record's ``[0, V_i)`` range.
            Boolean values are not valid blank IDs.

    Returns:
        None if all checked conditions pass. Inputs are not modified.

    Raises:
        ValueError: Batch sizes, tensor metadata, supported dimensions, blank IDs,
            target IDs, or minimum frame counts violate the input contract.
            Target-value and frame-feasibility errors list failing batch indices.

    Notes:
        After successful validation, the same valid inputs can be passed to
        ``align(..., check_inputs=False)`` without repeating GPU target checks.
        Changing target values, shapes, or blank IDs requires validating again.
        Validation does not guarantee sufficient GPU memory or that native
        scratch-buffer indexing limits are satisfied for an extreme batch.
    """
    blanks = _metadata(emissions, targets, blank_id)
    valid = []
    for e, t, blank in zip(emissions, targets, blanks, strict=True):
        tokens_ok = ((t >= 0) & (t < e.shape[1]) & (t != blank)).all()
        frames_ok = t.numel() + (t[1:] == t[:-1]).sum() <= e.shape[0]
        valid.append(tokens_ok & frames_ok)
    failures = [i for i, ok in enumerate(torch.stack(valid).cpu().tolist()) if not ok]
    if failures:
        raise ValueError(f"invalid target IDs or insufficient CTC frames at batch indices {failures}")


@torch.inference_mode()
def emission_validity(emissions: Sequence[Tensor]) -> Tensor:
    """Return a CUDA bool mask rejecting NaN/+inf in each emission recording.

    Accepts 1..65535 contiguous CUDA FP32 ``[T,V]`` tensors on one device,
    with the same dimension limits as ``align``. Finite values and negative
    infinity are valid. Every element is scanned, including vocabulary entries
    not visited by an alignment path. No GPU values are read by the host.

    This helper also supports records without alignable targets. Pass the mask
    (or an ordered subset) to ``align(..., emission_validity=mask)`` to exclude
    invalid recordings before any alignment recurrence or greedy shortcut. Copy
    the full mask to the host at the normal completion boundary if failed input
    rows need special handling. Metadata errors raise immediately; CUDA execution
    errors retain their usual asynchronous behavior.

    Work runs on the current CUDA stream. Inputs must not be mutated between
    this scan and alignment; establish stream dependencies when handing inputs,
    masks, or outputs between streams. Reading the returned mask on the host
    synchronizes, so throughput-sensitive callers should defer that read.
    """
    if not 0 < len(emissions) <= 65535:
        raise ValueError("provide 1..65535 emission tensors")
    device = emissions[0].device
    for e in emissions:
        if (e.device.type != "cuda" or e.device != device or e.dtype != torch.float32
                or e.ndim != 2 or not e.is_contiguous()
                or not 0 < e.shape[0] <= 2**31 - 1 or not 0 < e.shape[1] <= 8192):
            raise ValueError("emissions must be contiguous CUDA float32 [T,V] with supported dimensions on one device")
    with torch.cuda.device(device):
        load_extension()
        return torch.ops.fast_forced_aligner_cuda.emission_validity(list(emissions))


@torch.inference_mode()
def align(
    emissions: Sequence[Tensor], targets: Sequence[Tensor], *,
    blank_id: int | Sequence[int] = 0, mode: Mode = "optimized",
    check_inputs: bool = True, emission_validity: Tensor | None = None,
) -> AlignmentBatch:
    """Align known CTC targets to resident emissions and return CUDA frame paths.

    Each record is aligned independently against its complete target sequence.
    Both modes prepare the CTC topology, allocate scratch, and produce one label
    and selected emission score per input frame on every call. This operation
    does not run an acoustic model, copy input emissions between devices, or
    convert frame indices into timestamps.

    Args:
        emissions (Sequence[Tensor]): Sequence of ``B`` contiguous CUDA float32
            tensors with shape ``(T_i, V_i)``: frames by vocabulary, including
            the blank class. Supply unpadded records; ``T_i`` and ``V_i`` may vary.
            Supported sizes are ``1 <= B <= 65535``, ``1 <= T_i <= 2**31 - 1``,
            and ``1 <= V_i <= 8192``, subject to native indexing and memory limits.
            Normally these are log probabilities, e.g.
            ``logits.float().log_softmax(-1)``. Values must be finite or negative
            infinity, never NaN or positive infinity. Scores are used as supplied:
            no normalization is performed. Without an emission_validity mask,
            absence of NaN/+inf remains a caller-side precondition.
        targets (Sequence[Tensor]): Sequence of ``B`` contiguous CUDA int64
            tensors with shape ``(U_i,)``, in input order, containing the known
            transcript's vocabulary IDs without inserted blanks. All tensors
            in both sequences must reside on the same CUDA device. Each ID must
            be in ``[0, V_i)`` and differ from the record's blank ID, with
            ``0 <= U_i <= 2047``. Empty targets produce all-blank paths if their
            scores are finite. Adjacent repeated target IDs require intervening
            blank frames, so ``T_i`` must be at least ``U_i`` plus the number of
            adjacent equal target pairs.
        blank_id (int | Sequence[int]): Blank vocabulary ID, default ``0``.
            Supply one Python int for the entire batch, or a sequence of ``B``
            Python ints for per-record blanks. Each must lie in ``[0, V_i)``;
            Boolean values are not accepted.
        mode (Literal["optimized", "conventional"]): Alignment implementation,
            default ``"optimized"``. Optimized mode uses 128-frame chunks,
            greedy shortcuts, up to two repair rounds, and conventional DP
            fallback. Conventional mode directly runs the full FP32 dynamic
            program. Both return the same output structure. Different FP32
            accumulation orders can select different paths near numerical ties;
            optimized mode does not promise bitwise agreement or an identical
            objective to conventional DP for every input.
        check_inputs (bool): Whether to validate target values and CTC frame
            feasibility during GPU input preparation, default ``True``. No
            validation result is copied to the host. Invalid target IDs produce
            status three; insufficient frames produce status four. Rejected
            records skip alignment safely; call ``raise_for_status`` to raise
            on the host, or inspect ``status`` directly.
            Set ``False`` only for inputs already satisfying ``validate_inputs``
            and the emission-value preconditions. Metadata checks and all native
            input preparation still run. Invalid target IDs with checking disabled
            can cause out-of-bounds native reads; this is not a safe way to accept
            malformed inputs. This option does not scan emission values or
            guarantee a finite-scoring path.
            Must be a Python bool.
        emission_validity (Tensor | None): Optional contiguous CUDA bool ``[B]``
            mask on the input device, normally produced by ``emission_validity``.
            False excludes a record before greedy or DP execution, returning
            status two unless target validation detects invalid IDs (three) or
            insufficient frames (four). Rejected records have labels -1 and
            scores/objective negative infinity. True asserts that the input
            contains no NaN/+inf. The mask is not read
            on the host and does not replace target validation. It must match
            the current, unmodified emissions in input order. Default None
            preserves the original valid-emission caller precondition.

    Returns:
        AlignmentBatch: Packed CUDA outputs in input order. ``labels`` and
        ``scores`` have shape ``(sum(T_i),)`` with dtypes int64 and float32;
        ``paths[i]`` views record ``i``'s frame labels. Blanks and repetitions are
        retained. Per-record ``status`` is zero on success and one if no finite
        complete path was produced, including FP32 accumulation overflow. These
        failures, status two for a mask-rejected input, status three for invalid
        target IDs, and status four for insufficient frames do not automatically
        raise, even with ``check_inputs=True``;
        they fill labels with ``-1`` and scores with negative infinity. Call
        ``result.raise_for_status()`` to raise on any failed record, or inspect
        ``result.status`` to handle records individually. The object also contains
        native objectives and GPU diagnostic tensors; see ``AlignmentBatch``
        for their contracts.

    Raises:
        ValueError: The mode, metadata, blank IDs, or emission-validity mask
            metadata are invalid. Target-value errors are reported in ``status``.
        TypeError: ``check_inputs`` is not a Python bool.
        RuntimeError: CUDA or build prerequisites are missing, extension loading
            fails, native scratch indexing limits are exceeded, or CUDA execution
            or allocation fails. Asynchronous CUDA errors may surface later.

    Notes:
        Error handling has two stages. Invalid metadata and extension-loading
        failures raise during the call as described above. Target validity,
        frame feasibility, and whether a finite alignment exists are determined
        on the GPU and recorded per record in ``result.status``. The function
        leaves that status on the
        device so the host need not wait for the alignment outcome before
        submitting more work. CUDA execution faults remain runtime errors and
        may surface at a later CUDA operation; they are not converted to record
        statuses.

        For ordinary use, call ``result.raise_for_status()`` before consuming
        paths. It reads status on the host and synchronizes even on success.
        To preserve overlap, defer this call until a convenient completion
        boundary, or handle ``result.status`` directly. For example,
        ``result.status.cpu().tolist()`` synchronizes and lets you skip failed
        recordings without raising for the entire batch. ``result.status == 0``
        instead forms a GPU validity mask without a host read. Failed records
        must still be excluded or handled; a CUDA synchronization or CPU copy of
        the outputs alone does not check alignment status. Default target checks
        do not introduce a device-to-host transfer or validation synchronization.

        This is inference-only; autograd is disabled. Work is submitted to the
        current CUDA stream for the input device. Order input-producing work
        before this call, and establish stream dependencies before consuming
        outputs on a different stream. Do not mutate inputs while execution is
        in flight. Inputs are not modified by this function.

        ``load_extension`` runs lazily on first use. For steady-state timings,
        load and warm up first, then measure through GPU completion, including
        preparation and any validation you enable. Native preparation includes
        a small host-to-device metadata upload. Returning GPU tensors does not
        itself synchronize completion or copy frame paths to the CPU.

    Examples:
        >>> emissions = [torch.randn(8, 4, device="cuda").log_softmax(-1)]
        >>> targets = [torch.tensor([1, 2], device="cuda", dtype=torch.int64)]
        >>> result = align(emissions, targets, blank_id=0)
        >>> result.raise_for_status()
        >>> result.paths[0].shape
        torch.Size([8])
    """
    if mode not in ("optimized", "conventional"):
        raise ValueError("mode must be 'optimized' or 'conventional'")
    if type(check_inputs) is not bool:
        raise TypeError("check_inputs must be bool")
    blanks = _metadata(emissions, targets, blank_id)
    if emission_validity is not None:
        if (not isinstance(emission_validity, Tensor)
                or emission_validity.device != emissions[0].device
                or emission_validity.dtype != torch.bool or emission_validity.ndim != 1
                or emission_validity.numel() != len(emissions)
                or not emission_validity.is_contiguous()):
            raise ValueError("emission_validity must be contiguous CUDA bool [batch] on the input device")
    with torch.cuda.device(emissions[0].device):
        load_extension()
        optimized = mode == "optimized"
        tensors, lengths = torch.ops.fast_forced_aligner_cuda.align(
            list(emissions), list(targets), blanks,
            128, 2 if optimized else 0, 512, 2, 2, 2,
            512 if optimized else 0, 0, False, True, optimized,
            2048, 2048, optimized, 2048, emission_validity, check_inputs,
        )
    return AlignmentBatch(tuple(lengths), *tensors)
