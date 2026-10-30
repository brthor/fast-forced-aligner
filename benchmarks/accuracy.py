"""Audit measured frame paths against conventional DP outside the timed region.

Objective differences use exact rational sums of the supplied binary emissions.
A different path may be better, worse, or an equal-score alternative; none of
these labels claims agreement with human timestamps.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class CaseLike(Protocol):
    case_id: str
    emissions: NDArray[np.float32]
    targets: NDArray[np.int64]
    blank_id: int


Output = tuple[np.ndarray, np.ndarray]
VALID_CLASSIFICATIONS = frozenset(("identical", "better", "worse", "equal_score_alternative"))


def array_hash(array: np.ndarray) -> str:
    """Fingerprint dtype, shape, and contents without relying on archive metadata."""
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode()
                          + np.ascontiguousarray(array).tobytes()).hexdigest()


def validate(case: CaseLike, output: Output | None, status: str = "ok") -> tuple[dict[str, Any], np.ndarray | None]:
    """Validate CTC collapse and the score selected at every original frame."""
    result: dict[str, Any] = {"status": status, "path_valid": False, "frame_scores_valid": False}
    if status != "ok":
        return result, None
    try:
        if output is None:
            raise ValueError("successful status requires output")
        labels, scores = map(np.asarray, output)
        result.update(labels_sha256=array_hash(labels), scores_sha256=array_hash(scores))
        if labels.shape != (len(case.emissions),) or scores.shape != labels.shape:
            raise ValueError("labels and scores must contain one entry per frame")
        if labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= case.emissions.shape[1]):
            raise ValueError("labels must be integer vocabulary indices")
        nonblank = labels != case.blank_id
        starts = nonblank & np.r_[True, labels[1:] != labels[:-1]] if len(labels) else nonblank
        if not np.array_equal(labels[starts], case.targets):
            raise ValueError("path does not collapse to the complete target transcript")
        states = 2 * np.cumsum(starts, dtype=np.int64) - nonblank.astype(np.int64)
        result["path_valid"] = True
        expected = case.emissions[np.arange(len(labels)), labels]
        if not np.isfinite(expected).all():
            raise ValueError("successful path selects nonfinite emissions")
        if not np.array_equal(scores, expected):
            raise ValueError("returned scores differ from selected original emissions")
        result["frame_scores_valid"] = True
        return result, states
    except (ValueError, TypeError, IndexError) as error:
        result["error"] = str(error)
        return result, None


def compare_relative(case: CaseLike, candidate: Output | None, conventional: Output | None,
                     *, candidate_status: str = "ok", conventional_status: str = "ok") -> dict[str, Any]:
    """Classify output validity, exact objective difference, and token boundaries.

    Positive score_delta means a lower (worse) candidate objective. Boundaries
    are token occurrence starts and exclusive ends, measured in emission frames.
    Failure statuses are explicit; matching failures are not proofs of infeasibility.
    """
    c, cs = validate(case, candidate, candidate_status)
    d, ds = validate(case, conventional, conventional_status)
    result: dict[str, Any] = {"case_id": case.case_id, "candidate": c, "conventional": d}
    if candidate_status != "ok" or conventional_status != "ok":
        result["classification"] = ("matching_failure_status" if candidate_status == conventional_status
                                    else "candidate_failure" if ds is not None
                                    else "candidate_success_conventional_failure" if cs is not None
                                    else "unresolved_failure")
        return result
    if ds is None or cs is None:
        result["classification"] = "invalid_conventional_output" if ds is None else "invalid_candidate_output"
        return result
    assert candidate is not None and conventional is not None
    cl, dl = np.asarray(candidate[0]), np.asarray(conventional[0])
    changed = np.flatnonzero(cl != dl)
    delta = sum((Fraction(float(case.emissions[t, dl[t]])) - Fraction(float(case.emissions[t, cl[t]]))
                 for t in changed), Fraction(0))
    classification = ("identical" if not len(changed) else "worse" if delta > 0
                      else "better" if delta < 0 else "equal_score_alternative")
    boundaries: dict[str, Any] = {}
    for name, side in (("token_starts", "left"), ("token_ends", "right")):
        token_states = 2 * np.arange(len(case.targets)) + 1
        displacement = np.searchsorted(cs, token_states, side=side) - np.searchsorted(ds, token_states, side=side)
        boundaries[name] = {"signed_frames": displacement.tolist(), "count": len(displacement),
                            "different": int(np.count_nonzero(displacement)),
                            "max_absolute_frames": int(np.max(np.abs(displacement), initial=0))}
    result.update(classification=classification, different_frames=len(changed),
                  frame_disagreement_fraction=len(changed) / max(1, len(cl)),
                  exact_score_delta=str(delta), score_delta=float(delta),
                  score_delta_per_frame=float(delta / max(1, len(cl))), **boundaries)
    return result


def summarize(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observations, preserving their distinction from distinct inputs."""
    counts = Counter(row["classification"] for row in assessments)
    worse = [row for row in assessments if row["classification"] == "worse"]
    return {"observations": len(assessments), "classifications": dict(counts),
            "failed_cases": sum(count for kind, count in counts.items() if kind not in VALID_CLASSIFICATIONS),
            "maximum_additional_score_loss": max((row["score_delta"] for row in worse), default=0.),
            "maximum_additional_score_loss_per_frame": max((row["score_delta_per_frame"] for row in worse), default=0.),
            "maximum_boundary_displacement_frames": max((row.get(name, {}).get("max_absolute_frames", 0)
                for row in assessments for name in ("token_starts", "token_ends")), default=0)}
