"""Measure resident-input alignment queues and audit every measured output."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean
import subprocess
import time
from typing import Any

import numpy as np

from accuracy import VALID_CLASSIFICATIONS, array_hash, compare_relative, summarize, validate
from adapters import METHODS, make_backend
from inputs import load_cohorts, load_manifest
from upstream import SOURCES


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    """Atomically retain complete JSON even when a later measurement fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment() -> dict[str, Any]:
    """Record hardware and relevant software without unique device/process IDs."""
    import torch
    packages = {}
    for name in ("fast_forced_aligner", "torch", "torchaudio", "k2", "numpy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result: dict[str, Any] = {
        "platform": platform.system(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "packages": packages,
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        result["cpu_model"] = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                                    if line.startswith("model name")), "unknown")
    for name, command in {
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.limit,temperature.gpu", "--format=csv,noheader"],
    }.items():
        try:
            result[name] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
        except (OSError, subprocess.SubprocessError) as error:
            result[name] = f"unavailable: {type(error).__name__}"
    return result


def source_paths() -> list[Path]:
    import fast_forced_aligner
    root = Path(__file__).resolve().parent
    package = Path(fast_forced_aligner.__file__).resolve().parent
    paths = list(root.glob("*.py")) + list(root.glob("adapters/*.py")) + list(root.glob("patches/*")) + list(root.glob("manifests/*.json"))
    paths += [root / name for name in ("Dockerfile", "requirements.txt", "upstream.py", "run.sh")]
    paths += [path for suffix in ("*.py", "*.cu", "*.cpp", "*.h", "*.cuh") for path in package.rglob(suffix)]
    return sorted({path for path in paths if path.is_file()})


def aggregate(campaign: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Require complete identical coverage before producing common aggregates."""
    observations = campaign["arguments"]["rounds"] * campaign["arguments"]["repetitions"]
    expected_keys = {(trial, repetition) for trial in range(campaign["arguments"]["rounds"])
                     for repetition in range(campaign["arguments"]["repetitions"])}
    problems: list[str] = []
    results = []
    if not campaign["methods"] or not campaign["cohorts"] or len(set(campaign["methods"])) != len(campaign["methods"]):
        problems.append("empty or duplicate method/workload coverage")
    if any(row["method"] not in campaign["methods"] or row["cohort"] not in campaign["cohorts"]
           for row in campaign["runs"]):
        problems.append("observations outside the selected matrix")
    for method in campaign["methods"]:
        cells: dict[str, float] = {}
        counts: Counter[str] = Counter()
        for cohort in campaign["cohorts"]:
            rows = [row for row in campaign["runs"] if row["method"] == method and row["cohort"] == cohort]
            keys = {(row["round"], row["repetition"]) for row in rows}
            cases = campaign["cohorts"][cohort]["cases"]
            if len(rows) != observations or keys != expected_keys or any(
                row.get("error") or row.get("failed_cases") or not isinstance(row.get("seconds"), (int, float))
                or not math.isfinite(row["seconds"]) or row["seconds"] <= 0 for row in rows
            ):
                problems.append(f"{method}/{cohort}: missing, failed, invalid, or duplicate observations")
                continue
            if any(row.get("observations") != cases or row.get("cases") != cases
                   or sum(row.get("classifications", {}).values()) != cases
                   or any(kind not in VALID_CLASSIFICATIONS or type(count) is not int or count < 0
                          for kind, count in row.get("classifications", {}).items()) for row in rows):
                problems.append(f"{method}/{cohort}: incomplete or invalid per-record audit coverage")
                continue
            cells[cohort] = mean(row["seconds"] for row in rows)
            for row in rows:
                counts.update(row["classifications"])
        seconds = sum(cells.values())
        entries = sum(cohort["cases"] for cohort in campaign["cohorts"].values())
        method_rows = [row for row in campaign["runs"] if row["method"] == method]
        results.append({"method": method, "suite_mean_ms": seconds * 1000,
                        "rows_per_second": entries / seconds if seconds else None,
                        "suite_entries": entries, "per_cohort_mean_seconds": cells,
                        "classifications": dict(counts),
                        **{key: max((row.get(key, 0) for row in method_rows), default=0) for key in
                           ("maximum_additional_score_loss", "maximum_additional_score_loss_per_frame",
                            "maximum_boundary_displacement_frames")}})
    if campaign.get("sources_changed") or campaign.get("inputs_changed"):
        problems.append("sources or input arrays changed during execution")
    if problems:
        return [], problems
    optimized = next((row["suite_mean_ms"] for row in results if row["method"] == "optimized"), None)
    for row in results:
        row["speedup_optimized"] = row["suite_mean_ms"] / optimized if optimized else None
    return results, problems


def execute_queue(backend: Any, prepared: list[Any], groups: list[list[int]]) -> dict[str, Any]:
    """Wall-clock an entire queue, including all native per-call work and completion."""
    backend.synchronize()
    raw_results = []
    start = time.perf_counter_ns()
    for indices, resident in zip(groups, prepared, strict=True):
        try:
            raw_results.append((indices, backend.run_batch(resident), None))
        except Exception as error:
            raw_results.append((indices, None, f"{type(error).__name__}: {error}"))
    completion_error = None
    try:
        backend.synchronize()
    except Exception as error:
        completion_error = f"{type(error).__name__}: {error}"
    seconds = (time.perf_counter_ns() - start) * 1e-9
    return {"seconds": seconds, "raw_results": raw_results, "completion_error": completion_error}


def audit(backend: Any, timed: dict[str, Any], cases: list[Any], references: list[Any], artifact: Path) -> dict[str, Any]:
    assessments = []
    native_statuses: Counter[str] = Counter()
    for indices, raw, error in timed["raw_results"]:
        error = error or timed["completion_error"]
        outputs = None
        codes = None
        if error is None:
            try:
                if hasattr(backend, "status_codes"):
                    codes = backend.status_codes(raw)
                    if codes is not None:
                        if len(codes) != len(indices):
                            raise ValueError("wrong native status count")
                        native_statuses.update(map(str, codes))
                outputs = backend.materialize_batch(raw)
                if len(outputs) != len(indices):
                    raise ValueError("wrong output count")
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
        for local, index in enumerate(indices):
            pair = None if outputs is None else tuple(np.asarray(value) for value in outputs[local])
            status = (f"native_status_{codes[local]}" if codes is not None and codes[local] != 0
                      else "execution_failure" if error else "ok")
            assessment = compare_relative(cases[index], pair, references[index], candidate_status=status)
            if error:
                assessment["execution_error"] = error
            if pair is not None and assessment["classification"] != "identical":
                mismatch = artifact.parent / "outputs" / f"{artifact.stem}-{index}.npz"
                mismatch.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(mismatch, labels=pair[0], scores=pair[1],
                                    conventional_labels=references[index][0], conventional_scores=references[index][1])
                assessment["output_artifact"] = str(mismatch.relative_to(artifact.parent.parent))
                assessment["output_artifact_sha256"] = sha(mismatch)
            assessments.append(assessment)
    write(artifact, {"assessments": assessments})
    return {**summarize(assessments), "native_status_counts": dict(native_statuses),
            "artifact": str(artifact.relative_to(artifact.parent.parent)), "artifact_sha256": sha(artifact)}


def main() -> int:
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--suite", choices=("readme", "regenerated-readme", "regenerated-readme-partial", "synthetic"), default="regenerated-readme")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nemo-batch-order", choices=("input", "length"), default="input",
                        help="NeMo queue order; sorting, batching and restoring order are timed")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if min(args.batch_size, args.rounds, args.repetitions, args.workers) < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("batch size, rounds, repetitions, workers must be positive; methods must be distinct")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    sources = {str(path): sha(path) for path in source_paths()}
    campaign: dict[str, Any] = {
        "schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "methods": list(args.methods), "method_selection": "full" if set(args.methods) == set(METHODS) else "explicit subset",
        "source_hashes": sources, "environment": environment(), "upstream_sources": SOURCES, "backend_builds": {},
        "input_manifest": load_manifest(args.inputs) if args.suite != "synthetic" else {"suite": "synthetic", "generator_sha256": sha(Path(__file__).parent / "inputs.py")},
        "timing": "Host wall time from resident inputs through complete queue outputs and synchronization; per-call preparation, allocation, internal transfers and traceback included. Initial placement/admission, compilation, warmup, reference execution and output audits excluded. Checked lanes include repeated public input validation.",
        "aggregation": "Sum of per-cohort arithmetic mean complete-queue times; all observations retained. Speedup is comparator total / optimized total; repetitions and overlapping workloads are not distinct inputs.",
        "cohorts": {}, "runs": [], "summary": [], "errors": [], "sources_changed": [], "inputs_changed": [],
    }
    campaign_path = args.output / "campaign.json"
    write(campaign_path, campaign)
    try:
        cohorts = load_cohorts(args.inputs, suite=args.suite)
        for cohort, cases in cohorts:
            if not cases:
                raise ValueError(f"empty cohort: {cohort}")
            fingerprints = [(array_hash(case.emissions), array_hash(case.targets), int(case.blank_id)) for case in cases]
            campaign["cohorts"][cohort] = {"cases": len(cases), "frames": sum(len(case.emissions) for case in cases),
                "case_ids": [case.case_id for case in cases],
                "input_fingerprints": [hashlib.sha256(json.dumps(item).encode()).hexdigest() for item in fingerprints]}
            groups = [list(range(i, min(i + args.batch_size, len(cases)))) for i in range(0, len(cases), args.batch_size)]
            reference_backend = make_backend("conventional", workers=args.workers, device=args.device)
            references = []
            try:
                for indices in groups:
                    resident = reference_backend.prepare_batch([cases[index] for index in indices])
                    raw = reference_backend.run_batch(resident)
                    references.extend(reference_backend.materialize_batch(raw))
                if len(references) != len(cases):
                    raise ValueError("conventional reference returned wrong output count")
                for case, reference in zip(cases, references, strict=True):
                    checked, states = validate(case, reference)
                    if states is None:
                        raise ValueError(f"invalid conventional reference for {case.case_id}: {checked}")
            finally:
                reference_backend.close()
            del reference_backend, resident, raw
            for trial in range(args.rounds):
                for method in args.methods if trial % 2 == 0 else reversed(args.methods):
                    print(f"START {cohort} {method} round={trial}", flush=True)
                    torch.set_num_threads(args.workers if method == "nemo-cpu" else 1)
                    backend = None
                    recorded: set[int] = set()
                    prepared: list[Any] = []
                    measured: list[dict[str, Any]] = []
                    resident = None
                    timed = None
                    try:
                        backend = make_backend(method, workers=args.workers, device=args.device,
                                               batch_size=args.batch_size, nemo_batch_order=args.nemo_batch_order)
                        if hasattr(backend, "build_manifest"):
                            campaign["backend_builds"][method] = backend.build_manifest
                        # NeMo schedules from the whole resident queue inside its
                        # timed call so length sorting cannot become free preparation.
                        method_groups = [list(range(len(cases)))] if method.startswith("nemo-") else groups
                        prepared = [backend.prepare_batch([cases[index] for index in indices]) for indices in method_groups]
                        for resident in prepared:
                            raw = backend.run_batch(resident)
                            backend.synchronize()
                            del raw
                        # Keep all measured outputs alive; do not audit/log between timed repetitions.
                        measured = [execute_queue(backend, prepared, method_groups) for _ in range(args.repetitions)]
                        for repetition, timed in enumerate(measured):
                            artifact = args.output / "raw" / f"{cohort}-{method}-{trial}-{repetition}.json"
                            row = {"cohort": cohort, "method": method, "round": trial, "repetition": repetition,
                                   "seconds": timed["seconds"], "cases": len(cases),
                                   **audit(backend, timed, cases, references, artifact)}
                            if timed["completion_error"]:
                                row["error"] = timed["completion_error"]
                            campaign["runs"].append(row)
                            recorded.add(repetition)
                            write(campaign_path, campaign)
                            print(f"DONE {cohort} {method} {row['seconds'] * 1000:.4f} ms {row['classifications']}", flush=True)
                    except Exception as error:
                        message = f"{type(error).__name__}: {error}"
                        campaign["errors"].append({"cohort": cohort, "method": method, "round": trial, "error": message})
                        for repetition in range(args.repetitions):
                            if repetition not in recorded:
                                failed_timing = measured[repetition] if repetition < len(measured) else None
                                campaign["runs"].append({"cohort": cohort, "method": method, "round": trial,
                                    "repetition": repetition, "seconds": failed_timing["seconds"] if failed_timing else None,
                                    "cases": len(cases),
                                    "batch_errors": [error for _, _, error in failed_timing["raw_results"] if error] if failed_timing else [],
                                    "failed_cases": len(cases), "classifications": {"unmeasured_failure": len(cases)}, "error": message})
                        write(campaign_path, campaign)
                        print(f"FAILED {cohort} {method}: {message}", flush=True)
                    finally:
                        measured.clear()
                        prepared.clear()
                        resident = timed = None
                        if backend is not None:
                            backend.close()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            after = [(array_hash(case.emissions), array_hash(case.targets), int(case.blank_id)) for case in cases]
            if after != fingerprints:
                campaign["inputs_changed"].append(cohort)
        final_sources = {str(path): sha(path) for path in source_paths()}
        campaign["sources_changed"] = sorted(path for path in sources.keys() | final_sources.keys()
                                              if sources.get(path) != final_sources.get(path))
        campaign["summary"], campaign["incomplete_cells"] = aggregate(campaign)
        campaign["complete"] = not campaign["incomplete_cells"]
        campaign["completed_utc"] = datetime.now(timezone.utc).isoformat()
        campaign["final_environment"] = environment()
        write(campaign_path, campaign)
        return 0 if campaign["complete"] else 1
    except BaseException as error:
        campaign["complete"] = False
        campaign["fatal_error"] = f"{type(error).__name__}: {error}"
        write(campaign_path, campaign)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
