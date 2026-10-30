"""Compare fixed NeMo schedules, then independently remeasure the suite-wide winners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from benchmark import aggregate, write
from report import verify_audits

# B1 has no padding across records, so sorting it is a redundant configuration.
CONFIGURATIONS = ((1, "input"), (16, "input"), (64, "input"),
                  (16, "length"), (64, "length"))
METHODS = ("nemo-cpu", "nemo-cuda")


def verified_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads((path / "campaign.json").read_text())
    verify_audits(campaign, path)
    summary, errors = aggregate(campaign)
    if not campaign.get("complete") or errors or summary != campaign["summary"]:
        raise ValueError(f"incomplete or inconsistent campaign: {path}")
    return campaign


def select_configurations(candidates: list[dict[str, Any]], methods: list[str]) -> dict[str, str]:
    """Select one configuration per method by the entire suite, never per workload."""
    if len({row["configuration"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate configuration")
    selected = {}
    for method in methods:
        if not candidates or any(method not in row["totals_ms"] for row in candidates):
            raise ValueError(f"incomplete sweep for {method}")
        selected[method] = min(candidates, key=lambda row: row["totals_ms"][method])["configuration"]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("readme", "regenerated-readme", "regenerated-readme-partial", "synthetic"), default="regenerated-readme")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot-repetitions", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if min(args.workers, args.pilot_repetitions, args.rounds, args.repetitions) < 1:
        parser.error("worker and measurement counts must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be distinct")
    args.output.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "complete": False,
        "protocol": "Five fixed full-suite schedules; one global winner per method selected by "
                    "pilot total. Selected schedules and B64 input-order control are independently "
                    "remeasured. Sorting, batching, padding and output order restoration are timed. "
                    "Tuning and confirmation share inputs; this is configuration tuning, not held-out generalization.",
        "candidates": [], "selected": {}, "confirmations": [],
    }
    state_path = args.output / "sweep.json"
    write(state_path, state)
    identity: dict[str, Any] | None = None

    def run(name: str, batch: int, order: str, methods: list[str], rounds: int,
            repetitions: int) -> dict[str, Any]:
        nonlocal identity
        destination = args.output / name
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("benchmark.py")),
            "--inputs", str(args.inputs), "--output", str(destination), "--suite", args.suite,
            "--batch-size", str(batch), "--nemo-batch-order", order,
            "--rounds", str(rounds), "--repetitions", str(repetitions),
            "--workers", str(args.workers), "--device", args.device, "--methods", *methods,
        ], check=True)
        campaign = verified_campaign(destination)
        current = {key: campaign[key] for key in ("cohorts", "source_hashes", "upstream_sources")}
        if identity is not None and current != identity:
            raise ValueError("sweep inputs or source files differ across configurations")
        identity = current
        return {"configuration": f"b{batch}-{order}", "batch_size": batch, "order": order,
                "campaign": name, "totals_ms": {row["method"]: row["suite_mean_ms"]
                                                for row in campaign["summary"]}}

    try:
        for batch, order in CONFIGURATIONS:
            name = f"pilot-b{batch}-{order}"
            state["candidates"].append(run(name, batch, order, args.methods, 1, args.pilot_repetitions))
            write(state_path, state)
        state["selected"] = select_configurations(state["candidates"], args.methods)
        write(state_path, state)
        # Always retain a contemporaneous matched-B64 control, even when it loses.
        for batch, order in CONFIGURATIONS:
            configuration = f"b{batch}-{order}"
            methods = [method for method in args.methods
                       if state["selected"][method] == configuration or configuration == "b64-input"]
            if methods:
                state["confirmations"].append(run(f"confirm-{configuration}", batch, order, methods,
                                                  args.rounds, args.repetitions))
                write(state_path, state)
        state["complete"] = True
        write(state_path, state)
        for row in state["confirmations"]:
            print(row, flush=True)
        return 0
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
        write(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
