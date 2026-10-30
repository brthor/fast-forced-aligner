"""Configuration selection uses complete suite totals, not per-workload minima."""
import json
import sys

import pytest

import nemo_sweep
from nemo_sweep import select_configurations


def test_cpu_and_gpu_choose_independent_global_configurations():
    candidates = [
        {"configuration": "b1-input", "totals_ms": {"nemo-cpu": 10, "nemo-cuda": 30}},
        {"configuration": "b64-length", "totals_ms": {"nemo-cpu": 20, "nemo-cuda": 15}},
    ]
    assert select_configurations(candidates, ["nemo-cpu", "nemo-cuda"]) == {
        "nemo-cpu": "b1-input", "nemo-cuda": "b64-length"}


def test_missing_and_duplicate_configurations_fail_closed():
    row = {"configuration": "b64-input", "totals_ms": {"nemo-cpu": 10}}
    with pytest.raises(ValueError, match="incomplete"):
        select_configurations([row], ["nemo-cuda"])
    with pytest.raises(ValueError, match="duplicate"):
        select_configurations([row, row], ["nemo-cpu"])
    with pytest.raises(ValueError, match="incomplete"):
        select_configurations([], ["nemo-cpu"])


def test_sweep_remeasures_selected_schedules_and_control(tmp_path, monkeypatch):
    calls = []

    def execute(arguments, **kwargs):
        calls.append(arguments)

    def campaign(path):
        # CPU prefers B1, GPU prefers sorted B64. Confirmation values differ so
        # the test catches accidental reuse of pilot totals as final evidence.
        cpu = 10 if "b1-input" in path.name else 20
        gpu = 15 if "b64-length" in path.name else 30
        if path.name.startswith("confirm"):
            cpu += 1
            gpu += 1
        arguments = calls[-1]
        methods = arguments[arguments.index("--methods") + 1:]
        return {"cohorts": {"fixture": {"cases": 3}}, "source_hashes": {"fixture": "hash"},
                "upstream_sources": {}, "summary": [{"method": method, "suite_mean_ms":
                    cpu if method == "nemo-cpu" else gpu} for method in methods]}

    output = tmp_path / "sweep"
    monkeypatch.setattr(sys, "argv", ["sweep", "--inputs", str(tmp_path), "--output", str(output)])
    monkeypatch.setattr(nemo_sweep.subprocess, "run", execute)
    monkeypatch.setattr(nemo_sweep, "verified_campaign", campaign)
    assert nemo_sweep.main() == 0
    result = json.loads((output / "sweep.json").read_text())
    assert result["complete"]
    assert len(result["candidates"]) == 5
    assert len(result["confirmations"]) == 3
    assert result["selected"] == {"nemo-cpu": "b1-input", "nemo-cuda": "b64-length"}
    confirmations = {row["configuration"]: row["totals_ms"] for row in result["confirmations"]}
    assert confirmations == {"b1-input": {"nemo-cpu": 11},
                             "b64-input": {"nemo-cpu": 21, "nemo-cuda": 31},
                             "b64-length": {"nemo-cuda": 16}}
    assert all(call[call.index("--rounds") + 1] == "2" for call in calls[5:])
    assert all(call[call.index("--repetitions") + 1] == "3" for call in calls[5:])


def test_sweep_preserves_failure_without_selecting_partial_winner(tmp_path, monkeypatch):
    output = tmp_path / "sweep"
    monkeypatch.setattr(sys, "argv", ["sweep", "--inputs", str(tmp_path), "--output", str(output)])

    def fail(*args, **kwargs):
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(nemo_sweep.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="candidate failed"):
        nemo_sweep.main()
    result = json.loads((output / "sweep.json").read_text())
    assert not result["complete"]
    assert not result["selected"]
    assert not result["confirmations"]
    assert result["error"] == "RuntimeError: candidate failed"
