"""Behavioral checks for timing, exact rescores, and incomplete aggregation."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accuracy import compare_relative, summarize, validate
from benchmark import aggregate, environment, execute_queue


def case(emissions, targets, blank=0):
    return SimpleNamespace(case_id="case", emissions=np.array(emissions, dtype=np.float32),
                           targets=np.array(targets, dtype=np.int64), blank_id=blank)


def output(c, labels):
    labels = np.array(labels, dtype=np.int64)
    return labels, c.emissions[np.arange(len(labels)), labels]


class AccuracyTests(unittest.TestCase):
    def test_repeated_occurrences_and_nonzero_blank(self):
        c = case(np.zeros((5, 3)), [0, 0], blank=2)
        record, states = validate(c, output(c, [2, 0, 2, 0, 2]))
        self.assertTrue(record["frame_scores_valid"])
        np.testing.assert_array_equal(states, [0, 1, 2, 3, 4])
        record, states = validate(c, output(c, [2, 0, 0, 0, 2]))
        self.assertIsNone(states)

    def test_frame_scores_and_topology_are_both_checked(self):
        c = case(np.zeros((3, 3)), [1])
        labels, scores = output(c, [0, 1, 0])
        self.assertIsNone(validate(c, (labels, scores + 1))[1])
        self.assertIsNone(validate(c, output(c, [0, 2, 0]))[1])
        self.assertIsNone(validate(c, (labels.astype(np.float32), scores))[1])
        c.emissions[1, 1] = -np.inf
        self.assertIsNone(validate(c, output(c, [0, 1, 0]))[1])

    def test_exact_score_detects_loss_hidden_by_float64_summation(self):
        # Each path sums a huge value and a small value. Float64 sums tie,
        # whereas exact binary-emission rescoring preserves the unit loss.
        c = case([[0, 2**80], [0, 0], [0, 1]], [1])
        candidate, reference = output(c, [1, 1, 0]), output(c, [1, 1, 1])
        self.assertEqual(float(candidate[1].sum(dtype=np.float64)), float(reference[1].sum(dtype=np.float64)))
        row = compare_relative(c, candidate, reference)
        self.assertEqual(row["classification"], "worse")
        self.assertEqual(Fraction(row["exact_score_delta"]), Fraction(1))
        self.assertEqual(row["token_ends"]["max_absolute_frames"], 1)
        self.assertEqual(compare_relative(c, reference, candidate)["classification"], "better")

    def test_equal_score_alternative_is_not_invalid(self):
        c = case(np.zeros((4, 2)), [1])
        row = compare_relative(c, output(c, [0, 1, 0, 0]), output(c, [0, 0, 1, 0]))
        self.assertEqual(row["classification"], "equal_score_alternative")
        self.assertEqual(summarize([row])["failed_cases"], 0)
        self.assertEqual(row["token_starts"]["signed_frames"], [-1])

    def test_status_failure_is_not_valid_output(self):
        c = case(np.zeros((3, 2)), [1])
        row = compare_relative(c, None, output(c, [0, 1, 0]), candidate_status="native_status_1")
        self.assertEqual(row["classification"], "candidate_failure")
        self.assertEqual(summarize([row])["failed_cases"], 1)


class EnvironmentTests(unittest.TestCase):
    def test_collects_only_relevant_packages_and_nonunique_gpu_details(self):
        torch = SimpleNamespace(__version__="2.8.0", version=SimpleNamespace(cuda="12.8"),
                                get_num_threads=lambda: 1)
        with patch.dict(sys.modules, {"torch": torch}), \
             patch("benchmark.importlib.metadata.version", return_value="1.0") as versions, \
             patch("benchmark.subprocess.check_output", return_value="Example GPU") as command:
            result = environment()
        self.assertEqual(set(result["packages"]),
                         {"fast_forced_aligner", "torch", "torchaudio", "k2", "numpy", "matplotlib"})
        self.assertEqual(versions.call_count, 6)
        command.assert_called_once()
        query = command.call_args.args[0]
        self.assertFalse(any("uuid" in arg or "process" in arg for arg in query))
        self.assertNotIn("gpu_processes", result)
        self.assertNotIn("cpu_affinity", result)

    def test_failed_gpu_query_does_not_copy_diagnostic_output(self):
        torch = SimpleNamespace(__version__="2.8.0", version=SimpleNamespace(cuda="12.8"),
                                get_num_threads=lambda: 1)
        failure = subprocess.CalledProcessError(1, "nvidia-smi", output="private diagnostic marker")
        with patch.dict(sys.modules, {"torch": torch}), \
             patch("benchmark.importlib.metadata.version", return_value="1.0"), \
             patch("benchmark.subprocess.check_output", side_effect=failure):
            result = environment()
        self.assertEqual(result["nvidia_smi"], "unavailable: CalledProcessError")
        self.assertNotIn("private diagnostic marker", json.dumps(result))


class TimingTests(unittest.TestCase):
    def test_queue_preserves_error_and_completes_all_batches(self):
        events = []
        class Backend:
            def synchronize(self):
                events.append("sync")
            def run_batch(self, value):
                events.append(value)
                if value == "bad":
                    raise ValueError("bad input")
                return value
        result = execute_queue(Backend(), ["first", "bad", "last"], [[0], [1], [2]])
        self.assertEqual(events, ["sync", "first", "bad", "last", "sync"])
        self.assertEqual(len(result["raw_results"]), 3)
        self.assertIn("bad input", result["raw_results"][1][2])
        self.assertGreater(result["seconds"], 0)

    def test_aggregate_uses_total_queue_ratio_and_refuses_missing_or_failed_cells(self):
        campaign = {"arguments": {"rounds": 1, "repetitions": 1}, "methods": ["optimized", "other"],
                    "cohorts": {"short": {"cases": 2}, "long": {"cases": 3}}, "runs": []}
        for method, values in (("optimized", [1., 9.]), ("other", [4., 18.])):
            for cohort, seconds in zip(campaign["cohorts"], values):
                count = campaign["cohorts"][cohort]["cases"]
                campaign["runs"].append({"method": method, "cohort": cohort, "seconds": seconds,
                    "round": 0, "repetition": 0, "failed_cases": 0, "cases": count, "observations": count,
                    "classifications": {"identical": count}})
        rows, problems = aggregate(campaign)
        self.assertFalse(problems)
        self.assertEqual(rows[1]["speedup_optimized"], 2.2)
        self.assertEqual(rows[0]["rows_per_second"], .5)
        for mutation in ("missing", "failure", "duplicate", "nonfinite", "wrong_round", "unaudited", "invalid_classification"):
            broken = deepcopy(campaign)
            if mutation == "missing":
                broken["runs"].pop()
            elif mutation == "failure":
                broken["runs"][0]["failed_cases"] = 1
            elif mutation == "nonfinite":
                broken["runs"][0]["seconds"] = float("nan")
            elif mutation == "wrong_round":
                broken["runs"][0]["round"] = 20
            elif mutation == "unaudited":
                broken["runs"][0]["observations"] = 1
            elif mutation == "invalid_classification":
                broken["runs"][0]["classifications"] = {"candidate_failure": 2}
            else:
                broken["runs"].append(broken["runs"][0])
            rows, problems = aggregate(broken)
            self.assertFalse(rows, mutation)
            self.assertTrue(problems, mutation)


if __name__ == "__main__":
    unittest.main()
