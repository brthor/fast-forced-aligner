"""Checks for report integrity and explicit incomplete-run handling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report import main, digest, verify_audits, render_chart
from accuracy import summarize


class ReportTests(unittest.TestCase):
    def campaign(self):
        return {"arguments": {"rounds": 1, "repetitions": 1, "batch_size": 64, "workers": 8},
                "methods": ["optimized"], "method_selection": "explicit subset", "cohorts": {"test": {"cases": 2}},
                "runs": [], "complete": False, "environment": {}, "timing": "resident", "aggregation": "sum"}

    def test_incomplete_run_is_reported_without_fabricating_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "campaign.json").write_text(json.dumps(self.campaign()))
            (root / "report").mkdir()
            (root / "report/performance.png").write_bytes(b"stale image")
            with patch.object(sys, "argv", ["report.py", "--input", str(root), "--output", str(root / "report")]):
                main()
            summary = json.loads((root / "report/summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertFalse(summary["summary"])
            self.assertFalse((root / "report/performance.png").exists())

    def test_missing_audit_and_forged_summary_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = self.campaign()
            campaign["cohorts"]["test"]["case_ids"] = ["a", "b"]
            assessments = [{"case_id": name, "classification": "identical"} for name in ("a", "b")]
            artifact = root / "audit.json"
            artifact.write_text(json.dumps({"assessments": assessments}))
            row = {"cohort": "test", "seconds": 1., "cases": 2,
                   "artifact": "audit.json", "artifact_sha256": digest(artifact), **summarize(assessments)}
            campaign["runs"] = [row]
            verify_audits(campaign, root)
            row["classifications"] = {"better": 2}
            with self.assertRaisesRegex(ValueError, "audit summary mismatch"):
                verify_audits(campaign, root)
            row.update(summarize(assessments))
            campaign["cohorts"]["test"]["case_ids"].reverse()
            with self.assertRaisesRegex(ValueError, "membership/order mismatch"):
                verify_audits(campaign, root)
            row.pop("artifact")
            with self.assertRaisesRegex(ValueError, "missing its audit"):
                verify_audits(campaign, root)

    def test_failed_audit_retains_timing_without_claiming_valid_output(self):
        campaign = self.campaign()
        campaign["runs"] = [{"seconds": 1., "cases": 2, "failed_cases": 2, "error": "audit failed"}]
        verify_audits(campaign, Path("."))

    def test_modified_audit_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = self.campaign()
            campaign["runs"] = [{"artifact": "audit.json", "artifact_sha256": "0" * 64}]
            (root / "campaign.json").write_text(json.dumps(campaign))
            (root / "audit.json").write_text('{"assessments": []}')
            with patch.object(sys, "argv", ["report.py", "--input", str(root), "--output", str(root / "report")]):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    main()

    def test_incomplete_cells_override_claimed_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = self.campaign()
            campaign["complete"] = True
            campaign["summary"] = [{"method": "optimized", "suite_mean_ms": 1}]
            (root / "campaign.json").write_text(json.dumps(campaign))
            with patch.object(sys, "argv", ["report.py", "--input", str(root), "--output", str(root / "report")]):
                main()
            summary = json.loads((root / "report/summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertTrue(summary["incomplete_cells"])
            self.assertFalse(summary["summary"])

    def test_chart_identifies_partial_inputs_independently_of_workload_count(self):
        campaign = self.campaign()
        campaign["summary"] = [{"method": "optimized", "suite_mean_ms": 1}]
        campaign["input_manifest"] = {"suite": "regenerated-readme", "partial": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render_chart(campaign, root)
            self.assertIn("PARTIAL", (root / "performance.svg").read_text())
            campaign["input_manifest"]["partial"] = False
            render_chart(campaign, root)
            self.assertNotIn("PARTIAL", (root / "performance.svg").read_text())


if __name__ == "__main__":
    unittest.main()
