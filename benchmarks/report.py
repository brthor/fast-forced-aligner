"""Render a portable benchmark campaign as Markdown, JSON, and PNG/SVG charts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from accuracy import summarize

LABELS = {
    "optimized": "Ours · optimized", "conventional": "Ours · conventional DP",
    "optimized-checked": "Ours · optimized + input checks", "conventional-checked": "Ours · DP + input checks",
    "ctc-corrected": "ctc-forced-aligner · corrected CPU", "ctc-cpp": "ctc-forced-aligner · original CPU",
    "ta-cpu": "TorchAudio · CPU", "ta-cuda": "TorchAudio · CUDA", "k2-cuda": "k2 · CUDA",
    "nemo-cpu": "NeMo Viterbi · CPU", "nemo-cuda": "NeMo Viterbi · CUDA",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_artifact(root: Path, relative: str, expected: str) -> Path:
    """Resolve only campaign-local evidence, and verify it before use."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact escapes campaign directory: {relative}")
    if digest(path) != expected:
        raise ValueError(f"audit checksum mismatch: {path}")
    return path


def verify_audits(campaign: dict[str, Any], root: Path) -> None:
    """Require actual ordered per-record evidence for every timed observation."""
    for row in campaign["runs"]:
        if not row.get("artifact"):
            if not row.get("error") or row.get("failed_cases") != row.get("cases"):
                raise ValueError("timed observation is missing its audit artifact")
            continue
        artifact = checked_artifact(root, row["artifact"], row["artifact_sha256"])
        assessments = json.loads(artifact.read_text())["assessments"]
        cohort = campaign["cohorts"][row["cohort"]]
        if len(assessments) != cohort["cases"] or [item["case_id"] for item in assessments] != cohort["case_ids"]:
            raise ValueError(f"audit membership/order mismatch: {artifact}")
        for key, value in summarize(assessments).items():
            if row.get(key) != value:
                raise ValueError(f"audit summary mismatch ({key}): {artifact}")
        for assessment in assessments:
            if assessment.get("output_artifact"):
                checked_artifact(root, assessment["output_artifact"], assessment["output_artifact_sha256"])


def render_chart(campaign: dict[str, Any], output: Path) -> None:
    """Draw all available headline methods with spacious, mobile-readable rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    summary = {row["method"]: row for row in campaign["summary"]}
    omit = {"ta-cuda", "ctc-cpp", "optimized-checked", "conventional-checked"}
    methods = [name for name in ("optimized", "conventional") if name in summary]
    methods += sorted(set(summary) - set(methods) - omit, key=lambda name: summary[name]["suite_mean_ms"])
    if not methods:
        methods = list(summary)
    values = [summary[name]["suite_mean_ms"] / 1000 for name in methods]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("chart requires positive finite measured times")
    bg, ink, muted, green = "#F5F7FA", "#14283F", "#586D83", "#138458"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    height = 3.5 + 1.7 * len(methods)
    fig = plt.figure(figsize=(8, height), facecolor=bg)
    fig.text(.075, 1 - .5 / height, "FAST FORCED ALIGNER", fontsize=18, color=green, weight="bold")
    fig.text(.075, 1 - 1.15 / height, "Alignment time", fontsize=32, color=ink, weight="bold")
    qualifier = "selected methods" if campaign["method_selection"] != "full" else "lower is better"
    fig.text(.075, 1 - 1.65 / height, f"{len(campaign['cohorts'])} workloads · {qualifier}", fontsize=18, color=muted)
    manifest = campaign.get("input_manifest", {})
    if manifest.get("suite") == "regenerated-readme":
        coverage = "PARTIAL stock-model selection" if manifest.get("partial") else "Regenerated stock-model inputs"
        fig.text(.075, 1 - 2.12 / height, coverage, fontsize=15, color=muted)
    elif campaign.get("arguments", {}).get("suite") == "synthetic":
        fig.text(.075, 1 - 2.12 / height, "Synthetic smoke suite", fontsize=15, color=muted)
    ax = fig.add_axes((.075, 1.15 / height, .85, (height - 3.5) / height), facecolor=bg)
    ax.set_xlim(0, max(values) * 1.03)
    ax.set_ylim(len(methods) - .7, -1)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g} s"))
    ax.tick_params(axis="x", labelsize=16, colors=muted, length=0, pad=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, (method, seconds) in enumerate(zip(methods, values, strict=True)):
        ours = method.startswith(("optimized", "conventional"))
        color = green if ours else "#7390AC"
        ax.hlines(index + .05, 0, max(values) * 1.03, color="#DDE5ED", linewidth=1)
        ax.barh(index + .05, seconds, height=.08, color=color)
        ax.scatter([seconds], [index + .05], color=color, s=34, clip_on=False)
        y = fig.transFigure.inverted().transform(ax.transData.transform((0, index)))[1]
        fig.text(.075, y + .95 / height, LABELS.get(method, method), fontsize=20,
                 color=green if ours else ink, weight="bold" if ours else "medium")
        duration = f"{seconds * 1000:,.1f} ms" if seconds < 1 else f"{seconds:,.2f} s"
        fig.text(.075, y + .38 / height, duration, fontsize=21, weight="bold", color=green if ours else ink)
        speedup = summary[method].get("speedup_optimized")
        if speedup is not None:
            relative = ("1.00× baseline" if method == "optimized" else f"optimized {speedup:,.2f}× faster"
                        if speedup >= 1 else f"optimized {1 / speedup:,.2f}× slower")
            fig.text(.925, y + .38 / height, relative, fontsize=19, color=green if ours else muted, ha="right")
    fig.text(.075, .42 / height, "Total queue time · preparation included", fontsize=16, color=muted)
    for extension in ("png", "svg"):
        fig.savefig(output / f"performance.{extension}", dpi=160, facecolor=bg)
    plt.close(fig)


def make_report(campaign: dict[str, Any], source: Path) -> str:
    arguments = campaign["arguments"]
    cohorts = campaign["cohorts"]
    fingerprints = {value for cohort in cohorts.values() for value in cohort.get("input_fingerprints", [])}
    lines = ["# Reproducible forced-alignment benchmark", "",
             f"Run: `{campaign.get('started_utc', 'unknown')}`. Status: **{'complete' if campaign.get('complete') else 'incomplete'}**. "
             f"Method coverage: **{campaign.get('method_selection', 'unknown')}**.", "",
             f"{len(cohorts)} workloads; {sum(cohort['cases'] for cohort in cohorts.values()):,} suite entries; "
             f"{len(fingerprints):,} distinct input fingerprints. Batch limit {arguments['batch_size']}; "
             f"{arguments['rounds']} rounds × {arguments['repetitions']} repetitions; CPU worker setting {arguments['workers']}.", "",
             f"NeMo schedule: **{arguments.get('nemo_batch_order', 'input')} order**; queue scheduling and order restoration are timed.", "",
             "## Timing contract", "", campaign["timing"], "", campaign["aggregation"], "",
             "The conventional reference and accuracy checks run outside measurement. Input-checking lanes are explicitly labeled; "
             "skipping input validation requires previously admitted inputs. Completion synchronization is included for every lane. "
             "The native status arrays are read during the untimed audit. Per-record status checks are therefore not repeated in the measured call.", ""]
    manifest = campaign.get("input_manifest", {})
    if manifest.get("suite") == "regenerated-readme":
        coverage = "PARTIAL recipe selection" if manifest.get("partial") else "full recipe selection"
        lines += [f"Inputs: **stock-model regeneration; {coverage}**. "
                  f"Recipe `{manifest.get('recipe_version', 'unknown')}`. "
                  "Emissions may differ numerically from the historical README inputs. "
                  "Downloads and acoustic inference are outside alignment timing.", ""]
    elif arguments.get("suite") == "synthetic":
        lines += ["Inputs: **synthetic smoke suite**, not the full README workload.", ""]
    lines += ["The common timed output is the complete frame-label alignment. k2 and NeMo return paths; their per-frame emission "
              "scores are gathered only for the untimed accuracy audit. Methods whose native API returns frame scores include "
              "that native work in their timing.", ""]
    if campaign.get("complete") and campaign.get("summary"):
        lines += ["## Aggregate performance", "",
                  "The speedup column is each method's total divided by the optimized total, so it describes how much faster optimized execution is.", "",
                  "| Implementation | Total (ms) | Rows/s | Optimized speedup |",
                  "|---|---:|---:|---:|"]
        for row in campaign["summary"]:
            ratio = f"{row['speedup_optimized']:.2f}×" if row.get("speedup_optimized") is not None else "—"
            lines.append(f"| {LABELS.get(row['method'], row['method'])} | {row['suite_mean_ms']:.3f} | {row['rows_per_second']:,.1f} | {ratio} |")
        lines += ["", "![Measured aggregate alignment times](performance.png)", "",
                  "The chart shows the headline methods; the table retains every selected method, including validation-enabled lanes.", "",
                  "## Accuracy against conventional DP", "",
                  "Counts below are measured output observations, including timing repetitions and workload overlap; they are not independent recordings. "
                  "Exact objective differences rescore the original binary floating-point emissions without summation-rounding ambiguity. "
                  "Boundary differences concern token starts and exclusive ends on the emission-frame grid, not human timestamp accuracy.", "",
                  "| Implementation | Identical | Equal-score alternatives | Better | Worse | Max additional objective loss | Max boundary shift (frames) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in campaign["summary"]:
            counts = row["classifications"]
            lines.append(f"| {LABELS.get(row['method'], row['method'])} | {counts.get('identical', 0):,} | {counts.get('equal_score_alternative', 0):,} | "
                         f"{counts.get('better', 0):,} | {counts.get('worse', 0):,} | {row['maximum_additional_score_loss']:.9g} | {row['maximum_boundary_displacement_frames']} |")
    else:
        lines += ["## Incomplete coverage", "", "No common aggregate, relative speedup, or performance chart is produced when any selected cell is missing or fails validation.", ""]
        for error in campaign.get("incomplete_cells", []):
            lines.append(f"- {error}")
        if campaign.get("fatal_error"):
            lines.append(f"- Fatal error: {campaign['fatal_error']}")
    lines += ["", "## Per-workload observations", "", "All observations are retained; there is no outlier filtering. Failed observations remain visible.", "",
              "| Workload | Method | Valid observations | Mean (ms) | Min (ms) | Max (ms) | Failed outputs |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for cohort in cohorts:
        for method in campaign["methods"]:
            rows = [row for row in campaign["runs"] if row["cohort"] == cohort and row["method"] == method]
            seconds = [row["seconds"] for row in rows if row.get("seconds") is not None and not row.get("error") and not row.get("failed_cases")]
            values = [f"{operation(seconds) * 1000:.3f}" if seconds else "—" for operation in (mean, min, max)]
            lines.append(f"| {cohort} | {method} | {len(seconds)}/{arguments['rounds'] * arguments['repetitions']} | "
                         + " | ".join(values) + f" | {sum(row.get('failed_cases', 0) for row in rows):,} |")
    native: dict[str, Counter[str]] = {}
    for row in campaign["runs"]:
        native.setdefault(row["method"], Counter()).update(row.get("native_status_counts", {}))
    lines += ["", "## Status and provenance", "",
              "Native status counts are available only where a backend exposes them. Other failures are recorded as execution/materialization failures without inventing per-record native codes.", "",
              "```json", json.dumps({method: dict(counts) for method, counts in native.items()}, indent=2), "```", "",
              "Environment:", "", "```json", json.dumps(campaign["environment"], indent=2), "```", "",
              f"Campaign SHA-256: `{digest(source)}`. `campaign.json` records input provenance and checksums, package and harness source hashes, "
              "dependency versions, command arguments, every timing, and links to checksummed per-record audit artifacts. "
              "Different or failed paths are additionally saved as NumPy archives with their conventional reference. "
              "Report generation validates audit artifact checksums before summarizing.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input / "campaign.json" if args.input.is_dir() else args.input
    campaign = json.loads(source.read_text())
    verify_audits(campaign, source.parent)
    from benchmark import aggregate
    summary, problems = aggregate(campaign)
    if campaign.get("complete"):
        campaign["summary"] = summary
        campaign["incomplete_cells"] = problems
        campaign["complete"] = not problems
    else:
        campaign["summary"] = []
        campaign["incomplete_cells"] = problems
    args.output.mkdir(parents=True, exist_ok=True)
    if campaign.get("complete") and campaign.get("summary"):
        render_chart(campaign, args.output)
    else:
        for extension in ("png", "svg"):
            (args.output / f"performance.{extension}").unlink(missing_ok=True)
    (args.output / "report.md").write_text(make_report(campaign, source))
    (args.output / "summary.json").write_text(json.dumps({"complete": campaign.get("complete", False),
        "campaign_sha256": digest(source), "summary": campaign.get("summary", []),
        "incomplete_cells": campaign.get("incomplete_cells", [])}, indent=2) + "\n")
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
