"""Generate a traceable write-up table from saved real-study evidence only."""

import argparse
import csv
import json
from pathlib import Path


def summarize(bank_dir, study_dir):
    bank = json.loads((bank_dir / "bank.json").read_text())
    outputs = []
    for name in (
        "EXPERIMENT_1.json",
        "EXPERIMENT_2.json",
        "EXPERIMENT_3.json",
        "EXPERIMENT_5_VALIDATION.json",
    ):
        path = study_dir / name
        if path.exists():
            outputs.append((name, json.loads(path.read_text())))
    final = study_dir / "FINAL" / "COMPLETE.json"
    lines = [
        "# Real-graph study evidence",
        "",
        f"Dataset: `{bank['plan'].get('dataset', 'ogbn-arxiv')}`.",
        "",
        f"Final evaluation completed: **{final.exists()}**.",
        "Software fixture tests are separate from these numerical observations. CPU execution does not establish GPU performance.",
        "",
    ]
    lines += [
        "## Offline source bank",
        "",
        "| Architecture | Source fits | Total observed fit seconds |",
        "|---|---:|---:|",
    ]
    for arch in bank["plan"]["architectures"]:
        rows = [r for r in bank["models"] if r["architecture"] == arch]
        lines.append(
            f"| {arch} | {len(rows)} | {sum(r['fit_seconds'] for r in rows):.2f} |"
        )
    lines += [
        "",
        "Bank fit wall times include ordinary machine contention and checkpoint checks. Dataset download/construction and development search costs are additional. A generator requires extra source checkpoints that a one-checkpoint iterative deployment may not require.",
        "",
    ]
    for name, output in outputs:
        lines += [
            "## " + name.removesuffix(".json"),
            "",
            "| Architecture | Inputs | Update sites | Method | Seed | Validation accuracy | Selected epoch | Adaptation parameters | Generator parameters | Generator fit seconds | Teacher collection seconds |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in output.get("rows", []):
            lines.append(
                f"| {row['architecture']} | {row['conditioning']} | {row['update']} | {row['method']} | {row['seed']} | {100 * row['mean_validation_accuracy']:.3f}% | {row['selected_epoch']} | {row.get('adaptation_parameter_count', 'unrecorded')} | {row.get('generator_parameter_count', 'unrecorded')} | {row['generator_fit_seconds']:.2f} | {row['shared_teacher_collection_seconds']:.2f} |"
            )
        lines += [
            "",
            "Teacher time is a shared collection cost reported for every method that needs it; do not sum duplicated references as distinct collection runs. Epoch-zero selection is an identity update only when verified from its saved delta.",
            "",
        ]
    sensitivity = study_dir / "CONDITIONING_CONTROL" / "COMPLETE.json"
    if sensitivity.exists():
        lines += ["", "## Fixed-generator conditioning sensitivity", "",
                  "| Inputs | Original validation accuracy | Cyclic-donor accuracy | Prediction disagreement |",
                  "|---|---:|---:|---:|"]
        for row in json.loads(sensitivity.read_text())["rows"]:
            lines.append(f"| {row['conditioning']} | {100*row['mean_normal_accuracy']:.3f}% | {100*row['mean_donor_accuracy']:.3f}% | {100*row['mean_prediction_disagreement']:.3f}% |")
        lines += ["", "Each source model and actual target graph stays fixed while the descriptor uses the next validation environment in saved manifest order. This is an inference-only sensitivity diagnostic, not a retrained no-input baseline or deployment method; it does not revise design selection.", ""]
    if final.exists():
        scored = json.loads((study_dir / "FINAL" / "SCORES.json").read_text())["rows"]
        with (study_dir / "FINAL" / "all_scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
            writer.writeheader()
            writer.writerows(scored)
        lines += [
            "## Frozen-protocol final scores",
            "",
            "| Architecture | Method | Mean accuracy | Mean gain (pp) | Harmful cases |",
            "|---|---|---:|---:|---:|",
        ]
        for row in json.loads(final.read_text())["summary"]:
            lines.append(
                f"| {row['architecture']} | {row['method']} | {100 * row['mean_accuracy']:.3f}% | {100 * row['mean_gain']:+.3f} | {100 * row['harm_fraction']:.1f}% |"
            )
        timing = json.loads((study_dir / "FINAL" / "TIMINGS.json").read_text())["rows"]
        lines += [
            "",
            "## Complete prediction-pipeline latency",
            "",
            "| Architecture | Environment | Method | Median ms | Speedup vs head IM | Speedup vs LoRA IM |",
            "|---|---|---|---:|---:|---:|",
        ]
        for case in timing:
            methods = case["methods"]
            for method, row in methods.items():
                latency = row["median_seconds"]
                lines.append(
                    f"| {case['architecture']} | {case['environment_id']} | {method} | {1000 * latency:.3f} | {methods['iterative_im_head']['median_seconds'] / latency:.3f}x | {methods['iterative_im_lora']['median_seconds'] / latency:.3f}x |"
                )
        lines += [
            "",
            "Each event starts with a canonical CPU graph and resident models. Required descriptors, generation/optimization, update application and prediction are included. Diffusion includes all five predictions and selection. Scoring, artifact export and offline fitting are excluded. Timing ratios do not imply matched accuracy. Separate memory probes report sampled process RSS and, on CUDA, measured peak allocated/reserved tensor memory including resident tensors and warm cached reservations. CUDA allocator peaks are not whole-device usage or isolated fresh-process method allocations.",
            "",
        ]
    else:
        lines += [
            "## Pending evidence",
            "",
            "Final scoring and performance tables are not available yet. No final-result claim follows from a loader, fixture test, or partially trained bank.",
            "",
        ]
    diagnostic = study_dir / "DIAGNOSTIC" / "RESULTS.json"
    if diagnostic.exists():
        lines += [
            "## Development supervised-reference diagnostic",
            "",
            "| Architecture | Practice environment | Method | Accuracy | Gain (pp) | Fit seconds |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in json.loads(diagnostic.read_text())["rows"]:
            fit = (
                f"{row['fit_seconds']:.2f}"
                if "fit_seconds" in row
                else "not timed here"
            )
            lines.append(
                f"| {row['architecture']} | {row['environment_id']} | {row['method']} | {100 * row['accuracy']:.3f}% | {100 * row['gain']:+.3f} | {fit} |"
            )
        lines += [
            "",
            "These are seen practice checkpoint/environment comparisons, with disjoint diagnostic nodes. Supervised methods use practice fitting and inner-stopping labels. They are not deployment results or optimal accuracy ceilings.",
            "",
        ]
    unique_fits = []
    for path in study_dir.glob("*/RESULT.json"):
        unique_fits.append(json.loads(path.read_text()))
    if unique_fits:
        lines += [
            "## Observed generator fitting and search cost",
            "",
            f"Unique completed fits: {len(unique_fits)}; sum of recorded generator fit wall times: {sum(r['generator_fit_seconds'] for r in unique_fits):.2f} seconds.",
            "This includes completed exploratory input/site fits in this study folder. Teacher files record their own collection time separately; repeated references to one cached teacher file must not be summed as separate collections. Source-bank preparation and generator search are not free. Dataset/preparation time not explicitly measured in older bank artifacts remains unmeasured.",
            "",
        ]
    lines += [
        "## Interpretation limits",
        "",
        "Validation is used for design selection and must not be called an unbiased final estimate. Cases share underlying graphs; perturbation draws are not independent datasets. Report seed variation separately from case-to-case variation. Negative gains and unchanged controls must stay in the tables. Fixed-budget supervised references are not optimal accuracy ceilings. GTrans feature-only is a declared variant, not a reproduction of joint graph transformation.",
        "",
    ]
    (study_dir / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--study", type=Path, required=True)
    a = p.parse_args()
    summarize(a.bank, a.study)
