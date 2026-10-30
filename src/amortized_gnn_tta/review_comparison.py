"""Strictly matched supervised full/head/LoRA development comparisons."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

from .adaptation_artifacts import load_weight_delta
from .bank_artifacts import artifact_path, file_sha256


def _read_review(directory: Path) -> tuple[dict, dict]:
    report = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report["plan_sha256"] != file_sha256(directory / "plan.json"):
        raise ValueError("Comparison requires complete reviews with intact plans")
    return report, plan


def compare_supervised_reviews(full_dir: Path, reference_dirs: Sequence[Path], *, output_dir: Path) -> dict[str, Any]:
    """Require identical pairs/settings, validate saved deltas, retain regressions.

    This reads existing experiments only. It neither selects hyperparameters nor
    fits models. Prediction replay is a separate independent audit; checks here
    bind each row to its saved weights, graph, evaluation indices and metrics.
    """
    full_dir, output = Path(full_dir), Path(output_dir)
    full, full_plan = _read_review(full_dir)
    if (full["protocol"] != "full-finetuning-development-v1" or not reference_dirs
            or any(r["method"] != "full" or r["objective"] != "supervised_development"
                   for r in full["adaptations"])):
        raise ValueError("Choose a supervised full diagnostic and at least one reference review")
    expected = {(m, e) for m in full_plan["model_ids"] for e in full_plan["environment_ids"]}
    if not expected:
        raise ValueError("Comparison requires nonempty development pairs")
    groups, sources = {}, []
    pair_identity = {}
    for directory in [full_dir, *map(Path, reference_dirs)]:
        report, plan = _read_review(directory)
        for field in ("bank_sha256", "environment_manifest_sha256", "model_spec", "adaptation_config", "versions"):
            if plan[field] != full_plan[field]:
                raise ValueError(f"Unmatched review setting: {field}")
        if set(plan["model_ids"]) != set(full_plan["model_ids"]) or set(plan["environment_ids"]) != set(full_plan["environment_ids"]):
            raise ValueError("Reviews must contain the same model/environment pairs")
        local = defaultdict(dict)
        for row in report["adaptations"]:
            if row["objective"] != "supervised_development":
                continue
            method = row["method"]
            if method not in {"head", "lora", "full"}:
                raise ValueError("Unknown supervised update method")
            pair = row["model_id"], row["environment_id"]
            if pair in local[method]:
                raise ValueError("Duplicate method/pair in review")
            path = artifact_path(directory, row["weight_delta"])
            if file_sha256(path) != row["weight_delta_sha256"]:
                raise ValueError("Weight artifact checksum disagrees with review")
            _, payload = load_weight_delta(path)
            context = payload["context"]
            if (context["model_id"], context["environment_id"]) != pair or any(
                context[key] != row[key] for key in ("before", "after")
            ) or row["before_accuracy"] != row["before"]["accuracy"] or row["after_accuracy"] != row["after"]["accuracy"]:
                raise ValueError("Saved artifact identity/metrics disagree with review")
            if (payload["peft_config"]["method"] != method or payload["objective"] != row["objective"]
                    or payload["adaptation_config"] != plan["adaptation_config"]):
                raise ValueError("Saved fitting configuration disagrees with review")
            identity = (payload["initial_sha256"], context["graph_sha256"], context["evaluation_indices"],
                        row["before_accuracy"], row["family"], row["shift"], row["severity"])
            if pair in pair_identity and identity != pair_identity[pair]:
                raise ValueError("Unmatched initial weights, graph, split or baseline")
            pair_identity[pair] = identity
            local[method][pair] = row
        if not local:
            raise ValueError("Reference review has no supervised updates")
        for method, rows in local.items():
            if method in groups or set(rows) != expected:
                raise ValueError("Duplicate method or incomplete supervised pair grid")
            groups[method] = rows
        sources.append({"directory": str(directory), "review_sha256": file_sha256(directory / "review.json"),
                        "plan_sha256": file_sha256(directory / "plan.json")})
    pairs, summaries = [], []
    for pair in sorted(expected):
        base = groups["full"][pair]
        pairs.append({"model_id": pair[0], "environment_id": pair[1], "family": base["family"],
                      "shift": base["shift"], "before_accuracy": base["before_accuracy"],
                      "after_accuracy": {method: rows[pair]["after_accuracy"] for method, rows in groups.items()}})
    # Overall shifted mean excludes controls; individual shifts retain all cases.
    for name in ["all_shifted", "control", *sorted({r["shift"] for r in pairs if r["shift"] is not None})]:
        selected = [r for r in pairs if (r["family"] != "control" if name == "all_shifted" else
                    r["family"] == "control" if name == "control" else r["shift"] == name)]
        if not selected:
            continue
        before = sum(r["before_accuracy"] for r in selected) / len(selected)
        for method in groups:
            after = sum(r["after_accuracy"][method] for r in selected) / len(selected)
            full_diffs = [r["after_accuracy"]["full"] - r["after_accuracy"][method] for r in selected]
            summaries.append({"shift": name, "method": method, "pairs": len(selected),
                              "before_accuracy": before, "after_accuracy": after, "gain_pp": 100 * (after - before),
                              "full_minus_method_pp": 100 * sum(full_diffs) / len(selected),
                              "worsened_vs_frozen": sum(r["after_accuracy"][method] < r["before_accuracy"] for r in selected)})
    result = {"status": "complete", "protocol": "matched-supervised-review-comparison-v1",
              "sources": sources, "adaptation_config": full_plan["adaptation_config"],
              "pairs": pairs, "summary": summaries,
              "limitations": ["Supervised development diagnostic, not unlabeled adaptation or a proven ceiling",
                              "Related model/graph pairs are not independent statistical repetitions",
                              "Same optimizer settings need not be optimal for different update methods"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Matched supervised development comparison", "",
             "| Shift | Method | Pairs | Frozen | After | Gain (pp) | Full minus method (pp) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['shift']} | {row['method']} | {row['pairs']} | {100*row['before_accuracy']:.2f}% | "
                     f"{100*row['after_accuracy']:.2f}% | {row['gain_pp']:+.2f} | {row['full_minus_method_pp']:+.2f} |")
    lines += ["", *[f"- {note}" for note in result["limitations"]]]
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
