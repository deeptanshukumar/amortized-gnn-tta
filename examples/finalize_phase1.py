"""Audit a frozen source bank, partition it, measure diversity, then score source test.

    python examples/finalize_phase1.py --bank runs/gcn-bank-phase1 --output-dir runs/phase1-review

Creates a new review directory. No target data, adaptation, model filtering, or
test-based tuning occurs. CPU execution only for this initial audit command.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from amortized_gnn_tta.bank_artifacts import (
    create_model_partition, file_sha256, load_model_partition, load_verified_bank, save_model_partition,
)
from amortized_gnn_tta.bank_diagnostics import analyze_bank_diversity, evaluate_source_test
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.models import ARCHITECTURES, build_model, make_model_spec
from amortized_gnn_tta.training import source_training_data


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs" / "phase1_csbm_gcn.json")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() or args.threads < 1:
        parser.error("Use a new output directory and positive thread count")
    torch.set_num_threads(args.threads)
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    architecture = protocol.get("architecture", "gcn")  # Preserve the original GCN configuration.
    if architecture not in ARCHITECTURES or protocol.get("protocol") != f"phase1-csbm-{architecture}-v1":
        parser.error("Unsupported phase protocol")
    source_config = dict(protocol["source"])
    source_config["split_ratios"] = tuple(source_config["split_ratios"])
    bundle = load_dataset([CSBMEnvironment("source", "source_train", CSBMConfig.from_matcha(**source_config))])
    source = source_training_data(bundle, "source")

    bank = load_verified_bank(args.bank, source, build_model)
    expected_model = make_model_spec(architecture, source_config["num_features"], len(source.class_names),
                                    **protocol["model"])
    if (bank.manifest["model_spec"] != expected_model or bank.manifest["model_seeds"] != protocol["model_seeds"]
            or bank.manifest["training_config_without_seed"] != protocol["training"]
            or bank.subsets["label_fraction"] != protocol["label_fraction_of_training_pool"]
            or [item["seed"] for item in bank.subsets["subsets"]] != protocol["subset_seeds"]
            or protocol["partition"]["policy"] != "whole_source_subset"):
        raise ValueError("Bank does not match the frozen phase protocol")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol.json", protocol)
    partition = create_model_partition(bank, seed=protocol["partition"]["seed"],
                                        group_counts=tuple(protocol["partition"]["group_counts"]))
    save_model_partition(args.output_dir / "model_partition.json", bank, partition)
    assert load_model_partition(args.output_dir / "model_partition.json", bank) == partition
    # This file is saved before any source-test label access or source-test score.
    diversity = analyze_bank_diversity(bank, source, build_model,
                                       lambda model, graph: model.encode(graph.x, graph.edge_index))
    write_json(args.output_dir / "diversity.json", diversity)
    write_csv(args.output_dir / "diversity_pairs.csv", diversity["pairs"])
    evaluation = None
    issues = []
    if diversity["exact_parameter_duplicate_pairs"]:
        issues.append("Exact parameter duplicates require bank-construction review")
    if not diversity["functional_variation_observed"]:
        issues.append("No probability variation observed on the source probe")
    if diversity["undefined_cka_pairs"]:
        issues.append("Constant/degenerate representations made some CKA comparisons undefined")
    if not issues:
        evaluation = evaluate_source_test(
            bank, source, partition, build_model, test_indices=bundle.select("source", "test"),
            test_labels=bundle.evaluation_labels("source", "test"))
        write_json(args.output_dir / "source_test.json", evaluation)
        write_csv(args.output_dir / "source_test_models.csv", evaluation["models"])
    capped = sum(row["best_epoch"] == protocol["training"]["max_epochs"] for row in bank.manifest["models"])
    # Preserve the audit implementation separately from the bank's training Git
    # state. This snapshot is not retroactively claimed to be its training code.
    snapshots = {}
    paths = [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml", Path(__file__)]
    for path in paths:
        relative = path.resolve().relative_to(root)
        target = args.output_dir / "analysis_code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        snapshots[str(relative)] = file_sha256(target)
    display_name = "GraphSAGE" if architecture == "graphsage" else architecture.upper()
    report = {
        "protocol": protocol["protocol"], "status": "complete" if not issues else "needs_review",
        "scope": f"{display_name}/CSBM engineering phase 1; not adaptation or final-paper research completion",
        "architecture": architecture,
        "num_parameters": sum(parameter.numel() for parameter in build_model(expected_model).parameters()),
        "bank_sha256": bank.sha256, "protocol_sha256": file_sha256(args.output_dir / "protocol.json"),
        "partition_sha256": file_sha256(args.output_dir / "model_partition.json"),
        "models": len(bank.manifest["models"]), "partition_counts": {role: len(ids) for role, ids in partition["model_ids"].items()},
        "pair_count": diversity["pair_count"], "issues": issues, "selected_at_epoch_cap": capped,
        "convergence": "No selected checkpoint at cap; validation-based stopping observed" if not capped else
                       "Some checkpoints selected at cap; convergence is not established",
        "diversity_sufficiency": "Variation measured on a fixed source probe; useful adaptation diversity remains unproven",
        "analysis_code_snapshot_sha256": snapshots, "training_code_state": bank.manifest["run_metadata"].get("git"),
        "source_test_accuracy": evaluation["accuracy_descriptive"] if evaluation else None,
        "limitations": ["Same source graph and paired seeds, not new-graph or unseen-initialization generalization",
                        "Descriptive model spread, not an IID confidence interval",
                        "Development source graph/split reused; source-test scores were viewed in earlier examples, not a sealed final benchmark",
                        "CPU verified only; clean installation and other backends unverified",
                        "Bank training used an uncommitted working tree; audit snapshot is distinct from training provenance"],
    }
    write_json(args.output_dir / "phase1_report.json", report)
    lines = [f"# Phase 1 {display_name}/CSBM review", "", f"Status: **{report['status']}**. Models: {report['models']}.",
             "", "## Model partition", "", "| Role | Models | Whole label subsets |", "| --- | ---: | --- |"]
    for role, ids in partition["model_ids"].items():
        lines.append(f"| {role} | {len(ids)} | {', '.join(partition['groups'][role])} |")
    lines += ["", "## Diversity on native source-training nodes", "", "| Measure | Mean | Min | Max |",
              "| --- | ---: | ---: | ---: |"]
    for name, summary in diversity["summary"]["all"].items():
        if summary:
            lines.append(f"| {name} | {summary['mean']:.6f} | {summary['min']:.6f} | {summary['max']:.6f} |")
    if evaluation:
        lines += ["", "## Held-out source accuracy", "", "Descriptive spread across paired models; not an IID uncertainty estimate.",
                  "", "| Label subset | Models | Mean accuracy | Population SD |", "| --- | ---: | ---: | ---: |"]
        for subset, summary in evaluation["by_subset"].items():
            lines.append(f"| {subset} | {summary['count']} | {100*summary['mean']:.2f}% | {100*summary['std_population']:.2f} pp |")
    lines += ["", report["convergence"], "", report["diversity_sufficiency"], "", "## Limits", ""]
    lines += [f"- {item}" for item in report["limitations"] + issues]
    (args.output_dir / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "models": report["models"],
                      "partitions": report["partition_counts"], "pairs": report["pair_count"],
                      "selected_at_cap": capped, "issues": issues, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
