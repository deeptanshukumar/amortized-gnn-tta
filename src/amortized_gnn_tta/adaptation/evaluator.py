"""User-readable evaluation of amortized adaptation on shifted environments.

Strict Boundary Guarantee:
Target graphs are evaluated using only `bundle.unlabeled_input(env_id)` for
generating the descriptor, running the generator, and executing GNN forward passes.
Target labels are accessed strictly for offline scoring after predictions are made.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..data.api import DatasetBundle
from ..evaluation import validate_logits, score_logits
from ._validation import conditioning_contract, finite, positive_int, source_state
from .generator import (
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    adapted_forward,
    extract_target_parameters,
)


@dataclass(frozen=True)
class AdaptationRecord:
    """Detailed score record for one (model, environment) evaluation pair."""

    model_id: str
    environment_id: str
    role: str
    family: str
    severity: float
    is_seen_severity: bool
    baseline_accuracy: float
    adapted_accuracy: float
    accuracy_gain: float  # adapted_acc - baseline_acc
    weight_delta_norm: float


def evaluate_adaptation_pair(
    model: nn.Module,
    model_id: str,
    bundle: DatasetBundle,
    env_id: str,
    env_metadata: dict[str, Any],
    descriptor: GraphShiftDescriptor,
    generator: HeadAdapterGenerator,
    device: str = "cpu",
    target_scope: str = "head",
) -> AdaptationRecord:
    """Evaluate one model on one environment with and without adaptation.

    Strict boundary: model & generator only see unlabeled_input(env_id).
    """
    if hasattr(generator, "descriptor_dim"):
        conditioning_contract(descriptor, generator)
    unlabeled = bundle.unlabeled_input(env_id).to(device)
    with source_state(model), source_state(generator), torch.no_grad():
        base_logits = model(unlabeled.x, unlabeled.edge_index)
        validate_logits(base_logits, unlabeled.num_nodes)
        target_weights, schema = extract_target_parameters(model, scope=target_scope)
        desc_vec = descriptor(model, unlabeled).to(device)
        finite(desc_vec, "conditioning")
        delta = finite(generator(desc_vec, target_weights.to(device)), "generated delta")
        delta_norm = float(torch.linalg.vector_norm(delta))
        adapted_logits = adapted_forward(model, unlabeled, delta, schema)
        validate_logits(adapted_logits, unlabeled.num_nodes, base_logits.shape[1])

    # Label access happens only after all deployment predictions are complete.
    split_name = "eval" if "eval" in bundle.graphs[env_id].splits else "test"
    eval_idx = bundle.select(env_id, split_name)
    labels = bundle.evaluation_labels(env_id, split_name)
    base_acc = score_logits(base_logits, eval_idx, labels).accuracy
    adapted_acc = score_logits(adapted_logits, eval_idx, labels).accuracy

    gain = adapted_acc - base_acc

    return AdaptationRecord(
        model_id=model_id,
        environment_id=env_id,
        role=env_metadata.get("role", "unknown"),
        family=env_metadata.get("family", "unknown"),
        severity=float(env_metadata.get("severity", 0.0)),
        is_seen_severity=bool(env_metadata.get("seen_severity_in_meta_train", False)),
        baseline_accuracy=base_acc,
        adapted_accuracy=adapted_acc,
        accuracy_gain=gain,
        weight_delta_norm=delta_norm,
    )


def summarize_records(records: Sequence[AdaptationRecord]) -> dict[str, Any]:
    """Aggregate a sequence of evaluation records into summary metrics."""
    if not records:
        return {"count": 0}

    base_accs = [r.baseline_accuracy for r in records]
    adapted_accs = [r.adapted_accuracy for r in records]
    gains = [r.accuracy_gain for r in records]
    norms = [r.weight_delta_norm for r in records]

    for record in records:
        finite(torch.tensor([record.baseline_accuracy, record.adapted_accuracy, record.accuracy_gain, record.weight_delta_norm]), "score record")
    return {
        "harm_fraction": float(np.mean([g < 0 for g in gains])),
        "count": len(records),
        "mean_baseline_acc": float(np.mean(base_accs)),
        "mean_adapted_acc": float(np.mean(adapted_accs)),
        "mean_gain": float(np.mean(gains)),
        "std_gain": float(np.std(gains)),
        "positive_gain_fraction": float(np.mean([g > 0 for g in gains])),
        "mean_delta_norm": float(np.mean(norms)),
    }


def format_adaptation_report(records: Sequence[AdaptationRecord], title: str = "Adaptation Report") -> str:
    """Format evaluation records into a clean, human-readable Markdown report."""
    if not records:
        return f"# {title}\n\nNo records to display.\n"

    overall = summarize_records(records)
    lines: list[str] = [
        f"# {title}",
        "",
        "## Overall Summary Across All Evaluated Pairs",
        "Unweighted mean across model–environment pairs; pair SD is not training-seed uncertainty. Includes controls when present.",
        f"- **Evaluated Pairs:** {overall['count']} (Models × Environments)",
        f"- **Average Baseline Accuracy:** {overall['mean_baseline_acc'] * 100:.2f}%",
        f"- **Average Adapted Accuracy:**  {overall['mean_adapted_acc'] * 100:.2f}%",
        f"- **Net Accuracy Gain (Mean):**   {overall['mean_gain'] * 100:+.2f} pp (pair SD {overall['std_gain'] * 100:.2f} pp)",
        f"- **Fraction of Pairs Improved:** {overall['positive_gain_fraction'] * 100:.1f}%",
        f"- **Mean Weight Delta Norm:**     {overall['mean_delta_norm']:.4f}",
        "",
        "## Breakdown by Shift Family",
        "| Shift Family | Count | Baseline Acc | Adapted Acc | Net Gain | Pairs Improved |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]

    # Group by shift family
    by_family: dict[str, list[AdaptationRecord]] = defaultdict(list)
    for r in records:
        by_family[r.family].append(r)

    for fam in sorted(by_family.keys()):
        s = summarize_records(by_family[fam])
        lines.append(
            f"| `{fam}` | {s['count']} | {s['mean_baseline_acc'] * 100:.2f}% | "
            f"{s['mean_adapted_acc'] * 100:.2f}% | **{s['mean_gain'] * 100:+.2f} pp** | "
            f"{s['positive_gain_fraction'] * 100:.1f}% |"
        )

    lines.extend([
        "",
        "## Breakdown by Shift Severity",
        "| Severity | Extrapolation (OOD)? | Count | Baseline Acc | Adapted Acc | Net Gain |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ])

    # Group by severity
    by_sev: dict[float, list[AdaptationRecord]] = defaultdict(list)
    for r in records:
        by_sev[r.severity].append(r)

    for sev in sorted(by_sev.keys()):
        s = summarize_records(by_sev[sev])
        is_ood = not by_sev[sev][0].is_seen_severity and sev > 0
        tag = "**YES (Unseen Severity)**" if is_ood else "No (Seen in Train)"
        lines.append(
            f"| `s = {sev:.2f}` | {tag} | {s['count']} | {s['mean_baseline_acc'] * 100:.2f}% | "
            f"{s['mean_adapted_acc'] * 100:.2f}% | **{s['mean_gain'] * 100:+.2f} pp** |"
        )

    lines.append("")
    return "\n".join(lines)


def subsample_environments(
    envs: Sequence[dict[str, Any]],
    max_environments: int | None,
) -> list[dict[str, Any]]:
    """Family-round-robin debug cap; does not guarantee lineage/severity balance.

    Instead of taking the first N environments in raw manifest order (which
    biases toward a single lineage and early shift types), this groups by
    shift family and round-robins across them to ensure representation.
    """
    if max_environments is not None:
        positive_int(max_environments, "max_environments")
    if max_environments is None or len(envs) <= max_environments:
        return list(envs)

    by_family: dict[str, list[dict[str, Any]]] = {}
    for e in envs:
        fam = e.get("family", "unknown")
        by_family.setdefault(fam, []).append(e)

    selected: list[dict[str, Any]] = []
    family_queues = [list(lst) for lst in by_family.values()]
    idx = 0
    while len(selected) < max_environments and any(family_queues):
        queue = family_queues[idx % len(family_queues)]
        if queue:
            selected.append(queue.pop(0))
        idx += 1

    return selected


def evaluate_adaptation(
    models: dict[str, nn.Module],
    manifest: dict[str, Any],
    load_env_fn,  # callable (manifest, env_id) -> DatasetBundle
    descriptor: GraphShiftDescriptor,
    generator: HeadAdapterGenerator,
    role: str = "validation",
    max_environments: int | None = None,
    device: str = "cpu",
    target_scope: str = "head",
    allow_final_evaluation: bool = False,
) -> tuple[list[AdaptationRecord], str]:
    """Run full evaluation across multiple models and environments in a declared role.

    Returns:
        records: List of individual AdaptationRecord dataclass items.
        report: Formatted Markdown report string.
    """
    if role == "target_test" and not allow_final_evaluation:
        raise ValueError("Final evaluation requires explicit allow_final_evaluation=True")
    envs = [e for e in manifest["environments"] if e["role"] == role]
    envs = subsample_environments(envs, max_environments)

    records: list[AdaptationRecord] = []

    for env_meta in envs:
        env_id = env_meta["id"]
        bundle = load_env_fn(manifest, env_id)

        for model_id, model in models.items():
            record = evaluate_adaptation_pair(
                model=model,
                model_id=model_id,
                bundle=bundle,
                env_id=env_id,
                env_metadata=env_meta,
                descriptor=descriptor,
                generator=generator,
                device=device,
                target_scope=target_scope,
            )
            records.append(record)

    report = format_adaptation_report(records, title=f"Adaptation Evaluation Report ({role.upper()})")
    return records, report
