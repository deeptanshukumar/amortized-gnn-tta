"""Label-free bank diversity and a separate, explicit source-test evaluator.

Linear CKA follows the centered feature-space definition of Kornblith et al.
(ICML 2019, https://proceedings.mlr.press/v97/kornblith19a.html). This independent
PyTorch implementation uses H-by-H products, never a dense node Gram matrix.
"""

from collections.abc import Callable
from dataclasses import asdict
from itertools import combinations
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .bank_artifacts import VerifiedBank, artifact_path, file_sha256, validate_model_partition
from .evaluation import predict, score_logits
from .training import SourceTrainingData, load_checkpoint


def linear_cka(first: Tensor, second: Tensor) -> float | None:
    """Biased centered linear CKA; undefined constant representations return None.

    Columns are representation coordinates; rows are the same probe nodes in the
    same order. Invariance to column permutations/orthogonal rotations and global
    nonzero scaling makes this more informative than raw coordinate distance.
    It is a probe-dependent similarity, not a sufficient-diversity certificate.
    """
    if (first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]
            or first.shape[0] < 2 or min(first.shape[1], second.shape[1]) < 1
            or not torch.isfinite(first).all() or not torch.isfinite(second).all()):
        raise ValueError("CKA requires finite, aligned [probe_nodes, features] matrices")
    a, b = first.detach().cpu().double(), second.detach().cpu().double()
    a, b = a - a.mean(0, keepdim=True), b - b.mean(0, keepdim=True)
    scale = torch.linalg.vector_norm(a.T @ a) * torch.linalg.vector_norm(b.T @ b)
    if scale == 0:
        return None
    return float(((a.T @ b).square().sum() / scale).clamp(0, 1))


def describe(values: list[float]) -> dict[str, float | int] | None:
    """Descriptive population spread only, not a confidence interval."""
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(data.mean()), "std_population": float(data.std()),
            "min": float(data.min()), "median": float(np.median(data)), "max": float(data.max())}


def _load_model(bank: VerifiedBank, row: dict[str, Any], factory: Callable[[dict], nn.Module]) -> nn.Module:
    if file_sha256(bank.directory / "bank.json") != bank.sha256:
        raise ValueError("Bank changed since validation")
    path = artifact_path(bank.directory, row["checkpoint"])
    if file_sha256(path) != row["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since validation")
    model, _ = load_checkpoint(path, factory, device="cpu")
    return model


def analyze_bank_diversity(bank: VerifiedBank, source: SourceTrainingData,
                           model_factory: Callable[[dict], nn.Module],
                           representation: Callable[[nn.Module, Data], Tensor],
                           *, probe_indices: Tensor | None = None) -> dict[str, Any]:
    """Compare all model pairs on a fixed, label-free native training-pool probe.

    Stored source-validation scores do not influence the comparisons. Hard-label
    disagreement, probability RMSE, centered linear CKA, raw weight L2 and cosine
    are descriptive diagnostics. No universal diversity threshold is invented.
    """
    probe = source.train_indices.sort().values if probe_indices is None else probe_indices.clone()
    if (probe.device.type != "cpu" or probe.dtype != torch.long or probe.ndim != 1 or len(probe) < 2
            or probe.unique().numel() != len(probe) or not torch.isin(probe, source.train_indices).all()):
        raise ValueError("Probe must contain unique native source-training node IDs on CPU")
    rows = bank.manifest["models"]
    if len(rows) < 2:
        raise ValueError("Diversity requires at least two bank models")
    weights, probabilities, embeddings, parameter_names = [], [], [], None
    for row in rows:
        model = _load_model(bank, row, model_factory)
        parameters = sorted(model.named_parameters())
        names = [(name, tuple(value.shape)) for name, value in parameters]
        if parameter_names is not None and names != parameter_names:
            raise ValueError("Bank parameter schemas differ")
        parameter_names = names
        weights.append(torch.cat([value.detach().cpu().double().flatten() for _, value in parameters]))
        probabilities.append(predict(model, source.graph)[probe].double().softmax(dim=1))
        with torch.inference_mode():
            hidden = representation(model, source.graph.clone())
        if (hidden.ndim != 2 or hidden.shape[0] != source.graph.num_nodes or hidden.shape[1] < 1
                or not torch.isfinite(hidden).all()):
            raise ValueError("Representation callback must return finite [N,H] values")
        embeddings.append(hidden[probe].detach().cpu().double())
        del model, hidden
    pairs = []
    for i, j in combinations(range(len(rows)), 2):
        a, b = weights[i], weights[j]
        norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
        distance = float(torch.linalg.vector_norm(a - b))
        scale = float((norm_a + norm_b) / 2)
        relation = ("same_subset" if rows[i]["subset_id"] == rows[j]["subset_id"] else
                    "same_seed" if rows[i]["model_seed"] == rows[j]["model_seed"] else "different_both")
        pairs.append({
            "first": rows[i]["model_id"], "second": rows[j]["model_id"], "relation": relation,
            "weight_l2": distance, "weight_relative_l2": distance / scale if scale else 0.0,
            "weight_cosine": float((a @ b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else None,
            "prediction_disagreement": float((probabilities[i].argmax(1) != probabilities[j].argmax(1)).double().mean()),
            "probability_rmse": float((probabilities[i] - probabilities[j]).square().mean().sqrt()),
            "hidden_linear_cka": linear_cka(embeddings[i], embeddings[j]),
            "exact_parameter_duplicate": bool(torch.equal(a, b)),
        })
    metrics = ("weight_relative_l2", "weight_cosine", "prediction_disagreement", "probability_rmse", "hidden_linear_cka")
    groups = {"all": pairs, **{name: [p for p in pairs if p["relation"] == name]
                             for name in ("same_subset", "same_seed", "different_both")}}
    summaries = {name: {key: describe([p[key] for p in values if p[key] is not None]) for key in metrics}
                 for name, values in groups.items()}
    undefined = sum(p["hidden_linear_cka"] is None for p in pairs)
    duplicate_pairs = [[p["first"], p["second"]] for p in pairs if p["exact_parameter_duplicate"]]
    return {
        "bank_sha256": bank.sha256, "probe_indices": probe.tolist(), "probe_nodes": len(probe),
        "probe_policy": "fixed native source-training pool; no labels used",
        "model_ids": [row["model_id"] for row in rows], "pair_count": len(pairs),
        "summary": summaries, "pairs": pairs, "exact_parameter_duplicate_pairs": duplicate_pairs,
        "undefined_cka_pairs": undefined,
        "unique_hard_prediction_patterns": len({tuple(p.argmax(1).tolist()) for p in probabilities}),
        "functional_variation_observed": any(p["probability_rmse"] > 0 for p in pairs),
        "interpretation": "Probe-dependent variation, not proof of sufficient adaptation diversity. Raw weights have permutation symmetries. High CKA or identical hard predictions alone are not failure criteria.",
    }


def evaluate_source_test(bank: VerifiedBank, source: SourceTrainingData, partition: dict[str, Any],
                         model_factory: Callable[[dict], nn.Module], *, test_indices: Tensor,
                         test_labels: Tensor) -> dict[str, Any]:
    """Final offline scoring only, after partition/protocol selection is frozen.

    Test membership must be exactly the complement of native training+validation.
    Labels enter score_logits, never model construction or prediction. Summaries
    expose subset/seed grouping and do not treat thirty paired jobs as IID trials.
    """
    validate_model_partition(bank, partition)
    expected = torch.arange(source.graph.num_nodes)
    expected = expected[~torch.isin(expected, torch.cat((source.train_indices, source.val_indices)))]
    if (test_indices.device.type != "cpu" or test_indices.dtype != torch.long or test_indices.ndim != 1
            or not torch.equal(test_indices.sort().values, expected)):
        raise ValueError("Source-test indices must preserve the native held-out complement")
    roles = {model_id: role for role, ids in partition["model_ids"].items() for model_id in ids}
    results = []
    for row in bank.manifest["models"]:
        model = _load_model(bank, row, model_factory)
        scores = predict(model, source.graph)
        metric = score_logits(scores, test_indices, test_labels)
        results.append({"model_id": row["model_id"], "subset_id": row["subset_id"], "seed": row["model_seed"],
                        "partition": roles[row["model_id"]], **asdict(metric)})
        del model
    subset_ids = [item["id"] for item in bank.subsets["subsets"]]
    subset_summary = {subset: describe([row["accuracy"] for row in results if row["subset_id"] == subset])
                      for subset in subset_ids}
    return {
        "bank_sha256": bank.sha256, "test_indices": test_indices.tolist(), "models": results,
        "accuracy_descriptive": describe([row["accuracy"] for row in results]),
        "by_subset": subset_summary,
        "subset_mean_accuracy_descriptive": describe([value["mean"] for value in subset_summary.values()]),
        "by_model_seed": {str(seed): describe([row["accuracy"] for row in results if row["seed"] == seed])
                          for seed in bank.manifest["model_seeds"]},
        "interpretation": "Same held-out source nodes, paired model seeds and overlapping label subsets. Descriptive spreads only; no IID confidence interval, significance test, target or adaptation claim.",
    }
