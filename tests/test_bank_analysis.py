"""Artifact consistency, grouped holdouts, mathematical diagnostics and phase audit."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from amortized_gnn_tta.bank_artifacts import (
    artifact_path, create_model_partition, file_sha256, load_model_partition, load_verified_bank,
    save_model_partition, validate_model_partition,
)
from amortized_gnn_tta.bank_diagnostics import analyze_bank_diversity, evaluate_source_test, linear_cka
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.evaluation import predict
from amortized_gnn_tta.models import GCN, GCNConfig
from amortized_gnn_tta.source_subsets import create_source_subsets
from amortized_gnn_tta.training import TrainingConfig, source_training_data
from amortized_gnn_tta.weight_bank import train_weight_bank


def factory(spec):
    return GCN(GCNConfig(**spec["config"]))


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs" / "phase1_csbm_gcn.json").read_text())
    protocol["source"].update(num_nodes=100, num_features=8)
    protocol["model"] = {"hidden_channels": 8, "dropout": 0.2}
    protocol["training"].update(max_epochs=3, patience=2)
    protocol["model_seeds"] = [31, 32]
    values = dict(protocol["source"])
    values["split_ratios"] = tuple(values["split_ratios"])
    bundle = load_dataset([CSBMEnvironment("source", "source_train", CSBMConfig.from_matcha(**values))])
    source = source_training_data(bundle, "source")
    subsets = create_source_subsets(source, label_fraction=0.5, seeds=protocol["subset_seeds"])
    config = GCNConfig(8, 2, **protocol["model"])
    directory = tmp_path_factory.mktemp("phase1-bank") / "bank"
    train_weight_bank(lambda: GCN(config), source, subsets, TrainingConfig(**protocol["training"]),
                      model_spec={"architecture": "two_layer_gcn", "config": asdict(config)},
                      model_seeds=protocol["model_seeds"], output_dir=directory, device="cpu")
    bank = load_verified_bank(directory, source, factory)
    return bundle, source, bank, protocol


def test_complete_bank_audit_and_partition_roundtrip(prepared, tmp_path):
    _, source, bank, _ = prepared
    assert len(bank.manifest["models"]) == 10
    assert bank.sha256 == file_sha256(bank.directory / "bank.json")
    np.random.seed(211)
    numpy_before = np.random.get_state()
    partition = create_model_partition(bank)
    numpy_after = np.random.get_state()
    assert np.array_equal(numpy_before[1], numpy_after[1]) and numpy_before[2:] == numpy_after[2:]
    assert {key: len(value) for key, value in partition["model_ids"].items()} == {
        "meta_train": 6, "validation": 2, "test": 2}
    assert partition == create_model_partition(bank)
    # Permitting scores to affect assignment would leak selection into holdouts.
    changed = deepcopy(bank.manifest)
    for row in changed["models"]:
        row["source_val_accuracy"] = 1 - row["source_val_accuracy"]
    assert create_model_partition(replace(bank, manifest=changed)) == partition
    path = tmp_path / "partition.json"
    save_model_partition(path, bank, partition)
    assert load_model_partition(path, bank) == partition
    with pytest.raises(FileExistsError):
        save_model_partition(path, bank, partition)
    assert load_verified_bank(bank.directory, source, factory).sha256 == bank.sha256


@pytest.mark.parametrize("problem", ["partial", "missing_pair", "duplicate_pair", "hash", "path",
                                     "seed", "supervision", "nan", "shape", "epoch", "score"])
def test_corrupt_banks_rejected_even_when_changed_payload_hash_is_updated(prepared, tmp_path, problem):
    _, source, original, _ = prepared
    directory = tmp_path / "corrupt"
    shutil.copytree(original.directory, directory)
    manifest = json.loads((directory / "bank.json").read_text())
    row = manifest["models"][0]
    if problem == "partial":
        manifest["status"] = "running"
    elif problem == "missing_pair":
        manifest["models"].pop()
    elif problem == "duplicate_pair":
        manifest["models"][1] = deepcopy(row)
    elif problem == "hash":
        row["checkpoint_sha256"] = "0" * 64
    elif problem == "path":
        row["checkpoint"] = "../outside.pt"
    elif problem == "score":
        row["source_val_accuracy"] += 0.123
    else:
        path = directory / row["checkpoint"]
        payload = torch.load(path, weights_only=True)
        if problem == "seed":
            payload["training_config"]["model_seed"] += 1
        elif problem == "supervision":
            payload["source"]["train_indices"][0] = source.val_indices[0]
        elif problem == "nan":
            next(iter(payload["state_dict"].values())).flatten()[0] = float("nan")
        elif problem == "shape":
            payload["state_dict"]["conv1.lin.weight"] = torch.zeros(1, 1)
        else:
            payload["best_epoch"] = 999
        torch.save(payload, path)
        row["checkpoint_sha256"] = file_sha256(path)
    (directory / "bank.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_verified_bank(directory, source, factory)


def test_source_mismatch_and_escaping_artifact_rejected(prepared, tmp_path):
    _, source, bank, _ = prepared
    graph = source.graph.clone()
    graph.x[0, 0] += 1
    with pytest.raises(ValueError, match="does not match"):
        load_verified_bank(bank.directory, replace(source, graph=graph), factory)
    outside = tmp_path / "outside.pt"
    outside.write_text("not a checkpoint")
    with pytest.raises(ValueError):
        artifact_path(bank.directory, str(outside))


@pytest.mark.parametrize("problem", ["group_overlap", "model_leak", "wrong_bank", "missing_model", "missing_seed"])
def test_model_partitions_reject_leaks_and_incomplete_membership(prepared, problem):
    _, _, bank, _ = prepared
    partition = create_model_partition(bank)
    if problem == "group_overlap":
        partition["groups"]["test"] = partition["groups"]["validation"].copy()
    elif problem == "model_leak":
        partition["model_ids"]["test"].append(partition["model_ids"]["meta_train"][0])
    elif problem == "wrong_bank":
        partition["bank_sha256"] = "0" * 64
    elif problem == "missing_model":
        partition["model_ids"]["test"].pop()
    else:
        del partition["seed"]
    with pytest.raises(ValueError):
        validate_model_partition(bank, partition)


def test_linear_cka_matches_centered_gram_identity_and_invariances():
    a = torch.tensor([[1., 2., 0.], [3., 0., 1.], [0., 4., 2.], [2., 1., 4.]], dtype=torch.float64)
    b = torch.tensor([[2., 1.], [0., 3.], [4., 1.], [1., 2.]], dtype=torch.float64)
    center = torch.eye(4, dtype=torch.float64) - torch.ones(4, 4, dtype=torch.float64) / 4
    gram_a, gram_b = center @ (a @ a.T) @ center, center @ (b @ b.T) @ center
    expected = float((gram_a * gram_b).sum() / (gram_a.norm() * gram_b.norm()))
    assert linear_cka(a, b) == pytest.approx(expected, abs=1e-12)
    assert linear_cka(a, 3 * a[:, [2, 0, 1]] + 10) == pytest.approx(1, abs=1e-12)
    assert linear_cka(a, torch.ones(4, 2)) is None
    with pytest.raises(ValueError):
        linear_cka(a, b[:2])


def test_hidden_permutations_change_weights_but_preserve_predictions(prepared):
    _, source, _, _ = prepared
    first = GCN(GCNConfig(8, 2, 8, 0.2)).eval()
    second = deepcopy(first)
    perm = torch.arange(7, -1, -1)
    with torch.no_grad():
        second.conv1.lin.weight.copy_(first.conv1.lin.weight[perm])
        second.conv1.bias.copy_(first.conv1.bias[perm])
        second.conv2.lin.weight.copy_(first.conv2.lin.weight[:, perm])
    assert not torch.equal(first.conv1.lin.weight, second.conv1.lin.weight)
    torch.testing.assert_close(predict(first, source.graph), predict(second, source.graph), rtol=1e-6, atol=1e-6)
    assert linear_cka(first.encode(source.graph.x, source.graph.edge_index),
                      second.encode(source.graph.x, source.graph.edge_index)) == pytest.approx(1)


def test_encode_preserves_original_forward_formula(prepared):
    _, source, _, _ = prepared
    model = GCN(GCNConfig(8, 2, 8, 0.2)).train()
    torch.manual_seed(41)
    original = model.conv2(F.dropout(F.relu(model.conv1(source.graph.x, source.graph.edge_index)),
                                    p=model.config.dropout, training=True), source.graph.edge_index)
    torch.manual_seed(41)
    assert torch.equal(original, model(source.graph.x, source.graph.edge_index))


def test_diversity_is_label_free_and_uses_native_training_probe(prepared, monkeypatch):
    _, source, bank, _ = prepared
    import amortized_gnn_tta.bank_diagnostics as diagnostics
    def forbidden_score(*args, **kwargs):
        raise AssertionError("Diversity must not score labels")
    monkeypatch.setattr(diagnostics, "score_logits", forbidden_score)
    representation = lambda model, graph: model.encode(graph.x, graph.edge_index)
    first = analyze_bank_diversity(bank, source, factory, representation)
    relabeled = replace(source, train_labels=1-source.train_labels, val_labels=1-source.val_labels)
    second = analyze_bank_diversity(bank, relabeled, factory, representation)
    assert first == second and first["pair_count"] == 45
    assert first["probe_nodes"] == len(source.train_indices)
    assert first["functional_variation_observed"] and not first["exact_parameter_duplicate_pairs"]
    for pair in first["pairs"]:
        assert 0 <= pair["prediction_disagreement"] <= 1 and 0 <= pair["hidden_linear_cka"] <= 1
    with pytest.raises(ValueError, match="Probe"):
        analyze_bank_diversity(bank, source, factory, representation, probe_indices=source.val_indices)


def test_source_test_scores_are_explicit_and_cannot_change_partition(prepared):
    bundle, source, bank, _ = prepared
    partition = create_model_partition(bank)
    indices, labels = bundle.select("source", "test"), bundle.evaluation_labels("source", "test")
    first = evaluate_source_test(bank, source, partition, factory, test_indices=indices, test_labels=labels)
    second = evaluate_source_test(bank, source, partition, factory, test_indices=indices, test_labels=1-labels)
    assert create_model_partition(bank) == partition
    for a, b in zip(first["models"], second["models"]):
        assert a["accuracy"] + b["accuracy"] == pytest.approx(1)
    with pytest.raises(ValueError, match="held-out complement"):
        evaluate_source_test(bank, source, partition, factory, test_indices=source.train_indices, test_labels=source.train_labels)


def test_end_to_end_finalization_with_frozen_small_protocol(prepared, tmp_path):
    _, _, bank, protocol = prepared
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(protocol))
    output = tmp_path / "review"
    script = Path(__file__).resolve().parents[1] / "examples" / "finalize_phase1.py"
    result = subprocess.run([sys.executable, str(script), "--config", str(config_path), "--bank", str(bank.directory),
                             "--output-dir", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "phase1_report.json").read_text())
    assert report["status"] == "complete" and report["models"] == 10
    assert report["partition_counts"] == {"meta_train": 6, "validation": 2, "test": 2}
    assert (output / "REVIEW.md").exists() and (output / "diversity_pairs.csv").exists()
    assert json.loads((output / "source_test.json").read_text())["accuracy_descriptive"]["count"] == 10
    for filename, digest in report["analysis_code_snapshot_sha256"].items():
        assert file_sha256(output / "analysis_code_snapshot" / filename) == digest
