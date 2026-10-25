"""Bank membership, per-model evidence, reproducibility, and partial failures."""

from dataclasses import asdict
import hashlib
import json

import pytest
import torch

from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.models import GCN, GCNConfig
from amortized_gnn_tta.source_subsets import create_source_subsets
from amortized_gnn_tta.training import TrainingConfig, source_training_data
from amortized_gnn_tta.weight_bank import train_weight_bank


@pytest.fixture
def source_and_subsets():
    bundle = load_dataset([CSBMEnvironment("source", "source_train", CSBMConfig(
        num_nodes=100, num_features=8, p_in=0.2, p_out=0.01))])
    source = source_training_data(bundle, "source")
    subsets = create_source_subsets(source, label_fraction=0.5, seeds=[101, 102])
    return source, subsets


def run_bank(source, subsets, directory, seeds=(31, 32), factory=None):
    config = GCNConfig(8, 2, 8, 0.2)
    return train_weight_bank(factory or (lambda: GCN(config)), source, subsets,
                             TrainingConfig(max_epochs=4, patience=2),
                             model_spec={"architecture": "two_layer_gcn", "config": asdict(config)},
                             model_seeds=seeds, output_dir=directory, device="cpu")


def test_cartesian_bank_contains_only_selected_source_supervision(source_and_subsets, tmp_path):
    source, subsets = source_and_subsets
    original_x = source.graph.x.clone()
    original_indices = source.train_indices.clone()
    manifest = run_bank(source, subsets, tmp_path / "bank")
    directory = tmp_path / "bank"
    assert manifest == json.loads((directory / "bank.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_models"] == 4
    assert manifest["expected_models"] == 4
    assert {(row["subset_id"], row["model_seed"]) for row in manifest["models"]} == {
        (subset, seed) for subset in ("source_subset_0", "source_subset_1") for seed in (31, 32)}
    for row in manifest["models"]:
        assert row["status"] == "complete" and row["cpu_reload_predictions_close"]
        checkpoint = directory / row["checkpoint"]
        metrics_path = directory / row["metrics"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha256"]
        assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == row["metrics_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = next(item["train_indices"] for item in subsets["subsets"] if item["id"] == row["subset_id"])
        assert payload["source"]["train_indices"].tolist() == expected
        assert torch.equal(payload["source"]["val_indices"], source.val_indices)
        assert payload["training_config"]["model_seed"] == row["model_seed"]
        assert payload["source"]["source_subset"]["id"] == row["subset_id"]
        assert all(value.device.type == "cpu" for value in payload["state_dict"].values())
        metrics = json.loads(metrics_path.read_text())
        assert metrics["selected_source_train"]["num_nodes"] == 30
        assert metrics["source_val"]["num_nodes"] == 20
        assert metrics["best_val_loss"] == min(item["val_loss"] for item in metrics["history"])
        assert "source_test" not in metrics and "target_metrics" not in metrics
    assert torch.equal(source.graph.x, original_x)
    assert torch.equal(source.train_indices, original_indices)


def test_each_job_is_reproducible_independently_of_seed_order(source_and_subsets, tmp_path):
    source, subsets = source_and_subsets
    first = run_bank(source, subsets, tmp_path / "first")
    second = run_bank(source, subsets, tmp_path / "second", seeds=(32, 31))
    second_rows = {row["model_id"]: row for row in second["models"]}
    for row in first["models"]:
        a = json.loads((tmp_path / "first" / row["metrics"]).read_text())
        b = json.loads((tmp_path / "second" / second_rows[row["model_id"]]["metrics"]).read_text())
        assert a["history"] == b["history"] and a["best_epoch"] == b["best_epoch"]
        weights_a = torch.load(tmp_path / "first" / row["checkpoint"], weights_only=True)["state_dict"]
        weights_b = torch.load(tmp_path / "second" / row["checkpoint"], weights_only=True)["state_dict"]
        assert all(torch.equal(value, weights_b[key]) for key, value in weights_a.items())


def test_failed_job_preserves_completed_checkpoints_and_records_incomplete_bank(source_and_subsets, tmp_path):
    source, subsets = source_and_subsets
    calls = 0
    def fail_second_training_job():
        nonlocal calls
        calls += 1
        if calls == 3:  # First job constructs training+reload models; second fit fails.
            raise RuntimeError("Deliberate test failure")
        return GCN(GCNConfig(8, 2, 8, 0.2))
    directory = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Deliberate"):
        run_bank(source, subsets, directory, factory=fail_second_training_job)
    manifest = json.loads((directory / "bank.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_models"] == 1
    assert [row["status"] for row in manifest["models"]] == ["complete", "failed", "pending", "pending"]
    assert (directory / manifest["models"][0]["checkpoint"]).exists()
    assert manifest["error"]["type"] == "RuntimeError"


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [True], [2**32]])
def test_invalid_seeds_fail_before_creating_run_directory(source_and_subsets, tmp_path, seeds):
    source, subsets = source_and_subsets
    with pytest.raises(ValueError, match="seeds"):
        run_bank(source, subsets, tmp_path / "invalid", seeds=seeds)
    assert not (tmp_path / "invalid").exists()


def test_mismatched_subset_source_fails_before_training(source_and_subsets, tmp_path):
    source, subsets = source_and_subsets
    source.graph.x[0, 0] += 0.1
    with pytest.raises(ValueError, match="does not match"):
        run_bank(source, subsets, tmp_path / "mismatched")
    assert not (tmp_path / "mismatched").exists()


def test_existing_output_is_not_modified(source_and_subsets, tmp_path):
    source, subsets = source_and_subsets
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "bank.json"
    marker.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        run_bank(source, subsets, directory)
    assert marker.read_text() == "existing evidence"
    assert list(directory.iterdir()) == [marker]
