"""Real bank/partition integration, held-out rejection and review artifacts."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from amortized_gnn_tta.adaptation import AdaptationConfig, PEFTConfig
from amortized_gnn_tta.adaptation_artifacts import load_weight_delta
from amortized_gnn_tta.adaptation_review import run_development_review
from amortized_gnn_tta.bank_artifacts import (
    create_model_partition, load_verified_bank, save_model_partition,
)
from amortized_gnn_tta.data import (
    CSBMConfig, CSBMEnvironment, CSBMShift, create_environment_manifest, load_dataset,
    load_shift_environment, save_environment_manifest,
)
from amortized_gnn_tta.evaluation import predict
from amortized_gnn_tta.models import build_model, make_model_spec
from amortized_gnn_tta.source_subsets import create_source_subsets
from amortized_gnn_tta.training import TrainingConfig, source_training_data
from amortized_gnn_tta.weight_bank import train_weight_bank


@pytest.fixture
def setup(tmp_path):
    config = CSBMConfig.from_matcha(num_nodes=100, num_features=8, degree_scale=5,
                                   homophily_parameter=0.8, edge_seed=11, feature_seed=12, split_seed=13)
    source = source_training_data(load_dataset([CSBMEnvironment("source", "source_train", config)]), "source")
    subsets = create_source_subsets(source, label_fraction=0.5, seeds=(101, 102, 103, 104, 105))
    spec = make_model_spec("gcn", 8, 2, hidden_channels=8)
    directory = tmp_path / "bank"
    train_weight_bank(lambda: build_model(spec), source, subsets, TrainingConfig(max_epochs=2, patience=1),
                      model_spec=spec, model_seeds=(31,), output_dir=directory, device="cpu")
    bank = load_verified_bank(directory, source, build_model)
    partition = create_model_partition(bank)
    env = create_environment_manifest(config, [CSBMShift("attribute", attribute_offset=0.02),
                                               CSBMShift("homophily", homophily_parameter=0.2)],
                                      lineage_counts=(1, 1, 1), train_severities=(0.5,))
    return source, bank, partition, env


def selection(partition, env):
    return [partition["model_ids"]["meta_train"][0]], [e["id"] for e in env["environments"] if e["role"] == "meta_train"]


def test_real_adaptation_review_and_delta_reloads(setup, tmp_path):
    source, bank, partition, env = setup
    models, environments = selection(partition, env)
    output = tmp_path / "review"
    report = run_development_review(
        bank, partition, env, source, model_ids=models, environment_ids=environments, output_dir=output,
        peft_configs=(PEFTConfig("head"), PEFTConfig("lora", backbone_rank=2)), config=AdaptationConfig(steps=2))
    assert report["status"] == "complete" and len(report["baselines"]) == 3
    assert len(report["adaptations"]) == 18
    progress = json.loads((output / "progress.json").read_text())
    assert progress["completed_adaptations"] == progress["planned_adaptations"] == 18
    assert progress["status"] == "complete"
    for row in report["adaptations"]:
        path = output / row["weight_delta"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["weight_delta_sha256"]
        restored, payload = load_weight_delta(path)
        bundle = load_shift_environment(env, row["environment_id"])
        graph = bundle.unlabeled_input(row["environment_id"])
        torch.testing.assert_close(predict(restored, graph), payload["adapted_logits"])
        assert payload["context"]["evaluation_indices"] == bundle.select(row["environment_id"], "test").tolist()
        assert payload["context"]["model_role"] == payload["context"]["environment_role"] == "meta_train"
        if row["objective"] != "supervised_development":
            assert row["selected_step"] == row["steps_run"] == 2
    with pytest.raises(FileExistsError):
        run_development_review(bank, partition, env, source, model_ids=models, environment_ids=environments,
                               output_dir=output)


@pytest.mark.parametrize("bad_selection", ("model", "environment", "duplicate", "reference"))
def test_heldout_invalid_and_reference_mismatch_rejected_before_output(setup, tmp_path, bad_selection):
    source, bank, partition, env = setup
    models, environments = selection(partition, env)
    if bad_selection == "model":
        models = partition["model_ids"]["test"]
    elif bad_selection == "environment":
        environments = [next(e["id"] for e in env["environments"] if e["role"] == "target_test")]
    elif bad_selection == "duplicate":
        environments *= 2
    else:
        values = dict(env["reference"])
        values["split_ratios"] = tuple(values["split_ratios"])
        env = create_environment_manifest(replace(CSBMConfig(**values), positive_mean=0.04),
                                          [CSBMShift("offset", attribute_offset=0.02)])
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_development_review(bank, partition, env, source, model_ids=models, environment_ids=environments,
                               output_dir=output)
    assert not output.exists()


def test_baseline_only_makes_no_weight_update_artifacts(setup, tmp_path):
    source, bank, partition, env = setup
    models, environments = selection(partition, env)
    output = tmp_path / "baseline"
    report = run_development_review(bank, partition, env, source, model_ids=models,
                                    environment_ids=environments, output_dir=output, peft_configs=(), objectives=())
    assert len(report["baselines"]) == 3 and report["adaptations"] == []
    assert not list(output.rglob("*.pt"))


def test_actual_cli_and_pre_execution_snapshot(setup, tmp_path):
    source, bank, partition, env = setup
    root = Path(__file__).resolve().parents[1]
    partition_path, manifest_path = tmp_path / "partition.json", tmp_path / "environments.json"
    save_model_partition(partition_path, bank, partition)
    save_environment_manifest(manifest_path, env)
    output = tmp_path / "cli"
    result = subprocess.run([sys.executable, str(root / "examples" / "run_adaptation_review.py"),
                             "--bank", str(bank.directory), "--model-partition", str(partition_path),
                             "--environments", str(manifest_path), "--output-dir", str(output),
                             "--steps", "2", "--backbone-rank", "2"],
                            capture_output=True, text=True, check=True)
    report = json.loads((output / "review.json").read_text())
    assert report["status"] == "complete" and len(report["adaptations"]) == 54
    assert '"status": "complete"' in result.stdout
    implementation = json.loads((output / "implementation.json").read_text())
    assert implementation["snapshot_captured"] == "before execution"
    for relative, expected in implementation["adaptation_code_snapshot_sha256"].items():
        assert hashlib.sha256((output / "adaptation_code_snapshot" / relative).read_bytes()).hexdigest() == expected
