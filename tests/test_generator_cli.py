"""Real command-line save/reload and validation-default regressions."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from amortized_gnn_tta.bank_artifacts import create_model_partition, save_model_partition, load_verified_bank
from amortized_gnn_tta.data import (CSBMConfig, CSBMEnvironment, CSBMShift, load_dataset,
    create_environment_manifest, save_environment_manifest)
from amortized_gnn_tta.models import build_model, make_model_spec
from amortized_gnn_tta.source_subsets import create_source_subsets
from amortized_gnn_tta.training import source_training_data, TrainingConfig
from amortized_gnn_tta.weight_bank import train_weight_bank


@pytest.fixture(scope="module", params=["gcn", "gat", "graphsage"])
def prepared(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("generator-cli-" + request.param)
    config = CSBMConfig(num_nodes=100, num_features=8, p_in=.2, p_out=.03)
    source = source_training_data(load_dataset([CSBMEnvironment("source", "source_train", config)]), "source")
    subsets = create_source_subsets(source, label_fraction=.5, seeds=[101, 102, 103, 104, 105])
    spec = make_model_spec(request.param, in_channels=8, hidden_channels=8, out_channels=2)
    bank_dir = root / "bank"
    train_weight_bank(lambda: build_model(spec), source, subsets, TrainingConfig(max_epochs=2, patience=1),
                     model_spec=spec, model_seeds=[31], output_dir=bank_dir, device="cpu")
    bank = load_verified_bank(bank_dir, source, build_model)
    partition_path = root / "partition.json"
    save_model_partition(partition_path, bank, create_model_partition(bank))
    manifest = create_environment_manifest(config, [CSBMShift("attribute", attribute_offset=.1)],
        lineage_counts=(1, 1, 1), train_severities=(.5,), test_severities=(.5,), generation_seed=140601)
    manifest_path = root / "environments.json"
    save_environment_manifest(manifest_path, manifest)
    return root, request.param


def command(script, *args):
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run([sys.executable, str(repo / "examples" / script), *map(str, args)],
                          cwd=repo, env=env, text=True, capture_output=True, timeout=90)


@pytest.mark.parametrize("kind", ["hypernetwork", "diffusion"])
def test_actual_training_and_benchmark_cli(prepared, kind):
    root, architecture = prepared
    output = root / kind
    args = ["--bank", root / "bank", "--manifest", root / "environments.json",
            "--model-partition", root / "partition.json", "--output-dir", output,
            "--generator", kind, "--conditioning", "D", "--epochs", "1", "--max-models", "1",
            "--max-environments", "2", "--diffusion-steps", "7", "--sampling-steps", "3", "--oracle-steps", "1"]
    result = command("train_adapter.py", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete" and run["evaluation_role"] == "validation"
    assert set(run["selected_model_ids"]) == {"meta_train", "validation"}
    assert all(len(ids) == 1 for ids in run["selected_model_ids"].values())
    assert all(r["role"] == "validation" for r in json.loads((output / "pair_results.json").read_text()))
    assert (output / "code_snapshot.zip").is_file()
    assert (output / "generator_best.pt").is_file()
    again = command("train_adapter.py", *args)
    assert again.returncode != 0 and json.loads((output / "run.json").read_text())["status"] == "complete"
    if architecture == "gcn":
        timing_dir = root / (kind + "-timing")
        benchmark = command("benchmark_adapters.py", "--bank", root / "bank", "--manifest", root / "environments.json",
            "--model-partition", root / "partition.json", "--checkpoint", output / "generator_best.pt",
            "--output-dir", timing_dir, "--max-environments", "1", "--steps", "2", "--warmups", "0", "--repeats", "2")
        assert benchmark.returncode == 0, benchmark.stdout + benchmark.stderr
        rows = json.loads((timing_dir / "results.json").read_text())
        assert len(rows) == 1 and rows[0]["speedup"] > 0
        assert len(rows[0]["timing"]["generated"]["samples_seconds"]) == 2


def test_provenance_for_exported_code_without_git(monkeypatch):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "examples" / "train_adapter.py"
    spec = importlib.util.spec_from_file_location("generator_cli_for_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def absent_git(*args, **kwargs):
        raise FileNotFoundError("git executable absent in exported environment")
    monkeypatch.setattr(module.subprocess, "run", absent_git)
    provenance = module.code_provenance()
    assert provenance["revision"] is None and provenance["dirty"] is None
    assert provenance["git_status"].startswith("unavailable")
    assert "examples/train_adapter.py" in provenance["source_hashes"]


def test_training_cli_requires_durable_output_before_loading_inputs():
    result = command("train_adapter.py", "--model-partition", "does-not-exist.json")
    assert result.returncode == 2
    assert "--output-dir" in result.stderr
