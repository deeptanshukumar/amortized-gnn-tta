"""Real portable workflows for the contributor's experiment families.

Tiny randomly initialized/trained fixtures establish software behavior only;
they are not adaptation efficacy or hardware performance measurements.
"""

import json
from pathlib import Path

import pytest
import torch

from amortized_gnn_tta.adaptation.experiments import training_recipe
from test_generator_cli import prepared, command


@pytest.mark.parametrize("scope", ["head", "backbone", "all"])
def test_portable_review_and_independent_replay(prepared, scope):
    root, architecture = prepared
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads((repo / "configs/generator_csbm_review.json").read_text())
    plan.update(architectures=[architecture], conditioning=["compact", "A"], update_scope=scope,
                epochs=1, generator_seeds=[2101], diffusion_steps=7, sampling_steps=3,
                teacher_steps=1, iterative_steps=2, warmups=0, repeats=2,
                banks={architecture: ["bank", "partition.json"]}, environment_manifest="environments.json")
    config = root / f"{scope}-plan.json"
    config.write_text(json.dumps(plan))
    output = root / f"review-{scope}"
    run = command("run_generator_review.py", "--config", config, "--artifact-root", root, "--output-dir", output)
    assert run.returncode == 0, run.stdout + run.stderr
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "complete" and progress["completed"] == 4 and progress["replayed_pairs"] == 8
    replay = command("replay_generator_review.py", "--review-dir", output)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    audit = json.loads((output / "INDEPENDENT_REPLAY.json").read_text())
    assert audit["runs"] == 4 and audit["pairs"] == 8 and audit["accuracy_mismatches"] == 0
    assert audit["max_logit_error"] == 0
    for checkpoint in output.glob("*/generator_best.pt"):
        payload = torch.load(checkpoint, weights_only=True)
        assert payload["target_scope"] == scope
        assert payload["generator_config"]["include_head_weights"] == ("compact" in checkpoint.parent.name)
        summary = json.loads((checkpoint.parent / "summary.json").read_text())
        assert summary["target_parameter_count"] == payload["generator_config"]["head_dim"]
        assert (summary["speedup"] is not None) == (scope == "head")
        assert all(row["environment_id"] in payload["provenance"]["selected_environment_ids"]
                   for row in json.loads((checkpoint.parent / "pair_results.json").read_text()))
    if architecture == "gcn":
        # Exercise saved-checkpoint timing for broader sites: never compare it
        # silently against the head-only optimizer's different update space.
        checkpoint = next(output.glob("*/generator_best.pt"))
        timing_dir = root / f"benchmark-{scope}"
        result = command("benchmark_adapters.py", "--bank", root / "bank", "--manifest", root / "environments.json",
            "--model-partition", root / "partition.json", "--checkpoint", checkpoint,
            "--output-dir", timing_dir, "--steps", "2", "--max-environments", "1", "--warmups", "0", "--repeats", "2")
        assert result.returncode == 0, result.stdout + result.stderr
        row = json.loads((timing_dir / "results.json").read_text())[0]
        assert row["target_scope"] == scope
        assert (row["speedup"] is not None) == (scope == "head")
        assert ("iterative_im" in row["timing"]) == (scope == "head")
    if architecture == "gcn" and scope == "head":
        # A changed grid/timing protocol may reuse the very same scientific fit.
        plan.update(conditioning=["A"], warmups=1)
        reuse_config = root / "reuse-plan.json"
        reuse_config.write_text(json.dumps(plan))
        reused_dir = root / "review-reused"
        reused = command("run_generator_review.py", "--config", reuse_config, "--artifact-root", root,
            "--output-dir", reused_dir, "--reuse-completed-from", output)
        assert reused.returncode == 0, reused.stdout + reused.stderr
        for summary in reused_dir.glob("*/summary.json"):
            record = json.loads(summary.read_text())
            assert record["training_reused_from"] and "review-head" in record["training_reused_from"]
        replay = command("replay_generator_review.py", "--review-dir", reused_dir)
        assert replay.returncode == 0, replay.stdout + replay.stderr


@pytest.mark.parametrize("recipe", ["compact", "structural"])
def test_portable_training_cli_supports_legacy_inputs(prepared, recipe):
    root, _ = prepared
    output = root / f"legacy-cli-{recipe}"
    result = command("train_adapter.py", "--bank", root / "bank", "--manifest", root / "environments.json",
        "--model-partition", root / "partition.json", "--output-dir", output, "--conditioning", recipe,
        "--scope", "backbone", "--epochs", "1", "--max-models", "1", "--max-environments", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    payload = torch.load(output / "generator_best.pt", weights_only=True)
    assert payload["generator_config"]["include_head_weights"]
    assert payload["descriptor_config"]["conditioning"] is None
    assert payload["descriptor_config"]["descriptor_type"] == ("moments" if recipe == "compact" else "structural")
    assert payload["target_scope"] == "backbone"


def test_completed_checkpoint_reuse_does_not_relax_fitting_settings():
    base = {"epochs": 5, "teacher_steps": 30, "update_scope": "head", "threads": 1,
            "conditioning": ["A", "B"], "architectures": ["gcn"], "repeats": 15}
    assert training_recipe(base) == training_recipe({**base, "conditioning": ["compact", "A"], "repeats": 3})
    for key, value in [("epochs", 6), ("teacher_steps", 40), ("update_scope", "all"), ("threads", 4), ("new_fitting_option", True)]:
        assert training_recipe(base) != training_recipe({**base, key: value})


def test_generated_lora_review_and_factor_replay(prepared):
    root, architecture = prepared
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads((repo / "configs/generator_matched_lora.json").read_text())
    plan.update(architectures=[architecture], epochs=1, generator_seeds=[2101],
        diffusion_steps=7, sampling_steps=3, teacher_steps=1, iterative_steps=2,
        warmups=0, repeats=2, banks={architecture:["bank", "partition.json"]},
        environment_manifest="environments.json")
    plan["lora_config"]["backbone_rank"] = 2
    config = root / "lora-plan.json"
    config.write_text(json.dumps(plan))
    output = root / "review-generated-lora"
    run = command("run_generator_review.py", "--config", config, "--artifact-root", root, "--output-dir", output)
    assert run.returncode == 0, run.stdout+run.stderr
    replay = command("replay_generator_review.py", "--review-dir", output)
    assert replay.returncode == 0, replay.stdout+replay.stderr
    audit = json.loads((output/"INDEPENDENT_REPLAY.json").read_text())
    assert audit["pairs"] == 4 and audit["runs"] == 2 and audit["max_logit_error"] == 0
    for directory in output.glob("*/summary.json"):
        row = json.loads(directory.read_text())
        assert row["update_scope"] == "lora" and row["speedup"] > 0
        assert row["factor_parameter_count"] > 0 and row["target_parameter_count"] > row["factor_parameter_count"]
        saved = torch.load(directory.parent/"pair_00.pt", weights_only=True)
        assert saved["factor_offsets"].numel() == row["factor_parameter_count"]
        assert saved["update_representation"] == "lora-offset-factors-v1"
    head_plan = {**plan, "update_scope":"head", "update_representation":"direct"}
    head_config = root/"matched-head-plan.json"
    head_config.write_text(json.dumps(head_plan))
    head_output = root/"matched-head-study"
    result = command("run_generator_review.py", "--config", head_config, "--artifact-root", root, "--output-dir", head_output)
    assert result.returncode == 0, result.stdout+result.stderr
    replay = command("replay_generator_review.py", "--review-dir", head_output)
    assert replay.returncode == 0, replay.stdout+replay.stderr
    compared = root/"matched-comparison"
    result = command("compare_adapter_studies.py", "--head-study", head_output, "--lora-study", output, "--output-dir", compared)
    assert result.returncode == 0, result.stdout+result.stderr
    progress = json.loads((compared/"progress.json").read_text())
    assert progress["status"] == "complete" and progress["iterative_saved_replays"] == 8 and progress["rows"] == 9
    if architecture == "gcn":
        timing_dir = root/"lora-saved-benchmark"
        checkpoint = next(output.glob("*/generator_best.pt"))
        result = command("benchmark_adapters.py", "--bank", root/"bank", "--manifest", root/"environments.json",
            "--model-partition", root/"partition.json", "--checkpoint", checkpoint, "--output-dir", timing_dir,
            "--steps", "2", "--max-environments", "1", "--warmups", "0", "--repeats", "2")
        assert result.returncode == 0, result.stdout+result.stderr
        row = json.loads((timing_dir/"results.json").read_text())[0]
        assert row["speedup"] > 0 and row["factor_parameter_count"] > 0
