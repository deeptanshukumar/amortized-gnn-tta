"""Paired interventions, whole-lineage holdouts, leakage boundaries and real CLI."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from amortized_gnn_tta.data import (
    CSBMConfig, CSBMShift, audit_environment_suite, create_environment_manifest,
    default_csbm_shifts, generate_csbm, load_environment_manifest, load_shift_environment,
    save_environment_manifest, shifted_config, validate_environment_manifest,
)
from amortized_gnn_tta.evaluation import predict
from amortized_gnn_tta.models import ARCHITECTURES, build_model, make_model_spec
from amortized_gnn_tta.training import source_training_data


@pytest.fixture
def reference():
    return CSBMConfig.from_matcha(num_nodes=100, num_features=8, degree_scale=5,
                                 homophily_parameter=0.8, edge_seed=11, feature_seed=12, split_seed=13)


@pytest.fixture
def manifest(reference):
    return create_environment_manifest(reference, default_csbm_shifts(), lineage_counts=(1, 1, 1))


def record(manifest, row):
    return load_shift_environment(manifest, row["id"]).graphs[row["id"]]


def resign(manifest):
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_all_shift_families_and_factor_combinations(reference):
    shifts = default_csbm_shifts()
    assert len(shifts) == 11 and len({s.name for s in shifts}) == 11
    assert {s.family for s in shifts} == {"attribute", "structural", "combined"}
    for shift in shifts:
        assert shifted_config(reference, shift, 0) == reference
        halfway = shifted_config(reference, shift, 0.5)
        final = shifted_config(reference, shift, 1)
        assert halfway.negative_mean == pytest.approx(reference.negative_mean + shift.attribute_offset / 2)
        assert final.positive_mean - final.negative_mean == pytest.approx(0.06)
        assert final.noise_scale == reference.noise_scale
        degree = final.num_nodes * (final.p_in + final.p_out)
        assert degree == pytest.approx(shift.degree_scale if shift.degree_scale is not None else 5)
        assert final.p_in / (final.p_in + final.p_out) == pytest.approx(
            shift.homophily_parameter if shift.homophily_parameter is not None else 0.8)
        assert (halfway.edge_seed, halfway.feature_seed, halfway.split_seed) == (11, 12, 13)


def test_exact_pairing_and_feature_translation(reference):
    base = generate_csbm(reference, role="meta_train")
    for shift in default_csbm_shifts():
        variant = generate_csbm(shifted_config(reference, shift, 1), role="meta_train")
        assert torch.equal(base.labels, variant.labels)
        assert all(torch.equal(base.splits[k], variant.splits[k]) for k in base.splits)
        if shift.family == "attribute":
            assert torch.equal(base.graph.edge_index, variant.graph.edge_index)
        if shift.family == "structural":
            assert torch.equal(base.graph.x, variant.graph.x)
        else:
            torch.testing.assert_close(variant.graph.x - base.graph.x,
                                       torch.full_like(base.graph.x, shift.attribute_offset / math.sqrt(8)),
                                       atol=1e-7, rtol=1e-5)
    # Changes are drawn on a common latent-noise realization, not independent
    # redraws that merely preserve the marginal feature distribution.
    assert torch.equal(base.graph.x, generate_csbm(reference).graph.x)


def test_degree_shift_nests_edges_and_homophily_has_finite_size_drift(reference):
    def edges(config):
        return set(map(tuple, generate_csbm(config).graph.edge_index.t().tolist()))
    base = edges(reference)
    low = shifted_config(reference, CSBMShift("low", degree_scale=2), 1)
    high = shifted_config(reference, CSBMShift("high", degree_scale=10), 1)
    assert edges(low) < base < edges(high)
    hom = shifted_config(reference, CSBMShift("hom", homophily_parameter=0.2), 1)
    assert hom.expected_degree - reference.expected_degree == pytest.approx(5 * (0.8 - 0.2) / 100)
    assert hom.num_nodes * (hom.p_in + hom.p_out) == pytest.approx(5)


def test_default_counts_lineage_exclusivity_and_support(reference):
    manifest = create_environment_manifest(reference, default_csbm_shifts())
    assert len(manifest["environments"]) == 126
    expected = {"meta_train": (3, 69), "validation": (1, 23), "target_test": (1, 34)}
    for role, (base_count, variant_count) in expected.items():
        assert sum(r["role"] == role for r in manifest["lineages"]) == base_count
        assert sum(r["role"] == role for r in manifest["environments"]) == variant_count
    seen_seeds = [set(), set(), set()]
    for lineage in manifest["lineages"]:
        variants = [e for e in manifest["environments"] if e["lineage_id"] == lineage["id"]]
        assert {e["role"] for e in variants} == {lineage["role"]}
        for name, pool in zip(("edge_seed", "feature_seed", "split_seed"), seen_seeds):
            assert lineage["config"][name] not in pool | {getattr(reference, name)}
            pool.add(lineage["config"][name])
    test = [e for e in manifest["environments"] if e["role"] == "target_test" and e["family"] != "control"]
    assert {e["severity"] for e in test if e["seen_severity_in_meta_train"]} == {0.5}
    assert {e["severity"] for e in test if not e["seen_severity_in_meta_train"]} == {0.75, 1.0}
    assert all(e["seen_family_in_meta_train"] for e in test)


def test_optional_unseen_family_has_no_training_or_validation_exposure(reference):
    manifest = create_environment_manifest(reference, default_csbm_shifts(), lineage_counts=(1, 1, 1),
                                           held_out_families=("combined",))
    assert not any(e["family"] == "combined" and e["role"] != "target_test" for e in manifest["environments"])
    heldout = [e for e in manifest["environments"] if e["family"] == "combined"]
    assert heldout and all(not e["seen_family_in_meta_train"] for e in heldout)
    assert {e["seen_severity_in_meta_train"] for e in heldout} == {False, True}


def test_generation_and_partition_rngs_are_separate_and_global_states_untouched(reference):
    np.random.seed(20)
    torch.manual_seed(21)
    before_np, before_torch = np.random.get_state(), torch.random.get_rng_state().clone()
    first = create_environment_manifest(reference, default_csbm_shifts())
    changed_partition = create_environment_manifest(reference, default_csbm_shifts(), partition_seed=502)
    assert [e["config"] for e in first["lineages"]] == [e["config"] for e in changed_partition["lineages"]]
    assert [e["role"] for e in first["lineages"]] != [e["role"] for e in changed_partition["lineages"]]
    changed_generation = create_environment_manifest(reference, default_csbm_shifts(), generation_seed=602)
    assert [e["role"] for e in first["lineages"]] == [e["role"] for e in changed_generation["lineages"]]
    assert [e["config"] for e in first["lineages"]] != [e["config"] for e in changed_generation["lineages"]]
    graph_id = first["environments"][0]["id"]
    a, b = load_shift_environment(first, graph_id), load_shift_environment(first, graph_id)
    assert torch.equal(a.unlabeled_input(graph_id).x, b.unlabeled_input(graph_id).x)
    assert torch.equal(a.unlabeled_input(graph_id).edge_index, b.unlabeled_input(graph_id).edge_index)
    assert torch.equal(before_torch, torch.random.get_rng_state())
    after_np = np.random.get_state()
    assert before_np[0] == after_np[0] and np.array_equal(before_np[1], after_np[1])
    assert before_np[2:] == after_np[2:]


def test_manifest_persistence_tampering_and_no_overwrite(manifest, tmp_path):
    path = tmp_path / "environments.json"
    save_environment_manifest(path, manifest)
    assert load_environment_manifest(path) == manifest
    with pytest.raises(FileExistsError):
        save_environment_manifest(path, manifest)
    damaged = deepcopy(manifest)
    damaged["environments"][0]["role"] = "target_test"
    with pytest.raises(ValueError, match="checksum"):
        validate_environment_manifest(damaged)
    resign(damaged)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_environment_manifest(damaged)
    damaged = deepcopy(manifest)
    damaged["environments"][1]["config"]["feature_seed"] += 1
    resign(damaged)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_environment_manifest(damaged)
    with pytest.raises(ValueError, match="Unknown"):
        load_shift_environment(manifest, "absent")


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_model_inputs_do_not_expose_labels_or_shift_metadata(manifest, architecture):
    target = next(e for e in manifest["environments"] if e["role"] == "target_test" and e["family"] == "combined")
    bundle = load_shift_environment(manifest, target["id"])
    inputs = bundle.unlabeled_input(target["id"])
    assert set(inputs.keys()) == {"num_nodes", "x", "edge_index"}
    assert set(bundle.graphs[target["id"]].splits) == {"eval"}
    model = build_model(make_model_spec(architecture, 8, 2, hidden_channels=8))
    before = predict(model, inputs)
    bundle.graphs[target["id"]].labels ^= 1
    assert torch.equal(before, predict(model, bundle.unlabeled_input(target["id"])))
    with pytest.raises(ValueError, match="evaluation-only"):
        bundle.training_labels(target["id"])
    with pytest.raises(ValueError):
        source_training_data(bundle, target["id"])
    inputs.x.zero_()
    assert bundle.unlabeled_input(target["id"]).x.abs().sum() > 0


def test_meta_label_access_preserves_existing_source_runner_boundary(manifest):
    meta = next(e for e in manifest["environments"] if e["role"] == "meta_train" and e["family"] == "attribute")
    bundle = load_shift_environment(manifest, meta["id"])
    assert len(bundle.training_labels(meta["id"])) == 60
    assert {key: len(value) for key, value in bundle.graphs[meta["id"]].splits.items()} == {
        "train": 60, "val": 20, "test": 20}
    # A future meta-trainer must opt in explicitly. The existing SOURCE fitting
    # command must not silently consume a labeled shifted meta-environment.
    with pytest.raises(ValueError, match="source_train"):
        source_training_data(bundle, meta["id"])


def test_full_construction_audit_and_degenerate_cross_lineage_duplicates(manifest, reference):
    audit = audit_environment_suite(manifest)
    assert audit["status"] == "complete"
    assert audit["partition_counts"] == {"meta_train": 23, "validation": 23, "target_test": 34}
    assert all(e["labels_and_splits_unchanged"] for e in audit["environments"])
    assert len({e["fingerprint"] for e in audit["lineages"]}) == 3
    degenerate = replace(reference, p_in=0, p_out=0, noise_scale=0)
    recipe = create_environment_manifest(degenerate, [CSBMShift("offset", attribute_offset=0.02)],
                                          lineage_counts=(1, 1, 1))
    with pytest.raises(ValueError, match="Different lineages"):
        audit_environment_suite(recipe)


@pytest.mark.parametrize("kwargs", [
    {"attribute_offset": float("nan")}, {"attribute_offset": True}, {"attribute_offset": None},
    {"homophily_parameter": 1.1}, {"degree_scale": -1}, {}, {"name": "../bad", "attribute_offset": 0.02},
])
def test_invalid_shift_definitions(kwargs):
    with pytest.raises(ValueError):
        CSBMShift(**({"name": "shift"} | kwargs))


@pytest.mark.parametrize("severity", [-0.1, 1.1, float("nan"), True])
def test_invalid_severity(reference, severity):
    with pytest.raises(ValueError):
        shifted_config(reference, CSBMShift("offset", attribute_offset=0.02), severity)


@pytest.mark.parametrize("kwargs", [
    {"lineage_counts": (1, 0, 1)}, {"lineage_counts": (1, True, 1)},
    {"train_severities": (0,)}, {"test_severities": (0.5, 0.5)},
    {"partition_seed": -1}, {"generation_seed": True},
    {"held_out_families": ("unknown",)},
    {"held_out_families": ("attribute", "structural", "combined")},
])
def test_invalid_environment_design(reference, kwargs):
    with pytest.raises(ValueError):
        create_environment_manifest(reference, default_csbm_shifts(), **kwargs)


def test_noop_endpoints_impossible_probabilities_and_duplicate_conditions(reference):
    for shift in (CSBMShift("same_h", homophily_parameter=0.8),
                  CSBMShift("same_d", degree_scale=5), CSBMShift("huge_d", degree_scale=1000)):
        with pytest.raises(ValueError):
            create_environment_manifest(reference, [shift])
    with pytest.raises(ValueError, match="Duplicate shift definitions"):
        create_environment_manifest(reference, [CSBMShift("a", attribute_offset=0.02),
                                                 CSBMShift("b", attribute_offset=0.02)])


def test_real_small_cli_and_saved_config_matches_api(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "controlled_csbm_v1.json").read_text())
    assert [asdict(CSBMShift(**s)) for s in config["shifts"]] == [asdict(s) for s in default_csbm_shifts()]
    output = tmp_path / "suite"
    completed = subprocess.run([sys.executable, str(root / "examples" / "prepare_shift_environments.py"),
                                "--output-dir", str(output)], check=True, capture_output=True, text=True)
    summary = json.loads(completed.stdout)
    assert summary["status"] == "complete" and summary["nodes_per_graph"] == 300
    assert summary["environment_counts"] == {"meta_train": 69, "validation": 23, "target_test": 34}
    manifest = load_environment_manifest(output / "environment_manifest.json")
    audit = json.loads((output / "construction_audit.json").read_text())
    assert audit["manifest_sha256"] == manifest["manifest_sha256"]
    assert len(audit["environments"]) == 126
    repeated = subprocess.run([sys.executable, str(root / "examples" / "prepare_shift_environments.py"),
                               "--output-dir", str(output)], capture_output=True, text=True)
    assert repeated.returncode != 0 and "new output directory" in repeated.stderr
