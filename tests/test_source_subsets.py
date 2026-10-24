"""Fixed-holdout subset sampling, persistence, and integration with source fitting."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.models import GCN, GCNConfig
from amortized_gnn_tta.source_subsets import (
    create_source_subsets, load_source_subsets, save_source_subsets, source_subset_view,
)
from amortized_gnn_tta.training import (
    TrainingConfig, fit_source, load_checkpoint, save_checkpoint, source_training_data,
)


@pytest.fixture
def bundle():
    config = CSBMConfig(num_nodes=100, num_features=8, p_in=0.2, p_out=0.01)
    return load_dataset([
        CSBMEnvironment("source", "source_train", config),
        CSBMEnvironment("target", "target_test", replace(config, edge_seed=10, feature_seed=11)),
    ])


def make_manifest(source):
    return create_source_subsets(source, label_fraction=0.5, seeds=[101, 102, 103, 104, 105])


def test_five_distinct_stratified_subsets_preserve_native_graph_and_splits(bundle):
    source = source_training_data(bundle, "source")
    original = {key: value.clone() for key, value in bundle.graphs["source"].splits.items()}
    manifest = make_manifest(source)
    memberships = []
    for item in manifest["subsets"]:
        subset = source_subset_view(source, manifest, item["id"])
        assert subset.graph is source.graph  # Views do not duplicate graph storage.
        assert len(subset.train_indices) == 30
        assert torch.bincount(subset.train_labels).tolist() == [15, 15]
        assert torch.isin(subset.train_indices, source.train_indices).all()
        assert torch.equal(subset.val_indices, source.val_indices)
        assert torch.equal(subset.val_labels, source.val_labels)
        assert not torch.isin(subset.train_indices, original["test"]).any()
        memberships.append(set(subset.train_indices.tolist()))
    assert len({tuple(sorted(values)) for values in memberships}) == 5
    assert any(memberships[0] & other for other in memberships[1:])  # Overlap is permitted.
    assert len(source.train_indices) == 60
    assert all(torch.equal(value, bundle.graphs["source"].splits[key]) for key, value in original.items())


def test_exact_reload_training_and_checkpoint_provenance(bundle, tmp_path):
    source = source_training_data(bundle, "source")
    manifest = make_manifest(source)
    path = tmp_path / "subsets.json"
    save_source_subsets(path, manifest, source)
    loaded = load_source_subsets(path, source)
    assert loaded == manifest
    with pytest.raises(FileExistsError):
        save_source_subsets(path, manifest, source)
    subset = source_subset_view(source, loaded, "source_subset_0")
    before = source.graph.x.clone()
    result = fit_source(lambda: GCN(GCNConfig(8, 2, 8, 0.2)), subset,
                        TrainingConfig(max_epochs=3, patience=2), device="cpu")
    checkpoint = tmp_path / "subset.pt"
    save_checkpoint(checkpoint, result, {"architecture": "gcn"})
    _, payload = load_checkpoint(checkpoint, lambda spec: GCN(GCNConfig(8, 2, 8, 0.2)))
    assert torch.equal(payload["source"]["train_indices"], subset.train_indices)
    assert payload["source"]["source_subset"]["id"] == "source_subset_0"
    assert len(payload["source"]["source_subset"]["manifest_sha256"]) == 64
    assert torch.equal(before, source.graph.x)


def test_subset_selection_does_not_depend_on_held_out_labels(bundle):
    original = make_manifest(source_training_data(bundle, "source"))
    for split in ("val", "test"):
        bundle.graphs["source"].labels[bundle.select("source", split)] ^= 1
    bundle.graphs["target"].labels ^= 1
    assert make_manifest(source_training_data(bundle, "source")) == original


def test_local_rng_and_canonical_pool_order(bundle):
    source = source_training_data(bundle, "source")
    np.random.seed(50)
    torch.manual_seed(51)
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    manifest = make_manifest(source)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    shuffled = replace(source, train_indices=source.train_indices.flip(0), train_labels=source.train_labels.flip(0))
    assert make_manifest(shuffled) == manifest
    changed = create_source_subsets(source, label_fraction=0.5, seeds=[201, 202, 203, 204, 205])
    assert manifest["subsets"] != changed["subsets"]


def test_manifest_rejects_different_graph_pool_labels_or_validation(bundle, tmp_path):
    source = source_training_data(bundle, "source")
    path = tmp_path / "subsets.json"
    save_source_subsets(path, make_manifest(source), source)
    graph = source.graph.clone()
    graph.x[0, 0] += 0.1
    changed_views = [replace(source, graph=graph),
                     replace(source, train_labels=1 - source.train_labels),
                     replace(source, train_indices=source.train_indices[1:], train_labels=source.train_labels[1:]),
                     replace(source, val_indices=source.val_indices[1:], val_labels=source.val_labels[1:])]
    for changed in changed_views:
        with pytest.raises(ValueError, match="does not match"):
            load_source_subsets(path, changed)


@pytest.mark.parametrize("mutation", ["held_out_node", "duplicate_node", "duplicate_subset", "wrong_count",
                                     "wrong_class", "duplicate_seed", "wrong_id", "version"])
def test_invalid_stored_memberships_are_rejected(bundle, mutation):
    source = source_training_data(bundle, "source")
    manifest = make_manifest(source)
    item = manifest["subsets"][0]
    if mutation == "held_out_node":
        item["train_indices"][0] = int(source.val_indices[0])
        item["train_indices"].sort()
    elif mutation == "duplicate_node":
        item["train_indices"][0] = item["train_indices"][1]
    elif mutation == "duplicate_subset":
        manifest["subsets"][1]["train_indices"] = item["train_indices"].copy()
    elif mutation == "wrong_count":
        item["train_indices"].pop()
    elif mutation == "wrong_class":
        labels = dict(zip(source.train_indices.tolist(), source.train_labels.tolist()))
        remove = next(i for i in item["train_indices"] if labels[i] == 0)
        add = next(i for i, y in labels.items() if y == 1 and i not in item["train_indices"])
        item["train_indices"].remove(remove)
        item["train_indices"].append(add)
        item["train_indices"].sort()
    elif mutation == "duplicate_seed":
        manifest["subsets"][1]["seed"] = item["seed"]
    elif mutation == "wrong_id":
        item["id"] = "unknown"
    else:
        manifest["format_version"] = 2
    with pytest.raises(ValueError):
        source_subset_view(source, manifest, "source_subset_0")


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan"), True, 0.001])
def test_invalid_or_empty_label_budget(bundle, fraction):
    with pytest.raises(ValueError):
        create_source_subsets(source_training_data(bundle, "source"), label_fraction=fraction, seeds=[1])


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [2**32], [True], [1.2]])
def test_invalid_subset_seeds(bundle, seeds):
    with pytest.raises(ValueError):
        create_source_subsets(source_training_data(bundle, "source"), label_fraction=0.5, seeds=seeds)


def test_floor_budget_and_impossible_diversity_are_explicit(bundle):
    source = source_training_data(bundle, "source")
    manifest = create_source_subsets(source, label_fraction=0.51, seeds=[1])
    assert manifest["subset_class_counts"] == [15, 15]  # floor(30 * .51), not rounding up.
    assert source_subset_view(source, manifest, "source_subset_0").train_indices.numel() == 30
    full = create_source_subsets(source, label_fraction=1, seeds=[1])
    assert source_subset_view(source, full, "source_subset_0").train_indices.numel() == 60
    with pytest.raises(ValueError, match="distinct subsets"):
        create_source_subsets(source, label_fraction=1, seeds=[1, 2])
    with pytest.raises(ValueError, match="Unknown source subset"):
        source_subset_view(source, manifest, "does_not_exist")
