"""Distribution, supervision-boundary, and actual PyG compatibility checks."""

from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from amortized_gnn_tta.data import (
    CSBMConfig, CSBMEnvironment, DatasetBundle, generate_csbm, graph_diagnostics,
    load_dataset, validate_record,
)


def small(**kwargs):
    return CSBMConfig(num_nodes=100, num_features=8, p_in=0.2, p_out=0.05, **kwargs)


def test_graph_structure_and_stratified_splits():
    record = generate_csbm(small())
    validate_record(record, 2)
    assert record.graph.x.shape == (100, 8)
    assert record.graph.x.dtype == torch.float32
    assert record.labels.dtype == torch.long
    assert set(record.graph.keys()) == {"x", "edge_index", "num_nodes"}
    assert record.graph.is_undirected()
    assert not record.graph.has_self_loops()
    assert record.graph.edge_index.unique(dim=1).shape == record.graph.edge_index.shape
    for key, n in (("train", 60), ("val", 20), ("test", 20)):
        assert len(record.splits[key]) == n
        assert torch.bincount(record.labels[record.splits[key]]).tolist() == [n // 2, n // 2]
    assert torch.cat(list(record.splits.values())).sort().values.tolist() == list(range(100))


def test_empty_and_complete_graphs_keep_all_nodes():
    empty = generate_csbm(replace(small(), p_in=0, p_out=0))
    validate_record(empty, 2)
    assert empty.graph.edge_index.shape == (2, 0)
    assert empty.graph.num_nodes == 100
    report = graph_diagnostics(empty)
    assert report["isolated_nodes"] == 100
    assert report["edge_homophily"] is None
    full = generate_csbm(replace(small(), p_in=1, p_out=1))
    assert full.graph.edge_index.shape[1] == 100 * 99
    assert graph_diagnostics(full)["mean_out_degree"] == 99


def test_seed_repeatability_and_isolation_without_global_rng_changes():
    np.random.seed(123)
    torch.manual_seed(456)
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state().clone()
    config = small()
    a, b = generate_csbm(config), generate_csbm(config)
    assert torch.equal(a.graph.x, b.graph.x)
    assert torch.equal(a.graph.edge_index, b.graph.edge_index)
    assert all(torch.equal(a.splits[k], b.splits[k]) for k in a.splits)
    split_changed = generate_csbm(replace(config, split_seed=99))
    assert torch.equal(a.graph.x, split_changed.graph.x)
    assert torch.equal(a.graph.edge_index, split_changed.graph.edge_index)
    assert not torch.equal(a.splits["train"], split_changed.splits["train"])
    edges_changed = generate_csbm(replace(config, edge_seed=99))
    assert torch.equal(a.graph.x, edges_changed.graph.x)
    assert not torch.equal(a.graph.edge_index, edges_changed.graph.edge_index)
    features_changed = generate_csbm(replace(config, feature_seed=99))
    assert torch.equal(a.graph.edge_index, features_changed.graph.edge_index)
    assert not torch.equal(a.graph.x, features_changed.graph.x)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1])
    assert np_state[2:] == after[2:]


def test_matcha_conversion_and_exact_finite_degree():
    config = CSBMConfig.from_matcha(num_nodes=1000, num_features=16,
                                   degree_scale=10, homophily_parameter=0.8)
    assert config.p_in == pytest.approx(0.008)
    assert config.p_out == pytest.approx(0.002)
    assert config.expected_degree == pytest.approx(4.992)


def test_empirical_edge_probabilities_across_repeated_graphs():
    # Sum independent Bernoulli trials across 30 graphs; six standard deviations
    # tolerates random fluctuation while detecting the factor-of-two convention.
    config = small()
    within = between = 0
    repeats = 30
    for seed in range(repeats):
        record = generate_csbm(replace(config, edge_seed=seed))
        i, j = record.graph.edge_index
        unique = i < j
        same = record.labels[i[unique]] == record.labels[j[unique]]
        within += int(same.sum())
        between += int((~same).sum())
    half = config.num_nodes // 2
    for count, trials, probability in (
        (within, repeats * half * (half - 1), config.p_in),
        (between, repeats * half * half, config.p_out),
    ):
        assert abs(count - trials * probability) < 6 * math.sqrt(trials * probability * (1 - probability))


def test_gaussian_features_and_class_center_semantics():
    config = CSBMConfig(num_nodes=2000, num_features=32, p_in=0, p_out=0,
                        negative_mean=-2, positive_mean=2, noise_scale=0.5)
    record = generate_csbm(config)
    scaled = record.graph.x.numpy() * math.sqrt(config.num_features)
    for label, mean in ((0, -2), (1, 2)):
        residual = scaled[record.labels.numpy() == label] - mean
        assert abs(float(residual.mean())) < 0.015
        assert float(residual.var()) == pytest.approx(0.25, abs=0.01)
    noiseless = generate_csbm(replace(config, noise_scale=0))
    assert torch.allclose(noiseless.graph.x[0], torch.full((32,), 2 / math.sqrt(32)))


def make_bundle():
    return load_dataset([
        CSBMEnvironment("source", "source_train", small()),
        CSBMEnvironment("target", "target_test", small(edge_seed=100, feature_seed=101)),
    ])


def test_target_label_isolation_and_nonmutating_graph_access():
    bundle = make_bundle()
    assert set(bundle.graphs["target"].splits) == {"eval"}
    with pytest.raises(ValueError, match="evaluation-only"):
        bundle.training_labels("target")
    assert len(bundle.training_labels("source")) == 60
    assert len(bundle.evaluation_labels("target", "eval")) == 100
    before = bundle.unlabeled_input("target")
    bundle.graphs["target"].labels = 1 - bundle.graphs["target"].labels
    # Even aliases introduced by downstream code must not escape the allowlist.
    bundle.graphs["target"].graph.secret_labels = bundle.graphs["target"].labels
    after = bundle.unlabeled_input("target")
    assert set(after.keys()) == {"x", "edge_index", "num_nodes"}
    assert torch.equal(before.x, after.x)
    assert torch.equal(before.edge_index, after.edge_index)
    after.x.zero_()
    assert not torch.equal(after.x, bundle.graphs["target"].graph.x)


def test_environment_identity_schema_and_independence_validation():
    config = small()
    with pytest.raises(ValueError, match="unique"):
        load_dataset([CSBMEnvironment("a", "source_train", config)] * 2)
    with pytest.raises(ValueError, match="distinct"):
        load_dataset([CSBMEnvironment("a", "source_train", config),
                      CSBMEnvironment("b", "target_test", config)])
    with pytest.raises(ValueError, match="dimensions"):
        load_dataset([CSBMEnvironment("a", "source_train", config),
                      CSBMEnvironment("b", "target_test", replace(config, num_features=4, edge_seed=3, feature_seed=4))])
    record = generate_csbm(config)
    with pytest.raises(ValueError, match="Duplicate graph"):
        DatasetBundle({"a": record, "b": record}).validate()


def test_validation_and_meta_training_roles_have_distinct_supervision():
    bundle = load_dataset([
        CSBMEnvironment("meta", "meta_train", small()),
        CSBMEnvironment("validation", "validation", small(edge_seed=100, feature_seed=101)),
    ])
    assert len(bundle.training_labels("meta")) == 60
    assert bundle.select("validation", "eval").tolist() == list(range(100))
    with pytest.raises(ValueError, match="evaluation-only"):
        bundle.training_labels("validation")


def test_invalid_matcha_probabilities_are_rejected_not_clipped():
    with pytest.raises(ValueError):
        CSBMConfig.from_matcha(degree_scale=-1)
    with pytest.raises(ValueError):
        CSBMConfig.from_matcha(num_nodes=100, degree_scale=200, homophily_parameter=1)


@pytest.mark.parametrize("changes", [
    {"num_nodes": 7}, {"num_nodes": 0}, {"num_nodes": 6.0},
    {"num_features": 0}, {"p_in": 1.1}, {"p_out": -0.1}, {"p_in": float("nan")},
    {"edge_seed": -1}, {"noise_scale": -1}, {"positive_mean": float("inf")},
    {"split_ratios": (0.8, 0.2, 0.2)}, {"split_ratios": (1.0, 0.0, 0.0)},
    {"num_nodes": 6, "split_ratios": (0.98, 0.01, 0.01)},
])
def test_invalid_generation_configuration(changes):
    with pytest.raises(ValueError):
        replace(small(), **changes)


@pytest.mark.parametrize("damage", ["edge_bounds", "split_overlap", "label_range", "nan", "alias"])
def test_record_validation_rejects_invalid_data(damage):
    record = generate_csbm(small())
    if damage == "edge_bounds":
        record.graph.edge_index[0, 0] = 100
    elif damage == "split_overlap":
        record.splits["val"][0] = record.splits["train"][0]
    elif damage == "label_range":
        record.labels[0] = 2
    elif damage == "nan":
        record.graph.x[0, 0] = float("nan")
    else:
        record.graph.y = record.labels
    with pytest.raises(ValueError):
        validate_record(record, 2)


@pytest.mark.parametrize("layer", [GCNConv, GATConv, SAGEConv])
def test_real_pyg_layers_forward_backward_without_target_labels(layer):
    bundle = make_bundle()
    graph = bundle.unlabeled_input("source")
    model = layer(8, 2)
    logits = model(graph.x, graph.edge_index)
    assert logits.shape == (100, 2)
    loss = torch.nn.functional.cross_entropy(logits[bundle.select("source", "train")],
                                             bundle.training_labels("source"))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    target = bundle.unlabeled_input("target")
    with torch.no_grad():
        assert model(target.x, target.edge_index).shape == (100, 2)
