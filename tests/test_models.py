"""Cross-architecture semantics, real learning, gradients, and checkpoint reuse."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from amortized_gnn_tta.bank_artifacts import load_verified_bank
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.evaluation import evaluate, predict
from amortized_gnn_tta.models import (
    ARCHITECTURES, GAT, GATConfig, GCN, GCNConfig, GraphSAGE, GraphSAGEConfig, build_model, make_model_spec,
)
from amortized_gnn_tta.training import (
    TrainingConfig, fit_source, load_checkpoint, save_checkpoint, source_training_data,
)
from amortized_gnn_tta.source_subsets import create_source_subsets
from amortized_gnn_tta.weight_bank import train_weight_bank


@pytest.fixture
def bundle():
    # Deliberately strong signal for a correctness/learnability fixture, not our
    # research CSBM settings and not an attempt to optimize target accuracy.
    config = CSBMConfig.from_matcha(num_nodes=100, num_features=8, degree_scale=21,
                                   homophily_parameter=20/21, negative_mean=-1.5,
                                   positive_mean=1.5, noise_scale=0.3)
    return load_dataset([CSBMEnvironment("source", "source_train", config),
                        CSBMEnvironment("target", "target_test", replace(config, edge_seed=10, feature_seed=11))])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_shapes_gradients_empty_edges_and_eval_determinism(bundle, architecture):
    graph = bundle.unlabeled_input("source")
    spec = make_model_spec(architecture, 8, 2, hidden_channels=8, dropout=0.2)
    model = build_model(spec)
    assert model.encode(graph.x, graph.edge_index).shape == (100, 8)
    logits = model(graph.x, graph.edge_index)
    assert logits.shape == (100, 2)
    train_indices = bundle.select("source", "train")
    F.cross_entropy(logits[train_indices], bundle.training_labels("source")).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())
    model.train()
    assert torch.equal(predict(model, graph), predict(model, graph)) and model.training
    graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    assert torch.isfinite(predict(model, graph)).all()  # Includes all isolated nodes.


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_permutation_equivariance(bundle, architecture):
    graph = bundle.unlabeled_input("source")
    model = build_model(make_model_spec(architecture, 8, 2, hidden_channels=8)).eval()
    order = torch.randperm(100, generator=torch.Generator().manual_seed(91))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(100)
    changed = graph.clone()
    changed.x = graph.x[order]
    changed.edge_index = inverse[graph.edge_index]
    torch.testing.assert_close(predict(model, changed), predict(model, graph)[order], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("architecture", ("gat", "graphsage"))
def test_source_learning_and_cpu_checkpoint_roundtrip(bundle, tmp_path, architecture):
    spec = make_model_spec(architecture, 8, 2, hidden_channels=8, dropout=0.2)
    source = source_training_data(bundle, "source")
    result = fit_source(lambda: build_model(spec), source, TrainingConfig(max_epochs=60, patience=15), device="cpu")
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert evaluate(result.model, source.graph, source.val_indices, source.val_labels).accuracy > 0.9
    path = tmp_path / "best.pt"
    save_checkpoint(path, result, spec)
    restored, metadata = load_checkpoint(path, build_model)
    assert metadata["model_spec"] == spec
    assert torch.equal(predict(result.model, bundle.unlabeled_input("target")),
                        predict(restored, bundle.unlabeled_input("target")))


@pytest.mark.parametrize("architecture", ("gat", "graphsage"))
def test_held_out_labels_cannot_change_training(bundle, architecture):
    spec = make_model_spec(architecture, 8, 2, hidden_channels=8, dropout=0.2)
    config = TrainingConfig(max_epochs=5, patience=3)
    first = fit_source(lambda: build_model(spec), source_training_data(bundle, "source"), config, device="cpu")
    bundle.graphs["target"].labels ^= 1
    bundle.graphs["source"].labels[bundle.select("source", "test")] ^= 1
    second = fit_source(lambda: build_model(spec), source_training_data(bundle, "source"), config, device="cpu")
    assert first.history == second.history and first.best_epoch == second.best_epoch
    assert all(torch.equal(value, second.model.state_dict()[name]) for name, value in first.model.state_dict().items())


def test_gat_head_width_uniform_attention_and_self_loops():
    model = GAT(GATConfig(2, 2, hidden_channels=2, dropout=0, heads=1)).eval()
    with torch.no_grad():
        model.conv1.lin.weight.copy_(torch.eye(2))
        model.conv1.bias.zero_()
        model.conv1.att_src.zero_()
        model.conv1.att_dst.zero_()
    x = torch.tensor([[1., 0.], [0., 1.], [2., 2.], [5., -1.]])
    edges = torch.tensor([[0, 1], [2, 2]])
    output, (stored, alpha) = model.conv1(x, edges, return_attention_weights=True)
    torch.testing.assert_close(output, torch.tensor([[1., 0.], [0., 1.], [1., 1.], [5., -1.]]))
    sums = torch.zeros(4, 1).index_add_(0, stored[1], alpha)
    torch.testing.assert_close(sums, torch.ones(4, 1))
    assert stored.shape[1] == 6  # Two directed edges plus four layer-owned self-loops.
    torch.testing.assert_close(model.encode(x, edges), F.elu(output))
    default = GAT(GATConfig(8, 2))
    assert default.conv1.heads == 4 and default.conv1.out_channels == 8
    assert default.conv2.heads == 1 and not default.conv2.concat
    assert default.conv1.dropout == 0 and default.config.dropout == 0.5


def test_graphsage_mean_and_root_are_separate_without_hidden_normalization():
    model = GraphSAGE(GraphSAGEConfig(2, 2, hidden_channels=2, dropout=0))
    with torch.no_grad():
        model.conv1.lin_l.weight.copy_(torch.eye(2))
        model.conv1.lin_l.bias.zero_()
        model.conv1.lin_r.weight.copy_(2 * torch.eye(2))
    x = torch.tensor([[1., 0.], [0., 1.], [2., 2.], [5., -1.]])
    edges = torch.tensor([[0, 1], [2, 2]])
    expected = torch.tensor([[2., 0.], [0., 2.], [4.5, 4.5], [10., -2.]])
    torch.testing.assert_close(model.conv1(x, edges), expected)
    assert model.conv1.root_weight and not model.conv1.normalize and not model.conv1.project


def test_nonzero_gat_attention_dropout_is_configured_and_disabled_in_eval(bundle):
    graph = bundle.unlabeled_input("source")
    spec = make_model_spec("gat", 8, 2, hidden_channels=8, dropout=0, heads=2, attention_dropout=0.5)
    model = build_model(spec)
    assert model.conv1.dropout == model.conv2.dropout == 0.5
    model.train()
    first, second = predict(model, graph), predict(model, graph)
    assert torch.equal(first, second) and model.training
    torch.manual_seed(71)
    train_first = model(graph.x, graph.edge_index)
    train_second = model(graph.x, graph.edge_index)
    assert not torch.equal(train_first, train_second)


@pytest.mark.parametrize("changes", [{"heads": 0}, {"heads": True}, {"heads": 3},
                                    {"attention_dropout": 1}, {"attention_dropout": float("nan")}])
def test_invalid_gat_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        GATConfig(8, 2, **changes)


def test_model_spec_rejects_unknown_or_inapplicable_settings_and_keeps_gcn_compatible():
    old = {"architecture": "two_layer_gcn", "config": {"in_channels": 8, "out_channels": 2,
            "hidden_channels": 32, "dropout": 0.5}}
    assert make_model_spec("gcn", 8, 2) == old
    assert isinstance(build_model(old), GCN)
    from amortized_gnn_tta.models.gcn import GCNConfig as LegacyImport
    assert LegacyImport is GCNConfig
    for bad in ({"architecture": "not-a-model", "config": {}}, {},
                {"architecture": "two_layer_gat", "config": {"unknown": True}}):
        with pytest.raises(ValueError):
            build_model(bad)
    with pytest.raises(ValueError, match="GAT-only"):
        make_model_spec("graphsage", 8, 2, heads=4)
    with pytest.raises(ValueError):
        make_model_spec("not-a-model", 8, 2)


@pytest.mark.parametrize("architecture", ("gat", "graphsage"))
def test_bank_and_audit_cli_reuse_same_protocol(bundle, tmp_path, architecture):
    # A small bank validates the whole path without a research-sized experiment.
    source = source_training_data(bundle, "source")
    subsets = create_source_subsets(source, label_fraction=0.5, seeds=[101, 102, 103, 104, 105])
    spec = make_model_spec(architecture, 8, 2, hidden_channels=8, dropout=0.2)
    bankdir = tmp_path / "bank"
    training = TrainingConfig(max_epochs=4, patience=2)
    train_weight_bank(lambda: build_model(spec), source, subsets, training, model_spec=spec,
                      model_seeds=[31, 32], output_dir=bankdir, device="cpu")
    bank = load_verified_bank(bankdir, source, build_model)
    assert bank.manifest["completed_models"] == 10
    # Reuse the exact fixture parameters on both sides, without inverting rounded
    # probabilities and hoping that nearby floating-point values sample alike.
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs" / f"phase1_csbm_{architecture}.json").read_text())
    protocol["source"].update(num_nodes=100, num_features=8, degree_scale=21,
                              homophily_parameter=20/21, negative_mean=-1.5, positive_mean=1.5,
                              noise_scale=0.3, edge_seed=0, feature_seed=1, split_seed=2)
    protocol["model"] = {key: value for key, value in spec["config"].items() if key not in ("in_channels", "out_channels")}
    protocol["training"] = {"max_epochs": 4, "patience": 2, "learning_rate": 0.01, "weight_decay": 0.0005}
    protocol["model_seeds"] = [31, 32]
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(protocol))
    output = tmp_path / "review"
    process = subprocess.run([sys.executable, str(root / "examples" / "finalize_phase1.py"),
                               "--config", str(config_path), "--bank", str(bankdir), "--output-dir", str(output)],
                              capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((output / "phase1_report.json").read_text())
    assert report["architecture"] == architecture and report["models"] == 10
    assert report["partition_counts"] == {"meta_train": 6, "validation": 2, "test": 2}
    assert report["num_parameters"] == sum(parameter.numel() for parameter in build_model(spec).parameters())
