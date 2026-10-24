"""Source-only optimization, validation selection, and checkpoint integrity."""

from copy import deepcopy
from dataclasses import asdict, replace
import math

import pytest
import torch
from torch import nn

from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.evaluation import evaluate, predict, score_logits, select_device
from amortized_gnn_tta.models import GCN, GCNConfig
from amortized_gnn_tta.training import (
    TrainingConfig, fit_source, load_checkpoint, save_checkpoint, source_training_data,
)


@pytest.fixture
def bundle():
    # Strong signal makes a learnability test reliable; this is NOT a benchmark
    # setting or an attempt to improve the review run's target score.
    config = CSBMConfig(num_nodes=100, num_features=8, p_in=0.2, p_out=0.01,
                        negative_mean=-1.5, positive_mean=1.5, noise_scale=0.3)
    return load_dataset([
        CSBMEnvironment("source", "source_train", config),
        CSBMEnvironment("target", "target_test", replace(config, edge_seed=10, feature_seed=11)),
    ])


def factory():
    return GCN(GCNConfig(8, 2, hidden_channels=8, dropout=0.2))


def test_gcn_shapes_gradients_dropout_and_empty_edges(bundle):
    graph = bundle.unlabeled_input("source")
    model = factory()
    assert not model.conv1.cached and not model.conv2.cached
    logits = model(graph.x, graph.edge_index)
    assert logits.shape == (100, 2)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.train()
    first, second = predict(model, graph), predict(model, graph)
    assert model.training  # Inference restores the caller's mode.
    assert torch.equal(first, second)
    graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    assert predict(model, graph).shape == (100, 2)


def test_source_training_learns_and_restores_selected_epoch(bundle):
    view = source_training_data(bundle, "source")
    result = fit_source(factory, view, TrainingConfig(max_epochs=60, patience=15), device="cpu")
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    best = min(result.history, key=lambda row: row["val_loss"])
    assert result.best_epoch == best["epoch"]
    metric = evaluate(result.model, view.graph, view.val_indices, view.val_labels)
    assert metric.loss == pytest.approx(result.best_val_loss, abs=1e-7)
    assert metric.accuracy > 0.9


def test_source_test_and_target_labels_cannot_change_training(bundle):
    view = source_training_data(bundle, "source")
    original_graph = bundle.graphs["source"].graph.x.clone()
    config = TrainingConfig(max_epochs=12, patience=5)
    first = fit_source(factory, view, config, device="cpu")
    bundle.graphs["source"].labels[bundle.select("source", "test")] ^= 1
    bundle.graphs["target"].labels ^= 1

    # Catch any accidental traversal of a target record by the view extractor.
    original_get_labels = bundle.evaluation_labels
    def guarded_labels(graph_id, split):
        assert graph_id == "source" and split == "val"
        return original_get_labels(graph_id, split)
    bundle.evaluation_labels = guarded_labels
    second = fit_source(factory, source_training_data(bundle, "source"), config, device="cpu")
    assert first.history == second.history
    assert first.best_epoch == second.best_epoch
    assert all(torch.equal(v, second.model.state_dict()[k]) for k, v in first.model.state_dict().items())
    assert torch.equal(bundle.graphs["source"].graph.x, original_graph)
    assert not hasattr(view, "test_labels") and not hasattr(view, "target_labels")


def test_validation_loss_controls_selection_and_patience():
    # A scripted deterioration after step 1 forces a non-final checkpoint to be
    # restored, so this catches state_dict aliasing and last-epoch selection bugs.
    from amortized_gnn_tta.training import SourceTrainingData
    from torch_geometric.data import Data
    class WorseningModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))
        def forward(self, x, edge_index):
            return torch.stack((self.weight.expand(len(x)), torch.zeros(len(x))), dim=1)
    graph = Data(x=torch.ones(4, 1), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=4)
    view = SourceTrainingData(graph, torch.tensor([0, 1]), torch.tensor([0, 0]),
                              torch.tensor([2, 3]), torch.tensor([1, 1]), ("a", "b"), {})
    result = fit_source(WorseningModel, view, TrainingConfig(max_epochs=20, patience=2,
                        learning_rate=0.1, weight_decay=0), device="cpu")
    assert result.best_epoch == 1 and len(result.history) == 3
    assert result.history[-1]["val_loss"] > result.best_val_loss
    assert evaluate(result.model, graph, view.val_indices, view.val_labels).loss == pytest.approx(result.best_val_loss)


def test_checkpoint_cpu_roundtrip_and_no_overwrite(bundle, tmp_path):
    config = GCNConfig(8, 2, 8, 0.2)
    result = fit_source(lambda: GCN(config), source_training_data(bundle, "source"),
                        TrainingConfig(max_epochs=5, patience=3), device="cpu")
    path = tmp_path / "best.pt"
    save_checkpoint(path, result, {"architecture": "gcn", "config": asdict(config)})
    loaded, metadata = load_checkpoint(path, lambda spec: GCN(GCNConfig(**spec["config"])))
    graph = bundle.unlabeled_input("target")
    assert torch.equal(predict(result.model, graph), predict(loaded, graph))
    assert all(t.device.type == "cpu" for t in metadata["state_dict"].values())
    assert metadata["best_epoch"] == result.best_epoch
    assert metadata["training_config"]["model_seed"] == 31
    assert metadata["source"]["graph_metadata"]["fingerprint"] == bundle.graphs["source"].metadata["fingerprint"]
    with pytest.raises(FileExistsError):
        save_checkpoint(path, result, {})


def test_manual_metric_calculation_and_label_free_inference(bundle):
    logits = torch.tensor([[2.0, 1.0], [0.0, 0.0], [1.0, 2.0]])
    metric = score_logits(logits, torch.tensor([0, 2]), torch.tensor([0, 1]))
    assert metric.loss == pytest.approx(math.log1p(math.exp(-1)))
    assert metric.accuracy == 1 and metric.num_nodes == 2
    graph = bundle.unlabeled_input("target")
    model = factory()
    before = deepcopy(model.state_dict())
    evaluate(model, graph, bundle.select("target", "eval"), bundle.evaluation_labels("target", "eval"))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    graph.y = bundle.graphs["target"].labels
    with pytest.raises(ValueError, match="label-free"):
        predict(model, graph)


def test_training_view_rejects_target_and_invalid_supervision(bundle):
    with pytest.raises(ValueError, match="source_train"):
        source_training_data(bundle, "target")
    view = source_training_data(bundle, "source")
    view.val_indices[0] = view.train_indices[0]
    with pytest.raises(ValueError, match="overlap"):
        fit_source(factory, view, device="cpu")


@pytest.mark.parametrize("changes", [{"max_epochs": 0}, {"patience": 0}, {"learning_rate": 0},
                                    {"weight_decay": -1}, {"model_seed": -1}])
def test_invalid_training_configuration(changes):
    with pytest.raises(ValueError):
        TrainingConfig(**changes)


def test_invalid_model_and_device_configuration(monkeypatch):
    with pytest.raises(ValueError):
        GCNConfig(8, 2, dropout=1)
    assert select_device("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda")
