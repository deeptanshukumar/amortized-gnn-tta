"""Regression checks for executed audit failures, not only happy-path shapes."""

from dataclasses import replace
import pickle

import pytest
import torch
from torch_geometric.data import Data

from amortized_gnn_tta.adaptation import (
    DiffusionHeadGenerator, GraphShiftDescriptor, HeadAdapterGenerator, NoiseSchedule,
    PEFTConfig, adapted_forward, apply_target_delta, extract_target_parameters,
    train_adaptation_generator, train_diffusion_generator, fit_oracle_deltas,
)
from amortized_gnn_tta.adaptation.checkpoints import save_generator_checkpoint, load_generator_checkpoint
from amortized_gnn_tta.adaptation.descriptors import edge_statistics, extract_target_graph_descriptor, extract_compact_weight_descriptor
from amortized_gnn_tta.adaptation.evaluator import AdaptationRecord, evaluate_adaptation_pair
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.models import build_model, make_model_spec


@pytest.fixture
def fixture():
    config = CSBMConfig(num_nodes=50, num_features=8, p_in=0.2, p_out=0.03)
    bundle = load_dataset([CSBMEnvironment("train", "source_train", config),
                           CSBMEnvironment("validation", "validation", replace(config, edge_seed=59, feature_seed=60, split_seed=61))])
    manifest = {"environments": [{"id": "train", "role": "meta_train"}, {"id": "validation", "role": "validation"}]}
    return bundle, manifest


def model_for(architecture):
    return build_model(make_model_spec(architecture, in_channels=8, hidden_channels=8, out_channels=2))


@pytest.mark.parametrize("architecture", ["gcn", "gat", "graphsage"])
@pytest.mark.parametrize("scope", ["head", "backbone", "all"])
def test_schema_application_parity_and_source_integrity(architecture, scope, fixture):
    model = model_for(architecture).eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    graph = fixture[0].unlabeled_input("validation")
    flat, schema = extract_target_parameters(model, scope)
    delta = (torch.randn_like(flat) * 0.001).requires_grad_()
    logits = adapted_forward(model, graph, delta, schema)
    materialized = apply_target_delta(model, delta.detach(), schema)
    torch.testing.assert_close(logits, materialized(graph.x, graph.edge_index))
    logits.square().mean().backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
    for bad_delta, bad_schema in [(torch.cat([delta.detach(), torch.zeros(1)]), schema),
                                  (delta[:-1], schema), (delta, [("unknown", tuple(delta.shape))]),
                                  (torch.full_like(delta, float("nan")), schema),
                                  (delta, [schema[0], schema[0]])]:
        with pytest.raises(ValueError):
            adapted_forward(model, graph, bad_delta, bad_schema)
        with pytest.raises(ValueError):
            apply_target_delta(model, bad_delta, bad_schema)


@pytest.mark.parametrize("edges", [torch.tensor([[0, 1, 3], [1, 2, 0]]), torch.empty(2, 0, dtype=torch.long)])
def test_chunked_statistics_match_direct_arithmetic(edges):
    x = torch.tensor([[1., 2.], [3., 7.], [4., -2.], [8., 9.]])
    for chunk in (1, 2, 100):
        energy, mean, std = edge_statistics(x, edges, chunk)
        if edges.shape[1]:
            difference = x[edges[0]] - x[edges[1]]
            torch.testing.assert_close(energy, difference.square().mean(0))
            torch.testing.assert_close(mean[0], difference.norm(dim=1).mean())
            torch.testing.assert_close(std[0], difference.norm(dim=1).std(correction=0))
        else:
            assert energy.eq(0).all() and mean.eq(0).all() and std.eq(0).all()


def test_directed_isolation_and_dimension_rejection():
    graph = Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0], [1]]), num_nodes=3)
    assert extract_target_graph_descriptor(graph, 2)[2*2+10] == pytest.approx(1/3)
    assert extract_target_graph_descriptor(graph, 2, version="v1")[2*2+10] == pytest.approx(2/3)
    with pytest.raises(ValueError, match="dimension"):
        extract_target_graph_descriptor(graph, 1)


def test_weight_scope_and_version_are_explicit(fixture):
    model = model_for("gcn").eval()
    old = extract_compact_weight_descriptor(model, "backbone")
    with torch.no_grad():
        for parameter in model.conv2.parameters():
            parameter.mul_(1000)
    torch.testing.assert_close(old, extract_compact_weight_descriptor(model, "backbone"), rtol=0, atol=0)
    graph = fixture[0].unlabeled_input("validation")
    a = GraphShiftDescriptor(in_channels=8, hidden_channels=8, conditioning="D", target_scope="head")
    b = GraphShiftDescriptor(in_channels=8, hidden_channels=8, conditioning="D", target_scope="all")
    torch.testing.assert_close(a(model, graph), b(model, graph), rtol=0, atol=0)
    a.prepare_model(model)
    before = a(model, graph)
    with torch.no_grad():
        next(model.conv1.parameters()).add_(1)
    assert not torch.equal(before, a(model, graph))
    assert a._weight_cache is None


@pytest.mark.parametrize("offset,scale", [(0., 1.), (1e4, .001), (1e7, 2.), (1., 1e-5), (10., 0.)])
def test_bounded_moments_agree_with_stable_reference(offset, scale):
    from amortized_gnn_tta.adaptation.descriptors import _moments_centers
    generator = torch.Generator().manual_seed(1204)
    x = offset + scale * torch.randn(5000, 16, generator=generator)
    for chunk in (127, 1024, 10000):
        mean, std, center_mean, center_std = _moments_centers(x, chunk)
        torch.testing.assert_close(mean, x.mean(0))
        torch.testing.assert_close(std, x.double().std(0, correction=0).float(), rtol=1e-5, atol=1e-7)
        distance = (x-x.mean(0)).norm(dim=1)
        torch.testing.assert_close(center_mean[0], distance.mean(), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(center_std[0], distance.std(correction=0), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("kind", ["hypernetwork", "diffusion"])
def test_nondefault_checkpoint_restores_exact_inference(kind, fixture, tmp_path):
    model = model_for("gcn").eval()
    descriptor = GraphShiftDescriptor(in_channels=8, hidden_channels=8, conditioning="D", chunk_size=7)
    flat, schema = extract_target_parameters(model)
    kwargs = dict(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=12, num_layers=3, include_head_weights=False)
    generator = (HeadAdapterGenerator(**kwargs) if kind == "hypernetwork" else
                 DiffusionHeadGenerator(**kwargs, diffusion_steps=11, sampling_steps=3, time_dim=7, default_seed=821))
    with torch.no_grad():
        generator.net[-1].weight.normal_(std=0.001)
    path = tmp_path / "best.pt"
    save_generator_checkpoint(path, generator, descriptor, schema, provenance={"seed": 42})
    with pytest.raises(FileExistsError):
        save_generator_checkpoint(path, generator, descriptor, schema, provenance={})
    state = torch.random.get_rng_state().clone()
    restored, desc, payload = load_generator_checkpoint(path, model=model)
    assert torch.equal(state, torch.random.get_rng_state())
    graph = fixture[0].unlabeled_input("validation")
    torch.testing.assert_close(restored(desc(model, graph), flat), generator(descriptor(model, graph), flat), rtol=0, atol=0)
    assert payload["descriptor_config"]["descriptor_version"] == "v2"
    payload["state_dict"][next(iter(payload["state_dict"]))].view(-1)[0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="hash"):
        load_generator_checkpoint(path)


def test_nan_generator_fails_before_labels_are_scored(fixture, monkeypatch):
    bundle, _ = fixture
    model = model_for("gcn")
    descriptor = GraphShiftDescriptor(hidden_channels=8)
    flat, _ = extract_target_parameters(model)
    class Invalid(torch.nn.Module):
        def forward(self, *args):
            return torch.full_like(flat, float("nan"))
    monkeypatch.setattr(bundle, "evaluation_labels", lambda *a: pytest.fail("Labels read before validation"))
    with pytest.raises(ValueError, match="finite"):
        evaluate_adaptation_pair(model, "m", bundle, "validation", {}, descriptor, Invalid())


@pytest.mark.parametrize("kind", ["hypernetwork", "diffusion"])
@pytest.mark.parametrize("quality", [0.4, 0.5, 0.6])
def test_validation_selection_forced_paths(kind, quality, fixture, monkeypatch):
    import amortized_gnn_tta.adaptation.meta_trainer as hyper
    import amortized_gnn_tta.adaptation.diffusion as diffusion
    model = model_for("gcn").train()
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    gradients = [p.grad.clone() for p in model.parameters()]
    descriptor = GraphShiftDescriptor(hidden_channels=8)
    flat, _ = extract_target_parameters(model)
    kwargs = dict(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=8)
    generator = (HeadAdapterGenerator(**kwargs) if kind == "hypernetwork" else
                 DiffusionHeadGenerator(**kwargs, diffusion_steps=5, sampling_steps=2))
    initial = {k: v.clone() for k, v in generator.state_dict().items()}
    calls = []
    def evaluate(*args, **kwargs):
        accuracy = 0.5 if not calls else quality
        calls.append(accuracy)
        return AdaptationRecord("m", "validation", "validation", "control", 0., True, .5, accuracy, accuracy-.5, 0.)
    module = hyper if kind == "hypernetwork" else diffusion
    monkeypatch.setattr(module, "evaluate_adaptation_pair", evaluate)
    bundle, manifest = fixture
    train = train_adaptation_generator if kind == "hypernetwork" else train_diffusion_generator
    extra = {} if kind == "hypernetwork" else {"oracle_steps": 1}
    result = train({"m": model}, manifest, lambda *a: bundle, descriptor, generator,
                   epochs=1, val_models={"v": model}, verbose=False, **extra)
    assert calls == [.5, quality]
    assert model.training and [p.requires_grad for p in model.parameters()] == flags
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), gradients))
    if quality <= .5:
        assert torch.equal(result.best_state_dict["net.2.weight"], initial["net.2.weight"])
    else:
        assert not torch.equal(result.best_state_dict["net.2.weight"], initial["net.2.weight"])


def test_stable_config_pickle_and_short_schedule_rejection():
    assert pickle.loads(pickle.dumps(PEFTConfig())) == PEFTConfig()
    with pytest.raises(ValueError, match="Short linear"):
        NoiseSchedule(100, "linear")


def test_cost_only_amortization_and_raw_weight_bypass_rejected(fixture):
    from amortized_gnn_tta.adaptation.benchmark import amortization_break_even
    assert amortization_break_even(offline_generated=100., offline_iterative=0., online_generated=.01, online_iterative=.02) == 10000
    assert amortization_break_even(offline_generated=100., offline_iterative=0., online_generated=.03, online_iterative=.02) is None
    model = model_for("gcn")
    descriptor = GraphShiftDescriptor(in_channels=8, hidden_channels=8, conditioning="A")
    weights, _ = extract_target_parameters(model)
    generator = HeadAdapterGenerator(descriptor.output_dim, len(weights))
    with pytest.raises(ValueError, match="raw-weight"):
        train_adaptation_generator({"m": model}, fixture[1], lambda *a: fixture[0], descriptor, generator, epochs=1)


@pytest.mark.parametrize("architecture", ["gcn", "gat", "graphsage"])
def test_benchmark_iterative_path_matches_existing_adapter(architecture, fixture):
    from amortized_gnn_tta.adaptation import fit_unlabeled, AdaptationConfig
    from amortized_gnn_tta.adaptation.benchmark import iterative_predictions, generated_predictions, measure_methods
    model = model_for(architecture).eval()
    graph = fixture[0].unlabeled_input("validation")
    expected = fit_unlabeled(model, graph, PEFTConfig("head"), AdaptationConfig(steps=3, learning_rate=.01), objective="information_maximization")
    actual = iterative_predictions(model, graph, steps=3, profile=True)
    torch.testing.assert_close(actual.logits, expected.model(graph.x, graph.edge_index), rtol=1e-5, atol=1e-6)
    assert actual.stages_seconds["adaptation_ready"] > 0
    descriptor = GraphShiftDescriptor(in_channels=8, hidden_channels=8, conditioning="A")
    flat, _ = extract_target_parameters(model)
    generator = HeadAdapterGenerator(descriptor.output_dim, len(flat), include_head_weights=False)
    result = generated_predictions(model, graph, descriptor, generator, profile=True)
    torch.testing.assert_close(result.logits, model(graph.x, graph.edge_index))
    calls = []
    def event():
        calls.append(True)
        return result
    measured = measure_methods({"event": event}, warmups=1, repeats=3)
    assert len(calls) == 5
    assert len(measured["event"]["samples_seconds"]) == 3


@pytest.mark.parametrize("architecture", ["gcn", "gat", "graphsage"])
def test_cached_teacher_matches_uncached_optimization(architecture, fixture):
    from torch.nn import functional as F
    from amortized_gnn_tta.adaptation._validation import source_state
    bundle, manifest = fixture
    model = model_for(architecture)
    graph = bundle.unlabeled_input("train")
    indices, labels = bundle.select("train", "train"), bundle.training_labels("train")
    weights, schema = extract_target_parameters(model)
    delta = torch.nn.Parameter(torch.zeros_like(weights))
    optimizer = torch.optim.Adam([delta], lr=.01)
    with source_state(model, freeze=True):
        for _ in range(3):
            optimizer.zero_grad()
            logits = adapted_forward(model, graph, delta, schema)
            loss = F.cross_entropy(logits[indices], labels) + .001 * delta.square().sum()
            loss.backward()
            optimizer.step()
    descriptor = GraphShiftDescriptor(hidden_channels=8)
    targets = fit_oracle_deltas({"m": model}, manifest, lambda *a: bundle, descriptor,
                               steps=3, lr=.01, delta_l2_reg=.001, verbose=False)
    torch.testing.assert_close(targets[0].target_delta, delta.detach(), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("architecture", ["gcn", "gat", "graphsage"])
@pytest.mark.parametrize("conditioning", ["B", "C", "D"])
@pytest.mark.parametrize("scope", ["head", "all"])
def test_generated_prediction_reuses_only_valid_frozen_features(architecture, conditioning, scope, fixture, monkeypatch):
    from amortized_gnn_tta.adaptation.benchmark import generated_predictions
    from amortized_gnn_tta.adaptation._validation import source_state
    model = model_for(architecture).train()
    graph = fixture[0].unlabeled_input("validation")
    original_x, original_edges = graph.x.clone(), graph.edge_index.clone()
    descriptor = GraphShiftDescriptor(in_channels=8, hidden_channels=8, conditioning=conditioning)
    flat, schema = extract_target_parameters(model, scope)
    generator = HeadAdapterGenerator(descriptor.output_dim, len(flat), include_head_weights=False)
    with torch.no_grad():
        generator.net[-1].weight.normal_(std=.001)
    calls = []
    original_encode = model.encode
    def encode(*args):
        calls.append(True)
        return original_encode(*args)
    monkeypatch.setattr(model, "encode", encode)
    result = generated_predictions(model, graph, descriptor, generator, scope=scope)
    assert len(calls) == (1 if scope == "head" else 2)
    with source_state(model), torch.no_grad():
        expected = adapted_forward(model, graph, result.delta, schema)
    torch.testing.assert_close(result.logits, expected, rtol=1e-5, atol=1e-6)
    assert model.training
    assert torch.equal(graph.x, original_x) and torch.equal(graph.edge_index, original_edges)


@pytest.mark.parametrize("conditioning", [None, "A"])
def test_complete_checkpoint_ignores_unused_descriptor_dimensions(conditioning, fixture, tmp_path):
    model = model_for("gcn")
    # Legacy moments depend on H, not the unused raw-feature default F=64.
    descriptor = GraphShiftDescriptor(hidden_channels=8 if conditioning is None else 32,
        in_channels=64 if conditioning is None else 8, conditioning=conditioning, descriptor_version="v1")
    flat, schema = extract_target_parameters(model)
    generator = HeadAdapterGenerator(descriptor.output_dim, len(flat), include_head_weights=conditioning is None)
    path = tmp_path / "legacy-definition.pt"
    save_generator_checkpoint(path, generator, descriptor, schema, provenance={})
    restored, desc, _ = load_generator_checkpoint(path, model=model)
    graph = fixture[0].unlabeled_input("validation")
    torch.testing.assert_close(restored(desc(model, graph), flat), generator(descriptor(model, graph), flat))


@pytest.mark.parametrize("architecture", ["gcn", "gat", "graphsage"])
@pytest.mark.parametrize("conditioning", list("ABCD"))
@pytest.mark.parametrize("kind", ["hypernetwork", "diffusion"])
@pytest.mark.parametrize("scope", ["head", "backbone", "all"])
def test_training_matrix_preserves_source_and_label_boundary(architecture, conditioning, kind, scope, fixture, monkeypatch):
    from amortized_gnn_tta.adaptation_artifacts import state_sha256
    bundle, manifest = fixture
    model = model_for(architecture).train()
    original_hash = state_sha256(model.state_dict())
    descriptor = GraphShiftDescriptor(in_channels=8, hidden_channels=8, conditioning=conditioning, target_scope=scope)
    flat, _ = extract_target_parameters(model, scope)
    kwargs = dict(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=8, include_head_weights=False)
    generator = (HeadAdapterGenerator(**kwargs) if kind == "hypernetwork" else
                 DiffusionHeadGenerator(**kwargs, diffusion_steps=5, sampling_steps=2))
    original_training_labels = bundle.training_labels
    seen = []
    def labels(env_id):
        assert env_id == "train"
        seen.append(env_id)
        return original_training_labels(env_id)
    monkeypatch.setattr(bundle, "training_labels", labels)
    train = train_adaptation_generator if kind == "hypernetwork" else train_diffusion_generator
    result = train({"m": model}, manifest, lambda *a: bundle, descriptor, generator,
                   epochs=1, val_models={"v": model}, verbose=False, target_scope=scope,
                   **({} if kind == "hypernetwork" else {"oracle_steps": 1}))
    assert seen == ["train"] and model.training
    assert all(p.requires_grad and p.grad is None for p in model.parameters())
    assert state_sha256(model.state_dict()) == original_hash
    assert all(torch.isfinite(value).all() for value in result.best_state_dict.values())
