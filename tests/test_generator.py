<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 511141a (rename file)
"""Update algebra, gradients, leakage boundaries, learning and delta restoration."""

from copy import deepcopy
from dataclasses import replace
import inspect
import math

import pytest
import torch

from amortized_gnn_tta.adaptation import (
    AdaptationConfig, PEFTConfig, WeightAdapter, adaptation_targets, development_supervision,
    fit_supervised_development, fit_unlabeled, unlabeled_objective,
)
from amortized_gnn_tta.adaptation_artifacts import load_weight_delta, save_weight_delta, state_sha256
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.evaluation import predict
from amortized_gnn_tta.models import ARCHITECTURES, build_model, make_model_spec


@pytest.fixture
def bundle():
    config = CSBMConfig(num_nodes=100, num_features=8, p_in=0.12, p_out=0.02,
                        negative_mean=-1.5, positive_mean=1.5, noise_scale=0.3)
    return load_dataset([CSBMEnvironment("meta", "meta_train", config),
                        CSBMEnvironment("target", "target_test", replace(config, edge_seed=10, feature_seed=11)),
                        CSBMEnvironment("validation", "validation", replace(config, edge_seed=20, feature_seed=21))])


def model(architecture="gcn"):
    torch.manual_seed(31)
    return build_model(make_model_spec(architecture, 8, 2, hidden_channels=8, dropout=0.5))


def state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("method", ("head", "lora", "full"))
def test_zero_delta_gradients_frozen_model_and_merge(bundle, architecture, method):
    original = model(architecture).train()
    original_state = state(original)
    graph = bundle.unlabeled_input("meta")
    rng_before = torch.random.get_rng_state().clone()
    adapter = WeightAdapter(original, PEFTConfig(method, backbone_rank=2))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    adapter.train()
    assert not adapter.base.training and original.training
    assert torch.equal(predict(adapter, graph), predict(original, graph))
    train_idx = bundle.select("meta", "train")
    train_y = bundle.training_labels("meta")
    optimizer = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=0.01)
    loss = torch.nn.functional.cross_entropy(adapter(graph.x, graph.edge_index)[train_idx], train_y)
    loss.backward()
    assert all(p.grad is None for p in adapter.base.parameters())
    if method == "lora":
        assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in adapter.a.values())
        assert any(p.grad.abs().sum() > 0 for p in adapter.b.values())
    elif method == "head":
        assert any(p.grad.abs().sum() > 0 for p in adapter.offsets.values())
    else:
        assert all(p.requires_grad and p.grad is not None for p in adapter.full_model.parameters())
        assert not adapter.full_model.training
    optimizer.step()
    assert any(torch.count_nonzero(change) for change in adapter.updates().values())
    assert all(torch.equal(value, state(original)[key]) for key, value in original_state.items())
    assert all(torch.equal(value, state(adapter.base)[key]) for key, value in original_state.items())
    torch.testing.assert_close(predict(adapter, graph), predict(adapter.merged_model(), graph), rtol=1e-5, atol=1e-6)
    if method == "lora":
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(adapter(graph.x, graph.edge_index)[train_idx], train_y).backward()
        assert any(p.grad.abs().sum() > 0 for p in adapter.a.values())


def test_lora_algebra_scaling_rank_and_sites():
    original = model()
    config = PEFTConfig("lora", backbone_rank=2, backbone_alpha=6, head_alpha=2)
    adapter = WeightAdapter(original, config)
    with torch.no_grad():
        for p in adapter.b.values():
            p.normal_()
    for i, (name, change) in enumerate(adapter.updates().items()):
        scale = 3 if name.startswith("conv1") else 2
        torch.testing.assert_close(change, (adapter.b[f"p{i}"] @ adapter.a[f"p{i}"]) * scale)
        assert torch.linalg.matrix_rank(change).item() <= (2 if name.startswith("conv1") else 1)
    assert adaptation_targets(original, "head") == ("conv2.bias", "conv2.lin.weight")
    sage = WeightAdapter(model("graphsage"), config)
    assert len(sage.target_names) == 4
    gat_head = adaptation_targets(model("gat"), "head")
    assert "conv2.att_src" in gat_head and "conv2.att_dst" in gat_head and "conv2.bias" in gat_head
    assert not any("att" in name or "bias" in name for name in adaptation_targets(model("gat"), "lora"))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("method", ("head", "lora", "full"))
def test_supervised_learning_selection_and_heldout_label_invariance(bundle, architecture, method):
    original = model(architecture)
    graph = bundle.unlabeled_input("meta")
    view = development_supervision(bundle, "meta")
    config = AdaptationConfig(steps=25, patience=10)
    peft = PEFTConfig(method, backbone_rank=2)
    result = fit_supervised_development(original, graph, view, peft, config)
    losses = [r["validation_loss"] for r in result.history]
    assert result.selected_step == result.history[losses.index(min(losses))]["step"]
    assert min(losses) <= losses[0]
    assert min(losses) < losses[0]  # Deliberately learnable correctness fixture.
    bundle.graphs["meta"].labels[bundle.select("meta", "test")] ^= 1
    bundle.graphs["target"].labels ^= 1
    again = fit_supervised_development(original, graph, development_supervision(bundle, "meta"), peft, config)
    assert result.history == again.history and result.selected_step == again.selected_step
    assert all(torch.equal(v, again.model.state_dict()[k]) for k, v in result.model.state_dict().items())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("objective", ("entropy", "information_maximization"))
def test_unlabeled_fixed_budget_no_supervision_and_repeatability(bundle, architecture, objective, monkeypatch):
    original = model(architecture)
    graph = bundle.unlabeled_input("target")
    with pytest.raises(ValueError, match="meta_train"):
        development_supervision(bundle, "target")
    with pytest.raises(ValueError, match="meta_train"):
        development_supervision(bundle, "validation")
    assert "labels" not in inspect.signature(fit_unlabeled).parameters
    def reject_labels(*args, **kwargs):
        raise AssertionError("Unlabeled fitting must not call supervised cross entropy")
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", reject_labels)
    config = AdaptationConfig(steps=4)
    peft = PEFTConfig("lora", backbone_rank=2)
    before_rng = torch.random.get_rng_state().clone()
    result = fit_unlabeled(original, graph, peft, config, objective=objective)
    bundle.graphs["target"].labels ^= 1
    again = fit_unlabeled(original, bundle.unlabeled_input("target"), peft, config, objective=objective)
    assert result.selected_step == 4 and len(result.history) == 4
    assert result.history == again.history
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert torch.equal(predict(result.model, graph), predict(again.model, graph))


def test_information_maximization_and_entropy_arithmetic():
    p = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
    logits = p.log().requires_grad_()
    conditional = -(p * p.log()).sum(1).mean()
    marginal = p.mean(0)
    expected = conditional + (marginal * marginal.log()).sum()
    torch.testing.assert_close(unlabeled_objective(logits, "entropy"), conditional)
    loss = unlabeled_objective(logits, "information_maximization")
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    extreme = torch.tensor([[1e4, -1e4], [1e4, -1e4]], requires_grad=True)
    unlabeled_objective(extreme, "information_maximization").backward()
    assert torch.isfinite(extreme.grad).all()
    with pytest.raises(ValueError):
        unlabeled_objective(logits, "wrong")


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("method", ("head", "lora", "full"))
def test_delta_initial_final_reconstruction_and_tampering(bundle, architecture, method, tmp_path):
    original = model(architecture)
    graph = bundle.unlabeled_input("meta")
    result = fit_supervised_development(original, graph, development_supervision(bundle, "meta"),
                                        PEFTConfig(method, backbone_rank=2), AdaptationConfig(steps=3))
    spec = make_model_spec(architecture, 8, 2, hidden_channels=8, dropout=0.5)
    path = tmp_path / "delta.pt"
    summary = save_weight_delta(path, result, spec, graph, context={"purpose": "test"})
    restored, payload = load_weight_delta(path)
    assert summary["reconstruction_predictions_close"]
    assert payload["initial_sha256"] == state_sha256(original.state_dict())
    for name, initial in payload["initial_state"].items():
        assert torch.equal(payload["delta_state"][name], payload["adapted_state"][name] - initial)
        if name not in result.adapter.target_names:
            assert torch.count_nonzero(payload["delta_state"][name]) == 0
    torch.testing.assert_close(predict(restored, graph), predict(result.model, graph))
    with pytest.raises(FileExistsError):
        save_weight_delta(path, result, spec, graph, context={})
    payload["delta_state"][result.adapter.target_names[0]].flatten()[0] += 1
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="checksum"):
        load_weight_delta(bad)


def test_supervision_mismatch_overlap_and_input_allowlist(bundle):
    original = model()
    graph = bundle.unlabeled_input("meta")
    view = development_supervision(bundle, "meta")
    bad = replace(view, graph_sha256="wrong")
    with pytest.raises(ValueError, match="different graph"):
        fit_supervised_development(original, graph, bad, PEFTConfig())
    bad = replace(view, val_indices=view.train_indices.clone(), val_labels=view.train_labels.clone())
    with pytest.raises(ValueError, match="overlap"):
        fit_supervised_development(original, graph, bad, PEFTConfig())
    graph.y = bundle.graphs["meta"].labels
    with pytest.raises(ValueError, match="label-free"):
        fit_unlabeled(original, graph, PEFTConfig())


@pytest.mark.parametrize("changes", [
    {"steps": 0}, {"steps": True}, {"patience": 0}, {"initialization_seed": -1},
    {"learning_rate": float("nan")}, {"learning_rate": 0}, {"diversity_weight": -1},
])
def test_invalid_fit_config(changes):
    with pytest.raises(ValueError):
        AdaptationConfig(**changes)


@pytest.mark.parametrize("changes", [
    {"method": "unknown"}, {"backbone_rank": 0}, {"head_rank": True}, {"backbone_alpha": 0},
])
def test_invalid_peft_config(changes):
    with pytest.raises(ValueError):
        PEFTConfig(**changes)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_full_finetuning_matches_direct_pyg_adam_updates(bundle, architecture):
    """Independent ordinary-model loop verifies every weight/gradient, not just loss."""
    original = model(architecture)
    direct = deepcopy(original).eval()
    adapter = WeightAdapter(original, PEFTConfig("full")).train()
    graph = bundle.unlabeled_input("meta")
    adapter.prepare_episode(graph.x, graph.edge_index)
    assert adapter._head_cache is None
    assert adapter.target_names == tuple(dict(original.named_parameters()))
    assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == sum(p.numel() for p in original.parameters())
    indices, labels = bundle.select("meta", "train"), bundle.training_labels("meta")
    optimizers = [torch.optim.Adam(direct.parameters(), lr=0.01),
                  torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=0.01)]
    for _ in range(3):
        for module, optimizer in zip((direct, adapter), optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(module(graph.x, graph.edge_index)[indices], labels)
            loss.backward()
            optimizer.step()
        for name, parameter in direct.named_parameters():
            actual = dict(adapter.full_model.named_parameters())[name]
            torch.testing.assert_close(actual.grad, parameter.grad, rtol=0, atol=0)
            torch.testing.assert_close(actual, parameter, rtol=0, atol=0)
    for layer in ("conv1.", "conv2."):
        assert any(torch.count_nonzero(value) for name, value in adapter.updates().items() if name.startswith(layer))
    with pytest.raises(ValueError, match="supervised development"):
        fit_unlabeled(original, graph, PEFTConfig("full"))


def test_rank_is_not_silently_clipped_and_unknown_models_rejected():
    with pytest.raises(ValueError, match="strictly below"):
        WeightAdapter(model(), PEFTConfig("lora", backbone_rank=8))
    with pytest.raises(ValueError, match="strictly below"):
        WeightAdapter(model(), PEFTConfig("lora", head_rank=2))
    with pytest.raises(ValueError, match="supports"):
        WeightAdapter(torch.nn.Linear(8, 2), PEFTConfig())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_frozen_head_cache_preserves_gradients_and_rejects_other_or_mutated_inputs(bundle, architecture):
    original = model(architecture)
    graph = bundle.unlabeled_input("meta")
    uncached = WeightAdapter(original, PEFTConfig("head"))
    cached = WeightAdapter(original, PEFTConfig("head"))
    cached.prepare_episode(graph.x, graph.edge_index)
    labels, indices = bundle.training_labels("meta"), bundle.select("meta", "train")
    for adapter in (uncached, cached):
        torch.nn.functional.cross_entropy(adapter(graph.x, graph.edge_index)[indices], labels).backward()
    assert torch.equal(uncached(graph.x, graph.edge_index), cached(graph.x, graph.edge_index))
    for (a_name, a), (b_name, b) in zip(uncached.named_parameters(), cached.named_parameters()):
        assert a_name == b_name
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)
    other = bundle.unlabeled_input("target")
    assert torch.equal(cached(other.x, other.edge_index), uncached(other.x, other.edge_index))
    graph.x.add_(0.1)  # Tensor identity alone would incorrectly hit an old cache.
    assert torch.equal(cached(graph.x, graph.edge_index), uncached(graph.x, graph.edge_index))
    graph.edge_index = torch.empty(2, 0, dtype=torch.long)
    assert torch.equal(cached(graph.x, graph.edge_index), uncached(graph.x, graph.edge_index))


def test_adaptation_enables_gradients_inside_no_grad_and_restores_context(bundle):
    original = model()
    graph = bundle.unlabeled_input("target")
    with torch.no_grad():
        result = fit_unlabeled(original, graph, PEFTConfig("head"), AdaptationConfig(steps=2))
        assert not torch.is_grad_enabled()
    assert result.selected_step == 2
    assert any(torch.count_nonzero(delta) for delta in result.adapter.updates().values())


def test_inference_mode_is_rejected_before_fitting(bundle):
    original = model()
    graph = bundle.unlabeled_input("target")
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        fit_unlabeled(original, graph, PEFTConfig("head"), AdaptationConfig(steps=2))


def test_stale_source_gradients_are_cleared_only_on_the_frozen_copy():
    original = model()
    sum(p.sum() for p in original.parameters()).backward()
    before = {name: p.grad.clone() for name, p in original.named_parameters()}
    adapter = WeightAdapter(original, PEFTConfig("head"))
    assert all(p.grad is None for p in adapter.base.parameters())
    assert all(torch.equal(p.grad, before[name]) for name, p in original.named_parameters())
<<<<<<< HEAD
=======
"""Tests for the amortized adaptation generator, descriptor, and evaluator."""

from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F

from amortized_gnn_tta.adaptation import (
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    apply_head_delta,
    evaluate_adaptation_pair,
    extract_head_parameters,
    format_adaptation_report,
)
from amortized_gnn_tta.adaptation.generator import adapted_forward, unflatten_head_delta
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.models import build_model, make_model_spec


@pytest.fixture
def tiny_bundle():
    """Create a small 50-node fixture for adaptation testing."""
    config = CSBMConfig.from_matcha(
        num_nodes=50,
        num_features=16,
        degree_scale=5,
        homophily_parameter=0.8,
        negative_mean=-0.5,
        positive_mean=0.5,
        noise_scale=0.5,
        edge_seed=42,
        feature_seed=43,
        split_seed=44,
    )
    target_config = CSBMConfig.from_matcha(
        num_nodes=50,
        num_features=16,
        degree_scale=5,
        homophily_parameter=0.2,
        negative_mean=-0.5,
        positive_mean=0.5,
        noise_scale=0.5,
        edge_seed=52,
        feature_seed=53,
        split_seed=54,
    )
    return load_dataset([
        CSBMEnvironment("source", "source_train", config),
        CSBMEnvironment("target", "target_test", target_config),
    ])


def test_graph_shift_descriptor_shapes(tiny_bundle):
    """Descriptor should produce fixed-size 1D tensor [2*H + 1] regardless of graph size."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)

    graph = tiny_bundle.unlabeled_input("source")
    desc = descriptor(model, graph)

    assert desc.ndim == 1
    assert desc.shape[0] == 8 * 2 + 1  # mean(8) + std(8) + deg(1) = 17
    assert torch.isfinite(desc).all()


def test_head_parameters_extraction_and_unflattening(tiny_bundle):
    """Head parameter extraction and unflattening should match parameter schemas."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)

    flat, schema = extract_head_parameters(model)
    # For GCN conv2: lin.weight is [2, 8] = 16, bias is [2] = 2. Total = 18
    assert flat.ndim == 1
    assert len(flat) == 18

    unflattened = unflatten_head_delta(flat, schema)
    assert "lin.weight" in unflattened or "weight" in unflattened
    assert "bias" in unflattened


def test_generator_zero_initialization(tiny_bundle):
    """At initialization, the generator must output exact zeros (ΔW=0, Δb=0)."""
    descriptor_dim = 17
    head_dim = 18
    gen = HeadAdapterGenerator(descriptor_dim=descriptor_dim, head_dim=head_dim, hidden_dim=32)

    dummy_desc = torch.randn(descriptor_dim)
    dummy_head = torch.randn(head_dim)

    delta = gen(dummy_desc, dummy_head)
    assert delta.shape == (head_dim,)
    assert torch.equal(delta, torch.zeros_like(delta)), "Generator must zero-init so initial adaptation is identity."


def test_adapted_forward_matches_baseline_at_zero_delta(tiny_bundle):
    """Zero delta must produce identical forward outputs to the unadapted model."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()
    graph = tiny_bundle.unlabeled_input("target")

    flat, schema = extract_head_parameters(model)
    zero_delta = torch.zeros_like(flat)

    baseline_logits = model(graph.x, graph.edge_index)
    adapted_logits = adapted_forward(model, graph, zero_delta, schema)

    torch.testing.assert_close(baseline_logits, adapted_logits)


def test_adapted_forward_gradients_flow_to_generator(tiny_bundle):
    """Loss on adapted_forward should backpropagate non-zero gradients into the generator."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()
    graph = tiny_bundle.unlabeled_input("source")

    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)
    flat, schema = extract_head_parameters(model)

    gen = HeadAdapterGenerator(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=32)

    desc = descriptor(model, graph)
    delta = gen(desc, flat)

    logits = adapted_forward(model, graph, delta, schema)
    labels = tiny_bundle.training_labels("source")
    train_idx = tiny_bundle.select("source", "train")

    loss = F.cross_entropy(logits[train_idx], labels)
    loss.backward()

    # Generator should receive non-zero gradients
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gen.parameters())


def test_apply_head_delta_returns_independent_copy(tiny_bundle):
    """apply_head_delta should modify the returned model without mutating original weights."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()

    flat, schema = extract_head_parameters(model)
    random_delta = torch.ones_like(flat) * 0.5

    adapted_model = apply_head_delta(model, random_delta, schema)

    # Original model parameters must not have changed
    flat_orig, _ = extract_head_parameters(model)
    assert torch.equal(flat, flat_orig)

    # Adapted model parameters should be shifted by 0.5
    flat_adapted, _ = extract_head_parameters(adapted_model)
    torch.testing.assert_close(flat_adapted, flat + 0.5)


def test_evaluate_adaptation_pair_and_report_formatting(tiny_bundle):
    """Evaluation pair should correctly compute baseline acc, adapted acc, and gain."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)
    flat, _ = extract_head_parameters(model)
    gen = HeadAdapterGenerator(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=32)

    env_meta = {"role": "target_test", "family": "structural", "severity": 0.5, "seen_severity_in_meta_train": False}
    record = evaluate_adaptation_pair(
        model=model,
        model_id="test_model",
        bundle=tiny_bundle,
        env_id="target",
        env_metadata=env_meta,
        descriptor=descriptor,
        generator=gen,
    )

    assert 0.0 <= record.baseline_accuracy <= 1.0
    assert 0.0 <= record.adapted_accuracy <= 1.0
    # Because generator is zero-init, baseline and adapted should match exactly initially
    assert record.accuracy_gain == pytest.approx(0.0)
    assert record.weight_delta_norm == pytest.approx(0.0)

    report = format_adaptation_report([record])
    assert "Overall Summary Across All Evaluated Pairs" in report
    assert "Breakdown by Shift Family" in report
    assert "Breakdown by Shift Severity" in report
>>>>>>> a496aab (added hypernetwork generator code)
=======
>>>>>>> 511141a (rename file)
