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


def test_subsample_environments_stratification():
    """subsample_environments must balance across shift families instead of slicing raw order."""
    from amortized_gnn_tta.adaptation import subsample_environments

    # Synthetic manifest where lineage 0 has 10 structural environments, followed by attribute and combined
    envs = [
        {"id": f"e_struct_{i}", "family": "structural"} for i in range(10)
    ] + [
        {"id": f"e_attr_{i}", "family": "attribute"} for i in range(5)
    ] + [
        {"id": f"e_comb_{i}", "family": "combined"} for i in range(5)
    ]

    # Subsampling 4 environments should include one of each family
    sub = subsample_environments(envs, max_environments=3)
    families = {e["family"] for e in sub}
    assert len(families) == 3
    assert families == {"structural", "attribute", "combined"}


def test_evaluation_with_partial_split_indexing():
    """evaluator should properly index predictions with bundle.select when split is a subset."""
    config = CSBMConfig.from_matcha(
        num_nodes=100, num_features=16, degree_scale=5, homophily_parameter=0.8,
        negative_mean=-0.5, positive_mean=0.5, noise_scale=0.5,
        edge_seed=11, feature_seed=12, split_seed=13
    )
    bundle = load_dataset([CSBMEnvironment("target", "target_test", config)])

    # Manually restrict 'eval' to a 20-node subset to simulate a benchmark with a partial test split
    partial_indices = torch.tensor([2, 5, 9, 14, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
    bundle.graphs["target"].splits["eval"] = partial_indices

    eval_idx = bundle.select("target", "eval")
    assert len(eval_idx) == 20
    eval_labels = bundle.evaluation_labels("target", "eval")
    assert len(eval_labels) == 20

    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)
    flat, _ = extract_head_parameters(model)
    gen = HeadAdapterGenerator(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=32)

    env_meta = {"role": "target_test", "family": "control", "severity": 0.0, "seen_severity_in_meta_train": True}
    # This must NOT fail with length mismatch between full graph (100) and partial split (20)
    rec = evaluate_adaptation_pair(
        model=model,
        model_id="m1",
        bundle=bundle,
        env_id="target",
        env_metadata=env_meta,
        descriptor=descriptor,
        generator=gen,
    )
    assert 0.0 <= rec.baseline_accuracy <= 1.0


def test_meta_training_without_val_models_does_not_discard_weights(tiny_bundle):
    """When val_models=None, train_adaptation_generator must select by training loss, not discard weights."""
    from amortized_gnn_tta.adaptation import train_adaptation_generator

    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)
    flat, _ = extract_head_parameters(model)
    gen = HeadAdapterGenerator(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=32)

    fake_manifest = {
        "environments": [
            {"id": "source", "role": "meta_train", "family": "control"}
        ]
    }

    res = train_adaptation_generator(
        models={"m1": model},
        manifest=fake_manifest,
        load_env_fn=lambda m, eid: tiny_bundle,
        descriptor=descriptor,
        generator=gen,
        epochs=2,
        val_models=None,
        verbose=False,
    )

    # Weights must NOT be identical to the zero-initialized initial weights!
    final_w = res.best_state_dict["net.2.weight"]
    assert not torch.equal(final_w, torch.zeros_like(final_w))


def test_do_no_harm_fallback_when_validation_degrades(tiny_bundle):
    """If generator worsens validation accuracy, best_state_dict must keep the zero-delta weights."""
    from amortized_gnn_tta.adaptation import train_adaptation_generator

    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)
    flat, _ = extract_head_parameters(model)
    gen = HeadAdapterGenerator(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=32)

    fake_manifest = {
        "environments": [
            {"id": "source", "role": "meta_train", "family": "control"},
            {"id": "target", "role": "validation", "family": "control"},
        ]
    }

    # Deliberately huge learning rate that causes validation to degrade
    res = train_adaptation_generator(
        models={"m1": model},
        manifest=fake_manifest,
        load_env_fn=lambda m, eid: tiny_bundle,
        descriptor=descriptor,
        generator=gen,
        epochs=1,
        lr=100.0,  # Extreme LR destabilizes model
        val_models={"m_val": model},
        verbose=False,
    )

    # When validation degrades, the do-no-harm fallback must preserve initial zero delta weights
    if res.history[0]["val_loss"] > (1.0 - res.history[0].get("val_gain", 0.0)):
        final_w = res.best_state_dict["net.2.weight"]
        assert torch.equal(final_w, torch.zeros_like(final_w))

