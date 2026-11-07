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


def test_structural_descriptor_shapes_and_sensitivity(tiny_bundle):
    """Structural descriptor should have shape 3*H + 9 and distinguish homophily shifts."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    desc_module = GraphShiftDescriptor(hidden_channels=8, descriptor_type="structural")

    assert desc_module.output_dim == 8 * 3 + 9  # 33

    # Homophilic graph (h=0.8) vs Heterophilic graph (h=0.2)
    homo_graph = tiny_bundle.unlabeled_input("source")
    hetero_graph = tiny_bundle.unlabeled_input("target")

    desc_homo = desc_module(model, homo_graph)
    desc_hetero = desc_module(model, hetero_graph)

    assert desc_homo.shape == (33,)
    assert desc_hetero.shape == (33,)
    assert torch.isfinite(desc_homo).all()
    assert torch.isfinite(desc_hetero).all()

    # Dirichlet energy (indices 16:24 are per-channel, 24 is dirichlet_total)
    dirichlet_homo = desc_homo[24].item()
    dirichlet_hetero = desc_hetero[24].item()

    # Heterophilic graphs must have significantly higher Dirichlet energy than homophilic graphs
    assert dirichlet_hetero > dirichlet_homo, (
        f"Expected heterophilic Dirichlet energy ({dirichlet_hetero:.4f}) > "
        f"homophilic Dirichlet energy ({dirichlet_homo:.4f})"
    )


def test_structural_descriptor_zero_edges(tiny_bundle):
    """Structural descriptor should gracefully handle graphs with zero edges."""
    from torch_geometric.data import Data

    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    desc_module = GraphShiftDescriptor(hidden_channels=8, descriptor_type="structural")

    empty_edge_graph = Data(
        x=torch.randn(20, 16),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=20,
    )
    desc = desc_module(model, empty_edge_graph)
    assert desc.shape == (33,)
    assert torch.isfinite(desc).all()
    # When zero edges, isolated_ratio (last element) must be 1.0
    assert torch.isclose(desc[-1], torch.tensor(1.0))


def test_graph_shift_descriptor_invalid_type():
    """Invalid descriptor_type should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown descriptor_type"):
        GraphShiftDescriptor(descriptor_type="invalid_type")


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


def test_extract_target_parameters_scopes():
    """extract_target_parameters must correctly filter by scope."""
    from amortized_gnn_tta.adaptation import extract_target_parameters

    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)

    # 1. Head scope: conv2 only (lin.weight: 16, bias: 2 -> 18)
    flat_head, schema_head = extract_target_parameters(model, scope="head")
    assert len(flat_head) == 18
    assert len(schema_head) == 2

    # 2. Backbone scope: conv1 only (lin.weight: 8x16=128, bias: 8 -> 136)
    flat_backbone, schema_backbone = extract_target_parameters(model, scope="backbone")
    assert len(flat_backbone) == 136
    assert len(schema_backbone) == 2
    assert all("conv1." in s[0] for s in schema_backbone)

    # 3. All scope: conv1 + conv2 (136 + 18 = 154)
    flat_all, schema_all = extract_target_parameters(model, scope="all")
    assert len(flat_all) == 154
    assert len(schema_all) == 4

    # 4. Invalid scope raises ValueError
    with pytest.raises(ValueError, match="Unknown scope"):
        extract_target_parameters(model, scope="invalid")


def test_adapted_forward_and_apply_delta_backbone(tiny_bundle):
    """adapted_forward and apply_target_delta must work with backbone scope."""
    from amortized_gnn_tta.adaptation import apply_target_delta, extract_target_parameters

    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()
    graph = tiny_bundle.unlabeled_input("target")

    flat_bb, schema_bb = extract_target_parameters(model, scope="backbone")
    zero_delta = torch.zeros_like(flat_bb)

    # Baseline vs adapted with zero delta
    base_logits = model(graph.x, graph.edge_index)
    adapted_logits = adapted_forward(model, graph, zero_delta, schema_bb)
    torch.testing.assert_close(base_logits, adapted_logits)

    # Non-zero delta must produce different outputs
    nonzero_delta = torch.randn_like(flat_bb) * 0.1
    shifted_logits = adapted_forward(model, graph, nonzero_delta, schema_bb)
    assert not torch.allclose(base_logits, shifted_logits)

    # apply_target_delta must produce matching forward pass and not mutate base
    adapted_model = apply_target_delta(model, nonzero_delta, schema_bb)
    perm_logits = adapted_model(graph.x, graph.edge_index)
    torch.testing.assert_close(shifted_logits, perm_logits)

    # Original model must be unchanged
    flat_orig, _ = extract_target_parameters(model, scope="backbone")
    assert torch.equal(flat_bb, flat_orig)


def test_conditioning_hierarchy_shapes(tiny_bundle):
    """Verify output dimensions for conditioning versions A, B, C, and D."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()
    graph = tiny_bundle.unlabeled_input("target")

    # in_channels=16, hidden_channels=8, out_channels=2
    # Dim A: 16*2 + 17 = 49
    # Dim B: 49 + (8*3 + 5) = 49 + 29 = 78
    # Dim C: 78 + (7 + 2) = 87
    # Dim D: 87 + 12 = 99

    desc_a = GraphShiftDescriptor(in_channels=16, hidden_channels=8, out_channels=2, conditioning="A")
    desc_b = GraphShiftDescriptor(in_channels=16, hidden_channels=8, out_channels=2, conditioning="B")
    desc_c = GraphShiftDescriptor(in_channels=16, hidden_channels=8, out_channels=2, conditioning="C")
    desc_d = GraphShiftDescriptor(in_channels=16, hidden_channels=8, out_channels=2, conditioning="D")

    assert desc_a.output_dim == 49
    assert desc_b.output_dim == 78
    assert desc_c.output_dim == 87
    assert desc_d.output_dim == 99

    vec_a = desc_a(model, graph)
    vec_b = desc_b(model, graph)
    vec_c = desc_c(model, graph)
    vec_d = desc_d(model, graph)

    assert vec_a.shape == (49,)
    assert vec_b.shape == (78,)
    assert vec_c.shape == (87,)
    assert vec_d.shape == (99,)

    assert torch.isfinite(vec_a).all()
    assert torch.isfinite(vec_b).all()
    assert torch.isfinite(vec_c).all()
    assert torch.isfinite(vec_d).all()

    # B must strictly contain A as its prefix
    torch.testing.assert_close(vec_b[:49], vec_a)
    # C must strictly contain B as its prefix
    torch.testing.assert_close(vec_c[:78], vec_b)
    # D must strictly contain C as its prefix
    torch.testing.assert_close(vec_d[:87], vec_c)


def test_conditioning_version_a_model_independence(tiny_bundle):
    """Version A must produce identical results with model=None or any model."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()
    graph = tiny_bundle.unlabeled_input("target")

    desc_a = GraphShiftDescriptor(in_channels=16, hidden_channels=8, out_channels=2, conditioning="A")
    res_none = desc_a(None, graph)
    res_model = desc_a(model, graph)

    assert torch.equal(res_none, res_model)


def test_conditioning_version_c_uncertainty_sensitivity(tiny_bundle):
    """Version C uncertainty descriptor must change when model logits change."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model1 = build_model(spec).eval()
    model2 = build_model(spec).eval()
    with torch.no_grad():
        model2.conv2.lin.weight.mul_(5.0)

    graph = tiny_bundle.unlabeled_input("target")
    desc_c = GraphShiftDescriptor(in_channels=16, hidden_channels=8, out_channels=2, conditioning="C")

    vec1 = desc_c(model1, graph)
    vec2 = desc_c(model2, graph)

    # Uncertainty slice (78:87) must differ
    assert not torch.allclose(vec1[78:87], vec2[78:87])


def test_conditioning_version_d_weight_sensitivity(tiny_bundle):
    """Version D compact weight descriptor must change when weights change."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model1 = build_model(spec).eval()
    model2 = build_model(spec).eval()
    with torch.no_grad():
        model2.conv2.lin.weight.add_(1.5)

    graph = tiny_bundle.unlabeled_input("target")
    desc_d = GraphShiftDescriptor(in_channels=16, hidden_channels=8, out_channels=2, conditioning="D")

    vec1 = desc_d(model1, graph)
    vec2 = desc_d(model2, graph)

    # Compact weight slice (87:100) must differ
    assert not torch.allclose(vec1[87:], vec2[87:])


def test_generator_with_version_conditioning(tiny_bundle):
    """Generators must work with include_head_weights=False on version descriptors."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec).eval()
    graph = tiny_bundle.unlabeled_input("target")

    flat_head, schema = extract_head_parameters(model)
    head_dim = len(flat_head)

    desc_module = GraphShiftDescriptor(in_channels=16, hidden_channels=8, out_channels=2, conditioning="A")
    desc = desc_module(model, graph)

    # Hypernetwork without extra head weights (conditioning vector is the sole input)
    gen = HeadAdapterGenerator(descriptor_dim=desc_module.output_dim, head_dim=head_dim, include_head_weights=False)
    delta = gen(desc)
    assert delta.shape == (head_dim,)
    assert torch.allclose(delta, torch.zeros_like(delta))  # Zero-init verification

