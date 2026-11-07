"""Tests for conditional diffusion generator for Graph Test-Time Adaptation."""

import math
import pytest
import torch
from torch.nn import functional as F

from amortized_gnn_tta.adaptation import (
    DiffusionHeadGenerator,
    GraphShiftDescriptor,
    NoiseSchedule,
    SinusoidalPositionalEmbedding,
    extract_head_parameters,
    fit_oracle_deltas,
    train_diffusion_generator,
)
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.models import build_model, make_model_spec


@pytest.fixture
def tiny_bundle():
    """Create a small 50-node fixture for diffusion tests."""
    src_cfg = CSBMConfig.from_matcha(
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
    val_cfg = CSBMConfig.from_matcha(
        num_nodes=50,
        num_features=16,
        degree_scale=5,
        homophily_parameter=0.5,
        negative_mean=-0.5,
        positive_mean=0.5,
        noise_scale=0.5,
        edge_seed=52,
        feature_seed=53,
        split_seed=54,
    )
    return load_dataset([
        CSBMEnvironment("meta_train_env", "source_train", src_cfg),
        CSBMEnvironment("val_env", "target_test", val_cfg),
    ])


def test_noise_schedule_properties():
    """Noise schedule buffers must be bounded, positive, and alphas_cumprod must monotonically decrease."""
    for schedule_name in ("cosine", "linear"):
        sched = NoiseSchedule(timesteps=50, schedule=schedule_name)
        assert len(sched.betas) == 50
        assert len(sched.alphas) == 50
        assert len(sched.alphas_cumprod) == 50

        # Betas must be strictly in (0, 1)
        assert (sched.betas > 0).all()
        assert (sched.betas < 1.0).all()

        # Alphas cumprod starts near 1 and decreases monotonically
        assert sched.alphas_cumprod[0] > 0.9
        diffs = sched.alphas_cumprod[1:] - sched.alphas_cumprod[:-1]
        assert (diffs <= 0).all(), f"{schedule_name} schedule must be monotonically decreasing"
        assert sched.alphas_cumprod[-1] > 0.0


def test_noise_schedule_q_sample():
    """q_sample closed-form forward diffusion must match analytical formula."""
    sched = NoiseSchedule(timesteps=20, schedule="cosine")
    x0 = torch.ones(4, 10)
    zero_noise = torch.zeros_like(x0)

    # At any t, with zero noise, x_t should equal sqrt(alpha_bar_t) * x0
    t = torch.tensor([5, 5, 5, 5], dtype=torch.long)
    xt, _ = sched.q_sample(x0, t, noise=zero_noise)
    expected = torch.sqrt(sched.alphas_cumprod[5]) * x0
    assert torch.allclose(xt, expected, atol=1e-6)

    # Shape preservation for 1D single vector
    x0_single = torch.ones(10)
    xt_single, _ = sched.q_sample(x0_single, torch.tensor(3))
    assert xt_single.shape == (10,)


def test_sinusoidal_positional_embedding():
    """Sinusoidal embedding should produce distinct vectors for distinct timesteps."""
    embed = SinusoidalPositionalEmbedding(dim=32)

    # Scalar input
    emb_0 = embed(torch.tensor(0))
    emb_10 = embed(torch.tensor(10))
    assert emb_0.shape == (32,)
    assert emb_10.shape == (32,)
    assert not torch.allclose(emb_0, emb_10)

    # Batched 1D input
    t_batch = torch.tensor([0, 5, 10, 20])
    emb_batch = embed(t_batch)
    assert emb_batch.shape == (4, 32)


def test_diffusion_head_generator_zero_initialization():
    """Untrained DiffusionHeadGenerator must output exactly zero delta (identity adaptation)."""
    descriptor_dim = 17
    head_dim = 18
    gen = DiffusionHeadGenerator(
        descriptor_dim=descriptor_dim,
        head_dim=head_dim,
        hidden_dim=32,
        diffusion_steps=20,
        sampling_steps=5,
    )

    desc = torch.randn(descriptor_dim)
    head_w = torch.randn(head_dim)

    # Forward should produce exact zeros
    delta = gen(desc, head_w)
    assert delta.shape == (head_dim,)
    assert torch.equal(delta, torch.zeros_like(delta)), "Zero-init must ensure delta = 0 initially."


def test_target_normalizer_preserves_zero_init():
    """Setting target standardization scale must preserve the zero-init identity."""
    gen = DiffusionHeadGenerator(
        descriptor_dim=17,
        head_dim=18,
        hidden_dim=32,
        diffusion_steps=20,
        sampling_steps=5,
    )

    # Register arbitrary scale factor
    std = torch.full((18,), 0.05)
    gen.set_target_normalizer(std=std)

    desc = torch.randn(17)
    head_w = torch.randn(18)
    delta = gen(desc, head_w)

    assert torch.equal(delta, torch.zeros_like(delta)), "Scale normalization must preserve delta = 0."


def test_deterministic_ddim_sampling():
    """Sampling with a fixed seed must be 100% deterministic and reproducible."""
    gen = DiffusionHeadGenerator(
        descriptor_dim=17,
        head_dim=18,
        hidden_dim=32,
        diffusion_steps=20,
        sampling_steps=5,
        default_seed=123,
    )

    # Break zero-init intentionally to test sampling trajectory
    with torch.no_grad():
        for p in gen.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    desc = torch.randn(17)
    head_w = torch.randn(18)

    sample1 = gen.sample(desc, head_w, steps=5, seed=42)
    sample2 = gen.sample(desc, head_w, steps=5, seed=42)
    assert torch.equal(sample1, sample2), "Identical seed must produce identical DDIM trajectory."

    sample_diff = gen.sample(desc, head_w, steps=5, seed=999)
    assert not torch.equal(sample1, sample_diff), "Different seed should produce different sample."


def test_diffusion_loss_gradient_flow():
    """diffusion_loss must produce a scalar loss with valid gradient flow."""
    gen = DiffusionHeadGenerator(
        descriptor_dim=17,
        head_dim=18,
        hidden_dim=32,
        diffusion_steps=20,
        sampling_steps=5,
    )

    # Single vector
    desc = torch.randn(17)
    head_w = torch.randn(18)
    target_delta = torch.randn(18)

    loss = gen.diffusion_loss(desc, head_w, target_delta)
    assert loss.ndim == 0
    assert loss.item() >= 0.0

    loss.backward()
    for name, p in gen.net.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"

    # Batch test
    gen.zero_grad()
    desc_batch = torch.randn(4, 17)
    head_w_batch = torch.randn(4, 18)
    target_batch = torch.randn(4, 18)

    batch_loss = gen.diffusion_loss(desc_batch, head_w_batch, target_batch)
    assert batch_loss.ndim == 0
    batch_loss.backward()
    for name, p in gen.net.named_parameters():
        assert p.grad is not None


def test_fit_oracle_deltas(tiny_bundle):
    """fit_oracle_deltas should optimize deltas on meta_train training nodes."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)

    fake_manifest = {
        "environments": [
            {"id": "meta_train_env", "role": "meta_train", "family": "control"}
        ]
    }

    records = fit_oracle_deltas(
        models={"m1": model},
        manifest=fake_manifest,
        load_env_fn=lambda m, eid: tiny_bundle,
        descriptor=descriptor,
        steps=10,
        lr=0.05,
        verbose=False,
    )

    assert len(records) == 1
    r = records[0]
    assert r.model_id == "m1"
    assert r.environment_id == "meta_train_env"
    assert r.target_delta.shape == (18,)
    # Supervised optimization on training nodes should maintain or improve train accuracy
    assert r.oracle_acc >= r.baseline_acc - 1e-4

    # Base model parameters must NOT have accumulated gradients
    for p in model.parameters():
        assert p.grad is None


def test_train_diffusion_generator_end_to_end(tiny_bundle):
    """train_diffusion_generator should fit targets, train denoiser, and evaluate validation."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)
    flat, _ = extract_head_parameters(model)

    gen = DiffusionHeadGenerator(
        descriptor_dim=descriptor.output_dim,
        head_dim=len(flat),
        hidden_dim=32,
        diffusion_steps=10,
        sampling_steps=3,
    )

    fake_manifest = {
        "environments": [
            {"id": "meta_train_env", "role": "meta_train", "family": "control"},
            {"id": "val_env", "role": "validation", "family": "control"},
        ]
    }

    res = train_diffusion_generator(
        models={"m1": model},
        manifest=fake_manifest,
        load_env_fn=lambda m, eid: tiny_bundle,
        descriptor=descriptor,
        generator=gen,
        epochs=2,
        lr=1e-3,
        oracle_steps=5,
        val_models={"v1": model},
        verbose=False,
    )

    assert len(res.history) == 2
    assert "train_loss" in res.history[0]
    assert "val_gain" in res.history[0]
    assert "net.0.weight" in res.best_state_dict


def test_diffusion_do_no_harm_fallback(tiny_bundle):
    """When validation degrades, the diffusion generator must preserve initial zero-delta weights."""
    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)
    flat, _ = extract_head_parameters(model)

    gen = DiffusionHeadGenerator(
        descriptor_dim=descriptor.output_dim,
        head_dim=len(flat),
        hidden_dim=32,
        diffusion_steps=10,
        sampling_steps=3,
    )

    fake_manifest = {
        "environments": [
            {"id": "meta_train_env", "role": "meta_train", "family": "control"},
            {"id": "val_env", "role": "validation", "family": "control"},
        ]
    }

    # Deliberately huge learning rate to force divergence / degradation
    res = train_diffusion_generator(
        models={"m1": model},
        manifest=fake_manifest,
        load_env_fn=lambda m, eid: tiny_bundle,
        descriptor=descriptor,
        generator=gen,
        epochs=1,
        lr=100.0,
        oracle_steps=1,
        val_models={"v1": model},
        verbose=False,
    )

    # Initial weights of the final layer were zeros.
    # If validation degraded (gain <= 0), best_state_dict kept the initial zero weights!
    if res.history[0]["val_gain"] is not None and res.history[0]["val_gain"] <= 0.0:
        assert torch.equal(res.best_state_dict["net.2.weight"], torch.zeros_like(res.best_state_dict["net.2.weight"]))


def test_diffusion_backbone_adaptation_training(tiny_bundle):
    """Diffusion generator must correctly train and sample updates for backbone parameters."""
    from amortized_gnn_tta.adaptation import extract_target_parameters, evaluate_adaptation_pair

    spec = make_model_spec("gcn", in_channels=16, out_channels=2, hidden_channels=8)
    model = build_model(spec)
    descriptor = GraphShiftDescriptor(hidden_channels=8, include_degree=True)
    flat_bb, schema_bb = extract_target_parameters(model, scope="backbone")
    assert len(flat_bb) == 136

    gen = DiffusionHeadGenerator(
        descriptor_dim=descriptor.output_dim,
        head_dim=len(flat_bb),
        hidden_dim=32,
        diffusion_steps=10,
        sampling_steps=3,
    )

    fake_manifest = {
        "environments": [
            {"id": "meta_train_env", "role": "meta_train", "family": "control"},
            {"id": "val_env", "role": "validation", "family": "control"},
        ]
    }

    # 1. Oracle fitting on backbone
    records = fit_oracle_deltas(
        models={"m1": model},
        manifest=fake_manifest,
        load_env_fn=lambda m, eid: tiny_bundle,
        descriptor=descriptor,
        steps=5,
        target_scope="backbone",
        verbose=False,
    )
    assert len(records) == 1
    assert records[0].target_delta.shape == (136,)

    # 2. Diffusion denoiser training with backbone scope
    res = train_diffusion_generator(
        models={"m1": model},
        manifest=fake_manifest,
        load_env_fn=lambda m, eid: tiny_bundle,
        descriptor=descriptor,
        generator=gen,
        epochs=2,
        lr=1e-3,
        oracle_steps=5,
        val_models={"v1": model},
        verbose=False,
        target_scope="backbone",
    )
    assert len(res.history) == 2
    assert "train_loss" in res.history[0]

    # 3. Evaluation on pair with backbone scope
    rec = evaluate_adaptation_pair(
        model=model,
        model_id="m1",
        bundle=tiny_bundle,
        env_id="val_env",
        env_metadata=fake_manifest["environments"][1],
        descriptor=descriptor,
        generator=gen,
        target_scope="backbone",
    )
    assert rec.weight_delta_norm >= 0.0
    assert torch.isfinite(torch.tensor(rec.adapted_accuracy))

