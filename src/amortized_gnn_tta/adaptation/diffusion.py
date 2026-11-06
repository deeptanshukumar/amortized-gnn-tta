"""Conditional Diffusion Generator for Graph Test-Time Adaptation.

Provides:
1. NoiseSchedule: Cosine/Linear DDPM & DDIM beta schedules and closed-form q_sample.
2. SinusoidalPositionalEmbedding: Timestep embeddings for diffusion conditioning.
3. DiffusionHeadGenerator: Conditional denoiser neural network predicting x0 weight deltas.
4. fit_oracle_deltas: Supervision extraction on meta-training graphs.
5. train_diffusion_generator: Training loop with DDIM validation and do-no-harm fallback.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data

from .evaluator import evaluate_adaptation_pair, subsample_environments, summarize_records
from .generator import (
    GraphShiftDescriptor,
    adapted_forward,
    extract_head_parameters,
)
from .meta_trainer import MetaTrainResult


# ==============================================================================
# 1. Noise Schedule & Analytical Noising
# ==============================================================================

class NoiseSchedule(nn.Module):
    """Variance schedule supporting forward noising and DDIM sampling.

    Uses the cosine schedule (Nichol & Dhariwal, 2021) by default, which prevents
    over-corruption at the start of diffusion.
    """

    def __init__(self, timesteps: int = 100, schedule: str = "cosine"):
        super().__init__()
        self.timesteps = timesteps
        self.schedule = schedule

        if schedule == "cosine":
            # Cosine schedule: f(t) = cos(((t/T + s) / (1 + s)) * (pi / 2))^2
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, min=1e-5, max=0.999)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register schedule buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Closed-form forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
        if noise is None:
            noise = torch.randn_like(x_0)

        # Shape handling for batch or single vector
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha.ndim < x_0.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise


# ==============================================================================
# 2. Timestep Positional Embedding
# ==============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timestep conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Compute sinusoidal embedding for 1D tensor of timesteps t."""
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(-emb_scale * torch.arange(half_dim, device=device, dtype=torch.float32))

        # t can be scalar or [B]
        t_float = t.float().view(-1, 1)
        args = t_float * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        out = self.mlp(embedding)
        return out.squeeze(0) if t.ndim == 0 else out


# ==============================================================================
# 3. Diffusion Head Generator (Denoiser + DDIM Sampler)
# ==============================================================================

class DiffusionHeadGenerator(nn.Module):
    """Conditional Diffusion Model that predicts head parameter updates (ΔW, Δb).

    Architecture:
        Conditions on [descriptor || head_weights || time_emb(t)].
        Uses x0-parameterization: the denoiser directly predicts the clean target delta.
        Zero-initialization of the final layer guarantees delta = 0 at initialization
        (exact identity / do-no-harm start).
    """

    def __init__(
        self,
        descriptor_dim: int,
        head_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        diffusion_steps: int = 100,
        sampling_steps: int = 20,
        time_dim: int = 32,
        schedule: str = "cosine",
        default_seed: int = 42,
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.sampling_steps = sampling_steps
        self.default_seed = default_seed

        self.schedule = NoiseSchedule(timesteps=diffusion_steps, schedule=schedule)
        self.time_embed = SinusoidalPositionalEmbedding(time_dim)

        # Input: [x_t (head_dim) || desc (desc_dim) || head_w (head_dim) || t_emb (time_dim)]
        in_dim = head_dim * 2 + descriptor_dim + time_dim

        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim

        final_linear = nn.Linear(current_dim, head_dim)
        # Zero-initialize the final output layer (guarantees x0_hat = 0 at epoch 0)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

        # Target standardization statistics (mean and std across training oracle deltas)
        self.register_buffer("target_mean", torch.zeros(head_dim))
        self.register_buffer("target_std", torch.ones(head_dim))
        self.register_buffer("has_norm_stats", torch.tensor(False, dtype=torch.bool))

    def set_target_normalizer(self, std: Tensor, mean: Tensor | None = None) -> None:
        """Register standardization statistics so diffusion trains on unit-variance space."""
        safe_std = torch.clamp(torch.nan_to_num(std, nan=1.0), min=1e-4)
        if mean is not None:
            safe_mean = torch.nan_to_num(mean, nan=0.0)
        else:
            safe_mean = torch.zeros_like(safe_std)
        self.target_mean.copy_(safe_mean.detach().to(self.target_mean.device))
        self.target_std.copy_(safe_std.detach().to(self.target_std.device))
        self.has_norm_stats.copy_(torch.tensor(True, dtype=torch.bool))

    def normalize(self, delta: Tensor) -> Tensor:
        if self.has_norm_stats:
            return (delta - self.target_mean) / self.target_std
        return delta

    def unnormalize(self, delta_normed: Tensor) -> Tensor:
        if self.has_norm_stats:
            return delta_normed * self.target_std + self.target_mean
        return delta_normed

    def forward_denoiser(self, x_t: Tensor, t: Tensor, desc: Tensor, head_weights: Tensor) -> Tensor:
        """Denoiser forward pass predicting clean x0."""
        # Ensure 2D batch shape
        is_single = x_t.ndim == 1
        if is_single:
            x_t = x_t.unsqueeze(0)
            desc = desc.unsqueeze(0)
            head_weights = head_weights.unsqueeze(0)
            if t.ndim == 0:
                t = t.unsqueeze(0)

        t_emb = self.time_embed(t)
        if t_emb.ndim == 1:
            t_emb = t_emb.unsqueeze(0)

        combined = torch.cat([x_t, desc, head_weights, t_emb], dim=-1)
        x0_pred = self.net(combined)

        return x0_pred.squeeze(0) if is_single else x0_pred

    def diffusion_loss(self, desc: Tensor, head_weights: Tensor, target_delta: Tensor) -> Tensor:
        """Compute training MSE loss: ||x0 - x0_hat||^2 on random timestep t."""
        # Normalize target into unit-variance space
        x_0 = self.normalize(target_delta)
        device = target_delta.device

        batch_size = 1 if target_delta.ndim == 1 else target_delta.shape[0]
        t = torch.randint(0, self.schedule.timesteps, (batch_size,), device=device, dtype=torch.long)
        if target_delta.ndim == 1:
            t = t[0]

        x_t, _ = self.schedule.q_sample(x_0, t)
        x0_pred = self.forward_denoiser(x_t, t, desc, head_weights)

        return F.mse_loss(x0_pred, x_0)

    @torch.no_grad()
    def sample(
        self,
        desc: Tensor,
        head_weights: Tensor,
        steps: int | None = None,
        seed: int | None = None,
    ) -> Tensor:
        """Deterministic DDIM sampling from pure noise x_T to clean delta x_0.

        Args:
            desc: Graph shift descriptor vector.
            head_weights: Model head parameter vector.
            steps: Number of DDIM steps (default: self.sampling_steps).
            seed: Optional RNG seed for deterministic noise initialization (defaults to self.default_seed).

        Returns:
            Clean un-normalized parameter update delta.
        """
        if steps is None:
            steps = self.sampling_steps
        if seed is None:
            seed = self.default_seed

        device = head_weights.device
        steps = max(1, min(steps, self.schedule.timesteps))

        # Sample initial pure Gaussian noise
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            x_t = torch.randn(self.head_dim, device=device, generator=g)
        else:
            x_t = torch.randn(self.head_dim, device=device)

        # DDIM timestep sequence from T-1 down to 0
        timesteps = torch.linspace(self.schedule.timesteps - 1, 0, steps, dtype=torch.long, device=device)

        alphas_cumprod = self.schedule.alphas_cumprod

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else None

            # 1. Predict clean x0
            x0_pred = self.forward_denoiser(x_t, t, desc, head_weights)

            # 2. Extract implied noise direction eps_hat = (x_t - sqrt(alpha_bar_t)*x0) / sqrt(1 - alpha_bar_t)
            alpha_bar_t = alphas_cumprod[t]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_t = torch.sqrt(1.0 - alpha_bar_t)
            eps_hat = (x_t - sqrt_alpha_bar_t * x0_pred) / torch.clamp(sqrt_one_minus_t, min=1e-8)

            # 3. Take DDIM step to t_prev
            if t_prev is not None:
                alpha_bar_prev = alphas_cumprod[t_prev]
                x_t = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_hat
            else:
                x_t = x0_pred

        # Unnormalize back to raw weight update space
        clean_delta = self.unnormalize(x_t)
        return clean_delta

    def forward(self, desc: Tensor, head_weights: Tensor) -> Tensor:
        """Evaluate adaptation delta, matching HeadAdapterGenerator callable interface."""
        return self.sample(desc, head_weights, steps=self.sampling_steps, seed=self.default_seed)


# ==============================================================================
# 4. Supervised Oracle Targets Fitting (Meta-Train Phase)
# ==============================================================================

@dataclass
class OracleTargetRecord:
    """Stores fitted teacher delta and conditioning for one (model, environment) pair."""
    model_id: str
    environment_id: str
    descriptor: Tensor
    head_weights: Tensor
    target_delta: Tensor
    baseline_acc: float
    oracle_acc: float


def fit_oracle_deltas(
    models: dict[str, nn.Module],
    manifest: dict[str, Any],
    load_env_fn: Callable[[dict[str, Any], str], Any],
    descriptor: GraphShiftDescriptor,
    steps: int = 50,
    lr: float = 1e-2,
    delta_l2_reg: float = 1e-3,
    max_environments: int | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> list[OracleTargetRecord]:
    """Precompute supervised teacher updates (Delta*) for all meta-training pairs.

    Strict boundary: reads training_labels ONLY on environments where role == 'meta_train'.
    """
    meta_envs = [e for e in manifest["environments"] if e["role"] == "meta_train"]
    meta_envs = subsample_environments(meta_envs, max_environments)

    total_pairs = len(models) * len(meta_envs)
    if verbose:
        print(f"Fitting Teacher Oracle Targets:")
        print(f"  • Models: {len(models)} | Environments: {len(meta_envs)}")
        print(f"  • Total pairs to fit: {total_pairs} (Steps per pair: {steps}, LR: {lr})\n")

    records: list[OracleTargetRecord] = []
    start_time = time.time()

    for env_meta in meta_envs:
        env_id = env_meta["id"]
        bundle = load_env_fn(manifest, env_id)

        unlabeled = bundle.unlabeled_input(env_id).to(device)
        train_idx = bundle.select(env_id, "train").to(device)
        train_y = bundle.training_labels(env_id).to(device)

        for model_id, model in models.items():
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            flat_w, schema = extract_head_parameters(model)
            flat_w = flat_w.to(device)
            desc = descriptor(model, unlabeled).to(device)

            # Baseline accuracy on training nodes before fitting
            with torch.no_grad():
                init_logits = model(unlabeled.x, unlabeled.edge_index)[train_idx]
                base_acc = float((init_logits.argmax(-1) == train_y).float().mean().item())

            # Optimize delta directly with Adam
            delta_param = nn.Parameter(torch.zeros_like(flat_w, requires_grad=True))
            optimizer = torch.optim.Adam([delta_param], lr=lr)

            for _ in range(steps):
                optimizer.zero_grad()
                logits = adapted_forward(model, unlabeled, delta_param, schema)[train_idx]
                loss = F.cross_entropy(logits, train_y) + delta_l2_reg * torch.sum(delta_param ** 2)
                loss.backward()
                optimizer.step()

            # Record final oracle accuracy on training nodes
            with torch.no_grad():
                final_logits = adapted_forward(model, unlabeled, delta_param, schema)[train_idx]
                oracle_acc = float((final_logits.argmax(-1) == train_y).float().mean().item())

            records.append(OracleTargetRecord(
                model_id=model_id,
                environment_id=env_id,
                descriptor=desc.detach().cpu(),
                head_weights=flat_w.detach().cpu(),
                target_delta=delta_param.detach().cpu(),
                baseline_acc=base_acc,
                oracle_acc=oracle_acc,
            ))

    elapsed = time.time() - start_time
    mean_gain = np.mean([r.oracle_acc - r.baseline_acc for r in records])
    if verbose:
        print(f"  • Fitted {len(records)} targets in {elapsed:.2f}s (Average Oracle Train Gain: {mean_gain*100:+.2f}%)\n")

    return records


# ==============================================================================
# 5. Diffusion Training Loop
# ==============================================================================

def train_diffusion_generator(
    models: dict[str, nn.Module],
    manifest: dict[str, Any],
    load_env_fn: Callable[[dict[str, Any], str], Any],
    descriptor: GraphShiftDescriptor,
    generator: DiffusionHeadGenerator,
    epochs: int = 10,
    lr: float = 1e-3,
    oracle_steps: int = 50,
    oracle_lr: float = 1e-2,
    delta_l2_reg: float = 1e-3,
    max_environments: int | None = None,
    max_models: int | None = None,
    val_models: dict[str, nn.Module] | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> MetaTrainResult:
    """Train the conditional diffusion head generator using oracle distillation."""
    total_start_time = time.time()
    generator = generator.to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    # 1. Select subset of meta-training models
    model_items = dict(list(models.items())[:max_models] if max_models is not None else models)

    # 2. Fit supervised teacher target deltas
    targets = fit_oracle_deltas(
        models=model_items,
        manifest=manifest,
        load_env_fn=load_env_fn,
        descriptor=descriptor,
        steps=oracle_steps,
        lr=oracle_lr,
        delta_l2_reg=delta_l2_reg,
        max_environments=max_environments,
        device=device,
        verbose=verbose,
    )

    if not targets:
        raise ValueError("No oracle targets were fitted. Check manifest and max_environments.")

    # 3. Standardize targets to unit variance space (zero-centered to preserve zero-init identity)
    all_targets = torch.stack([r.target_delta for r in targets])
    if len(targets) > 1:
        std_delta = all_targets.std(dim=0)
    else:
        std_delta = torch.ones(generator.head_dim)
    std_delta = torch.nan_to_num(std_delta, nan=1.0)
    std_delta = torch.clamp(std_delta, min=1e-3)
    generator.set_target_normalizer(std=std_delta)

    # 4. Prepare "Do-No-Harm" Baseline Check on validation environments
    best_state_dict = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_val_loss = float("inf")
    best_train_loss = float("inf")

    if val_models:
        generator.eval()
        val_envs = [e for e in manifest["environments"] if e["role"] == "validation"]
        val_envs = subsample_environments(val_envs, max_environments)

        baseline_records = []
        for v_env in val_envs:
            v_bundle = load_env_fn(manifest, v_env["id"])
            for v_mid, v_m in val_models.items():
                r = evaluate_adaptation_pair(
                    v_m, v_mid, v_bundle, v_env["id"], v_env, descriptor, generator, device=device
                )
                baseline_records.append(r)
        baseline_summary = summarize_records(baseline_records)
        best_val_loss = 1.0 - baseline_summary.get("mean_adapted_acc", 0.0)
        if verbose:
            print(f"Starting Diffusion Denoiser Training:")
            print(f"  • Unadapted Baseline Val Acc (Epoch 0): {baseline_summary.get('mean_adapted_acc', 0.0)*100:.2f}% (Threshold for saving generator)\n")

    history: list[dict[str, Any]] = []

    # 5. Training loop over targets (batched)
    batch_size = 16
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        generator.train()
        train_losses = []

        # Shuffle targets each epoch
        indices = torch.randperm(len(targets))

        for b_start in range(0, len(targets), batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            d = torch.stack([targets[k].descriptor for k in b_idx]).to(device)
            w = torch.stack([targets[k].head_weights for k in b_idx]).to(device)
            td = torch.stack([targets[k].target_delta for k in b_idx]).to(device)

            optimizer.zero_grad()
            loss = generator.diffusion_loss(d, w, td)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_loss = float(np.mean(train_losses))

        # Validation check using DDIM sampling
        val_loss = None
        val_gain = None
        if val_models:
            generator.eval()
            val_envs = [e for e in manifest["environments"] if e["role"] == "validation"]
            val_envs = subsample_environments(val_envs, max_environments)

            val_records = []
            for v_env in val_envs:
                v_bundle = load_env_fn(manifest, v_env["id"])
                for v_mid, v_m in val_models.items():
                    r = evaluate_adaptation_pair(
                        v_m, v_mid, v_bundle, v_env["id"], v_env, descriptor, generator, device=device
                    )
                    val_records.append(r)

            val_summary = summarize_records(val_records)
            val_gain = val_summary.get("mean_gain", 0.0)
            val_loss = 1.0 - val_summary.get("mean_adapted_acc", 0.0)

            # Do-no-harm update rule: strictly require beating the unadapted baseline
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        else:
            if mean_loss < best_train_loss:
                best_train_loss = mean_loss
                best_state_dict = {k: v.cpu().clone() for k, v in generator.state_dict().items()}

        epoch_time = time.time() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "val_loss": val_loss,
            "val_gain": val_gain,
            "time_seconds": epoch_time,
        }
        history.append(epoch_record)

        if verbose:
            val_str = f"Val Acc Gain: {val_gain*100:+.2f}%" if val_gain is not None else "No Val"
            print(f"Epoch {epoch:02d}/{epochs:02d} | Diff MSE: {mean_loss:.5f} | {val_str} | Time: {epoch_time:.2f}s")

    total_time = time.time() - total_start_time

    return MetaTrainResult(
        generator=generator,
        best_state_dict=best_state_dict,
        history=history,
        total_time_seconds=total_time,
        pairs_trained=len(targets),
    )
