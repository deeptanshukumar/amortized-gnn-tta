"""Meta-training loop for the adaptation generator.

Pairs pretrained GNNs with meta-training shifted graphs, updates the hypernetwork
parameters via gradient descent on meta-training labels, and validates on unseen
validation pairs.
"""

from collections.abc import Callable
from dataclasses import dataclass
import time
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.api import DatasetBundle
from .evaluator import evaluate_adaptation_pair, subsample_environments, summarize_records
from ._validation import conditioning_contract, checked_backward, frozen_sources, training_options
from .generator import (
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    adapted_forward,
    extract_target_parameters,
)


@dataclass
class MetaTrainResult:
    """Artifacts and history from training the adaptation generator."""

    generator: nn.Module
    best_state_dict: dict[str, Tensor]
    history: list[dict[str, Any]]
    total_time_seconds: float
    pairs_trained: int
    teacher_time_seconds: float = 0.0
    optimizer_steps: int = 0
    selected_epoch: int = 0


@frozen_sources
def train_adaptation_generator(
    models: dict[str, nn.Module],
    manifest: dict[str, Any],
    load_env_fn: Callable[[dict[str, Any], str], DatasetBundle],
    descriptor: GraphShiftDescriptor,
    generator: HeadAdapterGenerator,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    delta_l2_reg: float = 1e-3,
    max_environments: int | None = None,
    max_models: int | None = None,
    val_models: dict[str, nn.Module] | None = None,
    device: str = "cpu",
    verbose: bool = True,
    target_scope: str = "head",
    checkpoint_callback: Callable | None = None,
) -> MetaTrainResult:
    """Train the HeadAdapterGenerator across (pretrained_model, shifted_graph) pairs.

    Args:
        models: Mapping of model_id -> pretrained GNN module (meta-training split).
        manifest: Loaded environment manifest JSON.
        load_env_fn: Callable to load one environment by ID (e.g. load_shift_environment).
        descriptor: GraphShiftDescriptor to pool graph representations.
        generator: HeadAdapterGenerator MLP to optimize.
        epochs: Number of training sweeps over the pairs.
        lr: Learning rate for Adam optimizer.
        weight_decay: L2 weight decay for the generator.
        delta_l2_reg: Regularization penalty on the magnitude of the predicted delta.
        max_environments: Optional cap on the number of meta-train environments.
        max_models: Optional cap on the number of meta-train models.
        val_models: Optional mapping of validation models for validation monitoring.
        device: 'cpu' or 'cuda'.
        verbose: Whether to print progress to stdout.

    Returns:
        MetaTrainResult containing the trained generator, best weights, and metrics history.
    """
    conditioning_contract(descriptor, generator)
    training_options(epochs, lr, delta_l2_reg, max_models, max_environments)
    if weight_decay < 0 or not math.isfinite(weight_decay):
        raise ValueError("weight_decay must be finite and nonnegative")
    if val_models is not None:
        val_models = dict(list(val_models.items())[:max_models])
    total_start = time.perf_counter()
    generator = generator.to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=weight_decay)

    # Filter meta-training environments with family round-robin debug subsampling (not lineage/severity balancing)
    meta_envs = [e for e in manifest["environments"] if e["role"] == "meta_train"]
    meta_envs = subsample_environments(meta_envs, max_environments)

    # Select meta-training models
    model_items = list(models.items())
    if max_models is not None:
        model_items = model_items[:max_models]

    total_pairs = len(model_items) * len(meta_envs)
    if total_pairs == 0:
        raise ValueError("No valid (model, environment) pairs available for meta-training.")

    if verbose:
        print(f"Starting Generator Meta-Training:")
        print(f"  • Models: {len(model_items)}")
        print(f"  • Environments: {len(meta_envs)}")
        print(f"  • Total training pairs per epoch: {total_pairs}")
        print(f"  • Epochs: {epochs} | Device: {device}\n")

    history: list[dict[str, Any]] = []
    best_state_dict = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    selected_epoch = 0

    # Evaluate Epoch 0 unadapted baseline if val_models provided (do-no-harm baseline check)
    if val_models:
        generator.eval()
        val_envs = [e for e in manifest["environments"] if e["role"] == "validation"]
        val_envs = subsample_environments(val_envs, max_environments)
        if not val_envs:
            raise ValueError("Validation models require validation environments")
        baseline_records = []
        for v_env in val_envs:
            v_bundle = load_env_fn(manifest, v_env["id"])
            for v_mid, v_m in val_models.items():
                r = evaluate_adaptation_pair(
                    v_m, v_mid, v_bundle, v_env["id"], v_env, descriptor, generator, device=device, target_scope=target_scope
                )
                baseline_records.append(r)
        baseline_summary = summarize_records(baseline_records)
        best_val_loss = 1.0 - baseline_summary.get("mean_adapted_acc", 0.0)
        if verbose:
            print(f"  • Initial Candidate Val Acc (Epoch 0): {baseline_summary.get('mean_adapted_acc', 0.0)*100:.2f}% (Threshold for saving generator)\n")

    if checkpoint_callback:
        checkpoint_callback(best_state_dict, history)
    start_time = total_start

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        generator.train()
        train_losses = []
        ce_losses = []
        delta_norms = []

        # Iterate over all (model, env) combinations
        for env_meta in meta_envs:
            env_id = env_meta["id"]
            bundle = load_env_fn(manifest, env_id)

            unlabeled = bundle.unlabeled_input(env_id).to(device)
            train_idx = bundle.select(env_id, "train").to(device)
            train_y = bundle.training_labels(env_id).to(device)

            for model_id, model in model_items:
                model.eval()
                target_weights, schema = extract_target_parameters(model, scope=target_scope)
                target_weights = target_weights.to(device)

                # 1. Compute size-invariant shift descriptor
                desc = descriptor(model, unlabeled).to(device)

                # 2. Generator predicts delta
                delta = generator(desc, target_weights)

                # 3. Forward pass with dynamic delta through adapted GNN
                logits = adapted_forward(model, unlabeled, delta, schema)

                # 4. Meta-loss: Cross-entropy on meta-training labels + L2 penalty on delta
                ce_loss = F.cross_entropy(logits[train_idx], train_y)
                l2_penalty = delta_l2_reg * torch.sum(delta ** 2)
                loss = ce_loss + l2_penalty

                # 5. Backpropagate into the generator ONLY
                optimizer.zero_grad()
                checked_backward(loss, generator.parameters())
                optimizer.step()

                train_losses.append(loss.item())
                ce_losses.append(ce_loss.item())
                delta_norms.append(torch.linalg.vector_norm(delta).item())

        mean_train_loss = float(sum(train_losses) / len(train_losses))
        mean_ce_loss = float(sum(ce_losses) / len(ce_losses))
        mean_norm = float(sum(delta_norms) / len(delta_norms))

        # Validation check or training-loss selection
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
                        v_m, v_mid, v_bundle, v_env["id"], v_env, descriptor, generator, device=device, target_scope=target_scope
                    )
                    val_records.append(r)

            val_summary = summarize_records(val_records)
            val_gain = val_summary.get("mean_gain", 0.0)
            val_loss = 1.0 - val_summary.get("mean_adapted_acc", 0.0)

            # Strictly require outperforming the unadapted baseline (do-no-harm)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                selected_epoch = epoch
                best_state_dict = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        else:
            # When no validation models are supplied, select by training loss so work is not discarded
            if mean_train_loss < best_train_loss:
                best_train_loss = mean_train_loss
                selected_epoch = epoch
                best_state_dict = {k: v.cpu().clone() for k, v in generator.state_dict().items()}

        epoch_time = time.time() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "ce_loss": mean_ce_loss,
            "delta_norm": mean_norm,
            "val_loss": val_loss,
            "val_gain": val_gain,
            "time_seconds": epoch_time,
        }
        epoch_record["validation_accuracy_error"] = val_loss
        history.append(epoch_record)
        if checkpoint_callback:
            checkpoint_callback(best_state_dict, history)

        if verbose:
            val_msg = f" | Val Acc Gain: {val_gain * 100:+.2f} pp" if val_gain is not None else ""
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {mean_train_loss:.4f} "
                f"(CE: {mean_ce_loss:.4f}) | ΔNorm: {mean_norm:.4f}{val_msg} | Time: {epoch_time:.2f}s"
            )

    total_time = time.perf_counter() - start_time
    if verbose:
        print(f"\nMeta-training finished in {total_time:.2f}s across {total_pairs * epochs} step evaluations.")

    return MetaTrainResult(
        generator=generator,
        best_state_dict=best_state_dict,
        history=history,
        total_time_seconds=total_time,
        pairs_trained=total_pairs,
        optimizer_steps=total_pairs * epochs,
        selected_epoch=selected_epoch,
    )
