"""Teacher-matched hypernetwork control, independent of target labels.

This control regresses the same standardized teacher coordinates used by the
conditional diffusion model. Factor MSE is a training loss, not weight-space or
prediction equivalence: LoRA factors have non-unique coordinates.
"""

import math
import time
import torch
from ._validation import (
    checked_backward,
    conditioning_contract,
    positive_int,
    training_options,
    frozen_sources,
)
from .evaluator import evaluate_adaptation_pair, summarize_records
from .meta_trainer import MetaTrainResult


@frozen_sources
def train_teacher_hypernetwork(
    models,
    manifest,
    load_env_fn,
    descriptor,
    generator,
    *,
    targets,
    val_models,
    epochs=5,
    lr=0.001,
    target_scope="head",
    device="cpu",
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
):
    training_options(epochs, lr, 0.0)
    if early_stopping_patience is not None:
        positive_int(early_stopping_patience, "early_stopping_patience")
    if (
        isinstance(early_stopping_min_delta, bool)
        or not math.isfinite(early_stopping_min_delta)
        or early_stopping_min_delta < 0
    ):
        raise ValueError("early_stopping_min_delta must be finite and nonnegative")
    conditioning_contract(descriptor, generator)
    expected = {
        (mid, e["id"])
        for mid in models
        for e in manifest["environments"]
        if e["role"] == "meta_train"
    }
    actual = [(r.model_id, r.environment_id) for r in targets]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Teacher identities disagree with training pairs")
    if not val_models:
        raise ValueError("Validation models required for teacher control")
    start = time.perf_counter()
    generator.to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    target_vectors = torch.stack([r.target_delta for r in targets]).to(device)
    scale = (
        target_vectors.std(dim=0).clamp_min(0.001)
        if len(targets) > 1
        else torch.ones_like(target_vectors[0])
    )
    if not torch.isfinite(target_vectors).all():
        raise ValueError("Nonfinite teachers")
    core = getattr(generator, "core", generator)

    def validation_error():
        records = []
        for env in manifest["environments"]:
            if env["role"] != "validation":
                continue
            bundle = load_env_fn(manifest, env["id"])
            for mid, model in val_models.items():
                records.append(
                    evaluate_adaptation_pair(
                        model,
                        mid,
                        bundle,
                        env["id"],
                        env,
                        descriptor,
                        generator,
                        device=device,
                        target_scope=target_scope,
                    )
                )
        if not records:
            raise ValueError("Validation environments required")
        return 1 - summarize_records(records)["mean_adapted_acc"]

    best_error = validation_error()
    best = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}
    history = []
    selected_epoch = 0
    stopped_epoch = 0
    early_stopped = False
    stale_epochs = 0
    best_trained_error = float("inf")
    steps = 0
    for epoch in range(1, epochs + 1):
        generator.train()
        losses = []
        order = torch.randperm(len(targets))
        for offset in range(0, len(targets), 16):
            batch = order[offset : offset + 16]
            d = torch.stack([targets[int(i)].descriptor for i in batch]).to(device)
            w = torch.stack([targets[int(i)].head_weights for i in batch]).to(device)
            prediction = core(d, w if core.include_head_weights else None)
            loss = (
                ((prediction - target_vectors[batch.to(device)]) / scale)
                .square()
                .mean()
            )
            optimizer.zero_grad(set_to_none=True)
            checked_backward(loss, generator.parameters())
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
        error = validation_error()
        if error < best_error - early_stopping_min_delta:
            best_error = error
            selected_epoch = epoch
            best = {
                k: v.detach().cpu().clone() for k, v in generator.state_dict().items()
            }
        if error < best_trained_error - early_stopping_min_delta:
            best_trained_error = error
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            dict(
                epoch=epoch,
                train_loss=sum(losses) / len(losses),
                validation_accuracy_error=error,
            )
        )
        if (
            early_stopping_patience is not None
            and stale_epochs >= early_stopping_patience
        ):
            stopped_epoch = epoch
            early_stopped = True
            break
    return MetaTrainResult(
        generator,
        best,
        history,
        time.perf_counter() - start,
        len(targets),
        0.0,
        steps,
        selected_epoch,
        stopped_epoch,
        early_stopped,
    )
