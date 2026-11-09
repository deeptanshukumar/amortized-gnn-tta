"""A fixed practice-shift-trained GNN control; no online adaptation."""

import copy
import math
import time
import torch
from torch.nn import functional as F
from ._validation import checked_backward, positive_int, training_options
from ..evaluation import score_logits


def fit_practice_model(
    source,
    manifest,
    load_env_fn,
    *,
    epochs=5,
    lr=0.001,
    device="cpu",
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
):
    """Fine-tune on allowed practice labels, select on validation, then freeze.

    This is an offline robust-training control. It is not source-only inference,
    target-label adaptation, or a learned generator. Its extra training cost must
    be reported separately from its prediction-only online cost.
    """
    training_options(epochs, lr, 0.0)
    if early_stopping_patience is not None:
        positive_int(early_stopping_patience, "early_stopping_patience")
    if (
        isinstance(early_stopping_min_delta, bool)
        or not math.isfinite(early_stopping_min_delta)
        or early_stopping_min_delta < 0
    ):
        raise ValueError("early_stopping_min_delta must be finite and nonnegative")
    model = copy.deepcopy(source).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()

    def evaluate():
        accuracies = []
        model.eval()
        with torch.no_grad():
            for env in manifest["environments"]:
                if env["role"] != "validation":
                    continue
                bundle = load_env_fn(manifest, env["id"])
                graph = bundle.unlabeled_input(env["id"]).to(device)
                predictions = model(graph.x, graph.edge_index)
                accuracies.append(
                    score_logits(
                        predictions,
                        bundle.select(env["id"], "eval"),
                        bundle.evaluation_labels(env["id"], "eval"),
                    ).accuracy
                )
        if not accuracies:
            raise ValueError("Validation environments required")
        return sum(accuracies) / len(accuracies)

    best_accuracy = evaluate()
    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = [dict(epoch=0, validation_accuracy=best_accuracy)]
    selected = 0
    stopped_epoch = 0
    early_stopped = False
    stale_epochs = 0
    best_trained_accuracy = float("-inf")
    steps = 0
    for epoch in range(1, epochs + 1):
        losses = []
        for env in manifest["environments"]:
            if env["role"] != "meta_train":
                continue
            bundle = load_env_fn(manifest, env["id"])
            graph = bundle.unlabeled_input(env["id"]).to(device)
            indices = bundle.select(env["id"], "train").to(device)
            labels = bundle.training_labels(env["id"]).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(graph.x, graph.edge_index)[indices], labels)
            checked_backward(loss, model.parameters())
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("Practice environments required")
        accuracy = evaluate()
        if accuracy > best_accuracy + early_stopping_min_delta:
            best_accuracy = accuracy
            selected = epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if accuracy > best_trained_accuracy + early_stopping_min_delta:
            best_trained_accuracy = accuracy
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            dict(
                epoch=epoch,
                validation_accuracy=accuracy,
                train_loss=sum(losses) / len(losses),
            )
        )
        if (
            early_stopping_patience is not None
            and stale_epochs >= early_stopping_patience
        ):
            stopped_epoch = epoch
            early_stopped = True
            break
    model.load_state_dict(best)
    model.eval()
    return model, dict(
        history=history,
        selected_epoch=selected,
        stopped_epoch=stopped_epoch,
        early_stopped=early_stopped,
        optimizer_steps=steps,
        fit_seconds=time.perf_counter() - start,
    )
