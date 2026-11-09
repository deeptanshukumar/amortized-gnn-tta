"""A fixed practice-shift-trained GNN control; no online adaptation."""

import copy
import time
import torch
from torch.nn import functional as F
from ._validation import checked_backward, training_options
from ..evaluation import score_logits


def fit_practice_model(
    source, manifest, load_env_fn, *, epochs=5, lr=0.001, device="cpu"
):
    """Fine-tune on allowed practice labels, select on validation, then freeze.

    This is an offline robust-training control. It is not source-only inference,
    target-label adaptation, or a learned generator. Its extra training cost must
    be reported separately from its prediction-only online cost.
    """
    training_options(epochs, lr, 0.0)
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
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            selected = epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(
            dict(
                epoch=epoch,
                validation_accuracy=accuracy,
                train_loss=sum(losses) / len(losses),
            )
        )
    model.load_state_dict(best)
    model.eval()
    return model, dict(
        history=history,
        selected_epoch=selected,
        optimizer_steps=steps,
        fit_seconds=time.perf_counter() - start,
    )
