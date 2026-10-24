"""Label-free inference and explicit offline single-label scoring."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data


def select_device(requested: str | torch.device = "auto") -> torch.device:
    """Auto chooses CUDA when available, otherwise CPU; MPS is explicit opt-in.

    Backend availability does not guarantee graph-kernel support. Unsupported
    operations fail visibly; there is no silent device fallback during a run.
    """
    if str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Use cpu, cuda, mps, or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def validate_input(graph: Data) -> None:
    """The initial runner handles unweighted graphs, never silently dropping data."""
    if set(graph.keys()) - {"x", "edge_index", "num_nodes"}:
        raise ValueError("Runner accepts only label-free, unweighted x/edge_index/num_nodes")
    if graph.x is None or graph.x.dtype != torch.float32 or graph.x.ndim != 2:
        raise ValueError("x must be float32 [N,F]")
    if graph.edge_index is None or graph.edge_index.dtype != torch.long:
        raise ValueError("edge_index must be int64 [2,E]")
    graph.validate(raise_on_error=True)
    if graph.x.shape[0] != graph.num_nodes or not torch.isfinite(graph.x).all():
        raise ValueError("Features must be finite and aligned with num_nodes")


def validate_logits(logits: Tensor, n: int, classes: int | None = None) -> None:
    if (logits.ndim != 2 or logits.shape[0] != n or logits.shape[1] < 2 or
            (classes is not None and logits.shape[1] != classes) or
            not torch.isfinite(logits).all()):
        raise ValueError("Model must return finite raw class logits of shape [N,C]")


def predict(model: nn.Module, graph: Data, *, device: str | torch.device | None = None) -> Tensor:
    """Run eval-mode inference without labels, restoring the previous train flag.

    Returns CPU logits. Moving a cloned input prevents modification of loader data.
    An explicit device moves the model too; otherwise its current device is used.
    """
    validate_input(graph)
    chosen = select_device(device) if device is not None else next(model.parameters()).device
    model.to(chosen)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            inputs = graph.clone().to(chosen)
            logits = model(inputs.x, inputs.edge_index)
            validate_logits(logits, graph.num_nodes)
            return logits.cpu()
    finally:
        model.train(was_training)


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    num_nodes: int


def score_logits(logits: Tensor, indices: Tensor, labels: Tensor) -> ClassificationMetrics:
    """Score selected rows against explicitly provided, already-selected labels."""
    validate_logits(logits, logits.shape[0])
    if indices.dtype != torch.long or indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("Scoring needs a nonempty int64 index vector")
    if indices.min() < 0 or indices.max() >= len(logits) or indices.unique().numel() != indices.numel():
        raise ValueError("Scoring indices must be unique and in range")
    if labels.dtype != torch.long or labels.shape != indices.shape:
        raise ValueError("Labels must be int64 and aligned with selected indices")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("Labels are outside the model's class schema")
    selected = logits.detach().cpu()[indices.cpu()]
    labels = labels.cpu()
    return ClassificationMetrics(
        loss=float(F.cross_entropy(selected, labels)),
        accuracy=int((selected.argmax(dim=1) == labels).sum()) / len(indices),
        num_nodes=len(indices),
    )


def evaluate(model: nn.Module, graph: Data, indices: Tensor, labels: Tensor,
             *, device: str | torch.device | None = None) -> ClassificationMetrics:
    """Offline scoring only; labels never enter the model's prediction function."""
    return score_logits(predict(model, graph, device=device), indices, labels)
