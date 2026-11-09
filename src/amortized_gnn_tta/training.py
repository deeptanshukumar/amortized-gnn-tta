"""Source-only full-graph training and portable inference checkpoints.

The trainer receives train/validation supervision only. Source-test and target
labels are absent from its input type; final evaluation is a separate operation.
The model factory keeps this loop independent of GCN/GAT/GraphSAGE construction.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib.metadata
import math
from pathlib import Path
import platform
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data

from .data import DatasetBundle
from .evaluation import select_device, validate_input, validate_logits


@dataclass
class SourceTrainingData:
    graph: Data
    train_indices: Tensor
    train_labels: Tensor
    val_indices: Tensor
    val_labels: Tensor
    class_names: tuple[str, ...]
    provenance: dict[str, Any]


def source_training_data(bundle: DatasetBundle, graph_id: str) -> SourceTrainingData:
    """Extract the source view without reading test labels or any target record."""
    record = bundle.graphs[graph_id]
    if record.role != "source_train":
        raise ValueError("Source training requires a source_train environment")
    return SourceTrainingData(
        bundle.unlabeled_input(graph_id), bundle.select(graph_id, "train"),
        bundle.training_labels(graph_id), bundle.select(graph_id, "val"),
        bundle.evaluation_labels(graph_id, "val"), bundle.class_names,
        {"graph_id": graph_id, "dataset_metadata": deepcopy(bundle.metadata),
         "graph_metadata": deepcopy(record.metadata)},
    )


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 200
    patience: int = 30
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    model_seed: int = 31

    def __post_init__(self) -> None:
        for name in ("max_epochs", "patience", "model_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.max_epochs < 1 or self.patience < 1 or not 0 <= self.model_seed < 2**32:
            raise ValueError("Epochs/patience must be positive and seed in [0,2**32)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")


def seed_model(seed: int) -> None:
    """Seed initialization/dropout; dataset RNGs remain separately configured.

    This does not promise identical numerical results across hardware/versions or
    force deterministic graph kernels on backends where those may be unavailable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_source_training_data(data: SourceTrainingData) -> None:
    """Validate a source supervision view without access to test/target records."""
    validate_input(data.graph)
    n, c = data.graph.num_nodes, len(data.class_names)
    if c < 2 or len(set(data.class_names)) != c:
        raise ValueError("A source view needs a valid class schema")
    for indices, labels in ((data.train_indices, data.train_labels), (data.val_indices, data.val_labels)):
        if indices.device.type != "cpu" or labels.device.type != "cpu":
            raise ValueError("Source view indices/labels must be stored on CPU")
        if indices.dtype != torch.long or indices.ndim != 1 or not indices.numel():
            raise ValueError("Train/validation indices must be nonempty int64 vectors")
        if indices.min() < 0 or indices.max() >= n or len(indices.unique()) != len(indices):
            raise ValueError("Train/validation indices must be unique and in bounds")
        if labels.dtype != torch.long or labels.shape != indices.shape or labels.min() < 0 or labels.max() >= c:
            raise ValueError("Source labels must align with indices and class schema")
    if torch.isin(data.train_indices, data.val_indices).any():
        raise ValueError("Source train and validation supervision must not overlap")


@dataclass
class TrainingResult:
    model: nn.Module
    config: TrainingConfig
    history: list[dict[str, float | int]]
    best_epoch: int
    best_val_loss: float
    device: str
    source: SourceTrainingData


def fit_source(model_factory: Callable[[], nn.Module], data: SourceTrainingData,
               config: TrainingConfig = TrainingConfig(), *, device: str | torch.device = "auto") -> TrainingResult:
    """Fit Adam on source train labels, restore minimum source-validation CE.

    Strict improvement resets patience; ties retain the earlier checkpoint. Train
    loss is the pre-step dropout objective; val loss is post-step eval-mode CE.
    The optimizer applies weight decay to all trainable parameters. This is a
    documented small baseline, not a reproduction of the original GCN tuning.
    """
    validate_source_training_data(data)
    chosen = select_device(device)
    seed_model(config.model_seed)  # Must precede the factory's weight initialization.
    model = model_factory().to(chosen)
    graph = data.graph.clone().to(chosen)
    train_idx, val_idx = data.train_indices.to(chosen), data.val_indices.to(chosen)
    train_y, val_y = data.train_labels.to(chosen), data.val_labels.to(chosen)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state, best_epoch, best_loss, stale = None, 0, math.inf, 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.x, graph.edge_index)
        validate_logits(logits, graph.num_nodes, len(data.class_names))
        loss = F.cross_entropy(logits[train_idx], train_y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite source-training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite source-training gradients")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(graph.x, graph.edge_index)
            validate_logits(val_logits, graph.num_nodes, len(data.class_names))
            val_loss = float(F.cross_entropy(val_logits[val_idx], val_y))
            if not math.isfinite(val_loss):
                raise RuntimeError("Nonfinite source-validation loss")
            val_accuracy = int((val_logits[val_idx].argmax(1) == val_y).sum()) / len(val_y)
        history.append({"epoch": epoch, "train_loss": float(loss.detach()),
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            # state_dict alone aliases live tensors, so snapshot an actual copy.
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return TrainingResult(model, config, history, best_epoch, best_loss, str(chosen), data)


def software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                **{name: importlib.metadata.version(name) for name in ("numpy", "torch-geometric")}}
    for name in ("ogb", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass  # Optional tooling must not become a core import dependency.
    return versions


def save_checkpoint(path: Path, result: TrainingResult, model_spec: dict[str, Any]) -> None:
    """Save CPU weights and audit metadata, without overwriting an existing file.

    This is an inference checkpoint, not an exact training-resumption snapshot:
    optimizer/RNG states are intentionally not saved in this small milestone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "state_dict": {k: v.detach().cpu().clone() for k, v in result.model.state_dict().items()},
        "model_spec": deepcopy(model_spec), "training_config": asdict(result.config),
        "best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss,
        "selection": "minimum source-validation cross-entropy; ties keep earliest",
        "history": result.history, "training_device": result.device,
        "source": {**deepcopy(result.source.provenance), "class_names": list(result.source.class_names),
                   "train_indices": result.source.train_indices.clone(), "val_indices": result.source.val_indices.clone()},
        "versions": software_versions(),
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_checkpoint(path: Path, model_factory: Callable[[dict[str, Any]], nn.Module],
                    *, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Restore using caller-supplied architecture construction; CPU map first.

    Only tensor/plain metadata checkpoints are supported; no pickled model object
    or dynamically imported architecture registry is needed.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = model_factory(payload["model_spec"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(select_device(device)).eval()
    return model, payload
