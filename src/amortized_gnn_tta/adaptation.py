"""Episodic parameter-efficient updates with explicit supervision boundaries.

LoRA algebra/initialization follows microsoft/LoRA loralib/layers.py (MIT).
Entropy and information-maximization objectives follow Tent and SHOT respectively
(MIT); changing their update sites makes these objective baselines, not complete
reproductions. See THIRD_PARTY_NOTICES.md and docs/DECISIONS.md, decision D8.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F
from torch_geometric.data import Data

from .data import DatasetBundle
from .data.api import graph_fingerprint
from .evaluation import select_device, validate_input, validate_logits
from .models import GAT, GCN, GraphSAGE


@dataclass(frozen=True)
class PEFTConfig:
    """Head = entire conv2; LoRA = matrix transforms in both graph layers.

    Output rank 1 remains strictly below the two-class matrix dimension. Hidden
    rank 4 is explicit, never silently clipped for another model size. LoRA does
    not update biases or GAT attention vectors. alpha/r controls each update.
    """

    method: Literal["head", "lora"] = "head"
    backbone_rank: int = 4
    head_rank: int = 1
    backbone_alpha: float = 4.0
    head_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.method not in {"head", "lora"}:
            raise ValueError("Use head or lora adaptation")
        if any(type(v) is not int or v < 1 for v in (self.backbone_rank, self.head_rank)):
            raise ValueError("LoRA ranks must be positive integers")
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (self.backbone_alpha, self.head_alpha)):
            raise ValueError("LoRA alphas must be finite and positive")


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int = 100
    learning_rate: float = 0.01
    patience: int = 20
    initialization_seed: int = 801
    diversity_weight: float = 1.0

    def __post_init__(self) -> None:
        if (type(self.steps) is not int or self.steps < 1 or type(self.patience) is not int
                or self.patience < 1 or type(self.initialization_seed) is not int
                or not 0 <= self.initialization_seed < 2**32):
            raise ValueError("Use positive steps/patience and an integer initialization seed")
        if (isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate)
                or self.learning_rate <= 0 or isinstance(self.diversity_weight, bool)
                or not math.isfinite(self.diversity_weight) or self.diversity_weight < 0):
            raise ValueError("Use finite positive learning rate and nonnegative diversity weight")


def adaptation_targets(model: nn.Module, method: str) -> tuple[str, ...]:
    """Explicit maps for our homogeneous two-layer PyG models; reject others."""
    if not isinstance(model, (GCN, GAT, GraphSAGE)):
        raise ValueError("Adaptation supports our GCN, GAT and GraphSAGE models")
    if method == "head":
        return tuple(name for name, _ in model.named_parameters() if name.startswith("conv2."))
    if method != "lora":
        raise ValueError("Unknown adaptation method")
    if isinstance(model, GraphSAGE):
        return ("conv1.lin_l.weight", "conv1.lin_r.weight", "conv2.lin_l.weight", "conv2.lin_r.weight")
    return ("conv1.lin.weight", "conv2.lin.weight")


class WeightAdapter(nn.Module):
    """Frozen model plus differentiable updates; caller's model is never mutated.

    functional_call preserves the original PyG computation, including attention
    and graph normalization. Only substituted parameter values differ. All model
    dropout remains disabled during adaptation; eval() does NOT disable gradients.
    Each episode starts from zero effective change. No global RNG is consumed.
    """

    def __init__(self, model: nn.Module, config: PEFTConfig, *, seed: int = 801):
        super().__init__()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Initialization seed must be in [0,2**32)")
        self.config = config
        self.target_names = adaptation_targets(model, config.method)
        self.base = deepcopy(model).cpu().requires_grad_(False).eval()
        self.offsets, self.a, self.b = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        self.scales: dict[str, float] = {}
        self._head_cache = None
        generator = torch.Generator(device="cpu").manual_seed(seed)
        params = dict(self.base.named_parameters())
        for i, name in enumerate(self.target_names):
            weight, key = params[name], f"p{i}"
            if config.method == "head":
                self.offsets[key] = nn.Parameter(torch.zeros_like(weight))
            else:
                rank = config.backbone_rank if name.startswith("conv1.") else config.head_rank
                alpha = config.backbone_alpha if name.startswith("conv1.") else config.head_alpha
                if weight.ndim != 2 or rank >= min(weight.shape):
                    raise ValueError(f"Rank {rank} must be strictly below both dimensions of {name}")
                a = torch.empty(rank, weight.shape[1], dtype=weight.dtype)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
                self.a[key] = nn.Parameter(a)
                self.b[key] = nn.Parameter(torch.zeros(weight.shape[0], rank, dtype=weight.dtype))
                self.scales[key] = alpha / rank

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def updates(self) -> dict[str, Tensor]:
        return {name: self.offsets[f"p{i}"] if self.config.method == "head" else
                (self.b[f"p{i}"] @ self.a[f"p{i}"]) * self.scales[f"p{i}"]
                for i, name in enumerate(self.target_names)}

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        original = dict(self.base.named_parameters())
        replacements = {name: original[name] + change for name, change in self.updates().items()}
        cache = self._head_cache
        if (cache is not None and x is cache[0] and edge_index is cache[1]
                and x._version == cache[2] and edge_index._version == cache[3]):
            # conv1 is frozen and dropout is off: its representation is constant
            # during this HEAD-only episode. Different/mutated inputs fall back
            # to the complete model, preventing reuse on a source/other graph.
            head = {name.removeprefix("conv2."): value for name, value in replacements.items()}
            return functional_call(self.base.conv2, head, (cache[4], edge_index), strict=False)
        return functional_call(self.base, replacements, (x, edge_index), strict=False)

    def prepare_episode(self, x: Tensor, edge_index: Tensor) -> None:
        """Cache only a frozen head-only backbone for these exact input tensors.

        This is a computation cache, never a cached source adjacency or label
        statistic. LoRA changes conv1 and deliberately takes the full path.
        Exported/merged models contain no episode cache.
        """
        self._head_cache = None
        if self.config.method == "head" and not x.requires_grad:
            with torch.no_grad():
                features = self.base.encode(x, edge_index).detach()
            self._head_cache = (x, edge_index, x._version, edge_index._version, features)

    def trainable_state(self) -> dict[str, Tensor]:
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def restore_trainable_state(self, state: dict[str, Tensor]) -> None:
        params = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(params):
            raise ValueError("Adapter-state keys disagree")
        with torch.no_grad():
            for name, p in params.items():
                if state[name].shape != p.shape or not torch.isfinite(state[name]).all():
                    raise ValueError("Invalid adapter tensor")
                p.copy_(state[name])

    def merged_model(self) -> nn.Module:
        """Materialize ordinary architecture/state keys for inference and deltas."""
        result = deepcopy(self.base)
        params = dict(result.named_parameters())
        with torch.no_grad():
            for name, delta in self.updates().items():
                params[name].add_(delta)
        return result.eval()


@dataclass(frozen=True)
class DevelopmentSupervision:
    """Only permitted meta-training train/validation labels, never test labels."""

    graph_sha256: str
    environment_id: str
    train_indices: Tensor
    train_labels: Tensor
    val_indices: Tensor
    val_labels: Tensor


def development_supervision(bundle: DatasetBundle, environment_id: str) -> DevelopmentSupervision:
    record = bundle.graphs[environment_id]
    if record.role != "meta_train":
        raise ValueError("Supervised development adaptation requires meta_train, never held-out environments")
    return DevelopmentSupervision(
        graph_fingerprint(bundle.unlabeled_input(environment_id)), environment_id,
        bundle.select(environment_id, "train"), bundle.training_labels(environment_id),
        bundle.select(environment_id, "val"), bundle.evaluation_labels(environment_id, "val"),
    )


def _validate_supervision(data: DevelopmentSupervision, graph: Data, classes: int) -> None:
    if data.graph_sha256 != graph_fingerprint(graph):
        raise ValueError("Development supervision belongs to a different graph")
    for indices, labels in ((data.train_indices, data.train_labels), (data.val_indices, data.val_labels)):
        if (indices.device.type != "cpu" or labels.device.type != "cpu" or indices.dtype != torch.long
                or indices.ndim != 1 or not indices.numel() or indices.min() < 0
                or indices.max() >= graph.num_nodes or len(indices.unique()) != len(indices)
                or labels.dtype != torch.long or labels.shape != indices.shape
                or labels.min() < 0 or labels.max() >= classes):
            raise ValueError("Malformed development supervision")
    if torch.isin(data.train_indices, data.val_indices).any():
        raise ValueError("Development train/validation nodes overlap")


def unlabeled_objective(logits: Tensor, objective: str, *, diversity_weight: float = 1.0) -> Tensor:
    """Mean prediction entropy, optionally minus entropy of mean probabilities.

    The SHOT-inspired diversity term discourages one-class collapse; it favors a
    balanced marginal and is NOT a universal correction for label imbalance.
    Stable log_softmax/logsumexp avoid the epsilon-biased reference arithmetic
    and undefined gradients at an underflowed zero marginal probability.
    No pseudo-label clustering, normalization adaptation or source access occurs.
    """
    if objective not in {"entropy", "information_maximization"}:
        raise ValueError("Use entropy or information_maximization")
    log_probabilities = logits.log_softmax(dim=1)
    probabilities = log_probabilities.exp()
    conditional = -(probabilities * log_probabilities).sum(dim=1).mean()
    if objective == "entropy":
        return conditional
    log_marginal = torch.logsumexp(log_probabilities, dim=0) - math.log(len(logits))
    return conditional + diversity_weight * (log_marginal.exp() * log_marginal).sum()


@dataclass
class AdaptationResult:
    adapter: WeightAdapter
    model: nn.Module
    config: AdaptationConfig
    objective: str
    selected_step: int
    history: list[dict[str, float | int]]
    selection: str


def fit_supervised_development(model: nn.Module, graph: Data, supervision: DevelopmentSupervision,
                               peft: PEFTConfig, config: AdaptationConfig = AdaptationConfig(), *,
                               device: str | torch.device = "cpu") -> AdaptationResult:
    """Supervised development reference; select validation CE, including step 0."""
    return _fit(model, graph, peft, config, "supervised_development", supervision, device)


def fit_unlabeled(model: nn.Module, graph: Data, peft: PEFTConfig,
                  config: AdaptationConfig = AdaptationConfig(), *,
                  objective: str = "information_maximization",
                  device: str | torch.device = "cpu") -> AdaptationResult:
    """Fixed-budget, source-free episode. This signature accepts NO labels."""
    if objective not in {"entropy", "information_maximization"}:
        raise ValueError("Unknown unlabeled objective")
    return _fit(model, graph, peft, config, objective, None, device)


def _fit(model, graph, peft, config, objective, supervision, device) -> AdaptationResult:
    validate_input(graph)
    if supervision is not None:
        _validate_supervision(supervision, graph, model.config.out_channels)
    chosen = select_device(device)
    adapter = WeightAdapter(model, peft, seed=config.initialization_seed).to(chosen).train()
    inputs = graph.clone().to(chosen)
    adapter.prepare_episode(inputs.x, inputs.edge_index)
    optimizer = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad],
                                 lr=config.learning_rate, weight_decay=0.0)
    train_idx = supervision.train_indices.to(chosen) if supervision else None
    train_y = supervision.train_labels.to(chosen) if supervision else None
    val_idx = supervision.val_indices.to(chosen) if supervision else None
    val_y = supervision.val_labels.to(chosen) if supervision else None
    best_state, best_step, stale = adapter.trainable_state(), 0, 0
    with torch.no_grad():
        initial = adapter(inputs.x, inputs.edge_index)
        validate_logits(initial, graph.num_nodes, model.config.out_channels)
        best_loss = float(F.cross_entropy(initial[val_idx], val_y)) if supervision else math.inf
    if supervision and not math.isfinite(best_loss):
        raise RuntimeError("Nonfinite initial development-validation loss")
    history = [{"step": 0, "validation_loss": best_loss}] if supervision else []
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = adapter(inputs.x, inputs.edge_index)
        validate_logits(logits, graph.num_nodes, model.config.out_channels)
        loss = (F.cross_entropy(logits[train_idx], train_y) if supervision else
                unlabeled_objective(logits, objective, diversity_weight=config.diversity_weight))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite adaptation objective")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise RuntimeError("Nonfinite adaptation gradients")
        optimizer.step()
        row = {"step": step, "objective_before_update": float(loss.detach())}
        if supervision:
            with torch.no_grad():
                val_logits = adapter(inputs.x, inputs.edge_index)
                validate_logits(val_logits, graph.num_nodes, model.config.out_channels)
                val_loss = float(F.cross_entropy(val_logits[val_idx], val_y))
                if not math.isfinite(val_loss):
                    raise RuntimeError("Nonfinite development-validation loss")
            row["validation_loss"] = val_loss
            if val_loss < best_loss:
                best_loss, best_step, best_state, stale = val_loss, step, adapter.trainable_state(), 0
            else:
                stale += 1
        history.append(row)
        if supervision and stale >= config.patience:
            break
    if supervision:
        adapter.restore_trainable_state(best_state)
    else:
        best_step = config.steps
    merged = adapter.merged_model()
    with torch.no_grad():
        validate_logits(merged(inputs.x, inputs.edge_index), graph.num_nodes, model.config.out_channels)
    return AdaptationResult(adapter.eval(), merged, config, objective, best_step, history,
                            "minimum development-validation CE, including step 0" if supervision else
                            "final fixed-budget step; no label-based selection")
