"""Adaptation generator (hypernetwork) and size-invariant graph descriptor.

This module provides:
1. GraphShiftDescriptor: extracts fixed-size summary statistics from node
   representations produced by model.encode(x, edge_index).
2. HeadAdapterGenerator: an MLP hypernetwork that conditions on the graph descriptor
   and the GNN's current head weights to predict parameter updates (ΔW, Δb).
3. Helper functions to extract head parameters, apply deltas in memory, and
   execute functional forward passes with gradient flow during meta-training.
"""

from collections.abc import Sequence
import copy

import math
import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch_geometric.data import Data


# Public compatibility exports; implementations live in descriptors.py.
from .descriptors import (
    GraphShiftDescriptor, extract_compact_weight_descriptor,
    extract_hidden_representation_descriptor, extract_predictive_uncertainty_descriptor,
    extract_target_graph_descriptor,
)
from ._validation import finite, positive_int
from ..evaluation import validate_logits


def get_head_module(model: nn.Module) -> nn.Module:
    """Return the final classification layer (conv2) of the GNN."""
    if hasattr(model, "conv2"):
        return model.conv2
    raise ValueError(f"Model {type(model)} has no recognized classification head 'conv2'")


def extract_target_parameters(
    model: nn.Module,
    scope: str = "head",
) -> tuple[Tensor, list[tuple[str, tuple[int, ...]]]]:
    """Flatten model parameters according to the requested adaptation scope.

    Args:
        model: GNN model instance.
        scope: Target adaptation scope:
            - "head": Classification head only (conv2.*, legacy default)
            - "backbone": Encoder backbone only (conv1.*)
            - "all": All trainable parameters (conv1.* and conv2.*)

    Returns:
        flat_params: 1D Tensor containing the flattened parameter values.
        schema: List of (param_name, shape) tuples for unflattening.
    """
    if scope not in ("head", "backbone", "all"):
        raise ValueError(f"Unknown scope '{scope}'. Expected 'head', 'backbone', or 'all'.")

    if scope == "head":
        # Keep exact legacy relative-name schema for head
        head = get_head_module(model)
        params = list(head.named_parameters())
        if not params:
            raise ValueError("Classification head contains no trainable parameters")
        schema = [(name, tuple(p.shape)) for name, p in params]
        flat = torch.cat([p.detach().flatten() for _, p in params])
        return flat, schema

    params = []
    for name, p in model.named_parameters():
        if scope == "backbone" and name.startswith("conv1."):
            params.append((name, p))
        elif scope == "all":
            params.append((name, p))

    if not params:
        raise ValueError(f"No trainable parameters found for scope '{scope}' in model {type(model)}")

    schema = [(name, tuple(p.shape)) for name, p in params]
    flat = torch.cat([p.detach().flatten() for _, p in params])
    return flat, schema


def extract_head_parameters(model: nn.Module) -> tuple[Tensor, list[tuple[str, tuple[int, ...]]]]:
    """Legacy helper: Flatten all parameters belonging to the model's classification head (conv2)."""
    return extract_target_parameters(model, scope="head")


def unflatten_head_delta(delta: Tensor, schema: Sequence[tuple[str, tuple[int, ...]]]) -> dict[str, Tensor]:
    """Slice a flat delta tensor into a dictionary of named parameter tensors."""
    if delta.ndim != 1:
        raise ValueError("Delta must be one-dimensional")
    finite(delta, "delta")
    result, offset = {}, 0
    if not schema:
        raise ValueError("Empty update schema")
    for name, shape in schema:
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("Schema names must be unique nonempty strings")
        if any(type(dim) is not int or dim < 1 for dim in shape):
            raise ValueError("Invalid parameter shape")
        count = math.prod(shape)
        if offset + count > delta.numel():
            raise ValueError("Delta length disagrees with schema")
        result[name] = delta[offset:offset + count].view(shape)
        offset += count
    if offset != delta.numel():
        raise ValueError("Delta length disagrees with schema")
    return result


def _resolved_updates(model, delta, schema):
    updates = unflatten_head_delta(delta, schema)
    parameters = dict(model.named_parameters())
    resolved = {}
    for name, value in updates.items():
        # Relative names are exclusively the historical conv2 schema. Never
        # guess conv1: the same name can exist in both graph layers.
        full = name if name in parameters else "conv2." + name
        if full not in parameters or full in resolved or value.shape != parameters[full].shape:
            raise ValueError(f"Unknown, duplicate or incompatible update parameter: {name}")
        if value.device != parameters[full].device or value.dtype != parameters[full].dtype:
            raise ValueError("Delta dtype/device must match source parameters")
        resolved[full] = value
    names = set(resolved)
    allowed = [{n for n in parameters if n.startswith("conv2.")},
               {n for n in parameters if n.startswith("conv1.")}, set(parameters)]
    if names not in allowed:
        raise ValueError("Schema must cover exactly head, backbone or all parameters")
    return resolved


class HeadAdapterGenerator(nn.Module):
    """MLP hypernetwork that predicts parameter updates (ΔW, Δb) for the head.

    Inputs:
        - graph_descriptor: vector of shape [descriptor_dim] from GraphShiftDescriptor
        - current_head_weights: vector of shape [head_dim] from extract_head_parameters

    Output:
        - delta: predicted update vector of shape [head_dim]

    Key trick:
    The final linear layer is initialized with zeros. Thus, at epoch 0 of
    generator training, ΔW = 0 and Δb = 0. The model starts at exact baseline
    performance and learns deviations from there; positive gains are not guaranteed.
    """

    def __init__(
        self,
        descriptor_dim: int,
        head_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        include_head_weights: bool = True,
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        for name, value in (("descriptor_dim", descriptor_dim), ("head_dim", head_dim), ("hidden_dim", hidden_dim), ("num_layers", num_layers)):
            positive_int(value, name)
        self.include_head_weights = include_head_weights

        in_dim = (descriptor_dim + head_dim) if include_head_weights else descriptor_dim
        layers: list[nn.Module] = []
        current_dim = in_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, head_dim))
        self.net = nn.Sequential(*layers)

        # Zero-initialize the final projection layer
        final_layer = layers[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, graph_descriptor: Tensor, current_head_weights: Tensor | None = None) -> Tensor:
        """Predict the parameter delta vector.

        Inputs should be 1D (or 2D batch) tensors.
        """
        finite(graph_descriptor, "conditioning")
        if graph_descriptor.ndim not in (1, 2) or graph_descriptor.shape[-1] != self.descriptor_dim:
            raise ValueError("Conditioning dimension mismatch")
        if self.include_head_weights:
            if current_head_weights is None or current_head_weights.shape != (*graph_descriptor.shape[:-1], self.head_dim):
                raise ValueError("Current weights are required with the configured shape")
            finite(current_head_weights, "source weights")
            combined = torch.cat([graph_descriptor, current_head_weights], dim=-1)
        else:
            combined = graph_descriptor
        return finite(self.net(combined), "generated delta")


def adapted_forward(
    model: nn.Module,
    graph: Data,
    delta: Tensor,
    schema: Sequence[tuple[str, tuple[int, ...]]],
) -> Tensor:
    """Execute a forward pass with the predicted delta applied, preserving autograd gradients.

    Uses torch.func.functional_call so gradients flow back through the adapted GNN
    into delta (and therefore into the generator network during meta-training).
    Supports head-only, backbone-only, or full-network parameter schemas.
    """
    updates = _resolved_updates(model, delta, schema)
    params = dict(model.named_parameters())
    params.update({name: params[name] + value for name, value in updates.items()})
    logits = functional_call(model, params, (graph.x, graph.edge_index))
    validate_logits(logits, graph.num_nodes)
    return logits


def head_forward(model, graph, delta, schema, hidden):
    """Reuse exact frozen encode output, while retaining head-delta gradients."""
    updates = _resolved_updates(model, delta, schema)
    if any(not name.startswith("conv2.") for name in updates):
        raise ValueError("Hidden feature reuse is valid only for head updates")
    parameters = {name.removeprefix("conv2."): dict(model.named_parameters())[name] + value
                  for name, value in updates.items()}
    logits = functional_call(model.conv2, parameters, (hidden, graph.edge_index), strict=True)
    validate_logits(logits, graph.num_nodes)
    return logits


def apply_target_delta(
    model: nn.Module,
    delta: Tensor,
    schema: Sequence[tuple[str, tuple[int, ...]]],
) -> nn.Module:
    """Return an in-memory copy of the model with the delta permanently applied.

    Useful for offline evaluation, saving checkpoints, or testing inference.
    Supports head-only, backbone-only, and full-network parameter schemas.
    """
    updates = _resolved_updates(model, delta, schema)
    adapted = copy.deepcopy(model)
    parameters = dict(adapted.named_parameters())
    with torch.no_grad():
        for name, value in updates.items():
            parameters[name].add_(value)
            finite(parameters[name], "adapted parameter")
    return adapted


def apply_head_delta(
    model: nn.Module,
    delta: Tensor,
    schema: Sequence[tuple[str, tuple[int, ...]]],
) -> nn.Module:
    """Legacy helper: Return an in-memory copy with head delta applied."""
    return apply_target_delta(model, delta, schema)
