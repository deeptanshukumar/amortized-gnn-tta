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
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch_geometric.data import Data


class GraphShiftDescriptor(nn.Module):
    """Summarizes an arbitrary-sized graph into a compact, fixed-size vector.

    Intuition:
    A hypernetwork cannot accept a variable-sized [N, F] tensor directly.
    We pass the graph through the pretrained GNN's encoder to observe how the
    model represents the shifted graph, then compute size-invariant moments
    (mean and standard deviation across nodes) plus average graph degree.

    The resulting vector is fixed in size regardless of node count N.
    """

    def __init__(self, hidden_channels: int = 32, include_degree: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.include_degree = include_degree
        # [mean (H) || std (H) || (optional mean_degree (1))]
        self.output_dim = hidden_channels * 2 + (1 if include_degree else 0)

    @torch.no_grad()
    def forward(self, model: nn.Module, graph: Data) -> Tensor:
        """Compute the descriptor vector for the given model and graph.

        Args:
            model: Pretrained GNN exposing .encode(x, edge_index).
            graph: Graph data containing x and edge_index.

        Returns:
            1D float Tensor of shape [output_dim].
        """
        model_was_training = model.training
        model.eval()

        # Extract first-layer hidden representations before dropout
        x = graph.x
        edge_index = graph.edge_index
        h = model.encode(x, edge_index)  # Shape: [N, hidden_channels]

        # Compute size-invariant moments across all N nodes
        mean = h.mean(dim=0)  # Shape: [H]
        std = h.std(dim=0, unbiased=False)  # Shape: [H]

        components = [mean, std]
        if self.include_degree:
            num_nodes = max(graph.num_nodes if graph.num_nodes is not None else x.shape[0], 1)
            mean_deg = torch.tensor([edge_index.shape[1] / num_nodes], dtype=x.dtype, device=x.device)
            components.append(mean_deg)

        descriptor = torch.cat(components, dim=-1)

        if model_was_training:
            model.train()

        return descriptor


def get_head_module(model: nn.Module) -> nn.Module:
    """Return the final classification layer (conv2) of the GNN."""
    if hasattr(model, "conv2"):
        return model.conv2
    raise ValueError(f"Model {type(model)} has no recognized classification head 'conv2'")


def extract_head_parameters(model: nn.Module) -> tuple[Tensor, list[tuple[str, tuple[int, ...]]]]:
    """Flatten all parameters belonging to the model's classification head (conv2).

    Returns:
        flat_params: 1D Tensor containing all head parameter values.
        schema: List of (param_name, shape) tuples for unflattening.
    """
    head = get_head_module(model)
    params = list(head.named_parameters())
    if not params:
        raise ValueError("Classification head contains no trainable parameters")

    schema = [(name, tuple(p.shape)) for name, p in params]
    flat = torch.cat([p.detach().flatten() for _, p in params])
    return flat, schema


def unflatten_head_delta(delta: Tensor, schema: Sequence[tuple[str, tuple[int, ...]]]) -> dict[str, Tensor]:
    """Slice a flat delta tensor into a dictionary of named parameter tensors."""
    result = {}
    offset = 0
    for name, shape in schema:
        numel = 1
        for dim in shape:
            numel *= dim
        chunk = delta[offset : offset + numel].view(shape)
        result[name] = chunk
        offset += numel
    return result


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
    performance and learns meaningful positive deviations from there.
    """

    def __init__(
        self,
        descriptor_dim: int,
        head_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        in_dim = descriptor_dim + head_dim
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

    def forward(self, graph_descriptor: Tensor, current_head_weights: Tensor) -> Tensor:
        """Predict the parameter delta vector.

        Both inputs should be 1D (or 2D batch) tensors.
        """
        combined = torch.cat([graph_descriptor, current_head_weights], dim=-1)
        return self.net(combined)


def adapted_forward(
    model: nn.Module,
    graph: Data,
    delta: Tensor,
    schema: Sequence[tuple[str, tuple[int, ...]]],
) -> Tensor:
    """Execute a forward pass with the head delta applied, preserving autograd gradients.

    Uses torch.func.functional_call so gradients flow back through the adapted GNN
    into delta (and therefore into the generator network during meta-training).
    """
    unflattened = unflatten_head_delta(delta, schema)

    # Build the full parameter dictionary with conv2 weights updated by delta
    params = dict(model.named_parameters())
    for name, delta_tensor in unflattened.items():
        full_key = f"conv2.{name}"
        if full_key in params:
            params[full_key] = params[full_key] + delta_tensor

    # Functional forward pass through the GNN
    return functional_call(model, params, (graph.x, graph.edge_index))


def apply_head_delta(
    model: nn.Module,
    delta: Tensor,
    schema: Sequence[tuple[str, tuple[int, ...]]],
) -> nn.Module:
    """Return an in-memory copy of the model with the delta permanently added to conv2.

    Useful for offline evaluation, saving checkpoints, or testing inference.
    """
    adapted = copy.deepcopy(model)
    head = get_head_module(adapted)
    unflattened = unflatten_head_delta(delta, schema)

    with torch.no_grad():
        for name, delta_tensor in unflattened.items():
            param = getattr(head, name.split(".")[0])
            # If the parameter is nested, traverse to it
            subnames = name.split(".")
            curr = head
            for s in subnames[:-1]:
                curr = getattr(curr, s)
            target_param = getattr(curr, subnames[-1])
            target_param.add_(delta_tensor.to(target_param.device, dtype=target_param.dtype))

    return adapted
