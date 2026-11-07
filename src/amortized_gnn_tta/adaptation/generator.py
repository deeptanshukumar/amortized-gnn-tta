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

import math
import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch_geometric.data import Data


def extract_target_graph_descriptor(graph: Data, in_channels: int = 64) -> Tensor:
    """Extract summary statistics purely from the target graph (X, E) without any model.

    Captures:
    1. Feature distributions (channel mean/std, global mean/std, L2 norm mean/std, sparsity, min/max).
    2. Connection and topological structure (degree mean/std/max, isolated ratio, graph density).
    3. Graph topology-feature coupling (raw Dirichlet energy, edge distance dispersion, center dispersion).
    All statistics are strictly label-free.
    """
    x = graph.x
    edge_index = graph.edge_index
    num_nodes = max(graph.num_nodes if graph.num_nodes is not None else x.shape[0], 1)
    num_edges = edge_index.shape[1]

    # 1. Feature statistics
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0, unbiased=False)
    if in_channels is not None:
        if x_mean.shape[0] < in_channels:
            pad_len = in_channels - x_mean.shape[0]
            x_mean = torch.cat([x_mean, torch.zeros(pad_len, dtype=x.dtype, device=x.device)])
            x_std = torch.cat([x_std, torch.zeros(pad_len, dtype=x.dtype, device=x.device)])
        elif x_mean.shape[0] > in_channels:
            x_mean = x_mean[:in_channels]
            x_std = x_std[:in_channels]

    x_glob_mean = x.mean().view(1)
    x_glob_std = x.std(unbiased=False).view(1)
    x_norms = torch.norm(x, p=2, dim=-1)
    x_norm_mean = x_norms.mean().view(1)
    x_norm_std = x_norms.std(unbiased=False).view(1)
    x_sparsity = ((x == 0).to(dtype=x.dtype).mean()).view(1)
    x_max = x.max().view(1)
    x_min = x.min().view(1)

    # 2. Connection and topology statistics
    if num_edges > 0:
        src, dst = edge_index[0], edge_index[1]
        diff = x[src] - x[dst]
        dirichlet = (diff ** 2).mean().view(1)
        edge_dist = torch.norm(diff, p=2, dim=-1)
        edge_dist_mean = edge_dist.mean().view(1)
        edge_dist_std = edge_dist.std(unbiased=False).view(1)

        deg = torch.bincount(edge_index[0], minlength=num_nodes).to(dtype=x.dtype)
        deg_mean = deg.mean().view(1)
        deg_std = deg.std(unbiased=False).view(1)
        deg_max_ratio = (deg.max() / torch.clamp(deg_mean, min=1.0)).view(1)
        isolated_ratio = ((deg == 0).to(dtype=x.dtype).sum() / num_nodes).view(1)
    else:
        dirichlet = torch.zeros(1, dtype=x.dtype, device=x.device)
        edge_dist_mean = torch.zeros(1, dtype=x.dtype, device=x.device)
        edge_dist_std = torch.zeros(1, dtype=x.dtype, device=x.device)
        deg_mean = torch.zeros(1, dtype=x.dtype, device=x.device)
        deg_std = torch.zeros(1, dtype=x.dtype, device=x.device)
        deg_max_ratio = torch.zeros(1, dtype=x.dtype, device=x.device)
        isolated_ratio = torch.ones(1, dtype=x.dtype, device=x.device)

    edge_ratio = torch.tensor([num_edges / num_nodes], dtype=x.dtype, device=x.device)
    center_diff = x - x.mean(dim=0, keepdim=True)
    center_dist = torch.norm(center_diff, p=2, dim=-1)
    center_dist_mean = center_dist.mean().view(1)
    center_dist_std = center_dist.std(unbiased=False).view(1)

    components = [
        x_mean, x_std,
        x_glob_mean, x_glob_std, x_norm_mean, x_norm_std, x_sparsity, x_max, x_min,
        deg_mean, deg_std, deg_max_ratio, isolated_ratio, edge_ratio,
        dirichlet, edge_dist_mean, edge_dist_std, center_dist_mean, center_dist_std
    ]
    return torch.cat(components, dim=-1)


def extract_hidden_representation_descriptor(
    model: nn.Module,
    graph: Data,
    hidden_channels: int = 32,
) -> Tensor:
    """Extract summary statistics from the GNN encoder representations h = model.encode(X, E)."""
    x = graph.x
    edge_index = graph.edge_index
    num_nodes = max(graph.num_nodes if graph.num_nodes is not None else x.shape[0], 1)
    num_edges = edge_index.shape[1]

    h = model.encode(x, edge_index)
    mean = h.mean(dim=0)
    std = h.std(dim=0, unbiased=False)

    if num_edges > 0:
        src, dst = edge_index[0], edge_index[1]
        edge_diff = h[src] - h[dst]
        dirichlet_channel = (edge_diff ** 2).mean(dim=0)
        dirichlet_total = dirichlet_channel.mean().view(1)
        edge_dist = torch.norm(edge_diff, p=2, dim=-1)
        edge_dist_mean = edge_dist.mean().view(1)
        edge_dist_std = edge_dist.std(unbiased=False).view(1)
    else:
        dirichlet_channel = torch.zeros(hidden_channels, dtype=x.dtype, device=x.device)
        dirichlet_total = torch.zeros(1, dtype=x.dtype, device=x.device)
        edge_dist_mean = torch.zeros(1, dtype=x.dtype, device=x.device)
        edge_dist_std = torch.zeros(1, dtype=x.dtype, device=x.device)

    center_diff = h - mean.unsqueeze(0)
    center_dist = torch.norm(center_diff, p=2, dim=-1)
    center_dist_mean = center_dist.mean().view(1)
    center_dist_std = center_dist.std(unbiased=False).view(1)

    components = [
        mean, std, dirichlet_channel, dirichlet_total,
        edge_dist_mean, edge_dist_std, center_dist_mean, center_dist_std
    ]
    return torch.cat(components, dim=-1)


def extract_predictive_uncertainty_descriptor(
    model: nn.Module,
    graph: Data,
    out_channels: int = 2,
) -> Tensor:
    """Extract predictive uncertainty statistics from the pretrained model's forward pass.

    All statistics are strictly label-free:
    - Predictive Shannon entropy: mean, std, high-entropy fraction (> 0.5 * ln C)
    - Prediction confidence: mean, std of max softmax probability
    - Decision margin: mean, low-margin fraction (< 0.1)
    - Empirical class prior: average softmax distribution across all nodes
    """
    logits = model(graph.x, graph.edge_index)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(torch.clamp(probs, min=1e-8))
    entropy = -torch.sum(probs * log_probs, dim=-1)
    c = probs.shape[-1]
    max_ent = math.log(c) if c > 1 else 1.0

    ent_mean = entropy.mean().view(1)
    ent_std = entropy.std(unbiased=False).view(1)
    high_ent_ratio = ((entropy > 0.5 * max_ent).float().mean()).view(1)

    top_probs, _ = torch.topk(probs, k=min(2, c), dim=-1)
    conf_mean = top_probs[:, 0].mean().view(1)
    conf_std = top_probs[:, 0].std(unbiased=False).view(1)
    if top_probs.shape[1] >= 2:
        margin = top_probs[:, 0] - top_probs[:, 1]
        margin_mean = margin.mean().view(1)
        low_margin_ratio = ((margin < 0.1).float().mean()).view(1)
    else:
        margin_mean = torch.zeros(1, dtype=probs.dtype, device=probs.device)
        low_margin_ratio = torch.zeros(1, dtype=probs.dtype, device=probs.device)

    class_dist = probs.mean(dim=0)
    if class_dist.shape[0] < out_channels:
        class_dist = torch.cat([class_dist, torch.zeros(out_channels - class_dist.shape[0], dtype=probs.dtype, device=probs.device)])
    elif class_dist.shape[0] > out_channels:
        class_dist = class_dist[:out_channels]

    components = [
        ent_mean, ent_std, high_ent_ratio,
        conf_mean, conf_std, margin_mean, low_margin_ratio,
        class_dist
    ]
    return torch.cat(components, dim=-1)


def extract_compact_weight_descriptor(model: nn.Module, scope: str = "head") -> Tensor:
    """Extract a compact, size-invariant fingerprint of pretrained target parameters.

    Computes:
    - Distribution moments: norm, mean, std, max, min, 25/50/75% quantiles (8 scalars).
    - Spectral properties: top singular values (2), mean singular value, condition ratio, Frobenius norm (5 scalars).
    Total: 13 compact scalars describing weight scale and geometry.
    """
    flat_w, _ = extract_target_parameters(model, scope)
    norm = torch.norm(flat_w).view(1)
    mean = flat_w.mean().view(1)
    std = flat_w.std().view(1)
    max_val = flat_w.max().view(1)
    min_val = flat_w.min().view(1)
    q = torch.tensor([0.25, 0.5, 0.75], dtype=flat_w.dtype, device=flat_w.device)
    quantiles = torch.quantile(flat_w, q)

    svd_vals = []
    matrix_norms = []
    for name, p in model.named_parameters():
        if scope == "head" and "conv2" not in name:
            continue
        if p.ndim == 2:
            s = torch.linalg.svdvals(p)
            s_top = s[:2]
            if len(s_top) < 2:
                s_top = torch.cat([s_top, torch.zeros(2 - len(s_top), dtype=p.dtype, device=p.device)])
            svd_vals.append(s_top)
            matrix_norms.append(torch.norm(p).view(1))

    if svd_vals:
        all_s = torch.cat(svd_vals)
        s_mean = all_s.mean().view(1)
        s_max = all_s.max().view(1)
        cond = (s_max / torch.clamp(all_s.min(), min=1e-6)).view(1)
        f_norm = torch.cat(matrix_norms).mean().view(1)
    else:
        s_mean = torch.zeros(1, dtype=flat_w.dtype, device=flat_w.device)
        s_max = torch.zeros(1, dtype=flat_w.dtype, device=flat_w.device)
        cond = torch.ones(1, dtype=flat_w.dtype, device=flat_w.device)
        f_norm = torch.zeros(1, dtype=flat_w.dtype, device=flat_w.device)

    components = [norm, mean, std, max_val, min_val, quantiles, s_mean, s_max, cond, f_norm]
    return torch.cat(components, dim=-1)


class GraphShiftDescriptor(nn.Module):
    """Summarizes an arbitrary-sized graph into a compact, fixed-size vector.

    Supports two API styles:
    1. Conditioning Version Hierarchy:
       - 'A': Target graph only (raw features & connections summary)
       - 'B': Target graph + hidden representations (A + latent moments/Dirichlet)
       - 'C': Target graph + hidden + predictive uncertainty (B + entropy/confidence/margin)
       - 'D': Target graph + hidden + predictive uncertainty + compact pretrained weights
    2. Legacy Mode:
       - 'moments': [dim = 2H + 1]
       - 'structural': [dim = 3H + 9]
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        include_degree: bool = True,
        descriptor_type: str = "moments",
        conditioning: str | None = None,
        in_channels: int = 64,
        out_channels: int = 2,
        target_scope: str = "head",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.include_degree = include_degree
        self.descriptor_type = descriptor_type
        self.conditioning = conditioning.upper() if conditioning is not None else None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_scope = target_scope

        if self.conditioning is not None:
            if self.conditioning not in ("A", "B", "C", "D"):
                raise ValueError(f"Unknown conditioning version '{self.conditioning}'. Expected 'A', 'B', 'C', or 'D'.")
            # Version A: 2*in_channels + 17
            dim_a = in_channels * 2 + 17
            # Version B: dim(A) + (3*H + 5)
            dim_b = dim_a + (hidden_channels * 3 + 5)
            # Version C: dim(B) + (7 + out_channels)
            dim_c = dim_b + (7 + out_channels)
            # Version D: dim(C) + 12
            dim_d = dim_c + 12

            dims = {"A": dim_a, "B": dim_b, "C": dim_c, "D": dim_d}
            self.output_dim = dims[self.conditioning]
        else:
            if descriptor_type not in ("moments", "structural"):
                raise ValueError(
                    f"Unknown descriptor_type '{descriptor_type}'. Expected 'moments' or 'structural'."
                )
            if descriptor_type == "moments":
                self.output_dim = hidden_channels * 2 + (1 if include_degree else 0)
            else:
                self.output_dim = hidden_channels * 3 + 9

    @torch.no_grad()
    def forward(self, model: nn.Module | None, graph: Data) -> Tensor:
        """Compute the descriptor vector for the given model and graph."""
        if self.conditioning is not None:
            # 1. Version A: Target graph only
            z_a = extract_target_graph_descriptor(graph, in_channels=self.in_channels)
            if self.conditioning == "A":
                return z_a

            if model is None:
                raise ValueError(f"Conditioning version '{self.conditioning}' requires a model instance.")

            model_was_training = model.training
            model.eval()

            # 2. Version B: Target graph + hidden representations
            z_b_part = extract_hidden_representation_descriptor(model, graph, hidden_channels=self.hidden_channels)
            z_b = torch.cat([z_a, z_b_part], dim=-1)
            if self.conditioning == "B":
                if model_was_training:
                    model.train()
                return z_b

            # 3. Version C: Target graph + hidden + predictive uncertainty
            z_c_part = extract_predictive_uncertainty_descriptor(model, graph, out_channels=self.out_channels)
            z_c = torch.cat([z_b, z_c_part], dim=-1)
            if self.conditioning == "C":
                if model_was_training:
                    model.train()
                return z_c

            # 4. Version D: Target graph + hidden + uncertainty + compact weights
            z_d_part = extract_compact_weight_descriptor(model, scope=self.target_scope)
            z_d = torch.cat([z_c, z_d_part], dim=-1)
            if model_was_training:
                model.train()
            return z_d

        # Legacy descriptor code
        model_was_training = model.training if model is not None else False
        if model is not None:
            model.eval()

        x = graph.x
        edge_index = graph.edge_index
        h = model.encode(x, edge_index) if model is not None else x
        num_nodes = max(graph.num_nodes if graph.num_nodes is not None else x.shape[0], 1)

        mean = h.mean(dim=0)
        std = h.std(dim=0, unbiased=False)

        if self.descriptor_type == "moments":
            components = [mean, std]
            if self.include_degree:
                mean_deg = torch.tensor([edge_index.shape[1] / num_nodes], dtype=x.dtype, device=x.device)
                components.append(mean_deg)
            descriptor = torch.cat(components, dim=-1)
        else:
            num_edges = edge_index.shape[1]
            if num_edges > 0:
                src, dst = edge_index[0], edge_index[1]
                edge_diff = h[src] - h[dst]
                dirichlet_channel = (edge_diff ** 2).mean(dim=0)
                dirichlet_total = dirichlet_channel.mean().view(1)

                edge_dist = torch.norm(edge_diff, p=2, dim=-1)
                edge_dist_mean = edge_dist.mean().view(1)
                edge_dist_std = edge_dist.std(unbiased=False).view(1)

                deg = torch.bincount(edge_index[0], minlength=num_nodes).to(dtype=x.dtype)
                deg_mean = deg.mean().view(1)
                deg_std = deg.std(unbiased=False).view(1)
                deg_max_ratio = (deg.max() / torch.clamp(deg_mean, min=1.0)).view(1)
                isolated_ratio = ((deg == 0).to(dtype=x.dtype).sum() / num_nodes).view(1)
            else:
                dirichlet_channel = torch.zeros(self.hidden_channels, dtype=x.dtype, device=x.device)
                dirichlet_total = torch.zeros(1, dtype=x.dtype, device=x.device)
                edge_dist_mean = torch.zeros(1, dtype=x.dtype, device=x.device)
                edge_dist_std = torch.zeros(1, dtype=x.dtype, device=x.device)
                deg_mean = torch.zeros(1, dtype=x.dtype, device=x.device)
                deg_std = torch.zeros(1, dtype=x.dtype, device=x.device)
                deg_max_ratio = torch.zeros(1, dtype=x.dtype, device=x.device)
                isolated_ratio = torch.ones(1, dtype=x.dtype, device=x.device)

            center_diff = h - mean.unsqueeze(0)
            center_dist = torch.norm(center_diff, p=2, dim=-1)
            center_dist_mean = center_dist.mean().view(1)
            center_dist_std = center_dist.std(unbiased=False).view(1)

            components = [
                mean,
                std,
                dirichlet_channel,
                dirichlet_total,
                edge_dist_mean,
                edge_dist_std,
                center_dist_mean,
                center_dist_std,
                deg_mean,
                deg_std,
                deg_max_ratio,
                isolated_ratio,
            ]
            descriptor = torch.cat(components, dim=-1)

        if model is not None and model_was_training:
            model.train()

        return descriptor


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
        include_head_weights: bool = True,
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
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
        if self.include_head_weights and current_head_weights is not None:
            combined = torch.cat([graph_descriptor, current_head_weights], dim=-1)
        else:
            combined = graph_descriptor
        return self.net(combined)


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
    unflattened = unflatten_head_delta(delta, schema)

    # Build the full parameter dictionary updated by delta
    params = dict(model.named_parameters())
    for name, delta_tensor in unflattened.items():
        if name in params:
            params[name] = params[name] + delta_tensor
        elif f"conv2.{name}" in params:
            params[f"conv2.{name}"] = params[f"conv2.{name}"] + delta_tensor
        elif f"conv1.{name}" in params:
            params[f"conv1.{name}"] = params[f"conv1.{name}"] + delta_tensor

    # Functional forward pass through the GNN
    return functional_call(model, params, (graph.x, graph.edge_index))


def apply_target_delta(
    model: nn.Module,
    delta: Tensor,
    schema: Sequence[tuple[str, tuple[int, ...]]],
) -> nn.Module:
    """Return an in-memory copy of the model with the delta permanently applied.

    Useful for offline evaluation, saving checkpoints, or testing inference.
    Supports head-only, backbone-only, and full-network parameter schemas.
    """
    adapted = copy.deepcopy(model)
    unflattened = unflatten_head_delta(delta, schema)
    all_param_names = dict(adapted.named_parameters())

    with torch.no_grad():
        for name, delta_tensor in unflattened.items():
            if name in all_param_names:
                full_name = name
            elif f"conv2.{name}" in all_param_names:
                full_name = f"conv2.{name}"
            elif f"conv1.{name}" in all_param_names:
                full_name = f"conv1.{name}"
            else:
                continue

            # Traverse nested module attributes
            subnames = full_name.split(".")
            curr = adapted
            for s in subnames[:-1]:
                curr = getattr(curr, s)
            target_param = getattr(curr, subnames[-1])
            target_param.add_(delta_tensor.to(target_param.device, dtype=target_param.dtype))

    return adapted


def apply_head_delta(
    model: nn.Module,
    delta: Tensor,
    schema: Sequence[tuple[str, tuple[int, ...]]],
) -> nn.Module:
    """Legacy helper: Return an in-memory copy with head delta applied."""
    return apply_target_delta(model, delta, schema)
