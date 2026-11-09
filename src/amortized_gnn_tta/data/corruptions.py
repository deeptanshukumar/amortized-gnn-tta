"""Paired, label-independent interventions on sparse real graphs."""

import math
import torch
from torch_geometric.data import Data


def perturb_graph(
    graph: Data,
    *,
    feature_mask_rate: float,
    edge_drop_rate: float,
    feature_seed: int,
    edge_seed: int,
    undirected: bool = False,
) -> Data:
    """Zero independent feature entries and remove edges with separate local RNGs.

    For a symmetric experiment view, a single draw applies to each unordered
    pair. Both directed entries are removed together. No label object is accepted.
    No normalization, self-loop insertion or implicit symmetrization occurs.
    """
    for rate in (feature_mask_rate, edge_drop_rate):
        if isinstance(rate, bool) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Corruption rates must be in [0,1]")
    if graph.x.device.type != "cpu" or graph.edge_index.device.type != "cpu":
        raise ValueError("Construct benchmark interventions on CPU")
    if set(graph.keys()) - {"x", "edge_index", "num_nodes"}:
        raise ValueError(
            "This corruption adapter supports unweighted feature graphs only"
        )
    x = graph.x.clone()
    edges = graph.edge_index.clone()
    if feature_mask_rate:
        rng = torch.Generator().manual_seed(feature_seed)
        x.masked_fill_(torch.rand(x.shape, generator=rng) < feature_mask_rate, 0)
    if edge_drop_rate:
        rng = torch.Generator().manual_seed(edge_seed)
        if undirected:
            if not graph.is_undirected():
                raise ValueError("Expected an explicitly symmetric input view")
            # Stable IDs couple reciprocal edges without allocating an adjacency.
            keys = torch.minimum(edges[0], edges[1]) * graph.num_nodes + torch.maximum(
                edges[0], edges[1]
            )
            unique, inverse = keys.unique(sorted=True, return_inverse=True)
            keep = (torch.rand(len(unique), generator=rng) >= edge_drop_rate)[inverse]
        else:
            keep = torch.rand(edges.shape[1], generator=rng) >= edge_drop_rate
        edges = edges[:, keep]
    return Data(x=x, edge_index=edges, num_nodes=graph.num_nodes)


def unfamiliar_perturbation(graph, *, operation, strength, seed, undirected=False):
    """Held-out label-free operations; strength definitions differ by operation.

    Gaussian strength multiplies each raw channel's population standard deviation.
    Degree-biased removal deletes exactly floor(strength * unique edge pairs),
    sampling without replacement with weights sqrt((degree_u+1)(degree_v+1)).
    These are new benchmark interventions, not naturally observed shift claims.
    """
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or strength < 0
    ):
        raise ValueError("Invalid unfamiliar perturbation strength")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid shift seed")
    from ..evaluation import validate_input

    validate_input(graph)
    if graph.x.device.type != "cpu" or graph.edge_index.device.type != "cpu":
        raise ValueError("Benchmark intervention construction requires CPU inputs")
    rng = torch.Generator().manual_seed(seed)
    if operation == "gaussian_features":
        noise = torch.randn(graph.x.shape, generator=rng, dtype=graph.x.dtype)
        x = graph.x + strength * graph.x.std(dim=0, correction=0) * noise
        return Data(x=x, edge_index=graph.edge_index.clone(), num_nodes=graph.num_nodes)
    if operation != "degree_biased_removal" or strength > 1:
        raise ValueError("Unknown operation or invalid removal fraction")
    edges = graph.edge_index
    if undirected and not graph.is_undirected():
        raise ValueError("Expected symmetric graph")
    if undirected:
        codes = torch.minimum(edges[0], edges[1]) * graph.num_nodes + torch.maximum(
            edges[0], edges[1]
        )
    else:
        codes = edges[0] * graph.num_nodes + edges[1]
    unique, inverse = torch.unique(codes, return_inverse=True)
    endpoints = edges[0] if undirected else edges.reshape(-1)
    degree = torch.bincount(endpoints, minlength=graph.num_nodes).float()
    weights = (
        (degree[unique // graph.num_nodes] + 1) * (degree[unique % graph.num_nodes] + 1)
    ).sqrt()
    count = int(strength * len(unique))
    keep = torch.ones(len(unique), dtype=torch.bool)
    if count:
        keep[torch.multinomial(weights, count, replacement=False, generator=rng)] = (
            False
        )
    return Data(
        x=graph.x.clone(),
        edge_index=edges[:, keep[inverse]].clone(),
        num_nodes=graph.num_nodes,
    )
