"""Versioned, label-free conditioning statistics.

v2 fixes directed isolation and decouples D's weight scope from update sites.
v1 explicitly retains the historical scope/isolation semantics for replay.
Edge energy is mean_{stored edges, channels}(x_u-x_v)^2, not a
degree-normalized Laplacian energy. Reverse edge entries count separately.
"""

import math
import weakref

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from ..evaluation import validate_input, validate_logits
from ._validation import finite, positive_int, source_state


def edge_statistics(x: Tensor, edges: Tensor, chunk_size: int = 1024):
    """Exact per-channel energy and edge-distance moments, bounded E×F memory.

    Population moments use chunked Welford combination to avoid cancellation.
    Reduction order changes floating-point rounding, not the statistic.
    """
    positive_int(chunk_size, "chunk_size")
    energy = x.new_zeros(x.shape[1])
    mean = x.new_zeros(())
    m2 = x.new_zeros(())
    count = 0
    if edges.shape[1] == 0:
        return energy, mean.view(1), m2.view(1)
    for edge in edges.split(chunk_size, dim=1):
        squared = x.index_select(0, edge[0])
        squared.sub_(x.index_select(0, edge[1]))
        squared.square_()
        energy += squared.sum(dim=0)
        distance = squared.sum(dim=1).sqrt()
        n = len(distance)
        variance, block_mean = torch.var_mean(distance, correction=0)
        shift = block_mean - mean
        total = count + n
        m2 = m2 + variance * n + shift.square() * (count * n / total)
        mean = mean + shift * (n / total)
        count = total
    if count:
        energy /= count
    return energy, mean.view(1), (m2 / max(count, 1)).sqrt().view(1)


def _moments_centers(x, chunk_size=1024):
    """Population moments via centered two-pass sums, with round-off checks.

    For any center c, Var(x)=mean((x-c)^2)-mean(x-c)^2. The correction
    accounts for a rounded floating-point center. Chunk reduction is contiguous
    instead of a costly strided columnwise Welford scan. Only negative variance
    within a stated round-off bound is floored to zero; input values are intact.
    """
    mean = x.mean(dim=0)
    residual_sum = x.new_zeros(x.shape[1], dtype=torch.float64)
    squared_sum = x.new_zeros(x.shape[1], dtype=torch.float64)
    distances = []
    for block in x.split(chunk_size):
        residual = block - mean
        residual_sum += residual.sum(dim=0).double()
        residual.square_()
        squared_sum += residual.sum(dim=0).double()
        distances.append(residual.sum(dim=1).sqrt())
    second = squared_sum / len(x)
    variance = second - (residual_sum / len(x)).square()
    tolerance = 32 * torch.finfo(x.dtype).eps * second.abs()
    if (variance < -tolerance).any():
        raise ValueError("Variance exceeds the allowed floating-point round-off bound")
    std = variance.clamp_min(0).sqrt().to(x.dtype)
    distance = torch.cat(distances)
    return mean, std, distance.mean().view(1), distance.std(correction=0).view(1)


@torch.no_grad()
def _target_graph_descriptor(graph: Data, in_channels: int = 64, *, version="v2", chunk_size=1024) -> Tensor:
    """2F+17 fields. All standard deviations are population statistics.

    Order: channel mean/std; global mean/std; norm mean/std; zero fraction;
    max/min; outgoing-degree mean/std/max-to-max(mean,1); isolated fraction;
    E/N (a retained duplicate degree summary, NOT density); mean edge energy;
    edge-distance mean/std; distance-to-feature-center mean/std.
    """
    x, edges = graph.x, graph.edge_index
    if len(x) == 0 or x.shape[1] != in_channels:
        raise ValueError("Feature dimension mismatch or empty graph")
    if version not in {"v1", "v2"}:
        raise ValueError("Unknown descriptor version")
    mean, std, center_mean, center_std = _moments_centers(x, chunk_size)
    norms = torch.linalg.vector_norm(x, dim=1)
    energy, distance_mean, distance_std = edge_statistics(x, edges, chunk_size)
    degree = torch.bincount(edges[0], minlength=len(x)).to(x.dtype)
    incident = degree + torch.bincount(edges[1], minlength=len(x)).to(x.dtype)
    isolation = ((degree if version == "v1" else incident) == 0).to(x.dtype).mean()
    fields = [mean, std, x.mean().view(1), x.std(correction=0).view(1),
              norms.mean().view(1), norms.std(correction=0).view(1), ((x.numel() - torch.count_nonzero(x)).to(x.dtype) / x.numel()).view(1),
              x.max().view(1), x.min().view(1), degree.mean().view(1), degree.std(correction=0).view(1),
              (degree.max() / degree.mean().clamp_min(1)).view(1), isolation.view(1),
              x.new_tensor([edges.shape[1] / len(x)]), energy.mean().view(1),
              distance_mean, distance_std, center_mean, center_std]
    return finite(torch.cat(fields), "graph descriptor")


@torch.no_grad()
def extract_target_graph_descriptor(graph: Data, in_channels: int = 64, *, version="v2", chunk_size=1024):
    """Validated public raw-graph summary; see module documentation for units."""
    validate_input(graph)
    return _target_graph_descriptor(graph, in_channels, version=version, chunk_size=chunk_size)


def _hidden_statistics(hidden, edges, expected, chunk_size):
    if hidden.ndim != 2 or hidden.shape[1] != expected:
        raise ValueError("Hidden feature dimension mismatch")
    finite(hidden, "hidden representations")
    mean, std, center_mean, center_std = _moments_centers(hidden, chunk_size)
    energy, distance_mean, distance_std = edge_statistics(hidden, edges, chunk_size)
    return torch.cat([mean, std, energy, energy.mean().view(1),
                      distance_mean, distance_std, center_mean, center_std])


@torch.no_grad()
def extract_hidden_representation_descriptor(model, graph, hidden_channels=32):
    with source_state(model):
        return _hidden_statistics(model.encode(graph.x, graph.edge_index), graph.edge_index, hidden_channels, 1024)


def _uncertainty(logits, classes):
    validate_logits(logits, logits.shape[0], classes)
    probabilities = logits.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    top = probabilities.topk(2, dim=-1).values
    confidence, margin = top[:, 0], top[:, 0] - top[:, 1]
    return torch.cat([entropy.mean().view(1), entropy.std(correction=0).view(1),
                      (entropy > 0.5 * math.log(classes)).to(logits.dtype).mean().view(1),
                      confidence.mean().view(1), confidence.std(correction=0).view(1),
                      margin.mean().view(1), (margin < 0.1).to(logits.dtype).mean().view(1),
                      probabilities.mean(dim=0)])


@torch.no_grad()
def extract_predictive_uncertainty_descriptor(model, graph, out_channels=2):
    """7+C fields; last C are the predicted marginal, never a true class prior."""
    with source_state(model):
        return _uncertainty(model(graph.x, graph.edge_index), out_channels)


@torch.no_grad()
def extract_compact_weight_descriptor(model, scope="head", *, version="v2"):
    """12 fields, with precisely named retained-spectrum heuristics.

    Flat-weight L2 norm, mean, sample std, max, min, q25/q50/q75;
    retained-top-two singular-value mean/max/max-to-clamped-min ratio;
    mean matrix Frobenius norm. The ratio is NOT a condition number.
    Rank-one matrices retain the historical zero padding; denominator floor 1e-6.
    """
    if scope not in {"head", "backbone", "all"} or version not in {"v1", "v2"}:
        raise ValueError("Unknown weight scope or descriptor version")
    def allowed(name):
        return scope == "all" or name.startswith("conv2." if scope == "head" else "conv1.")
    selected = [(name, p.detach()) for name, p in model.named_parameters() if allowed(name)]
    if not selected:
        raise ValueError("No parameters in weight descriptor scope")
    flat = finite(torch.cat([p.flatten() for _, p in selected]), "source weights")
    spectral = [(name, p.detach()) for name, p in model.named_parameters()
                if (allowed(name) if version == "v2" else (scope != "head" or name.startswith("conv2."))) and p.ndim == 2]
    values, norms = [], []
    for _, parameter in spectral:
        singular = torch.linalg.svdvals(parameter)[:2]
        values.append(torch.nn.functional.pad(singular, (0, 2 - len(singular))))
        norms.append(parameter.norm())
    if values:
        singular = torch.cat(values)
        tail = torch.stack([singular.mean(), singular.max(), singular.max() / singular.min().clamp_min(1e-6), torch.stack(norms).mean()])
    else:
        tail = flat.new_tensor([0, 0, 1, 0])
    result = torch.cat([torch.stack([flat.norm(), flat.mean(), flat.std(correction=1) if len(flat) > 1 else flat.new_zeros(()), flat.max(), flat.min()]),
                        flat.quantile(flat.new_tensor([0.25, 0.5, 0.75])), tail])
    return finite(result, "weight descriptor")


class GraphShiftDescriptor(nn.Module):
    """A=graph; B=A+hidden; C=B+uncertainty; D=C+weight fingerprint.

    Legacy moments/structural layouts remain explicit. No persistent target or
    model cache is hidden in this module; each call charges its computations.
    """
    def __init__(self, hidden_channels=32, include_degree=True, descriptor_type="moments",
                 conditioning=None, in_channels=64, out_channels=2, target_scope="head",
                 descriptor_version="v2", weight_scope="all", chunk_size=1024):
        super().__init__()
        for name, value in (("hidden_channels", hidden_channels), ("in_channels", in_channels), ("out_channels", out_channels), ("chunk_size", chunk_size)):
            positive_int(value, name)
        if descriptor_version not in {"v1", "v2"} or target_scope not in {"head", "backbone", "all"} or weight_scope not in {"head", "backbone", "all"}:
            raise ValueError("Invalid descriptor version or parameter scope")
        self.hidden_channels, self.include_degree = hidden_channels, include_degree
        self.descriptor_type = descriptor_type
        self.conditioning = conditioning.upper() if conditioning else None
        self.in_channels, self.out_channels = in_channels, out_channels
        self.target_scope, self.weight_scope = target_scope, weight_scope
        self.descriptor_version, self.chunk_size = descriptor_version, chunk_size
        self._weight_cache = None
        if self.conditioning:
            dims = [2 * in_channels + 17, 3 * hidden_channels + 5, 7 + out_channels, 12]
            if self.conditioning not in "ABCD" or len(self.conditioning) != 1:
                raise ValueError("Unknown conditioning version")
            self.output_dim = sum(dims[:"ABCD".index(self.conditioning) + 1])
        elif descriptor_type == "moments":
            self.output_dim = 2 * hidden_channels + int(include_degree)
        elif descriptor_type == "structural":
            self.output_dim = 3 * hidden_channels + 9
        else:
            raise ValueError("Unknown descriptor_type")

    def configuration(self):
        return {name: getattr(self, name) for name in ("hidden_channels", "include_degree", "descriptor_type", "conditioning", "in_channels", "out_channels", "target_scope", "descriptor_version", "weight_scope", "chunk_size")}

    def _weight_signature(self, model):
        return (self.descriptor_version, self.weight_scope, self.target_scope,
                tuple((name, id(p), p.data_ptr(), p._version, p.device, p.dtype) for name, p in model.named_parameters()))

    @torch.no_grad()
    def prepare_model(self, model):
        """Explicit one-model static setup; caller must report its cost.

        No target information is cached. Normal in-place weight changes or
        parameter/device replacements invalidate this cache. Mutating tensors
        through .data bypasses PyTorch's version tracking and is unsupported.
        """
        if self.conditioning == "D":
            scope = self.target_scope if self.descriptor_version == "v1" else self.weight_scope
            value = extract_compact_weight_descriptor(model, scope, version=self.descriptor_version)
            self._weight_cache = (weakref.ref(model), self._weight_signature(model), value)

    def forward(self, model, graph):
        return self.describe_with_hidden(model, graph)[0]

    @torch.no_grad()
    def describe_with_hidden(self, model, graph):
        """Return summary and its frozen hidden features for this exact event.

        A has no model-dependent summary and returns no hidden cache. Callers
        may reuse returned features only for supported frozen-backbone heads.
        Nothing is retained across target events.
        """
        validate_input(graph)
        if not graph.num_nodes:
            raise ValueError("Descriptor requires a nonempty graph")
        parts = []
        if self.conditioning:
            parts.append(_target_graph_descriptor(graph, self.in_channels, version=self.descriptor_version, chunk_size=self.chunk_size))
            if self.conditioning == "A":
                return parts[0], None
        if model is None:
            raise ValueError("This descriptor requires a model")
        with source_state(model):
            hidden = model.encode(graph.x, graph.edge_index)
            if self.conditioning:
                parts.append(_hidden_statistics(hidden, graph.edge_index, self.hidden_channels, self.chunk_size))
                if self.conditioning in {"C", "D"}:
                    # Our supported two-layer wrappers have no eval-mode dropout
                    # between encode and conv2. This reuses the same first pass.
                    from ..models import GCN, GAT, GraphSAGE
                    logits = model.conv2(hidden, graph.edge_index) if isinstance(model, (GCN, GAT, GraphSAGE)) else model(graph.x, graph.edge_index)
                    parts.append(_uncertainty(logits, self.out_channels))
                if self.conditioning == "D":
                    scope = self.target_scope if self.descriptor_version == "v1" else self.weight_scope
                    cache = self._weight_cache
                    if cache is not None and cache[0]() is model and cache[1] == self._weight_signature(model):
                        parts.append(cache[2])
                    else:
                        self._weight_cache = None
                        parts.append(extract_compact_weight_descriptor(model, scope, version=self.descriptor_version))
                result = torch.cat(parts)
            elif self.descriptor_type == "moments":
                if hidden.shape[1] != self.hidden_channels:
                    raise ValueError("Hidden feature dimension mismatch")
                parts = [hidden.mean(0), hidden.std(0, correction=0)]
                if self.include_degree:
                    parts.append(hidden.new_tensor([graph.edge_index.shape[1] / len(hidden)]))
                result = torch.cat(parts)
            else:
                statistics = _hidden_statistics(hidden, graph.edge_index, self.hidden_channels, self.chunk_size)
                degree = torch.bincount(graph.edge_index[0], minlength=len(hidden)).to(hidden.dtype)
                incident = degree + torch.bincount(graph.edge_index[1], minlength=len(hidden)).to(hidden.dtype)
                isolation = ((degree if self.descriptor_version == "v1" else incident) == 0).to(hidden.dtype).mean()
                result = torch.cat([statistics, torch.stack([degree.mean(), degree.std(correction=0), degree.max() / degree.mean().clamp_min(1), isolation])])
        if result.shape != (self.output_dim,):
            raise ValueError("Descriptor shape disagrees with its configuration")
        return finite(result, "descriptor"), hidden
