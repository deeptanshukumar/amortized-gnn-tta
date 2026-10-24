"""Model-neutral graph records and explicit supervision boundaries.

The label-bearing bundle belongs to training/evaluation orchestration. Adaptation
must receive only ``unlabeled_input()``, never the bundle or a GraphRecord.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

EnvironmentRole = Literal["source_train", "meta_train", "validation", "target_test"]
ROLES = {"source_train", "meta_train", "validation", "target_test"}
INPUT_FIELDS = {"x", "edge_index", "num_nodes", "edge_weight", "edge_attr"}


@dataclass
class GraphRecord:
    """One CPU graph, separate ground truth, named splits, and offline metadata."""

    graph: Data
    labels: Tensor
    splits: dict[str, Tensor]
    role: EnvironmentRole
    metadata: dict[str, Any] = field(default_factory=dict)


def graph_fingerprint(graph: Data) -> str:
    """Identify graph input content, independently of labels and split seeds."""
    digest = hashlib.sha256()
    digest.update(str(graph.num_nodes).encode())
    for key in sorted(INPUT_FIELDS - {"num_nodes"}):
        value = graph.get(key)
        if value is not None:
            value = value.detach().cpu().contiguous()
            digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_record(record: GraphRecord, num_classes: int) -> None:
    """Reject malformed tensors and supervision splits without transforming them.

    This initial single-label contract requires disjoint named node splits. Shared
    graph context is intentional and is not confused with label-set overlap.
    """
    graph, labels = record.graph, record.labels
    if any(value.device.type != "cpu" for _, value in graph if isinstance(value, Tensor)):
        raise ValueError("Dataset storage must remain on CPU; move model inputs explicitly")
    if labels.device.type != "cpu" or any(value.device.type != "cpu" for value in record.splits.values()):
        raise ValueError("Labels and split indices must remain on CPU")
    if record.role not in ROLES:
        raise ValueError(f"Unknown environment role: {record.role}")
    if set(graph.keys()) - INPUT_FIELDS:
        raise ValueError("Graph contains fields outside the model-input allowlist")
    if graph.num_nodes is None or graph.num_nodes < 1:
        raise ValueError("num_nodes must be explicitly positive")
    n = graph.num_nodes
    if graph.x is None or graph.x.dtype != torch.float32 or graph.x.ndim != 2:
        raise ValueError("x must be a float32 [N,F] tensor")
    if graph.x.shape[0] != n or graph.x.shape[1] < 1 or not torch.isfinite(graph.x).all():
        raise ValueError("x must have N rows, positive feature dimension and finite values")
    edges = graph.edge_index
    if edges is None or edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must be int64 [2,E], including when E=0")
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Edge endpoints are outside [0,N)")
    if labels.dtype != torch.long or tuple(labels.shape) != (n,):
        raise ValueError("labels must be int64 [N]")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("Labels are outside the declared class schema")
    tensors = [graph.x, edges, labels]
    for key in ("edge_weight", "edge_attr"):
        value = graph.get(key)
        if value is not None:
            if value.ndim < 1 or value.shape[0] != edges.shape[1] or not torch.isfinite(value).all():
                raise ValueError(f"{key} must be finite and aligned with edge entries")
            if key == "edge_weight" and value.ndim != 1:
                raise ValueError("edge_weight must have shape [E]")
            tensors.append(value)
    seen = torch.zeros(n, dtype=torch.bool)
    if not record.splits:
        raise ValueError("At least one named split is required")
    for name, indices in record.splits.items():
        if not name or indices.dtype != torch.long or indices.ndim != 1:
            raise ValueError("Splits must have nonempty names and int64 index vectors")
        if indices.numel() and (indices.min() < 0 or indices.max() >= n):
            raise ValueError(f"Split {name} has out-of-range indices")
        if indices.unique().numel() != indices.numel() or seen[indices].any():
            raise ValueError(f"Split {name} contains duplicates or overlaps another split")
        seen[indices] = True
        tensors.append(indices)
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("Dataset storage must remain on CPU; move model inputs explicitly")
    if record.role in {"source_train", "meta_train"}:
        if not {"train", "val", "test"} <= record.splits.keys() or not seen.all():
            raise ValueError("Training environments require covering train/val/test splits")
    elif set(record.splits) != {"eval"} or not seen.all():
        raise ValueError("Validation/target environments require an eval-only covering split")


@dataclass
class DatasetBundle:
    """Graphs with a shared feature/class schema; native data are never cropped."""

    graphs: dict[str, GraphRecord]
    class_names: tuple[str, ...] = ("negative", "positive")
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if (not self.graphs or len(self.class_names) < 2 or
                len(set(self.class_names)) != len(self.class_names)):
            raise ValueError("A bundle needs graphs and a declared class schema")
        dimensions = set()
        fingerprints = set()
        for graph_id, record in self.graphs.items():
            if not graph_id:
                raise ValueError("Graph IDs must be nonempty")
            validate_record(record, len(self.class_names))
            dimensions.add(record.graph.x.shape[1])
            fingerprint = graph_fingerprint(record.graph)
            if fingerprint in fingerprints:
                raise ValueError("Duplicate graph realization across environments")
            fingerprints.add(fingerprint)
        if len(dimensions) != 1:
            raise ValueError("Source/target feature dimensions must agree")

    def unlabeled_input(self, graph_id: str) -> Data:
        """Return an independent, allowlisted graph without labels or metadata.

        Clones also prevent a downstream in-place transform from modifying the
        stored native graph. This API prevents accidents, not malicious access.
        """
        original = self.graphs[graph_id].graph
        result = Data(num_nodes=original.num_nodes)
        for key in INPUT_FIELDS - {"num_nodes"}:
            value = original.get(key)
            if value is not None:
                result[key] = value.clone()
        return result

    def select(self, graph_id: str, split: str) -> Tensor:
        """Return node indices only; a selection does not alter graph visibility."""
        return self.graphs[graph_id].splits[split].clone()

    def training_labels(self, graph_id: str) -> Tensor:
        """Return train labels only, rejecting validation and target environments."""
        record = self.graphs[graph_id]
        if record.role not in {"source_train", "meta_train"}:
            raise ValueError("Held-out environment labels are evaluation-only")
        return record.labels[record.splits["train"]].clone()

    def evaluation_labels(self, graph_id: str, split: str) -> Tensor:
        """Explicit offline label access for scoring, never adaptation input."""
        record = self.graphs[graph_id]
        return record.labels[record.splits[split]].clone()


def graph_diagnostics(record: GraphRecord) -> dict[str, Any]:
    """Offline inspection including label-derived edge homophily.

    Never feed this report into target adaptation or generator conditioning.
    Homophily is an edge fraction, not an average over node-level ratios.
    """
    graph = record.graph
    edges = graph.edge_index
    degree = torch.bincount(edges[0], minlength=graph.num_nodes)
    e = edges.shape[1]
    return {
        "num_nodes": graph.num_nodes,
        "num_edge_entries": e,
        "num_features": graph.x.shape[1],
        "is_undirected": bool(graph.is_undirected()),
        "self_loop_entries": int((edges[0] == edges[1]).sum()),
        "isolated_nodes": int(((degree + torch.bincount(edges[1], minlength=graph.num_nodes)) == 0).sum()),
        "mean_out_degree": e / graph.num_nodes,
        "edge_homophily": float((record.labels[edges[0]] == record.labels[edges[1]]).float().mean()) if e else None,
        "class_counts": torch.bincount(record.labels, minlength=2).tolist(),
        "split_counts": {key: len(value) for key, value in record.splits.items()},
    }
