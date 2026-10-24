"""Reproducible label subsets inside a fixed source-training pool.

These are supervision selections, not new graph realizations or new train/val/test
partitions. No source-test or target labels are accepted by this module.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.api import graph_fingerprint
from .training import SourceTrainingData, software_versions, validate_source_training_data

PROTOCOL = "source-label-subsets-v1"
RNG = "PCG64(SeedSequence([subset_seed, class_id])); sorted class-node IDs"


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_identity(source: SourceTrainingData) -> dict[str, Any]:
    order = source.train_indices.argsort()
    return {
        "graph_id": source.provenance.get("graph_id"),
        "graph_fingerprint": graph_fingerprint(source.graph),
        "class_names": list(source.class_names),
        "train_pool_indices": source.train_indices[order].tolist(),
        "train_pool_label_sha256": _digest(source.train_labels[order].tolist()),
        "validation_indices": source.val_indices.sort().values.tolist(),
    }


def _class_counts(source: SourceTrainingData, fraction: float) -> tuple[list[int], list[int]]:
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
            not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("label_fraction must be finite and in (0,1]")
    pool = torch.bincount(source.train_labels, minlength=len(source.class_names)).tolist()
    selected = [math.floor(fraction * count) for count in pool]
    if min(selected) < 1:
        raise ValueError("The fraction must select at least one training node per class")
    return pool, selected


def _seeds(values: Sequence[int]) -> list[int]:
    values = list(values)
    if (not values or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values)
            or len(set(values)) != len(values)):
        raise ValueError("Subset seeds must be distinct integers in [0,2**32)")
    return values


def create_source_subsets(source: SourceTrainingData, *, label_fraction: float,
                          seeds: Sequence[int]) -> dict[str, Any]:
    """Sample each class without replacement from the existing training pool.

    Each class contributes floor(label_fraction * its pool size) labels. Different
    subsets may overlap, but identical subsets fail visibly; we never retry draws
    secretly. The fraction is required, not a hidden benchmark/default budget.
    Local RNGs do not consume model initialization or global NumPy RNG state.
    """
    validate_source_training_data(source)
    seeds = _seeds(seeds)
    pool_counts, counts = _class_counts(source, label_fraction)
    capacity = math.prod(math.comb(n, k) for n, k in zip(pool_counts, counts))
    if len(seeds) > capacity:
        raise ValueError("This budget/pool cannot produce enough distinct subsets")
    subsets, seen = [], set()
    for position, seed in enumerate(seeds):
        selected = []
        for class_id, count in enumerate(counts):
            pool = source.train_indices[source.train_labels == class_id].sort().values.numpy()
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, class_id])))
            selected.extend(rng.choice(pool, size=count, replace=False).tolist())
        indices = tuple(sorted(selected))
        if indices in seen:
            raise ValueError("Seeds produced duplicate subsets; choose different seeds or budget")
        seen.add(indices)
        subsets.append({"id": f"source_subset_{position}", "seed": seed,
                        "train_indices": list(indices)})
    return {
        "format_version": 1, "protocol": PROTOCOL, "source": _source_identity(source),
        # Normalize tuples to JSON arrays now, so persistence preserves equality
        # and the manifest digest is identical before and after reloading.
        "source_provenance": json.loads(json.dumps(source.provenance, allow_nan=False)),
        "label_fraction": float(label_fraction), "rounding": "floor per class; reject empty classes",
        "pool_class_counts": pool_counts, "subset_class_counts": counts,
        "rng": RNG, "versions": software_versions(), "subsets": subsets,
    }


def _validate_manifest(manifest: dict[str, Any], source: SourceTrainingData) -> None:
    """Check stored membership against its source; never redraw on loading."""
    validate_source_training_data(source)
    if (not isinstance(manifest, dict) or type(manifest.get("format_version")) is not int
            or manifest["format_version"] != 1 or manifest.get("protocol") != PROTOCOL):
        raise ValueError("Unsupported source-subset manifest")
    if manifest.get("source") != _source_identity(source):
        raise ValueError("Manifest does not match the source graph, label pool, or validation indices")
    pool_counts, counts = _class_counts(source, manifest.get("label_fraction"))
    if manifest.get("pool_class_counts") != pool_counts or manifest.get("subset_class_counts") != counts:
        raise ValueError("Manifest class counts do not match its label budget")
    if manifest.get("rng") != RNG or manifest.get("rounding") != "floor per class; reject empty classes":
        raise ValueError("Manifest sampling convention is unsupported")
    if not isinstance(manifest.get("versions"), dict) or not isinstance(manifest.get("source_provenance"), dict):
        raise ValueError("Manifest needs version and source provenance records")
    subsets = manifest.get("subsets")
    if not isinstance(subsets, list) or not subsets:
        raise ValueError("Manifest must contain source subsets")
    if any(not isinstance(item, dict) for item in subsets):
        raise ValueError("Each subset must be a record")
    _seeds([item.get("seed") for item in subsets])
    labels = dict(zip(source.train_indices.tolist(), source.train_labels.tolist()))
    seen = set()
    for position, item in enumerate(subsets):
        if item.get("id") != f"source_subset_{position}":
            raise ValueError("Subset IDs/order must be unique and canonical")
        indices = item.get("train_indices")
        if (not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices)
                or indices != sorted(set(indices)) or any(i not in labels for i in indices)):
            raise ValueError("Subset indices must be sorted, unique members of the source training pool")
        actual = [sum(labels[i] == c for i in indices) for c in range(len(counts))]
        if actual != counts:
            raise ValueError("Subset membership violates its per-class label budget")
        key = tuple(indices)
        if key in seen:
            raise ValueError("Manifest contains duplicate subsets")
        seen.add(key)


def source_subset_view(source: SourceTrainingData, manifest: dict[str, Any],
                       subset_id: str) -> SourceTrainingData:
    """Select supervision for the existing trainer, preserving validation labels.

    Views share the source graph, which callers must treat as read-only. fit_source
    clones it before execution. Native bundle splits are never mutated.
    """
    _validate_manifest(manifest, source)
    item = next((item for item in manifest["subsets"] if item["id"] == subset_id), None)
    if item is None:
        raise ValueError(f"Unknown source subset: {subset_id}")
    indices = torch.tensor(item["train_indices"], dtype=torch.long)
    order = source.train_indices.argsort()
    positions = torch.searchsorted(source.train_indices[order], indices)
    provenance = deepcopy(source.provenance)
    provenance["source_subset"] = {
        "protocol": PROTOCOL, "id": subset_id, "seed": item["seed"],
        "label_fraction": manifest["label_fraction"], "class_counts": manifest["subset_class_counts"].copy(),
        "manifest_sha256": _digest(manifest),
    }
    return replace(source, train_indices=indices, train_labels=source.train_labels[order][positions].clone(),
                   val_indices=source.val_indices.clone(), val_labels=source.val_labels.clone(),
                   provenance=provenance)


def save_source_subsets(path: Path, manifest: dict[str, Any], source: SourceTrainingData) -> None:
    """Write an audited JSON manifest; never overwrite an existing file."""
    _validate_manifest(manifest, source)
    encoded = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def load_source_subsets(path: Path, source: SourceTrainingData) -> dict[str, Any]:
    """Load saved node memberships, rejecting a different source or invalid budget."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_manifest(manifest, source)
    return manifest
