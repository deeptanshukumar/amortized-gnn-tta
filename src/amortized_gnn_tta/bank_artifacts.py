"""Validated bank loading and score-independent, whole-subset model partitions."""

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .evaluation import predict, score_logits, validate_logits
from .source_subsets import load_source_subsets
from .training import SourceTrainingData, TrainingConfig


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_path(directory: Path, relative: str) -> Path:
    """Reject missing artifacts and paths that escape their bank directory."""
    root = Path(directory).resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Artifact paths must be nonempty and relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-bank artifact: {relative}")
    return path


@dataclass(frozen=True)
class VerifiedBank:
    """Read-only-by-convention metadata for an audited on-disk bank snapshot."""

    directory: Path
    manifest: dict[str, Any]
    subsets: dict[str, Any]
    sha256: str


def load_verified_bank(directory: Path, source: SourceTrainingData,
                       model_factory: Callable[[dict[str, Any]], nn.Module]) -> VerifiedBank:
    """Check every artifact before exposing a complete source-only bank.

    Loading is CPU-only and uses weights_only=True. Checkpoint identity, model
    shape/finite state, supervision, selected epoch, configs, and metric histories
    must agree with their manifests. No source-test/target labels are accepted.
    """
    directory = Path(directory).resolve()
    manifest_path = directory / "bank.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        if (manifest["format_version"] != 1 or manifest["protocol"] != "source-weight-bank-v1"
                or manifest["status"] != "complete"):
            raise ValueError("Bank must be a supported, complete bank")
        subset_path = artifact_path(directory, manifest["subset_manifest"])
        if file_sha256(subset_path) != manifest["subset_manifest_sha256"]:
            raise ValueError("Subset manifest hash mismatch")
        subsets = load_source_subsets(subset_path, source)
        seeds = manifest["model_seeds"]
        if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int for seed in seeds)):
            raise ValueError("Invalid model-seed list")
        for seed in seeds:
            TrainingConfig(**manifest["training_config_without_seed"], model_seed=seed)
        expected = {(item["id"], seed) for item in subsets["subsets"] for seed in seeds}
        rows = manifest["models"]
        pairs = [(row["subset_id"], row["model_seed"]) for row in rows]
        if (len(pairs) != len(expected) or set(pairs) != expected or
                manifest["expected_models"] != len(expected) or manifest["completed_models"] != len(expected)):
            raise ValueError("Bank must contain every subset/seed pair exactly once")
        canonical = json.dumps(subsets, sort_keys=True, separators=(",", ":"), allow_nan=False)
        subset_digest = hashlib.sha256(canonical.encode()).hexdigest()
        used_paths = set()
        for row in rows:
            model_id = f"{row['subset_id']}_seed_{row['model_seed']}"
            if row["model_id"] != model_id or row["status"] != "complete" or row["cpu_reload_predictions_close"] is not True:
                raise ValueError("Incomplete or misidentified bank model")
            paths = {}
            for field in ("checkpoint", "metrics"):
                path = artifact_path(directory, row[field])
                if path in used_paths or file_sha256(path) != row[field + "_sha256"]:
                    raise ValueError("Duplicate artifact path or artifact hash mismatch")
                used_paths.add(path)
                paths[field] = path
            payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
            metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            config = {**manifest["training_config_without_seed"], "model_seed": row["model_seed"]}
            if (payload["format_version"] != 1 or payload["model_spec"] != manifest["model_spec"]
                    or payload["training_config"] != config or metrics["training_config"] != config
                    or metrics["model_spec"] != manifest["model_spec"]
                    or metrics["model_id"] != model_id or metrics["subset_id"] != row["subset_id"]
                    or metrics["cpu_reload_predictions_close"] is not True):
                raise ValueError("Checkpoint/metric configuration or identity mismatch")
            state = payload["state_dict"]
            if not state or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                                for value in state.values()):
                raise ValueError("Checkpoint state must contain finite tensors")
            model = model_factory(manifest["model_spec"])
            expected_state = model.state_dict()
            if (set(expected_state) != set(state) or any(expected_state[key].shape != value.shape or
                    expected_state[key].dtype != value.dtype for key, value in state.items())):
                raise ValueError("Checkpoint tensor schema does not match its architecture")
            model.load_state_dict(state, strict=True)
            model.eval()
            subset = next(item for item in subsets["subsets"] if item["id"] == row["subset_id"])
            provenance = payload["source"]["source_subset"]
            if (provenance["id"] != subset["id"] or provenance["seed"] != subset["seed"]
                    or provenance["manifest_sha256"] != subset_digest
                    or provenance["label_fraction"] != subsets["label_fraction"]
                    or provenance["class_counts"] != subsets["subset_class_counts"]
                    or payload["source"]["class_names"] != list(source.class_names)
                    or payload["source"]["train_indices"].dtype != torch.long
                    or payload["source"]["train_indices"].tolist() != subset["train_indices"]
                    or not torch.equal(payload["source"]["val_indices"], source.val_indices)):
                raise ValueError("Checkpoint supervision or subset provenance mismatch")
            history = payload["history"]
            if not history or history != metrics["history"] or len(history) != row["epochs_run"]:
                raise ValueError("Checkpoint training history mismatch")
            for epoch, item in enumerate(history, 1):
                if (item["epoch"] != epoch or any(not math.isfinite(item[key]) for key in
                        ("train_loss", "val_loss", "val_accuracy")) or not 0 <= item["val_accuracy"] <= 1):
                    raise ValueError("Invalid training history")
            best = min(history, key=lambda item: item["val_loss"])
            for record in (row, payload, metrics):
                if record["best_epoch"] != best["epoch"] or record["best_val_loss"] != best["val_loss"]:
                    raise ValueError("Selected epoch must minimize source-validation loss, ties earliest")
            if (metrics["epochs_run"] != len(history) or len(history) > config["max_epochs"]
                    or metrics["selected_source_train"]["num_nodes"] != len(subset["train_indices"])
                    or metrics["source_val"]["num_nodes"] != len(source.val_indices)):
                raise ValueError("Metric counts or epoch budget mismatch")
            for metric in (metrics["selected_source_train"], metrics["source_val"]):
                if not math.isfinite(metric["loss"]) or metric["loss"] < 0 or not 0 <= metric["accuracy"] <= 1:
                    raise ValueError("Invalid classification metrics")
            if not math.isclose(metrics["source_val"]["loss"], best["val_loss"], rel_tol=1e-4, abs_tol=1e-5):
                raise ValueError("Saved source-validation score does not match selected checkpoint")
            logits = predict(model, source.graph)
            validate_logits(logits, source.graph.num_nodes, len(source.class_names))
            actual = score_logits(logits, source.val_indices, source.val_labels)
            if (not math.isclose(actual.loss, metrics["source_val"]["loss"], rel_tol=1e-4, abs_tol=1e-5)
                    or actual.accuracy != metrics["source_val"]["accuracy"]
                    or row["source_val_accuracy"] != actual.accuracy or best["val_accuracy"] != actual.accuracy):
                raise ValueError("Restored checkpoint does not reproduce its source-validation metrics")
            del model, payload
    except (KeyError, TypeError, RuntimeError, StopIteration) as error:
        raise ValueError(f"Malformed or incompatible bank: {error}") from error
    return VerifiedBank(directory, manifest, subsets, file_sha256(manifest_path))


def create_model_partition(bank: VerifiedBank, *, seed: int = 401,
                           group_counts: tuple[int, int, int] = (3, 1, 1)) -> dict[str, Any]:
    """Assign whole label subsets without using scores or checkpoint weights."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Partition seed must be an integer in [0,2**32)")
    ids = sorted(item["id"] for item in bank.subsets["subsets"])
    if len(group_counts) != 3 or any(type(n) is not int or n < 1 for n in group_counts) or sum(group_counts) != len(ids):
        raise ValueError("Three positive group counts must cover every source subset")
    rng = np.random.Generator(np.random.PCG64(seed))
    ordered = rng.permutation(ids).tolist()
    groups, roles, position = {}, {}, 0
    for role, count in zip(("meta_train", "validation", "test"), group_counts):
        groups[role] = ordered[position:position + count]
        position += count
        for subset in groups[role]:
            roles[subset] = role
    result = {"format_version": 1, "protocol": "whole-source-subset-partition-v1", "seed": seed,
              "bank_sha256": bank.sha256, "groups": groups,
              "model_ids": {role: [row["model_id"] for row in bank.manifest["models"]
                                   if roles[row["subset_id"]] == role] for role in groups},
              "claim": "held-out label selections on the same source graph; seed values shared across roles"}
    validate_model_partition(bank, result)
    return result


def validate_model_partition(bank: VerifiedBank, partition: dict[str, Any]) -> None:
    if file_sha256(bank.directory / "bank.json") != bank.sha256:
        raise ValueError("Bank changed since validation")
    if (partition.get("format_version") != 1 or partition.get("protocol") != "whole-source-subset-partition-v1"
            or partition.get("bank_sha256") != bank.sha256 or type(partition.get("seed")) is not int
            or not 0 <= partition["seed"] < 2**32):
        raise ValueError("Partition must refer to this bank snapshot")
    expected_roles = {"meta_train", "validation", "test"}
    groups, members = partition.get("groups", {}), partition.get("model_ids", {})
    if set(groups) != expected_roles or set(members) != expected_roles:
        raise ValueError("Partition needs meta_train/validation/test groups")
    all_groups = [name for names in groups.values() for name in names]
    expected = {item["id"] for item in bank.subsets["subsets"]}
    if len(all_groups) != len(expected) or set(all_groups) != expected or any(not names for names in groups.values()):
        raise ValueError("Source subsets must be disjoint and completely partitioned")
    for role in expected_roles:
        required = {row["model_id"] for row in bank.manifest["models"] if row["subset_id"] in groups[role]}
        if len(members[role]) != len(required) or set(members[role]) != required:
            raise ValueError("All models from one source subset must stay in the same partition")


def save_model_partition(path: Path, bank: VerifiedBank, partition: dict[str, Any]) -> None:
    validate_model_partition(bank, partition)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(partition, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_model_partition(path: Path, bank: VerifiedBank) -> dict[str, Any]:
    partition = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_model_partition(bank, partition)
    return partition
