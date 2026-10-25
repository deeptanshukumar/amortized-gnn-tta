"""Versioned, whole-lineage environment partitions with lazy CSBM generation.

The manifest contains offline recipes, not model conditioning. All variants of
one fresh draw share one role. No trained model, held-out score, or target-label
information is used to assign roles, severities, or optional held-out families.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .api import DatasetBundle, graph_diagnostics
from .csbm import CSBMConfig, PROTOCOL as CSBM_PROTOCOL, generate_csbm
from .shifts import CSBMShift, SHIFT_FAMILIES, SHIFT_PROTOCOL, shifted_config

ENVIRONMENT_PROTOCOL = "csbm-environment-partitions-v1"
PARTITION_ROLES = ("meta_train", "validation", "target_test")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _seed(value: int) -> None:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Seeds must be integers in [0,2**32)")


def _severities(values: Sequence[float]) -> list[float]:
    values = list(values)
    if (not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                         or not math.isfinite(v) or not 0 < v <= 1 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("Severities must be distinct finite numbers in (0,1]")
    return sorted(float(v) for v in values)


def create_environment_manifest(
    reference: CSBMConfig, shifts: Sequence[CSBMShift], *,
    lineage_counts: tuple[int, int, int] = (3, 1, 1),
    train_severities: Sequence[float] = (0.25, 0.5),
    test_severities: Sequence[float] = (0.5, 0.75, 1.0),
    generation_seed: int = 601, partition_seed: int = 501,
    held_out_families: Sequence[str] = (),
) -> dict[str, Any]:
    """Plan fresh draws and allocate whole paired lineages before any scoring.

    Validation uses the SAME family/severity support as meta-training, on fresh
    lineages. Test can include seen/unseen severity and optional unseen broad
    families. Counts refer to BASE DRAWS, not variants. The reference describes
    the existing source distribution; its edge/feature seeds are excluded from
    new lineages. Existing source graphs/checkpoints are neither loaded nor used.
    """
    _seed(generation_seed)
    _seed(partition_seed)
    if (len(lineage_counts) != 3 or any(type(n) is not int or n < 1 for n in lineage_counts)):
        raise ValueError("Use three positive lineage counts: meta_train, validation, target_test")
    train_levels, test_levels = _severities(train_severities), _severities(test_severities)
    shifts = list(shifts)
    if (not shifts or any(not isinstance(s, CSBMShift) for s in shifts)
            or len({s.name for s in shifts}) != len(shifts)):
        raise ValueError("Supply shifts with distinct names")
    # Reject duplicate definitions under different names; they are not independent
    # environments and must not receive extra experimental weight accidentally.
    signatures = {(s.attribute_offset, s.homophily_parameter, s.degree_scale) for s in shifts}
    if len(signatures) != len(shifts):
        raise ValueError("Duplicate shift definitions are not distinct conditions")
    heldout = list(held_out_families)
    families = {s.family for s in shifts}
    if (any(f not in SHIFT_FAMILIES for f in heldout) or len(set(heldout)) != len(heldout)
            or not set(heldout) <= families or not families - set(heldout)):
        raise ValueError("Held-out families must exist and leave a meta-training shift family")
    for shift in shifts:
        shifted_config(reference, shift, 1.0)  # Validate the complete endpoint too.
    count = sum(lineage_counts)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([partition_seed, 3])))
    assignments = {}
    for role, indices in zip(PARTITION_ROLES, np.split(rng.permutation(count), np.cumsum(lineage_counts)[:-1])):
        assignments.update({int(i): role for i in indices})
    lineages, environments = [], []
    used = [{reference.edge_seed}, {reference.feature_seed}, {reference.split_seed}]
    for i in range(count):
        seeds = [int(v) for v in np.random.SeedSequence([generation_seed, i, 4]).generate_state(3)]
        if any(seed in pool for seed, pool in zip(seeds, used)):
            raise ValueError("Generation seed produced a seed collision; select another generation_seed")
        for seed, pool in zip(seeds, used):
            pool.add(seed)
        base = replace(reference, edge_seed=seeds[0], feature_seed=seeds[1], split_seed=seeds[2])
        lineage_id, role = f"lineage_{i:04d}", assignments[i]
        lineages.append({"id": lineage_id, "role": role, "config": _json_value(asdict(base)),
                         "recipe_sha256": _digest(asdict(base))})
        levels = test_levels if role == "target_test" else train_levels
        conditions = [(None, 0.0)] + [(s, level) for s in shifts
                      if role == "target_test" or s.family not in heldout for level in levels]
        for shift, level in conditions:
            family = shift.family if shift else "control"
            condition = shift.name if shift else "control"
            environments.append({
                "id": f"{lineage_id}__{condition}__s{level:g}",
                "lineage_id": lineage_id, "role": role,
                "shift": shift.name if shift else None, "family": family, "severity": level,
                "seen_family_in_meta_train": family == "control" or family not in heldout,
                "seen_severity_in_meta_train": level == 0 or level in train_levels,
                "config": _json_value(asdict(shifted_config(base, shift, level) if shift else base)),
            })
    payload = {
        "format_version": 1, "protocol": ENVIRONMENT_PROTOCOL, "shift_protocol": SHIFT_PROTOCOL,
        "csbm_protocol": CSBM_PROTOCOL,
        "purpose": "development controlled shifts; not a sealed final benchmark",
        "creation_software": {"numpy": np.__version__, "torch": str(torch.__version__)},
        "reference": _json_value(asdict(reference)), "shifts": [asdict(s) for s in shifts],
        "lineage_counts": list(lineage_counts), "train_severities": train_levels,
        "validation_severities": train_levels, "test_severities": test_levels,
        "held_out_families": sorted(heldout), "generation_seed": generation_seed,
        "partition_seed": partition_seed,
        "partition_policy": "whole_base_lineage; validation matches meta-training support",
        "coupling": "shared edge uniforms and Gaussian feature noise within each lineage",
        "lineages": lineages, "environments": environments,
    }
    return {**payload, "manifest_sha256": _digest(payload)}


def _config(value: dict[str, Any]) -> CSBMConfig:
    return CSBMConfig(**{**value, "split_ratios": tuple(value["split_ratios"])})


def validate_environment_manifest(manifest: dict[str, Any]) -> None:
    """Verify the checksum AND all deterministic recipes/assignments.

    Re-signing a malformed partition does not bypass validation. This is content
    validation, not an authentication mechanism or guarantee about a file's owner.
    No graphs/labels/scores need to be materialized to perform this check.
    """
    try:
        payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if manifest["manifest_sha256"] != _digest(payload):
            raise ValueError("Environment manifest checksum mismatch")
        expected = create_environment_manifest(
            _config(manifest["reference"]), [CSBMShift(**s) for s in manifest["shifts"]],
            lineage_counts=tuple(manifest["lineage_counts"]),
            train_severities=manifest["train_severities"], test_severities=manifest["test_severities"],
            generation_seed=manifest["generation_seed"], partition_seed=manifest["partition_seed"],
            held_out_families=manifest["held_out_families"],
        )
        software = manifest["creation_software"]
        if (not isinstance(software, dict) or set(software) != {"numpy", "torch"}
                or any(not isinstance(v, str) or not v for v in software.values())):
            raise ValueError("Manifest needs its creation software versions")
        # Versions are provenance, not partition inputs. Preserve old versions
        # while checking deterministic recipes on the CURRENT stack. Re-audit
        # regenerated tensor fingerprints after changing the software stack.
        expected["creation_software"] = software
        expected["manifest_sha256"] = _digest({k: v for k, v in expected.items() if k != "manifest_sha256"})
        if manifest != expected:
            raise ValueError("Environment manifest recipes or partition assignments are inconsistent")
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed environment manifest") from exc


def save_environment_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Save recipes before inspecting model performance; refuse overwrites."""
    validate_environment_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_environment_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_environment_manifest(manifest)
    return manifest


def load_shift_environment(manifest: dict[str, Any], environment_id: str) -> DatasetBundle:
    """Materialize ONE paired variant through the existing common data boundary.

    Call unlabeled_input(id) before passing data to a model/adaptation method.
    Construction uses latent synthetic classes offline; no shift name, severity,
    generation parameter, ground truth or lineage metadata enters that graph.
    Regeneration uses the existing sampler: unchanged components are bitwise
    identical within the tested stack. No persistent or global graph cache exists.
    """
    validate_environment_manifest(manifest)
    item = next((e for e in manifest["environments"] if e["id"] == environment_id), None)
    if item is None:
        raise ValueError(f"Unknown environment ID: {environment_id}")
    record = generate_csbm(_config(item["config"]), role=item["role"])
    record.metadata["controlled_shift"] = {key: item[key] for key in (
        "lineage_id", "shift", "family", "severity", "seen_family_in_meta_train",
        "seen_severity_in_meta_train")}
    record.metadata["controlled_shift"]["manifest_sha256"] = manifest["manifest_sha256"]
    bundle = DatasetBundle({environment_id: record}, metadata={
        "dataset": "csbm", "protocol": CSBM_PROTOCOL, "shift_protocol": SHIFT_PROTOCOL,
        "environment_protocol": ENVIRONMENT_PROTOCOL, "task": "single_label_node_classification",
        "metric": "accuracy", "software": {"numpy": np.__version__, "torch": str(torch.__version__)},
    })
    bundle.validate()
    return bundle


def audit_environment_suite(manifest: dict[str, Any]) -> dict[str, Any]:
    """Execute all recipes sequentially and check pairing and input identities.

    Also detects exact duplicate inputs across fresh lineages (possible for
    degenerate/noiseless configurations despite distinct seeds). Identical
    variants within ONE lineage are reported rather than secretly resampled:
    finite Bernoulli draws need not change at every small parameter step.
    The report contains offline label-derived diagnostics, NOT model inputs.
    No model accuracy or adaptation result is computed.
    """
    validate_environment_manifest(manifest)
    rows, fingerprints, bases = [], {}, []
    for lineage in manifest["lineages"]:
        base = generate_csbm(_config(lineage["config"]), role=lineage["role"])
        bases.append({"id": lineage["id"], "role": lineage["role"],
                      "fingerprint": base.metadata["fingerprint"]})
        for item in manifest["environments"]:
            if item["lineage_id"] != lineage["id"]:
                continue
            bundle = load_shift_environment(manifest, item["id"])
            record = bundle.graphs[item["id"]]
            same_features = torch.equal(base.graph.x, record.graph.x)
            same_edges = torch.equal(base.graph.edge_index, record.graph.edge_index)
            if not torch.equal(base.labels, record.labels) or any(
                not torch.equal(base.splits[key], record.splits[key]) for key in base.splits
            ):
                raise ValueError("Paired shift changed labels or native split membership")
            if ((item["family"] in {"control", "structural"} and not same_features)
                    or (item["family"] in {"control", "attribute"} and not same_edges)):
                raise ValueError("Paired shift changed a component required to stay fixed")
            fingerprint = record.metadata["fingerprint"]
            previous = fingerprints.setdefault(fingerprint, [])
            if any(row["lineage_id"] != lineage["id"] for row in previous):
                raise ValueError("Different lineages produced duplicate graph inputs; revise the protocol explicitly")
            previous.append({"id": item["id"], "lineage_id": lineage["id"]})
            rows.append({key: item[key] for key in ("id", "lineage_id", "role", "family", "severity",
                                                   "seen_family_in_meta_train", "seen_severity_in_meta_train")}
                        | {"fingerprint": fingerprint, "features_unchanged": same_features,
                           "edges_unchanged": same_edges, "labels_and_splits_unchanged": True,
                           "expected_degree": record.metadata["expected_degree"],
                           "diagnostics": graph_diagnostics(record)})
    return {
        "status": "complete", "purpose": "offline construction audit; no model performance measured",
        "manifest_sha256": manifest["manifest_sha256"],
        "creation_software": manifest["creation_software"],
        "software": {"numpy": np.__version__, "torch": str(torch.__version__)},
        "partition_counts": {role: sum(e["role"] == role for e in rows) for role in PARTITION_ROLES},
        "lineage_counts": {role: sum(e["role"] == role for e in bases) for role in PARTITION_ROLES},
        "duplicate_variants_within_lineage": [group for group in fingerprints.values() if len(group) > 1],
        "lineages": bases, "environments": rows,
    }
