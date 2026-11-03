"""Freeze and audit controlled CSBM environments, loading one graph at a time.

Default: 300 nodes x 32 features for a quick construction check. --paper-scale
uses dimensions in the config (5000 x 2000). Both save the exact effective
configuration BEFORE construction auditing. No models are trained or scored.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from amortized_gnn_tta.data import (
    CSBMConfig, CSBMShift, audit_environment_suite, create_environment_manifest,
    load_environment_manifest, save_environment_manifest,
)
from amortized_gnn_tta.data.environments import ENVIRONMENT_PROTOCOL
from amortized_gnn_tta.training import software_versions


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]
                        / "configs" / "controlled_csbm_v1.json")
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--num-nodes", type=int, default=None, help="Override node count for non-paper runs")
    parser.add_argument("--num-features", type=int, default=None, help="Override feature count for non-paper runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() or args.threads < 1:
        parser.error("Use a new output directory and a positive thread count")
    torch.set_num_threads(args.threads)
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    if protocol["protocol"] != ENVIRONMENT_PROTOCOL:
        parser.error("Unsupported environment protocol")
    reference = dict(protocol["reference"])
    reference["split_ratios"] = tuple(reference["split_ratios"])
    if not args.paper_scale:
        nodes = args.num_nodes if args.num_nodes is not None else 300
        features = args.num_features if args.num_features is not None else 32
        reference.update(num_nodes=nodes, num_features=features)
    manifest = create_environment_manifest(
        CSBMConfig.from_matcha(**reference), [CSBMShift(**s) for s in protocol["shifts"]],
        lineage_counts=tuple(protocol["lineage_counts"]), train_severities=protocol["train_severities"],
        test_severities=protocol["test_severities"], generation_seed=protocol["generation_seed"],
        partition_seed=protocol["partition_seed"], held_out_families=protocol["held_out_families"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_environment_manifest(args.output_dir / "environment_manifest.json", manifest)
    assert load_environment_manifest(args.output_dir / "environment_manifest.json") == manifest
    root = Path(__file__).resolve().parents[1]
    snapshots = {}
    for path in [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml", Path(__file__)]:
        relative = path.resolve().relative_to(root)
        target = args.output_dir / "construction_code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        snapshots[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                  capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                capture_output=True, text=True).stdout
        git = {"revision": revision, "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        git = {"revision": None, "dirty": None, "note": "Git state unavailable"}
    write_json(args.output_dir / "run.json", {
        "stage": "configuration_frozen_before_audit", "effective_reference": reference, "paper_scale": args.paper_scale,
        "versions": software_versions(), "threads": args.threads,
        "manifest_sha256": manifest["manifest_sha256"], "requested_protocol": protocol,
        "git": git, "construction_code_snapshot_sha256": snapshots,
    })
    start = time.perf_counter()
    audit = audit_environment_suite(manifest)
    audit["audit_seconds"] = time.perf_counter() - start
    write_json(args.output_dir / "construction_audit.json", audit)
    summary = {
        "status": audit["status"], "output_dir": str(args.output_dir),
        "nodes_per_graph": reference["num_nodes"], "features_per_node": reference["num_features"],
        "lineage_counts": audit["lineage_counts"], "environment_counts": audit["partition_counts"],
        "shift_definitions": len(manifest["shifts"]),
        "family_counts": dict(Counter(e["family"] for e in manifest["environments"])),
        "duplicate_variant_groups": len(audit["duplicate_variants_within_lineage"]),
        "audit_seconds": audit["audit_seconds"], "versions": software_versions(),
        "scope": "Construction/partition audit only; no model accuracy or adaptation measured",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
