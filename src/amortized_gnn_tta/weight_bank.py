"""Sequential source-only model banks with per-model evidence and no overwrites.

The bank is a Cartesian product of saved label subsets and explicit model seeds.
Source-test/target supervision is absent. Generator partitions and diversity
diagnostics are separate steps; a completed bank does not establish either.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from .evaluation import predict, score_logits, select_device
from .source_subsets import save_source_subsets, source_subset_view, validate_source_subsets
from .training import (
    SourceTrainingData, TrainingConfig, fit_source, load_checkpoint, save_checkpoint, software_versions,
)

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _write_progress(directory: Path, manifest: dict[str, Any]) -> None:
    # Only replace our own progress file inside a newly created run directory.
    # Same-directory rename avoids readers seeing a half-written JSON document.
    temporary = directory / "bank.json.tmp"
    _write_json(temporary, manifest)
    temporary.replace(directory / "bank.json")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_weight_bank(model_factory: Callable[[], nn.Module], source: SourceTrainingData,
                      subsets: dict[str, Any], training: TrainingConfig, *, model_spec: dict[str, Any],
                      model_seeds: Sequence[int], output_dir: Path,
                      device: str | torch.device = "auto",
                      run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit every subset/seed pair, saving restored source-val-selected weights.

    The same seed list is reused across subsets as a paired factorial design,
    not a claim of independent IID model draws. Generic API sizes are arbitrary;
    the example enforces five subsets and six seeds. No model is pruned by score.
    Each checkpoint is reloaded on CPU and source predictions checked for agreement.

    Existing output directories are rejected. Failures preserve completed files
    and mark the bank failed when possible; this milestone does not resume jobs.
    """
    validate_source_subsets(subsets, source)
    seeds = list(model_seeds)
    if (not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Model seeds must be distinct integers in [0,2**32)")
    chosen = select_device(device)
    # Normalize audit metadata before creating output or starting expensive work.
    spec = json.loads(json.dumps(model_spec, allow_nan=False))
    metadata = json.loads(json.dumps(run_metadata or {}, allow_nan=False))
    config = asdict(training)
    del config["model_seed"]  # Actual seeds are explicit per model, not this unused template value.
    models = []
    for subset in subsets["subsets"]:
        for seed in seeds:
            model_id = f"{subset['id']}_seed_{seed}"
            models.append({"model_id": model_id, "subset_id": subset["id"], "model_seed": seed,
                           "checkpoint": f"models/{model_id}/best_model.pt",
                           "metrics": f"models/{model_id}/metrics.json", "status": "pending"})
    manifest = {
        "format_version": 1, "protocol": "source-weight-bank-v1", "status": "running",
        "purpose": "source-only pretrained bank; no target selection, adaptation, or generator partition",
        "model_spec": spec, "training_config_without_seed": config, "model_seeds": seeds,
        "seed_design": "same seed list reused across subsets; paired initialization/training streams",
        "requested_device": str(device), "execution_device": str(chosen),
        "cpu_threads": torch.get_num_threads(), "versions": software_versions(), "run_metadata": metadata,
        "subset_manifest": "source_subsets.json", "expected_models": len(models),
        "completed_models": 0, "models": models,
        "selection": "minimum source-validation CE per model; retain every subset/seed pair",
        "cpu_reload_tolerance": {"rtol": 1e-4, "atol": 1e-5},
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    _write_progress(directory, manifest)
    current = None
    try:
        save_source_subsets(directory / "source_subsets.json", subsets, source)
        manifest["subset_manifest_sha256"] = _file_hash(directory / "source_subsets.json")
        for subset in subsets["subsets"]:
            view = source_subset_view(source, subsets, subset["id"])
            for current in (row for row in models if row["subset_id"] == subset["id"]):
                current["status"] = "running"
                _write_progress(directory, manifest)
                started = time.perf_counter()
                result = fit_source(model_factory, view, replace(training, model_seed=current["model_seed"]),
                                    device=chosen)
                fit_seconds = time.perf_counter() - started
                checkpoint = directory / current["checkpoint"]
                save_checkpoint(checkpoint, result, spec)
                restored, _ = load_checkpoint(checkpoint, lambda _: model_factory(), device="cpu")
                logits = predict(result.model, view.graph)
                torch.testing.assert_close(logits, predict(restored, view.graph), rtol=1e-4, atol=1e-5)
                metrics = {
                    "model_id": current["model_id"], "subset_id": subset["id"],
                    "training_config": asdict(result.config), "model_spec": spec,
                    "best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss,
                    "epochs_run": len(result.history), "fit_seconds": fit_seconds,
                    "selected_source_train": asdict(score_logits(logits, view.train_indices, view.train_labels)),
                    "source_val": asdict(score_logits(logits, view.val_indices, view.val_labels)),
                    "cpu_reload_predictions_close": True, "history": result.history,
                }
                _write_json(directory / current["metrics"], metrics)
                current.update({"status": "complete", "best_epoch": result.best_epoch,
                                "epochs_run": len(result.history), "best_val_loss": result.best_val_loss,
                                "source_val_accuracy": metrics["source_val"]["accuracy"],
                                "fit_seconds": fit_seconds, "cpu_reload_predictions_close": True,
                                "checkpoint_sha256": _file_hash(checkpoint),
                                "metrics_sha256": _file_hash(directory / current["metrics"])})
                manifest["completed_models"] += 1
                _write_progress(directory, manifest)
                logger.info("Completed %s (%d/%d), selected epoch %d", current["model_id"],
                            manifest["completed_models"], len(models), result.best_epoch)
                # Keep memory bounded to this job, not all model instances in the bank.
                del result, restored, logits
        manifest["status"] = "complete"
        _write_progress(directory, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        if current is not None and current["status"] != "complete":
            current["status"] = "failed"
        try:
            _write_progress(directory, manifest)
        except OSError:
            logger.exception("Could not persist failed-bank status")
        raise
    return deepcopy(manifest)
