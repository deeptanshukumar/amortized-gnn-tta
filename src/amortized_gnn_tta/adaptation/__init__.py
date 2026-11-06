"""Amortized weight adaptation for GNNs under distribution shift."""

import importlib.util
from pathlib import Path

# 1. Re-export baseline episodic adaptation symbols from adaptation.py if present
_sibling_file = Path(__file__).resolve().parent.parent / "adaptation.py"
if _sibling_file.exists():
    _spec = importlib.util.spec_from_file_location("amortized_gnn_tta._adaptation_file", _sibling_file)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        for _k, _v in _mod.__dict__.items():
            if not _k.startswith("_"):
                globals()[_k] = _v

# 2. Amortized generator adaptation symbols
from .generator import (
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    apply_head_delta,
    extract_head_parameters,
)
from .meta_trainer import train_adaptation_generator
from .evaluator import (
    AdaptationRecord,
    evaluate_adaptation,
    evaluate_adaptation_pair,
    format_adaptation_report,
    subsample_environments,
)

__all__ = [
    # Generator symbols
    "AdaptationRecord",
    "GraphShiftDescriptor",
    "HeadAdapterGenerator",
    "apply_head_delta",
    "extract_head_parameters",
    "train_adaptation_generator",
    "evaluate_adaptation",
    "evaluate_adaptation_pair",
    "format_adaptation_report",
    "subsample_environments",
    # Baseline symbols (if present)
    "AdaptationConfig",
    "PEFTConfig",
    "WeightAdapter",
    "fit_unlabeled",
    "fit_supervised_development",
]
