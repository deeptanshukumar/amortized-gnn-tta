"""Amortized weight adaptation for GNNs under distribution shift."""

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
)

__all__ = [
    "AdaptationRecord",
    "GraphShiftDescriptor",
    "HeadAdapterGenerator",
    "apply_head_delta",
    "extract_head_parameters",
    "train_adaptation_generator",
    "evaluate_adaptation",
    "evaluate_adaptation_pair",
    "format_adaptation_report",
]
