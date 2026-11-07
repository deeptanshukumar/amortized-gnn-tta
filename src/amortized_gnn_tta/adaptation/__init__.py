"""Amortized weight adaptation for GNNs under distribution shift."""

from .iterative import (AdaptationConfig, AdaptationResult, DevelopmentSupervision, PEFTConfig, WeightAdapter, adaptation_targets, development_supervision, unlabeled_objective, fit_unlabeled, fit_supervised_development)

# 2. Amortized generator adaptation symbols
from .generator import (
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    adapted_forward,
    apply_head_delta,
    apply_target_delta,
    extract_compact_weight_descriptor,
    extract_head_parameters,
    extract_hidden_representation_descriptor,
    extract_predictive_uncertainty_descriptor,
    extract_target_graph_descriptor,
    extract_target_parameters,
)
from .diffusion import (
    DiffusionHeadGenerator,
    NoiseSchedule,
    OracleTargetRecord,
    SinusoidalPositionalEmbedding,
    fit_oracle_deltas,
    train_diffusion_generator,
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
    "DiffusionHeadGenerator",
    "GraphShiftDescriptor",
    "HeadAdapterGenerator",
    "NoiseSchedule",
    "adapted_forward",
    "apply_head_delta",
    "apply_target_delta",
    "extract_compact_weight_descriptor",
    "extract_head_parameters",
    "extract_hidden_representation_descriptor",
    "extract_predictive_uncertainty_descriptor",
    "extract_target_graph_descriptor",
    "extract_target_parameters",
    "fit_oracle_deltas",
    "train_adaptation_generator",
    "train_diffusion_generator",
    "evaluate_adaptation",
    "evaluate_adaptation_pair",
    "format_adaptation_report",
    "subsample_environments",
    # Baseline symbols (if present)
    "unlabeled_objective",
    "AdaptationResult",
    "DevelopmentSupervision",
    "adaptation_targets",
    "development_supervision",
    "AdaptationConfig",
    "PEFTConfig",
    "WeightAdapter",
    "fit_unlabeled",
    "fit_supervised_development",
]
