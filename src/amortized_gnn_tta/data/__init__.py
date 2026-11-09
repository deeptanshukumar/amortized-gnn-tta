"""Model-neutral graph interface and explicit dataset adapters."""

from .loaders import make_environment_loader
from .api import DatasetBundle, GraphRecord, graph_diagnostics, validate_record
from .csbm import CSBMConfig, CSBMEnvironment, generate_csbm, load_dataset
from .shifts import CSBMShift, default_csbm_shifts, shifted_config
from .environments import (
    audit_environment_suite, create_environment_manifest, load_environment_manifest,
    load_shift_environment, save_environment_manifest, validate_environment_manifest,
)

__all__ = ["make_environment_loader", "CSBMConfig", "CSBMEnvironment", "DatasetBundle", "GraphRecord",
           "generate_csbm", "graph_diagnostics", "load_dataset", "validate_record",
           "CSBMShift", "default_csbm_shifts", "shifted_config", "audit_environment_suite",
           "create_environment_manifest", "load_environment_manifest", "load_shift_environment",
           "save_environment_manifest", "validate_environment_manifest"]
