"""Explicit construction for the three supported models; no dynamic imports."""

from dataclasses import asdict
from typing import Any

from torch import nn

from .config import GATConfig, GCNConfig, GraphSAGEConfig
from .gat import GAT
from .gcn import GCN
from .graphsage import GraphSAGE

ARCHITECTURES = ("gcn", "gat", "graphsage")


def make_model_spec(architecture: str, in_channels: int, out_channels: int, *,
                    hidden_channels: int = 32, dropout: float = 0.5,
                    heads: int | None = None, attention_dropout: float | None = None) -> dict[str, Any]:
    """Return validated, plain checkpoint metadata using total hidden width."""
    common = dict(in_channels=in_channels, out_channels=out_channels,
                  hidden_channels=hidden_channels, dropout=dropout)
    if architecture == "gat":
        extra = {}
        if heads is not None:
            extra["heads"] = heads
        if attention_dropout is not None:
            extra["attention_dropout"] = attention_dropout
        config = GATConfig(**common, **extra)
    elif architecture in ("gcn", "graphsage"):
        if heads is not None or attention_dropout is not None:
            raise ValueError("heads/attention_dropout are GAT-only settings")
        config = GCNConfig(**common) if architecture == "gcn" else GraphSAGEConfig(**common)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return {"architecture": f"two_layer_{architecture}", "config": asdict(config)}


def build_model(spec: dict[str, Any]) -> nn.Module:
    """Restore one known architecture from its explicit configuration.

    The original two_layer_gcn spec and its state_dict keys remain compatible.
    Nothing in a checkpoint is treated as an import path or executable model.
    """
    if not isinstance(spec, dict) or set(spec) != {"architecture", "config"} or not isinstance(spec["config"], dict):
        raise ValueError("Model spec requires architecture and configuration records")
    try:
        if spec["architecture"] == "two_layer_gcn":
            return GCN(GCNConfig(**spec["config"]))
        if spec["architecture"] == "two_layer_gat":
            return GAT(GATConfig(**spec["config"]))
        if spec["architecture"] == "two_layer_graphsage":
            return GraphSAGE(GraphSAGEConfig(**spec["config"]))
    except TypeError as error:
        raise ValueError(f"Invalid model configuration: {error}") from error
    raise ValueError(f"Unknown checkpoint architecture: {spec['architecture']}")
