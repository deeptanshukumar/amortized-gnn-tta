"""Small model configurations with a common total-hidden-width convention."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NodeClassifierConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int = 32
    dropout: float = 0.5

    def __post_init__(self) -> None:
        for name in ("in_channels", "out_channels", "hidden_channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.out_channels < 2:
            raise ValueError("This single-label classifier requires at least two logits")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")


@dataclass(frozen=True)
class GCNConfig(NodeClassifierConfig):
    """Original GCN settings; serialized field names/defaults are unchanged."""


@dataclass(frozen=True)
class GATConfig(NodeClassifierConfig):
    """hidden_channels is TOTAL concatenated width, not width per head."""

    heads: int = 4
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.heads, bool) or not isinstance(self.heads, int) or self.heads < 1:
            raise ValueError("heads must be a positive integer")
        if self.hidden_channels % self.heads:
            raise ValueError("Total hidden_channels must be divisible by heads")
        if not math.isfinite(self.attention_dropout) or not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be finite and in [0,1)")


@dataclass(frozen=True)
class GraphSAGEConfig(NodeClassifierConfig):
    """Full-neighborhood mean aggregation with root weights, no L2 normalization."""
