"""A small, standard two-layer GCN; no adaptation or benchmark-specific logic."""

from dataclasses import dataclass
import math

from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


@dataclass(frozen=True)
class GCNConfig:
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


class GCN(nn.Module):
    """GCNConv → ReLU → dropout → GCNConv, returning raw class logits.

    Each layer handles self-loops and symmetric normalization. Caching is disabled:
    a source normalization must not be reused on a different target graph.
    """

    def __init__(self, config: GCNConfig):
        super().__init__()
        self.config = config
        self.conv1 = GCNConv(config.in_channels, config.hidden_channels, cached=False)
        self.conv2 = GCNConv(config.hidden_channels, config.out_channels, cached=False)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """First-layer post-ReLU representations, before training dropout."""
        return F.relu(self.conv1(x, edge_index))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.encode(x, edge_index)
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        return self.conv2(x, edge_index)
