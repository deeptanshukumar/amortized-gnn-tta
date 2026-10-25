"""A small, standard two-layer GCN; no adaptation or benchmark-specific logic."""

from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from .config import GCNConfig


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
