"""Two-layer original-style GAT attention via PyG, with explicit head semantics."""

from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

from .config import GATConfig


class GAT(nn.Module):
    """Multihead GATConv → ELU → hidden dropout → single-head GATConv.

    GATConv handles self-loops and neighborhood attention normalization. This
    wrapper adds no graph preprocessing, input dropout, residual or output softmax.
    Attention-coefficient dropout is separate from hidden-feature dropout.
    """

    def __init__(self, config: GATConfig):
        super().__init__()
        self.config = config
        self.conv1 = GATConv(config.in_channels, config.hidden_channels // config.heads,
                             heads=config.heads, concat=True, negative_slope=0.2,
                             dropout=config.attention_dropout, add_self_loops=True, bias=True)
        self.conv2 = GATConv(config.hidden_channels, config.out_channels, heads=1, concat=False,
                             negative_slope=0.2, dropout=config.attention_dropout,
                             add_self_loops=True, bias=True)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """First-layer post-ELU representations, before hidden-feature dropout."""
        return F.elu(self.conv1(x, edge_index))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(self.encode(x, edge_index), p=self.config.dropout, training=self.training)
        return self.conv2(x, edge_index)
