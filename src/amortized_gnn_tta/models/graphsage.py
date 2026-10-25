"""Two-layer full-neighborhood mean GraphSAGE via PyG."""

from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from .config import GraphSAGEConfig


class GraphSAGE(nn.Module):
    """SAGEConv(mean) → ReLU → hidden dropout → SAGEConv(mean), raw logits.

    Root-node features have their own learned transform. We do not add graph
    self-loops, normalize embeddings, project neighbors, or sample neighborhoods.
    Full-batch aggregation matches the current small-CSBM visibility protocol;
    it is not a reproduction of the original sampled training procedure.
    """

    def __init__(self, config: GraphSAGEConfig):
        super().__init__()
        self.config = config
        self.conv1 = SAGEConv(config.in_channels, config.hidden_channels, aggr="mean",
                              normalize=False, root_weight=True, project=False, bias=True)
        self.conv2 = SAGEConv(config.hidden_channels, config.out_channels, aggr="mean",
                              normalize=False, root_weight=True, project=False, bias=True)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """First-layer post-ReLU representations, before hidden dropout."""
        return F.relu(self.conv1(x, edge_index))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(self.encode(x, edge_index), p=self.config.dropout, training=self.training)
        return self.conv2(x, edge_index)
