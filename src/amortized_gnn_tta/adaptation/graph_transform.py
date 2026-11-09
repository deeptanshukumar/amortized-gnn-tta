"""Independent feature-only GTrans contrastive objective implementation.

Reference: Jin et al., ICLR 2023, arXiv:2210.03561, and official
ChandlerBang/GTrans gtransform_feat.py (11e5ae5). This variant optimizes features
only, using our one-hidden-layer embeddings and explicit target node pool. It is
not a reproduction of GTrans's joint feature/structure experiment. No upstream
source text is copied. The canonical input and pretrained weights stay frozen.
"""

import math
import torch
from torch.nn import functional as F
from ._validation import source_state, checked_backward, positive_int
from .benchmark import PipelineResult, _inputs
from ..data.corruptions import perturb_graph
from ..evaluation import validate_logits


@torch.enable_grad()
def feature_gtrans_predictions(
    model,
    graph,
    *,
    indices,
    steps=30,
    lr=0.001,
    edge_drop=0.05,
    seed=1771,
    device="cpu",
):
    positive_int(steps, "steps")
    if isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid feature learning rate")
    if (
        indices.dtype != torch.long
        or indices.ndim != 1
        or not len(indices)
        or indices.unique().numel() != len(indices)
    ):
        raise ValueError("Invalid unlabeled node pool")
    if indices.min() < 0 or indices.max() >= graph.num_nodes:
        raise ValueError("Node pool outside graph")
    positive = perturb_graph(
        graph,
        feature_mask_rate=0,
        edge_drop_rate=edge_drop,
        feature_seed=seed,
        edge_seed=seed + 1,
        undirected=graph.is_undirected(),
    )
    rng = torch.Generator().manual_seed(seed)
    with source_state(model, freeze=True):
        inputs = _inputs(graph, device)
        positive_edges = positive.edge_index.to(device)
        indices = indices.to(device)
        delta = torch.nn.Parameter(torch.full_like(inputs.x, 1e-7))
        optimizer = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            x = inputs.x + delta
            permutation = torch.randperm(inputs.num_nodes, generator=rng).to(device)
            original = model.encode(x, inputs.edge_index)[indices]
            augmented = model.encode(x, positive_edges)[indices]
            negative = model.encode(x[permutation], inputs.edge_index)[indices]
            # Positive distance - negative distance; fixed unit-temperature cosine.
            loss = (1 - F.cosine_similarity(augmented, original, eps=1e-15)).mean() - (
                1 - F.cosine_similarity(original, negative, eps=1e-15)
            ).mean()
            checked_backward(loss, [delta])
            optimizer.step()
        with torch.no_grad():
            logits = model(inputs.x + delta, inputs.edge_index)
            validate_logits(logits, inputs.num_nodes, model.config.out_channels)
    # This delta changes features, not GNN weights; do not export as weight delta.
    return PipelineResult(logits, None, {})
