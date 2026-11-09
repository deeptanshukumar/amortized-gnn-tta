"""Label-free candidate-agreement selection, deploying one actual update."""

import torch
from ._validation import finite, source_state
from .benchmark import PipelineResult, _inputs
from .generator import extract_target_parameters, adapted_forward, head_forward
from .lora_generator import LoRAGenerator


@torch.no_grad()
def agreement_predictions(
    model,
    graph,
    descriptor,
    generator,
    *,
    indices,
    seeds=(42, 43, 44, 45, 46),
    scope="head",
    device="cpu",
):
    """Include conditioning, every DDIM sample/prediction, and MSE selection.

    Agreement can select a common mistake. Seeds, node pool and sample count
    belong to the frozen protocol. Exact ties select the lowest candidate index.
    No labels or label-bearing bundle are accepted by this interface.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Candidate seeds must be nonempty and unique")
    if (
        indices.dtype != torch.long
        or indices.ndim != 1
        or not len(indices)
        or indices.unique().numel() != len(indices)
    ):
        raise ValueError("Invalid candidate node pool")
    if indices.min() < 0 or indices.max() >= graph.num_nodes:
        raise ValueError("Candidate indices outside graph")
    if isinstance(generator, LoRAGenerator) and scope != "all":
        raise ValueError("LoRA uses full functional schema")
    with source_state(model), source_state(generator):
        inputs = _inputs(graph, device)
        weights, schema = extract_target_parameters(model, scope)
        desc, hidden = descriptor.describe_with_hidden(model, inputs)
        core = generator.core if isinstance(generator, LoRAGenerator) else generator
        deltas, logits, probabilities, factors = [], [], [], []
        for seed in seeds:
            sampled = core.sample(
                desc, weights if core.include_head_weights else None, seed=seed
            )
            delta = (
                generator.update_space.dense_delta(sampled)
                if isinstance(generator, LoRAGenerator)
                else sampled
            )
            prediction = (
                head_forward(model, inputs, delta, schema, hidden)
                if scope == "head" and hidden is not None
                else adapted_forward(model, inputs, delta, schema)
            )
            factors.append(sampled if isinstance(generator, LoRAGenerator) else None)
            deltas.append(delta)
            logits.append(prediction)
            probabilities.append(prediction[indices.to(prediction.device)].softmax(-1))
        stack = finite(torch.stack(probabilities), "candidate probabilities")
        scores = (stack - stack.mean(0)).square().mean((1, 2))
        selected = int(scores.argmin())
        audit = dict(
            selected_index=selected,
            selected_seed=seeds[selected],
            seeds=list(seeds),
            mse=scores.cpu().tolist(),
            candidate_count=len(seeds),
            metric="mean squared probability difference over fixed node/class pool",
            tie_break="lowest index",
        )
        return PipelineResult(
            logits[selected],
            deltas[selected],
            {},
            factor_vector=factors[selected],
            selection=audit,
        ), audit
