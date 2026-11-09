"""Inference-only descriptor mismatch diagnostic; never a fitting objective."""
import torch
from ._validation import source_state, finite
from .benchmark import PipelineResult, _inputs
from .generator import extract_target_parameters, adapted_forward
from .lora_generator import LoRAGenerator


@torch.no_grad()
def donor_conditioning_predictions(model, target_graph, donor_graph, descriptor,
                                    generator, *, scope='all', device='cpu'):
    """Use this same source model's summary of a different unlabeled graph.

    The source weights and graph receiving the update remain fixed. Descriptor D
    therefore retains the same source-weight component. No labels enter this API.
    This is a sensitivity diagnostic, not a new adaptation method or timing case.
    """
    with source_state(model), source_state(generator):
        target, donor = _inputs(target_graph, device), _inputs(donor_graph, device)
        weights, schema = extract_target_parameters(model, scope)
        conditioning = descriptor(model, donor)
        factors = generator.forward_factors(conditioning) if isinstance(generator, LoRAGenerator) else None
        delta = finite(generator.update_space.dense_delta(factors) if factors is not None
                       else generator(conditioning, weights), 'donor-conditioned delta')
        logits = adapted_forward(model, target, delta, schema)
    return PipelineResult(logits, delta, {}, factor_vector=factors)
