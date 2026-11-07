"""Complete, versioned inference recipes; no optimizer-resumption claim."""

import json
import os
from pathlib import Path
import tempfile

import torch

from ..adaptation_artifacts import state_sha256
from ..training import software_versions
from ._validation import finite
from .descriptors import GraphShiftDescriptor
from .diffusion import DiffusionHeadGenerator
from .generator import HeadAdapterGenerator, extract_target_parameters, unflatten_head_delta

PROTOCOL = "csbm-direct-generator-v2"


def atomic_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_generator_checkpoint(path, generator, descriptor, schema, *, target_scope="head",
                              provenance, history=(), state_dict=None, overwrite=False):
    """Atomically save all constructor, conditioning and sampling settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    config = {name: getattr(generator, name) for name in
              ("descriptor_dim", "head_dim", "hidden_dim", "num_layers", "include_head_weights")}
    if isinstance(generator, DiffusionHeadGenerator):
        kind = "diffusion"
        config.update(diffusion_steps=generator.schedule.timesteps, sampling_steps=generator.sampling_steps,
                      time_dim=generator.time_dim, schedule=generator.schedule.schedule, default_seed=generator.default_seed)
    elif isinstance(generator, HeadAdapterGenerator):
        kind = "hypernetwork"
    else:
        raise ValueError("Unsupported generator type")
    if descriptor.conditioning and generator.include_head_weights:
        raise ValueError("A-D must not receive a separate raw-weight input")
    if descriptor.output_dim != generator.descriptor_dim:
        raise ValueError("Generator and descriptor dimensions disagree")
    unflatten_head_delta(torch.zeros(generator.head_dim), schema)
    state = {name: value.detach().cpu().clone() for name, value in
             (state_dict if state_dict is not None else generator.state_dict()).items()}
    for value in state.values():
        finite(value, "checkpoint state")
    # Keep metadata weights_only-compatible and ensure NaNs cannot enter JSON.
    metadata = json.loads(json.dumps({"provenance": provenance, "history": list(history)}, allow_nan=False))
    payload = {"protocol": PROTOCOL, "generator_type": kind, "generator_config": config,
               "descriptor_config": descriptor.configuration(), "target_scope": target_scope,
               "schema": list(schema), "state_dict": state, "state_sha256": state_sha256(state),
               "software": software_versions(), **metadata}
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Exclusive creation without exposing a partially written checkpoint.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_generator_checkpoint(path, *, model=None, device="cpu"):
    """Reconstruct inference with strict state/schema validation and CPU loading.

    Legacy incomplete checkpoints require an explicit external recipe; their
    missing sampling or conditioning settings must never be guessed here.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported/incomplete checkpoint; supply an explicit legacy migration recipe")
    state = payload["state_dict"]
    for value in state.values():
        finite(value, "checkpoint state")
    if state_sha256(state) != payload["state_sha256"]:
        raise ValueError("Checkpoint state hash mismatch")
    descriptor = GraphShiftDescriptor(**payload["descriptor_config"])
    constructors = {"hypernetwork": HeadAdapterGenerator, "diffusion": DiffusionHeadGenerator}
    if payload["generator_type"] not in constructors:
        raise ValueError("Unknown generator type")
    # Reconstructing weights should not perturb the caller's CPU initialization RNG.
    with torch.random.fork_rng(devices=[]):
        generator = constructors[payload["generator_type"]](**payload["generator_config"])
    if generator.descriptor_dim != descriptor.output_dim or (descriptor.conditioning and generator.include_head_weights):
        raise ValueError("Inconsistent conditioning recipe")
    generator.load_state_dict(state, strict=True)
    unflatten_head_delta(torch.zeros(generator.head_dim), payload["schema"])
    if model is not None:
        flat, schema = extract_target_parameters(model, payload["target_scope"])
        if schema != [(name, tuple(shape)) for name, shape in payload["schema"]] or len(flat) != generator.head_dim:
            raise ValueError("Source model disagrees with checkpoint schema")
        if ((descriptor.conditioning != "A" and model.config.hidden_channels != descriptor.hidden_channels) or
                (descriptor.conditioning is not None and model.config.in_channels != descriptor.in_channels) or
                (descriptor.conditioning in {"C", "D"} and model.config.out_channels != descriptor.out_channels)):
            raise ValueError("Source model disagrees with conditioning dimensions")
    return generator.to(device).eval(), descriptor.to(device).eval(), payload
