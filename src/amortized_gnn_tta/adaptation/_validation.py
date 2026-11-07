"""Small shared guards and scoped source-model state management."""

from contextlib import contextmanager, ExitStack
from functools import wraps
import inspect
import math

import torch


def finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def training_options(epochs, lr, delta_l2_reg, max_models=None, max_environments=None):
    positive_int(epochs, "epochs/steps")
    for name, value in (("lr", lr), ("delta_l2_reg", delta_l2_reg)):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
            raise ValueError(f"Invalid {name}")
    for name, value in (("max_models", max_models), ("max_environments", max_environments)):
        if value is not None:
            positive_int(value, name)


@contextmanager
def source_state(model, *, freeze=False):
    """Restore every submodule mode and parameter flag, including on failure.

    Freezing source parameters does not disable autograd through functional
    source+delta parameters. Existing caller gradients are left untouched.
    """
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    if freeze:
        for p, _ in flags:
            p.requires_grad_(False)
    try:
        yield model
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def frozen_sources(function):
    """Keep teacher/meta-training from changing a caller's model state."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        models = list((bound.arguments.get("models") or {}).values())
        models += list((bound.arguments.get("val_models") or {}).values())
        with ExitStack() as stack:
            for model in {id(m): m for m in models}.values():
                stack.enter_context(source_state(model, freeze=True))
            return function(*args, **kwargs)
    return wrapped


def checked_backward(loss, parameters):
    finite(loss, "training loss")
    loss.backward()
    for parameter in parameters:
        if parameter.grad is not None:
            finite(parameter.grad, "training gradient")


def conditioning_contract(descriptor, generator):
    if descriptor.output_dim != generator.descriptor_dim:
        raise ValueError("Generator and descriptor dimensions disagree")
    if descriptor.conditioning and generator.include_head_weights:
        raise ValueError("A-D conditioning must disable the separate raw-weight input")
