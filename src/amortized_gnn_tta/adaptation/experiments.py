"""Shared conditioning recipes for the portable CSBM experiment commands."""

from .descriptors import GraphShiftDescriptor

CONDITIONING_CHOICES = ("compact", "structural", "A", "B", "C", "D")


def make_descriptor(model, conditioning, *, scope="head", descriptor_version="v2",
                    weight_scope="all", chunk_size=1024):
    """Keep legacy inputs explicit; A-D must not receive extra raw weights.

    Compact/structural reproduce the contributor's model-summary families and
    retain their separate raw update-site weights. They are additional baselines,
    not replacements for the planned A-D input ablation.
    """
    if conditioning not in CONDITIONING_CHOICES:
        raise ValueError(f"Unknown conditioning recipe: {conditioning}")
    descriptor = GraphShiftDescriptor(
        in_channels=model.config.in_channels, hidden_channels=model.config.hidden_channels,
        out_channels=model.config.out_channels,
        conditioning=conditioning if conditioning in ("A", "B", "C", "D") else None,
        descriptor_type="structural" if conditioning == "structural" else "moments",
        target_scope=scope, descriptor_version=descriptor_version,
        weight_scope=weight_scope, chunk_size=chunk_size)
    return descriptor, conditioning in ("compact", "structural")


def review_run_name(architecture, method, conditioning, seed, scope="head"):
    # Preserve completed head-review names so their checkpoints can be reused.
    suffix = "" if scope == "head" else "-" + scope
    return f"{architecture}-{method}-{conditioning}{suffix}-{seed}"


def training_recipe(plan):
    """Allow a smaller grid or new timing repetitions, never a new fitting recipe.

    Per-run model/environment identities, method, input and seeds are checked
    separately, and regenerated deltas must also agree before reuse is accepted.
    Unknown future configuration keys remain part of this conservative check.
    """
    orchestration = {"protocol", "purpose", "architectures", "methods", "conditioning",
                     "generator_seeds", "warmups", "repeats", "timed_shifts"}
    return {key: value for key, value in plan.items() if key not in orchestration}
