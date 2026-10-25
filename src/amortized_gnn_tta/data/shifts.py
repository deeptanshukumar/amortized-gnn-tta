"""Offline, paired interventions on the existing Matcha-code CSBM distribution.

These are synthetic benchmark construction tools, never deployment transforms.
Pairs reuse the edge uniforms and feature noise through the existing independent
RNG streams. Changing features therefore preserves the exact edge realization,
and changing structure preserves the exact feature tensor. No labels are changed.
"""

from dataclasses import dataclass, replace
import math
import re

from .csbm import CSBMConfig

SHIFT_PROTOCOL = "csbm-paired-shifts-v1"
SHIFT_FAMILIES = {"attribute", "structural", "combined"}


@dataclass(frozen=True)
class CSBMShift:
    """Endpoints for a linear path from a reference CSBM configuration.

    attribute_offset translates BOTH class-center coefficients, before division
    by sqrt(F). Optional structural endpoints use Matcha's degree_scale and h
    conventions. Severity is a path coordinate, not a promised accuracy loss.
    """

    name: str
    attribute_offset: float = 0.0
    homophily_parameter: float | None = None
    degree_scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("Shift names must be lowercase identifiers")
        for name in ("attribute_offset", "homophily_parameter", "degree_scale"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.attribute_offset is None:
            raise ValueError("attribute_offset must be a finite number")
        if self.homophily_parameter is not None and not 0 <= self.homophily_parameter <= 1:
            raise ValueError("homophily_parameter must be in [0,1]")
        if self.degree_scale is not None and self.degree_scale < 0:
            raise ValueError("degree_scale must be nonnegative")
        if self.attribute_offset == 0 and self.homophily_parameter is None and self.degree_scale is None:
            raise ValueError("A shift needs a changed feature or structure parameter")

    @property
    def family(self) -> str:
        attribute = self.attribute_offset != 0
        structure = self.homophily_parameter is not None or self.degree_scale is not None
        return "combined" if attribute and structure else "attribute" if attribute else "structural"


def shifted_config(reference: CSBMConfig, shift: CSBMShift, severity: float) -> CSBMConfig:
    """Interpolate parameters, retaining node identities, labels, RNGs and splits.

    Common random numbers couple the draws without changing their individual
    Bernoulli/Gaussian marginals. With h fixed, increasing degree_scale nests the
    old edges inside the new graph. h alone keeps degree_scale fixed, NOT the
    exact finite-size expected or observed degree. Invalid probabilities fail;
    none are clipped. A zero-severity control is exactly the reference config.
    """
    if (isinstance(severity, bool) or not isinstance(severity, (int, float))
            or not math.isfinite(severity) or not 0 <= severity <= 1):
        raise ValueError("severity must be finite and in [0,1]")
    n = reference.num_nodes
    degree = n * (reference.p_in + reference.p_out)
    h = reference.p_in / (reference.p_in + reference.p_out) if degree else None
    if shift.homophily_parameter is not None and h is None:
        raise ValueError("Homophily shifts need a nonzero reference degree scale")
    if shift.degree_scale is not None and h is None:
        raise ValueError("Degree shifts need a defined reference homophily parameter")
    if shift.homophily_parameter is not None and math.isclose(shift.homophily_parameter, h, abs_tol=1e-12):
        raise ValueError("Homophily endpoint must differ from the reference")
    if shift.degree_scale is not None and math.isclose(shift.degree_scale, degree, abs_tol=1e-12):
        raise ValueError("Degree endpoint must differ from the reference")
    if severity == 0:
        return reference
    changes = {"negative_mean": reference.negative_mean + severity * shift.attribute_offset,
               "positive_mean": reference.positive_mean + severity * shift.attribute_offset}
    # Attribute-only construction preserves primitive probabilities bit for bit;
    # a needless round trip through d/h could introduce floating-point changes.
    if shift.homophily_parameter is not None or shift.degree_scale is not None:
        target_degree = degree + severity * (shift.degree_scale - degree) if shift.degree_scale is not None else degree
        target_h = h + severity * (shift.homophily_parameter - h) if shift.homophily_parameter is not None else h
        changes.update(p_in=target_degree * target_h / n, p_out=target_degree * (1 - target_h) / n)
    return replace(reference, **changes)


def default_csbm_shifts() -> tuple[CSBMShift, ...]:
    """Eleven explicit conditions for the Phase-1 d=5/h=.8 reference.

    Nonempty combinations of feature translation (+.02), h endpoint .2 and
    degree endpoint 2 OR 10. Endpoint choices draw on Matcha's released recipe;
    this full factorial pairing/interpolation design is OUR development protocol.
    Callers with other references should supply their own meaningful endpoints.
    """
    result = []
    for attribute in (False, True):
        for homophily in (False, True):
            for degree_name, degree in (("", None), ("degree_low", 2.0), ("degree_high", 10.0)):
                parts = (["attribute"] if attribute else []) + (["homophily"] if homophily else [])
                if degree_name:
                    parts.append(degree_name)
                if parts:
                    result.append(CSBMShift("_".join(parts), 0.02 if attribute else 0.0,
                                            0.2 if homophily else None, degree))
    return tuple(result)
