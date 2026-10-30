"""Portable initial/final/delta evidence, tied to model and environment identity."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .adaptation import AdaptationResult, PEFTConfig, WeightAdapter, adaptation_targets
from .evaluation import predict
from .models import build_model
from .training import software_versions

DELTA_PROTOCOL = "peft-weight-delta-v1"


def state_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def delta_summary(initial: dict[str, Tensor], final: dict[str, Tensor]) -> dict[str, Any]:
    rows = []
    for name, value in initial.items():
        change = final[name].double() - value.double()
        norm = float(torch.linalg.vector_norm(change))
        initial_norm = float(torch.linalg.vector_norm(value.double()))
        rows.append({"name": name, "shape": list(value.shape), "numel": value.numel(),
                     "changed_elements": int(torch.count_nonzero(change)), "delta_l2": norm,
                     "initial_l2": initial_norm,
                     "relative_delta_l2": norm / initial_norm if initial_norm else None})
    total = sum(r["delta_l2"] ** 2 for r in rows) ** 0.5
    initial_total = sum(r["initial_l2"] ** 2 for r in rows) ** 0.5
    return {"delta_l2": total, "relative_delta_l2": total / initial_total if initial_total else None,
            "changed_elements": sum(r["changed_elements"] for r in rows),
            "changed_tensors": [r["name"] for r in rows if r["changed_elements"]], "tensors": rows}


def save_weight_delta(path: Path, result: AdaptationResult, model_spec: dict[str, Any],
                      graph: Data, *, context: dict[str, Any]) -> dict[str, Any]:
    """Save actual float weights and final-minus-initial changes, not just factors.

    A/B factorization is also retained for audit/future generators. The full
    effective delta is the unambiguous quantity: factor pairs are non-unique.
    Original keys are used, so reconstructed models need no adapter library.
    """
    initial, final = _cpu_state(result.adapter.base), _cpu_state(result.model)
    if set(initial) != set(final):
        raise ValueError("Adaptation changed the model state schema")
    if any(not torch.isfinite(value).all() for state in (initial, final) for value in state.values()):
        raise ValueError("Nonfinite adaptation weights")
    delta = {name: final[name] - value for name, value in initial.items()}
    report = delta_summary(initial, final)
    if set(report["changed_tensors"]) - set(result.adapter.target_names):
        raise ValueError("A frozen parameter changed")
    # Independent, ordinary model reconstruction must reproduce the merged result.
    reconstructed = build_model(model_spec)
    reconstructed.load_state_dict({name: initial[name] + delta[name] for name in initial}, strict=True)
    before = predict(result.adapter.base, graph)
    after = predict(result.model, graph)
    torch.testing.assert_close(after, predict(reconstructed, graph), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(after, predict(result.adapter, graph), rtol=1e-4, atol=1e-5)
    payload = {
        "format_version": 1, "protocol": DELTA_PROTOCOL, "model_spec": model_spec,
        "initial_state": initial, "adapted_state": final, "delta_state": delta,
        "initial_sha256": state_sha256(initial), "adapted_sha256": state_sha256(final),
        "delta_sha256": state_sha256(delta), "peft_config": asdict(result.adapter.config),
        "adaptation_config": asdict(result.config), "objective": result.objective,
        "selection": result.selection, "selected_step": result.selected_step,
        "history": result.history, "adapter_state": result.adapter.trainable_state(),
        "target_names": list(result.adapter.target_names),
        "trainable_parameters": sum(p.numel() for p in result.adapter.parameters() if p.requires_grad),
        "model_parameters": sum(p.numel() for p in result.model.parameters()),
        "delta_summary": report, "initial_logits": before, "adapted_logits": after,
        "context": json.loads(json.dumps(context, allow_nan=False)), "versions": software_versions(),
        "reconstruction_predictions_close": True,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return {key: payload[key] for key in ("initial_sha256", "adapted_sha256", "delta_sha256",
                                         "trainable_parameters", "model_parameters", "delta_summary",
                                         "reconstruction_predictions_close")}


def load_weight_delta(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    """Validate saved states/factors and restore a vanilla CPU model safely."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("protocol") != DELTA_PROTOCOL:
        raise ValueError("Unsupported weight-delta artifact")
    model = build_model(payload["model_spec"])
    expected = model.state_dict()
    for name, hash_key in (("initial_state", "initial_sha256"), ("adapted_state", "adapted_sha256"),
                           ("delta_state", "delta_sha256")):
        state = payload[name]
        if (set(state) != set(expected) or any(state[k].shape != v.shape or state[k].dtype != v.dtype
                or not torch.isfinite(state[k]).all() for k, v in expected.items())
                or state_sha256(state) != payload[hash_key]):
            raise ValueError(f"Invalid {name} schema, finite values or checksum")
    initial, final, delta = payload["initial_state"], payload["adapted_state"], payload["delta_state"]
    peft = PEFTConfig(**payload["peft_config"])
    if list(adaptation_targets(model, peft.method)) != payload["target_names"]:
        raise ValueError("Delta target sites disagree with model/method")
    for key in expected:
        if not torch.equal(delta[key], final[key] - initial[key]):
            raise ValueError("Stored delta is not exactly final minus initial")
        if key not in payload["target_names"] and not torch.equal(final[key], initial[key]):
            raise ValueError("Frozen state changed")
    model.load_state_dict(initial, strict=True)
    adapter = WeightAdapter(model, peft)
    adapter.restore_trainable_state(payload["adapter_state"])
    factored = adapter.merged_model().state_dict()
    for key in expected:
        try:
            torch.testing.assert_close(factored[key], final[key], rtol=1e-5, atol=1e-6)
        except AssertionError as exc:
            raise ValueError("Saved adapter does not reconstruct final weights") from exc
    if (payload["delta_summary"] != delta_summary(initial, final)
            or payload["trainable_parameters"] != sum(p.numel() for p in adapter.parameters() if p.requires_grad)
            or payload["model_parameters"] != sum(p.numel() for p in model.parameters())):
        raise ValueError("Saved parameter counts or delta summary disagree with weights")
    model.load_state_dict(final, strict=True)
    return model.eval(), payload
