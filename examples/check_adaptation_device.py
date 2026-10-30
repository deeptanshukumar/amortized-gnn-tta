"""Exercise all model/update/objective combinations on an explicit backend.

This is a small correctness smoke check on random models, NOT an accuracy result.
CUDA/MPS failures remain visible; there is no silent fallback for explicit devices.
"""

import argparse
import json
from pathlib import Path

import torch

from amortized_gnn_tta.adaptation import (
    AdaptationConfig, PEFTConfig, development_supervision, fit_supervised_development, fit_unlabeled,
)
from amortized_gnn_tta.adaptation_artifacts import load_weight_delta, save_weight_delta
from amortized_gnn_tta.data import CSBMConfig, DatasetBundle, generate_csbm
from amortized_gnn_tta.evaluation import predict, select_device
from amortized_gnn_tta.models import ARCHITECTURES, build_model, make_model_spec
from amortized_gnn_tta.training import software_versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    device = select_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(31)
    bundle = DatasetBundle({"meta": generate_csbm(CSBMConfig(num_nodes=100, num_features=8), role="meta_train")})
    bundle.validate()
    graph = bundle.unlabeled_input("meta")
    supervision = development_supervision(bundle, "meta")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for architecture in ARCHITECTURES:
        spec = make_model_spec(architecture, 8, 2, hidden_channels=8)
        model = build_model(spec)
        for method in ("head", "lora"):
            for objective in ("supervised_development", "entropy", "information_maximization"):
                peft, config = PEFTConfig(method, backbone_rank=2), AdaptationConfig(steps=2)
                if objective == "supervised_development":
                    result = fit_supervised_development(model, graph, supervision, peft, config, device=device)
                else:
                    result = fit_unlabeled(model, graph, peft, config, objective=objective, device=device)
                path = args.output_dir / f"{architecture}-{method}-{objective}.pt"
                artifact = save_weight_delta(path, result, spec, graph, context={"purpose": "device smoke check"})
                restored, _ = load_weight_delta(path)
                torch.testing.assert_close(predict(restored, graph), predict(result.model, graph), rtol=1e-4, atol=1e-5)
                rows.append({"architecture": architecture, "method": method, "objective": objective,
                             "finite_outputs": True, "cpu_reload_predictions_close": True,
                             "delta_l2": artifact["delta_summary"]["delta_l2"]})
    report = {"status": "complete", "device": str(device), "versions": software_versions(),
              "purpose": "correctness check only; random models; no accuracy claim", "checks": rows,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (args.output_dir / "device_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "checks": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
