"""Tiny software integration checks; these are not Arxiv scientific results."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import pytest
from torch_geometric.data import Data
from amortized_gnn_tta.data.ogb import ArxivDataset
from amortized_gnn_tta.data.real_environments import (
    prepare_arxiv_protocol,
    ArxivEnvironments,
)
from amortized_gnn_tta.models import make_model_spec, build_model
from amortized_gnn_tta.training import (
    SourceTrainingData,
    TrainingConfig,
    fit_source,
    save_checkpoint,
)
from amortized_gnn_tta.adaptation_artifacts import state_sha256


@pytest.mark.parametrize("architecture", ["gcn", "gat", "graphsage"])
def test_real_study_fits_restores_and_replays_both_generator_families(
    tmp_path, monkeypatch, architecture
):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location(
        "arxiv_study_runner", examples / "run_arxiv_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n = 1600
    graph = Data(
        x=torch.randn(n, 8, generator=torch.Generator().manual_seed(19)),
        edge_index=torch.stack((torch.arange(n - 1), torch.arange(1, n))),
        num_nodes=n,
    )
    dataset = ArxivDataset(
        graph,
        torch.arange(n) % 40,
        torch.zeros(n, dtype=torch.long),
        dict(
            train=torch.arange(1200),
            valid=torch.arange(1200, 1400),
            test=torch.arange(1400, 1600),
        ),
        {},
    )
    manifest = prepare_arxiv_protocol(
        dataset, training_rates=(0.1,), validation_rates=(0.2,)
    )
    loader = ArxivEnvironments(dataset, manifest)
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "environment_manifest.json").write_text(json.dumps(manifest))
    plan = dict(
        architectures=[architecture],
        threads=1,
        generator_seeds=[7],
        generator_epochs=1,
        generator_lr=0.001,
        teacher_steps=1,
        teacher_lr=0.01,
        delta_l2=0.001,
        lora_rank=2,
        diffusion_steps=7,
        sampling_steps=3,
        candidate_seeds=[42, 43, 44, 45, 46],
    )
    rows = []
    view = SourceTrainingData(
        loader.graph,
        loader.pools["source_train"],
        dataset.labels[loader.pools["source_train"]],
        loader.pools["source_stop"],
        dataset.labels[loader.pools["source_stop"]],
        tuple(map(str, range(40))),
        {},
    )
    for role, seed in [("meta_train", 31), ("validation", 34), ("test", 35)]:
        model_spec = make_model_spec(architecture, 8, 40, hidden_channels=8)
        result = fit_source(
            lambda: build_model(model_spec),
            view,
            TrainingConfig(max_epochs=1, patience=1, model_seed=seed),
        )
        name = f"{architecture}-{role}-{seed}"
        path = bank / (name + ".pt")
        save_checkpoint(path, result, model_spec)
        rows.append(
            dict(
                id=name,
                architecture=architecture,
                role=role,
                path=path.name,
                checkpoint_sha256=module.file_hash(path),
                state_sha256=state_sha256(result.model.state_dict()),
            )
        )
    (bank / "bank.json").write_text(
        json.dumps(dict(status="complete", models=rows, plan=plan))
    )
    monkeypatch.setattr("amortized_gnn_tta.data.ogb.load_arxiv", lambda _: dataset)
    study = module.Study(
        SimpleNamespace(bank=bank, output_dir=tmp_path / "study", data_root=tmp_path)
    )
    for method in ["hypernetwork", "teacher_hypernetwork", "diffusion"]:
        row = study.fit(architecture, "A", "lora", method, 7)
        assert (
            0 <= row["mean_validation_accuracy"] <= 1
            and row["final_test_scored"] is False
        )
        assert len(row["records"]) == 4
        assert 0 < row["adaptation_parameter_count"] < row["source_parameter_count"]
        assert row["generator_parameter_count"] > 0
        if method == "diffusion":
            assert all(r["selection"]["candidate_count"] == 5 for r in row["records"])
        assert study.fit(architecture, "A", "lora", method, 7) == row
        from amortized_gnn_tta.adaptation.generator import (
            adapted_forward,
            extract_target_parameters,
        )

        model = next(iter(study.sources(architecture, "validation").values()))
        for record in row["records"]:
            artifact = torch.load(
                study.root / row["name"] / record["artifact"], weights_only=True
            )
            assert torch.equal(artifact["prediction_logits"].argmax(-1), artifact["predicted_classes"])
            eid = record["environment_id"]
            graph = study.loader(study.manifest, eid).unlabeled_input(eid)
            _, schema = extract_target_parameters(model, "all")
            with torch.no_grad():
                logits = adapted_forward(model, graph, artifact["delta"], schema)
            assert torch.equal(
                logits[artifact["indices"]].argmax(-1), artifact["predicted_classes"]
            )

    # All architectures validate factor/replay contracts above. The ordered
    # experiment starts with GCN by research design, so exercise it on GCN.
    if architecture != "gcn":
        return
    # GCN design can finish while unrelated bank jobs remain pending. Existing
    # checkpoint pins must survive adding later bank rows without retraining.
    saved_bank = json.loads((bank / "bank.json").read_text())
    partial = dict(
        saved_bank,
        status="running",
        models=[r for r in saved_bank["models"] if r["role"] != "test"],
    )
    (bank / "bank.json").write_text(json.dumps(partial))
    study.args.stop_after = 2
    early = module.Study(study.args)
    early.run()
    assert (study.root / "EXPERIMENT_2.json").exists()
    assert not (study.root / "FROZEN_DESIGN.json").exists()
    before_design = (study.root / "EXPERIMENT_1.json").read_bytes()
    control_spec = importlib.util.spec_from_file_location(
        "conditioning_diagnostic", examples / "diagnose_conditioning.py"
    )
    control_module = importlib.util.module_from_spec(control_spec)
    control_spec.loader.exec_module(control_module)
    control = control_module.run(study.args)
    assert control["changes_design_selection"] is False
    assert control["final_test_scored"] is False
    assert len(control["rows"]) == 4
    assert (study.root / "EXPERIMENT_1.json").read_bytes() == before_design
    (bank / "bank.json").write_text(json.dumps(saved_bank))
    study.args.stop_after = None
    study = module.Study(study.args)
    # Exercise the ordered design selection and final gate on mock data only.
    study.run()
    assert (study.root / "FROZEN_DESIGN.json").exists()
    monkeypatch.setitem(sys.modules, "run_arxiv_experiments", module)
    final_spec = importlib.util.spec_from_file_location(
        "real_final_runner", examples / "evaluate_real_generalization.py"
    )
    final_module = importlib.util.module_from_spec(final_spec)
    final_spec.loader.exec_module(final_module)
    args = SimpleNamespace(
        bank=bank,
        output_dir=study.root,
        data_root=tmp_path,
        final_evaluation=False,
        warmups=0,
        repeats=1,
        measure_memory=False,
    )
    import pytest

    with pytest.raises(ValueError, match="Explicit final"):
        final_module.run(args)
    args.final_evaluation = True
    final_module.run(args)
    completed = json.loads((study.root / "FINAL" / "COMPLETE.json").read_text())
    assert completed["final_test_scored"] is True
    assert completed["independent_delta_replays"] > 0
    assert (
        len(completed["summary"]) == 9
    )  # frozen, practice, GTrans, 2 IM, 3 generators

    diagnostic_spec = importlib.util.spec_from_file_location(
        "real_diagnostic_runner", examples / "diagnose_real_adaptation.py"
    )
    diagnostic_module = importlib.util.module_from_spec(diagnostic_spec)
    diagnostic_spec.loader.exec_module(diagnostic_module)
    diagnostic_module.run(args)
    diagnostic = json.loads((study.root / "DIAGNOSTIC" / "RESULTS.json").read_text())
    assert diagnostic["final_test_scored"] is False and len(diagnostic["rows"]) == 24

    altered = json.loads((bank / "bank.json").read_text())
    altered["models"][0]["state_sha256"] = "0" * 64
    (bank / "bank.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="checkpoint identity changed"):
        module.Study(args)
