"""Natural-study wiring on mock graphs, never numerical benchmark evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from amortized_gnn_tta.data.ogb import ArxivDataset
from amortized_gnn_tta.data.natural import (
    prepare_temporal_protocol,
    TemporalEnvironments,
    prepare_country_protocol,
)
from amortized_gnn_tta.models import make_model_spec, build_model
from amortized_gnn_tta.training import (
    SourceTrainingData,
    TrainingConfig,
    fit_source,
    save_checkpoint,
)
from amortized_gnn_tta.adaptation_artifacts import state_sha256


@pytest.mark.parametrize("kind", ["temporal", "country"])
def test_natural_study_and_final_gate(tmp_path, monkeypatch, kind):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location(
        "natural_study_runner", examples / "run_arxiv_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = torch.Generator().manual_seed(2)
    n = 1600
    graph = Data(
        x=torch.randn(n, 8, generator=rng),
        edge_index=torch.stack((torch.arange(n - 1), torch.arange(1, n))),
        num_nodes=n,
    )
    if kind == "temporal":
        labels = torch.arange(n) % 40
        years = torch.tensor(
            [2014] * 400
            + [2015] * 400
            + [2016] * 200
            + [2017] * 200
            + [2018] * 200
            + [2019] * 100
            + [2020] * 100
        )
        data = ArxivDataset(
            graph,
            labels,
            years,
            dict(
                train=torch.arange(1200),
                valid=torch.arange(1200, 1400),
                test=torch.arange(1400, 1600),
            ),
            {},
        )
        manifest = prepare_temporal_protocol(data)
        loader = TemporalEnvironments(data, manifest)
        source_graph = loader.graph
        source_labels = labels[loader.source_ids]
        classes = 40
        dataset = "ogbn-arxiv-temporal"
        monkeypatch.setattr("amortized_gnn_tta.data.ogb.load_arxiv", lambda _: data)
    else:
        labels = torch.arange(n) % 20
        monkeypatch.setattr(
            "amortized_gnn_tta.data.natural.load_country",
            lambda root, country: (graph.clone(), labels.clone()),
        )
        manifest = prepare_country_protocol(tmp_path)
        source_graph = graph.clone()
        source_graph.edge_index = to_undirected(graph.edge_index)
        source_labels = labels
        classes = 20
        dataset = "pair-align-mag-country"
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "environment_manifest.json").write_text(json.dumps(manifest))
    train, stop = (torch.tensor(manifest["source_splits"][k]) for k in ("train", "val"))
    view = SourceTrainingData(
        source_graph,
        train,
        source_labels[train],
        stop,
        source_labels[stop],
        tuple(map(str, range(classes))),
        {},
    )
    rows = []
    for role, seed in [("meta_train", 31), ("validation", 34), ("test", 35)]:
        start = time.perf_counter()
        model_spec = make_model_spec("gcn", 8, classes, hidden_channels=8)
        result = fit_source(
            lambda: build_model(model_spec),
            view,
            TrainingConfig(max_epochs=1, patience=1, model_seed=seed),
        )
        name = f"gcn-{role}-{seed}"
        path = bank / (name + ".pt")
        save_checkpoint(path, result, model_spec)
        rows.append(
            dict(
                id=name,
                role=role,
                seed=seed,
                architecture="gcn",
                path=path.name,
                checkpoint_sha256=module.file_hash(path),
                state_sha256=state_sha256(result.model.state_dict()),
                fit_seconds=time.perf_counter() - start,
            )
        )
    plan = dict(
        dataset=dataset,
        threads=1,
        architectures=["gcn"],
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
    (bank / "bank.json").write_text(
        json.dumps(dict(plan=plan, models=rows, status="complete"))
    )
    design = tmp_path / "CONTROLLED_DESIGN.json"
    design.write_text(
        json.dumps(
            dict(
                selections={"gcn": dict(conditioning="A", update="lora")},
                validation_selection_only=True,
            )
        )
    )
    args = SimpleNamespace(
        bank=bank,
        data_root=tmp_path,
        output_dir=tmp_path / "study",
        final_evaluation=True,
        warmups=0,
        repeats=1,
        measure_memory=False,
    )
    study = module.Study(args)
    study.run_natural(design)
    assert (study.root / "FROZEN_DESIGN.json").exists()
    monkeypatch.setitem(sys.modules, "run_arxiv_experiments", module)
    final_spec = importlib.util.spec_from_file_location(
        "natural_final_runner", examples / "evaluate_real_generalization.py"
    )
    final = importlib.util.module_from_spec(final_spec)
    final_spec.loader.exec_module(final)
    final.run(args)
    complete = json.loads((study.root / "FINAL" / "COMPLETE.json").read_text())
    assert complete["final_test_scored"] is True and len(complete["summary"]) == 9
    summary_spec = importlib.util.spec_from_file_location(
        "natural_report_runner", examples / "summarize_real_study.py"
    )
    summary = importlib.util.module_from_spec(summary_spec)
    summary_spec.loader.exec_module(summary)
    summary.summarize(bank, study.root)
    assert (
        "Complete prediction-pipeline latency" in (study.root / "REPORT.md").read_text()
    )
