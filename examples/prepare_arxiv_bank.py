"""Predeclared fresh Arxiv checkpoints; stopping uses source-pool labels only."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from amortized_gnn_tta.data.ogb import load_arxiv
from amortized_gnn_tta.data.real_environments import (
    prepare_arxiv_protocol,
    ArxivEnvironments,
)
from amortized_gnn_tta.data.api import graph_fingerprint
from amortized_gnn_tta.training import (
    SourceTrainingData,
    TrainingConfig,
    fit_source,
    save_checkpoint,
    load_checkpoint,
    software_versions,
)
from amortized_gnn_tta.models import make_model_spec, build_model
from amortized_gnn_tta.adaptation.checkpoints import atomic_json
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from train_adapter import code_provenance, file_hash


def load_bank_inputs(a, plan):
    """Source-only labels plus explicit local-to-native IDs for either bank size."""
    dataset_name = plan.get("dataset", "ogbn-arxiv")
    if dataset_name == "ogbn-arxiv":
        d = load_arxiv(a.data_root)
        protocol = prepare_arxiv_protocol(
            d,
            source_fraction=plan["source_fraction"],
            pool_seed=plan["pool_seed"],
            direction=plan["direction"],
            training_rates=plan["training_rates"],
            validation_rates=plan["validation_rates"],
        )
        loader = ArxivEnvironments(d, protocol)
        pools = loader.pools
        graph = loader.graph
        labels = d.labels
        classes = 40
        train_ids, stop_ids = pools["source_train"], pools["source_stop"]
    elif dataset_name == "ogbn-arxiv-temporal":
        from amortized_gnn_tta.data.natural import (
            prepare_temporal_protocol,
            TemporalEnvironments,
        )

        d = load_arxiv(a.data_root)
        protocol = prepare_temporal_protocol(
            d, seed=plan["pool_seed"], direction=plan["direction"]
        )
        loader = TemporalEnvironments(d, protocol)
        graph = loader.graph
        labels = d.labels[loader.source_ids]
        classes = 40
        train_ids, stop_ids = (
            torch.tensor(protocol["source_splits"][k], dtype=torch.long)
            for k in ("train", "val")
        )
    elif dataset_name == "pair-align-mag-country":
        from amortized_gnn_tta.data.natural import (
            prepare_country_protocol,
            load_country,
        )
        from torch_geometric.utils import to_undirected

        protocol = prepare_country_protocol(
            a.data_root, seed=plan["pool_seed"], direction=plan["direction"]
        )
        graph, labels = load_country(a.data_root, protocol["source_country"])
        classes = 20
        if plan["direction"] == "undirected":
            graph.edge_index = to_undirected(
                graph.edge_index, num_nodes=graph.num_nodes
            )
        train_ids, stop_ids = (
            torch.tensor(protocol["source_splits"][k], dtype=torch.long)
            for k in ("train", "val")
        )
    else:
        raise ValueError("Unknown research dataset")
    view = SourceTrainingData(
        graph,
        train_ids,
        labels[train_ids],
        stop_ids,
        labels[stop_ids],
        tuple(map(str, range(classes))),
        dict(
            dataset=dataset_name,
            graph_sha256=graph_fingerprint(graph),
            protocol_sha256=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        ),
    )
    native_ids = loader.source_ids if dataset_name == "ogbn-arxiv-temporal" else torch.arange(graph.num_nodes)
    return protocol, view, native_ids


def run(a):
    preparation_start = time.perf_counter()
    plan = json.loads(a.config.read_text())
    torch.set_num_threads(plan["threads"])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    protocol, view, _ = load_bank_inputs(a, plan)
    graph = view.graph
    classes = len(view.class_names)
    atomic_json(a.output_dir / "environment_manifest.json", protocol)
    provenance = code_provenance()
    atomic_json(
        a.output_dir / "PLAN.json",
        dict(
            plan=plan,
            code=provenance,
            software=software_versions(),
            final_test_access=False,
            preparation_seconds=time.perf_counter() - preparation_start,
        ),
    )
    import zipfile

    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(a.output_dir / "code_snapshot.zip", "x") as archive:
        for name in provenance["source_hashes"]:
            archive.write(root / name, name)
        archive.write(a.config, "configs/" + a.config.name)
    records = []
    try:
        for arch in plan["architectures"]:
            for role, seeds in plan["checkpoint_seeds"].items():
                for seed in seeds:
                    name = f"{arch}-{role}-{seed}"
                    print("SOURCE START", name, flush=True)
                    spec = make_model_spec(
                        arch,
                        graph.x.shape[1],
                        classes,
                        hidden_channels=plan["hidden_channels"],
                    )
                    start = time.perf_counter()
                    result = fit_source(
                        lambda: build_model(spec),
                        view,
                        TrainingConfig(
                            max_epochs=plan["source_epochs"],
                            patience=plan["source_patience"],
                            model_seed=seed,
                        ),
                        device=plan["device"],
                    )
                    path = a.output_dir / (name + ".pt")
                    save_checkpoint(path, result, spec)
                    model, _ = load_checkpoint(path, build_model)
                    if state_sha256(model.state_dict()) != state_sha256(
                        result.model.state_dict()
                    ):
                        raise ValueError("Checkpoint mismatch")
                    row = dict(
                        id=name,
                        architecture=arch,
                        role=role,
                        seed=seed,
                        path=path.name,
                        checkpoint_sha256=file_hash(path),
                        state_sha256=state_sha256(model.state_dict()),
                        selected_epoch=result.best_epoch,
                        fit_seconds=time.perf_counter() - start,
                    )
                    records.append(row)
                    atomic_json(
                        a.output_dir / "bank.json",
                        dict(models=records, plan=plan, status="running"),
                    )
                    print("SOURCE DONE", row, flush=True)
        atomic_json(
            a.output_dir / "bank.json",
            dict(models=records, plan=plan, status="complete"),
        )
    except BaseException as error:
        atomic_json(
            a.output_dir / "bank.json",
            dict(models=records, plan=plan, status="failed", error=str(error)),
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
