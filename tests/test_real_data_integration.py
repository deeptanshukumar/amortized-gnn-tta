"""Opt-in checks against official local artifacts; never download during tests."""

import os
from pathlib import Path
import pytest
import torch
from torch_geometric.data import Data
from amortized_gnn_tta.data.ogb import load_arxiv
from amortized_gnn_tta.data.real_environments import (
    prepare_arxiv_protocol,
    ArxivEnvironments,
)
from amortized_gnn_tta.data.natural import (
    prepare_temporal_protocol,
    TemporalEnvironments,
    load_country,
)
from amortized_gnn_tta.data.corruptions import perturb_graph
from amortized_gnn_tta.models import make_model_spec, build_model


@pytest.mark.skipif(
    not os.environ.get("AFML_OGB_ROOT"),
    reason="Set AFML_OGB_ROOT to verified official Arxiv data",
)
def test_official_real_graph_protocols_and_all_architecture_forward():
    dataset = load_arxiv(Path(os.environ["AFML_OGB_ROOT"]))
    manifest = prepare_arxiv_protocol(dataset)
    loader = ArxivEnvironments(dataset, manifest)
    assert loader.graph.num_nodes == 169343 and loader.graph.num_edges == 2315598
    options = dict(feature_seed=1701, edge_seed=1702, undirected=True)
    feature = perturb_graph(
        loader.graph, feature_mask_rate=0.2, edge_drop_rate=0, **options
    )
    edge = perturb_graph(
        loader.graph, feature_mask_rate=0, edge_drop_rate=0.2, **options
    )
    both = perturb_graph(
        loader.graph, feature_mask_rate=0.2, edge_drop_rate=0.2, **options
    )
    assert torch.equal(feature.x, both.x) and torch.equal(
        edge.edge_index, both.edge_index
    )
    assert torch.equal(feature.edge_index, loader.graph.edge_index) and torch.equal(
        edge.x, loader.graph.x
    )
    del feature, edge, both
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for architecture in ("gcn", "gat", "graphsage"):
            model = build_model(
                make_model_spec(architecture, 128, 40, hidden_channels=32)
            ).eval()
            logits = model(loader.graph.x, loader.graph.edge_index)
            assert logits.shape == (169343, 40) and torch.isfinite(logits).all()
    temporal = prepare_temporal_protocol(dataset)
    arrival = TemporalEnvironments(dataset, temporal)
    assert arrival.graph.num_nodes == 41125
    for eid, expected in [
        ("arrival-2015", 53160),
        ("arrival-2016", 69499),
        ("arrival-2017", 90941),
        ("arrival-2018", 120740),
    ]:
        bundle = arrival(temporal, eid)
        assert bundle.unlabeled_input(eid).num_nodes == expected
    with pytest.raises(ValueError, match="frozen-design"):
        arrival(temporal, "arrival-2020")


@pytest.mark.skipif(
    not os.environ.get("AFML_MAG_ROOT"),
    reason="Set AFML_MAG_ROOT to published Pair-Align country artifacts",
)
def test_published_country_checksums_and_schema():
    expected = {
        "CN": 101952,
        "DE": 43032,
        "FR": 29262,
        "JP": 37498,
        "RU": 32833,
        "US": 132558,
    }
    for country, count in expected.items():
        graph, labels = load_country(Path(os.environ["AFML_MAG_ROOT"]), country)
        assert (
            graph.num_nodes == count
            and graph.x.shape == (count, 128)
            and labels.shape == (count,)
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA artifact probe requires an actual GPU"
)
def test_cuda_exports_exact_generated_factors(tmp_path):
    from copy import deepcopy
    from amortized_gnn_tta.adaptation.generator import HeadAdapterGenerator
    from amortized_gnn_tta.adaptation.diffusion import DiffusionHeadGenerator
    from amortized_gnn_tta.adaptation.experiments import make_descriptor
    from amortized_gnn_tta.adaptation.lora_generator import (
        LoRAUpdateSpace,
        LoRAGenerator,
    )
    from amortized_gnn_tta.adaptation.iterative import PEFTConfig
    from amortized_gnn_tta.adaptation.benchmark import generated_predictions
    from amortized_gnn_tta.adaptation.candidates import agreement_predictions
    from amortized_gnn_tta.adaptation.generator import adapted_forward, extract_target_parameters

    exports = Path(os.environ.get("AFML_CUDA_EXPORT_DIR", str(tmp_path)))
    exports.mkdir(parents=True, exist_ok=True)

    graph = Data(
        x=torch.randn(400, 8),
        edge_index=torch.stack((torch.arange(399), torch.arange(1, 400))),
        num_nodes=400,
    )
    for architecture in ("gcn", "gat", "graphsage"):
        model = (
            build_model(make_model_spec(architecture, 8, 40, hidden_channels=8))
            .cuda()
            .eval()
        )
        for method in ("hypernetwork", "diffusion"):
            descriptor, _ = make_descriptor(model, "B", scope="all")
            space = LoRAUpdateSpace.from_model(
                model, PEFTConfig("lora", backbone_rank=2, head_rank=2)
            )
            options = (
                {}
                if method == "hypernetwork"
                else dict(diffusion_steps=7, sampling_steps=3)
            )
            cls = (
                HeadAdapterGenerator
                if method == "hypernetwork"
                else DiffusionHeadGenerator
            )
            core = cls(
                descriptor.output_dim,
                space.factor_dim,
                include_head_weights=False,
                **options,
            )
            # Nonzero outputs exercise the export contract beyond identity initialization.
            with torch.no_grad():
                core.net[-1].weight.normal_(0, 0.005)
            generator = LoRAGenerator(core, space).cuda().eval()
            if method == "hypernetwork":
                output = generated_predictions(
                    model, graph, descriptor, generator, scope="all", device="cuda"
                )
            else:
                output, audit = agreement_predictions(
                    model,
                    graph,
                    descriptor,
                    generator,
                    scope="all",
                    indices=torch.arange(50),
                    device="cuda",
                )
                assert output.selection == audit and audit["candidate_count"] == 5
            assert output.factor_vector is not None
            torch.testing.assert_close(
                generator.update_space.dense_delta(output.factor_vector),
                output.delta,
                rtol=0,
                atol=0,
            )
            _, schema = extract_target_parameters(model, "all")
            path = exports / f"{architecture}-{method}.pt"
            torch.save(dict(factors=output.factor_vector.detach().cpu(),
                            delta=output.delta.detach().cpu(),
                            logits=output.logits.detach().cpu(),
                            selection=output.selection, schema=schema,
                            source_state={k: v.cpu() for k, v in model.state_dict().items()},
                            model_spec=make_model_spec(architecture, 8, 40, hidden_channels=8),
                            x=graph.x, edge_index=graph.edge_index,
                            lora_space=generator.update_space.configuration()), path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            cpu_model = deepcopy(model).cpu().eval()
            cpu_model.load_state_dict(saved["source_state"])
            with torch.no_grad():
                replay = adapted_forward(cpu_model, graph, saved["delta"], saved["schema"])
            torch.testing.assert_close(replay, saved["logits"], rtol=1e-4, atol=1e-5)
