import pytest
import torch
from torch_geometric.data import Data
from amortized_gnn_tta.data.api import DatasetBundle, GraphRecord
from amortized_gnn_tta.data.corruptions import perturb_graph
from amortized_gnn_tta.data.ogb import ArxivDataset, partition_training_labels


def fixture():
    x = torch.arange(800, dtype=torch.float32).reshape(200, 4)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    return Data(x=x, edge_index=edges, num_nodes=200)


def test_partial_views_preserve_context_and_hide_labels():
    graph = fixture()
    splits = {
        "train": torch.tensor([0, 1]),
        "val": torch.tensor([2]),
        "test": torch.empty(0, dtype=torch.long),
    }
    record = GraphRecord(
        graph, torch.arange(200) % 40, splits, "source_train", split_policy="partial"
    )
    bundle = DatasetBundle({"arxiv": record}, tuple(map(str, range(40))))
    bundle.validate()
    unlabeled = bundle.unlabeled_input("arxiv")
    assert set(unlabeled.keys()) == {"x", "edge_index", "num_nodes"}
    assert unlabeled.num_nodes == 200
    record.split_policy = "complete"
    with pytest.raises(ValueError):
        bundle.validate()


def test_paired_corruptions_and_rng_isolation():
    graph = fixture()
    before = torch.get_rng_state().clone()
    options = dict(feature_seed=4, edge_seed=5, undirected=True)
    f = perturb_graph(graph, feature_mask_rate=0.5, edge_drop_rate=0, **options)
    e = perturb_graph(graph, feature_mask_rate=0, edge_drop_rate=0.5, **options)
    both = perturb_graph(graph, feature_mask_rate=0.5, edge_drop_rate=0.5, **options)
    assert torch.equal(f.x, both.x) and torch.equal(e.edge_index, both.edge_index)
    assert torch.equal(f.edge_index, graph.edge_index) and torch.equal(e.x, graph.x)
    assert e.is_undirected() and torch.equal(before, torch.get_rng_state())
    empty = perturb_graph(graph, feature_mask_rate=1, edge_drop_rate=1, **options)
    assert (
        empty.edge_index.shape == (2, 0)
        and empty.x.eq(0).all()
        and empty.num_nodes == 200
    )


def test_pools_are_disjoint_and_ignore_nontraining_labels():
    graph = fixture()
    labels = torch.arange(200) % 40
    d = ArxivDataset(graph, labels, torch.zeros(200), {"train": torch.arange(160)}, {})
    pools = partition_training_labels(d, source_fraction=0.75, seed=7)
    joined = torch.cat(list(pools.values()))
    assert joined.unique().numel() == 160 and joined.sort().values.equal(
        torch.arange(160)
    )
    labels[160:] = -999
    repeated = partition_training_labels(d, source_fraction=0.75, seed=7)
    assert all(torch.equal(v, repeated[k]) for k, v in pools.items())
    assert all(
        torch.bincount(labels[v], minlength=40).gt(0).all() for v in pools.values()
    )


def test_ogb_views_reject_final_labels_in_training():
    d = ArxivDataset(
        fixture(),
        torch.arange(200) % 40,
        torch.zeros(200),
        {
            "train": torch.arange(160),
            "valid": torch.arange(160, 180),
            "test": torch.arange(180, 200),
        },
        {},
    )
    with pytest.raises(ValueError, match="official node pool"):
        d.view(
            "source_train",
            {
                "train": torch.tensor([180]),
                "val": torch.tensor([0]),
                "test": torch.empty(0, dtype=torch.long),
            },
        )
    valid = d.view("validation", {"eval": torch.arange(160, 180)})
    assert valid.select("arxiv", "eval").numel() == 20
    with pytest.raises(ValueError):
        valid.training_labels("arxiv")


def test_real_environment_views_and_pairing():
    from amortized_gnn_tta.data.real_environments import (
        prepare_arxiv_protocol,
        ArxivEnvironments,
    )

    n = 1600
    g = Data(
        x=torch.randn(n, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        num_nodes=n,
    )
    d = ArxivDataset(
        g,
        torch.arange(n) % 40,
        torch.zeros(n),
        {
            "train": torch.arange(1200),
            "valid": torch.arange(1200, 1400),
            "test": torch.arange(1400, 1600),
        },
        {},
    )
    manifest = prepare_arxiv_protocol(
        d, source_fraction=0.5, training_rates=(0.3,), validation_rates=(0.2,)
    )
    loader = ArxivEnvironments(d, manifest)
    f = loader(manifest, "meta_train-attribute-0").unlabeled_input(
        "meta_train-attribute-0"
    )
    e = loader(manifest, "meta_train-structural-0").unlabeled_input(
        "meta_train-structural-0"
    )
    c = loader(manifest, "meta_train-combined-0").unlabeled_input(
        "meta_train-combined-0"
    )
    assert torch.equal(f.x, c.x) and torch.equal(e.edge_index, c.edge_index)
    assert e.is_undirected() and torch.equal(e.x, g.x)
    assert (
        loader(manifest, "validation-control")
        .select("validation-control", "eval")
        .equal(d.official_splits["valid"])
    )
    assert len(loader.pools["generator_stop"]) and len(
        loader.pools["generator_diagnostic"]
    )


def test_five_candidate_agreement_matches_manual_selection():
    from amortized_gnn_tta.models import make_model_spec, build_model
    from amortized_gnn_tta.adaptation import (
        DiffusionHeadGenerator,
        GraphShiftDescriptor,
        extract_target_parameters,
        adapted_forward,
    )
    from amortized_gnn_tta.adaptation.candidates import agreement_predictions

    model = build_model(make_model_spec("gcn", 4, 2, hidden_channels=8)).eval()
    graph = fixture()
    graph.x = (graph.x - 400) / 400
    descriptor = GraphShiftDescriptor(hidden_channels=8)
    weights, schema = extract_target_parameters(model)
    generator = DiffusionHeadGenerator(
        descriptor.output_dim,
        len(weights),
        hidden_dim=12,
        diffusion_steps=7,
        sampling_steps=3,
        include_head_weights=False,
    )
    with torch.no_grad():
        generator.net[-1].weight.copy_(
            torch.randn(
                generator.net[-1].weight.shape,
                generator=torch.Generator().manual_seed(8),
            )
            * 0.2
        )
    indices = torch.arange(20)
    before = torch.get_rng_state().clone()
    output, audit = agreement_predictions(
        model, graph, descriptor, generator, indices=indices
    )
    assert audit["candidate_count"] == 5
    assert torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        desc = descriptor(model, graph)
        predictions = [
            adapted_forward(model, graph, generator.sample(desc, seed=s), schema)
            for s in audit["seeds"]
        ]
        p = torch.stack([z[indices].softmax(-1) for z in predictions])
        score = (p - p.mean(0)).square().mean((1, 2))
    assert len(set(audit["mse"])) > 1
    assert audit["selected_index"] == int(score.argmin())
    torch.testing.assert_close(output.logits, predictions[audit["selected_index"]])


def test_temporal_arrivals_keep_future_nodes_out_and_gate_final():
    from amortized_gnn_tta.data.natural import (
        prepare_temporal_protocol,
        TemporalEnvironments,
    )

    n = 1600
    years = torch.tensor(
        [2014] * 400
        + [2015] * 400
        + [2016] * 200
        + [2017] * 200
        + [2018] * 200
        + [2019] * 100
        + [2020] * 100
    )
    graph = Data(
        x=torch.randn(n, 4),
        edge_index=torch.stack((torch.arange(n - 1), torch.arange(1, n))),
        num_nodes=n,
    )
    data = ArxivDataset(
        graph,
        torch.arange(n) % 40,
        years,
        {
            "train": torch.arange(1200),
            "valid": torch.arange(1200, 1400),
            "test": torch.arange(1400, 1600),
        },
        {},
    )
    manifest = prepare_temporal_protocol(data)
    loader = TemporalEnvironments(data, manifest)
    assert loader.graph.num_nodes == 400 and loader.graph.edge_index.max() < 400
    practice = loader(manifest, "arrival-2015")
    assert practice.unlabeled_input("arrival-2015").num_nodes == 800
    assert practice.select("arrival-2015", "train").min() >= 400
    val = loader(manifest, "arrival-2018")
    assert val.select("arrival-2018", "eval").equal(torch.arange(1200, 1400))
    with pytest.raises(ValueError, match="frozen-design"):
        loader(manifest, "arrival-2020")


def test_unfamiliar_operations_preserve_required_components():
    from amortized_gnn_tta.data.corruptions import unfamiliar_perturbation

    graph = fixture()
    rng = torch.get_rng_state().clone()
    noisy = unfamiliar_perturbation(
        graph, operation="gaussian_features", strength=0.5, seed=7
    )
    edges = unfamiliar_perturbation(
        graph, operation="degree_biased_removal", strength=0.5, seed=7, undirected=True
    )
    assert torch.equal(noisy.edge_index, graph.edge_index) and not torch.equal(
        noisy.x, graph.x
    )
    assert (
        torch.equal(edges.x, graph.x) and edges.is_undirected() and edges.num_edges == 4
    )
    assert torch.equal(rng, torch.get_rng_state())


def test_unlabeled_methods_preserve_source_and_graph():
    from amortized_gnn_tta.adaptation.benchmark import iterative_predictions
    from amortized_gnn_tta.adaptation.graph_transform import feature_gtrans_predictions
    from amortized_gnn_tta.models import make_model_spec, build_model

    graph = fixture()
    graph.x = graph.x / 800
    model = build_model(make_model_spec("gcn", 4, 2, hidden_channels=8)).eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    x = graph.x.clone()
    for method in (iterative_predictions, feature_gtrans_predictions):
        output = method(model, graph, indices=torch.arange(10), steps=2)
        assert output.logits.shape == (200, 2) and torch.isfinite(output.logits).all()
        assert torch.equal(graph.x, x)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
        assert all(p.requires_grad for p in model.parameters())
        with pytest.raises(ValueError):
            method(model, graph, indices=torch.tensor([0, 0]), steps=1)


def test_arxiv_cache_rejects_changed_tensor_content(tmp_path):
    from amortized_gnn_tta.data.ogb import _payload_sha256, load_arxiv

    payload = dict(
        x=torch.ones(2, 4),
        edge_index=torch.tensor([[0], [1]]),
        years=torch.tensor([2017, 2018]),
        labels=torch.tensor([0, 1]),
        splits={
            "train": torch.tensor([0]),
            "valid": torch.tensor([1]),
            "test": torch.empty(0, dtype=torch.long),
        },
        metadata={"dataset": "tiny_corruption_check"},
        cache_version=2,
    )
    payload["content_sha256"] = _payload_sha256(payload)
    payload["x"][0, 0] = 10
    torch.save(payload, tmp_path / "arxiv-tensors-v2.pt")
    with pytest.raises(ValueError, match="integrity"):
        load_arxiv(tmp_path)
