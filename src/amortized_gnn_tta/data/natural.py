"""Explicit temporal-arrival and country-shift protocols.

Country artifacts are the Pair-Align 20-class paper graphs, not native ogbn-mag.
They contain no official node splits. Our seed-controlled supervision splits and
country assignments therefore belong to a new declared experimental protocol.
"""

from pathlib import Path
import hashlib
import tempfile
import urllib.request

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.utils import subgraph, to_undirected

from .api import DatasetBundle, GraphRecord, graph_fingerprint

COUNTRY_MD5 = {
    "CN": "3e09b899d12d5801f39bf9cd187edcad",
    "JP": "7910d054a972897fc2466f177cb9fed4",
    "DE": "3e3830bd6102db954f1b0163761aebc3",
    "FR": "a2387bdff7841edb395f23d224b0b1c5",
    "US": "677b46f78e5fb946b2d9d2e4f76418fb",
    "RU": "3c86cf9b3b2052d31a433d5422a7ec5f",
}


def stratified_three_way(indices, labels, seed, fractions=(0.8, 0.1)):
    """Fit/stop/diagnostic allocation; never called on final country/year labels."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Split seed must be an integer in [0,2**32)")
    if (
        len(fractions) != 2
        or any(isinstance(f, bool) or not 0 < f < 1 for f in fractions)
        or sum(fractions) >= 1
    ):
        raise ValueError(
            "Fit/stop fractions must be positive and leave a diagnostic pool"
        )
    if (
        indices.dtype != torch.long
        or indices.ndim != 1
        or not len(indices)
        or indices.unique().numel() != len(indices)
    ):
        raise ValueError("Split nodes must be unique nonempty int64 IDs")
    if indices.min() < 0 or indices.max() >= len(labels):
        raise ValueError("Split IDs exceed label table")
    rng = torch.Generator().manual_seed(seed)
    pieces = [[], [], []]
    for category in labels[indices].unique().tolist():
        group = indices[labels[indices] == category]
        group = group[torch.randperm(len(group), generator=rng)]
        if len(group) < 3:
            # It is impossible to cover all three splits for rare arrival classes.
            # Preserve every observation; prioritize train, then inner validation.
            pieces[0].append(group[:1])
            if len(group) == 2:
                pieces[1].append(group[1:])
            continue
        a = min(max(1, int(fractions[0] * len(group))), len(group) - 2)
        b = min(max(1, int(fractions[1] * len(group))), len(group) - a - 1)
        for destination, part in zip(
            pieces, (group[:a], group[a : a + b], group[a + b :])
        ):
            destination.append(part)
    return dict(
        zip(
            ("train", "val", "test"),
            [
                torch.cat(p).sort().values if p else torch.empty(0, dtype=torch.long)
                for p in pieces
            ],
        )
    )


def arrival_view(dataset, end_year, direction="undirected"):
    """Induced graph through end_year, preserving an explicit global-ID map."""
    if direction not in {"directed", "undirected"}:
        raise ValueError("Unknown direction policy")
    ids = torch.where(dataset.years <= end_year)[0]
    edges, _ = subgraph(
        ids,
        dataset.graph.edge_index,
        relabel_nodes=True,
        num_nodes=dataset.graph.num_nodes,
    )
    if direction == "undirected":
        edges = to_undirected(edges, num_nodes=len(ids))
    graph = Data(x=dataset.graph.x[ids].clone(), edge_index=edges, num_nodes=len(ids))
    return graph, ids


def prepare_temporal_protocol(dataset, *, seed=20281109, direction="undirected"):
    """Approved 2014 source / 2015–17 practice / 2018 val / 2019–20 test."""
    graph, ids = arrival_view(dataset, 2014, direction)
    source = stratified_three_way(torch.arange(len(ids)), dataset.labels[ids], seed)
    environments = []
    for year in (2015, 2016, 2017, 2018, 2020):
        role = (
            "meta_train"
            if year <= 2017
            else "validation"
            if year == 2018
            else "target_test"
        )
        environments.append(
            dict(
                id=f"arrival-{year}",
                role=role,
                end_year=year,
                first_year=year if year != 2020 else 2019,
                family="natural_temporal",
                severity=0.0,
                shift=None,
            )
        )
    return dict(
        protocol="arxiv-arrival-v1",
        dataset="ogbn-arxiv-temporal",
        direction=direction,
        source_end_year=2014,
        split_seed=seed,
        source_graph_sha256=graph_fingerprint(graph),
        source_global_ids=ids.tolist(),
        source_splits={k: v.tolist() for k, v in source.items()},
        environments=environments,
        interpretation="Cumulative induced contexts; supervision only on newly arrived papers",
    )


class TemporalEnvironments:
    def __init__(self, dataset, manifest, *, allow_final=False):
        self.dataset, self.manifest, self.allow_final = dataset, manifest, allow_final
        self.graph, self.source_ids = arrival_view(
            dataset, manifest["source_end_year"], manifest["direction"]
        )
        if graph_fingerprint(self.graph) != manifest["source_graph_sha256"]:
            raise ValueError("Temporal source graph changed")
        if self.source_ids.tolist() != manifest["source_global_ids"]:
            raise ValueError("Temporal global IDs changed")
        self.environments = {e["id"]: e for e in manifest["environments"]}
        self.class_names = tuple(f"category_{i}" for i in range(40))

    def __call__(self, manifest, environment_id):
        if manifest != self.manifest:
            raise ValueError("Environment manifest differs from loader identity")
        env = self.environments[environment_id]
        if env["role"] == "target_test" and not self.allow_final:
            raise ValueError("Final temporal evaluation requires frozen-design gate")
        graph, ids = arrival_view(
            self.dataset, env["end_year"], self.manifest["direction"]
        )
        arrival_ids = torch.where(self.dataset.years[ids] >= env["first_year"])[0]
        labels = self.dataset.labels[ids]
        splits = (
            stratified_three_way(
                arrival_ids, labels, self.manifest["split_seed"] + env["end_year"]
            )
            if env["role"] == "meta_train"
            else {"eval": arrival_ids}
        )
        record = GraphRecord(
            graph,
            labels,
            splits,
            env["role"],
            dict(env, global_node_ids=ids.tolist()),
            split_policy="partial",
        )
        bundle = DatasetBundle({environment_id: record}, self.class_names)
        bundle.validate()
        return bundle


def load_country(root, country, *, download=False):
    """Verify published checksum, restrict pickle globals, preserve raw direction."""
    if country not in COUNTRY_MD5:
        raise ValueError("Unknown Pair-Align country")
    root = Path(root)
    path = root / f"{country}_labels_20.pt"
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            urllib.request.urlretrieve(
                f"https://zenodo.org/api/records/10681285/files/{path.name}/content",
                temporary,
            )
            if hashlib.md5(temporary.read_bytes()).hexdigest() != COUNTRY_MD5[country]:
                raise ValueError("Published MAG artifact checksum mismatch")
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.md5(path.read_bytes()).hexdigest() != COUNTRY_MD5[country]:
        raise ValueError("Published MAG artifact checksum mismatch")
    with torch.serialization.safe_globals(
        [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]
    ):
        raw = torch.load(path, map_location="cpu", weights_only=True)
    if set(raw.keys()) != {"x", "edge_index", "y"}:
        raise ValueError("Country artifact schema changed")
    graph = Data(
        x=raw.x.clone().float(),
        edge_index=raw.edge_index.clone().long(),
        num_nodes=len(raw.x),
    )
    labels = raw.y.reshape(-1).long()
    record = GraphRecord(
        graph, labels, {"eval": torch.arange(graph.num_nodes)}, "validation"
    )
    DatasetBundle({"country": record}, tuple(map(str, range(20)))).validate()
    if graph.x.shape[1] != 128:
        raise ValueError("Unexpected country feature schema")
    return graph, labels


def prepare_country_protocol(root, *, seed=20281110, direction="undirected"):
    """US source, CN/DE practice, FR validation, JP/RU final; new protocol."""
    graph, labels = load_country(root, "US")
    if direction == "undirected":
        graph.edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
    elif direction != "directed":
        raise ValueError("Unknown direction policy")
    source = stratified_three_way(torch.arange(len(labels)), labels, seed)
    environments = [
        dict(
            id=c, country=c, role=r, family="natural_country", severity=0.0, shift=None
        )
        for c, r in (
            ("CN", "meta_train"),
            ("DE", "meta_train"),
            ("FR", "validation"),
            ("JP", "target_test"),
            ("RU", "target_test"),
        )
    ]
    return dict(
        protocol="pair-align-country-v1",
        dataset="pair-align-mag-country",
        source_country="US",
        direction=direction,
        split_seed=seed,
        source_graph_sha256=graph_fingerprint(graph),
        source_splits={k: v.tolist() for k, v in source.items()},
        environments=environments,
        artifact_license="CC-BY-4.0",
        origin="https://zenodo.org/records/10681285",
        interpretation="Separate generators trained on MAG countries; no direct Arxiv-generator transfer claim",
    )


class CountryEnvironments:
    def __init__(self, root, manifest, *, allow_final=False):
        self.root, self.manifest, self.allow_final = root, manifest, allow_final
        self.environments = {e["id"]: e for e in manifest["environments"]}
        self.class_names = tuple(f"venue_group_{i}" for i in range(20))

    def __call__(self, manifest, environment_id):
        if manifest != self.manifest:
            raise ValueError("Environment manifest differs from loader identity")
        env = self.environments[environment_id]
        if env["role"] == "target_test" and not self.allow_final:
            raise ValueError("Final country evaluation requires frozen-design gate")
        graph, labels = load_country(self.root, env["country"])
        if self.manifest["direction"] == "undirected":
            graph.edge_index = to_undirected(
                graph.edge_index, num_nodes=graph.num_nodes
            )
        splits = (
            stratified_three_way(
                torch.arange(graph.num_nodes),
                labels,
                self.manifest["split_seed"] + ord(env["country"][0]),
            )
            if env["role"] == "meta_train"
            else {"eval": torch.arange(graph.num_nodes)}
        )
        record = GraphRecord(
            graph, labels, splits, env["role"], dict(env), split_policy="partial"
        )
        bundle = DatasetBundle({environment_id: record}, self.class_names)
        bundle.validate()
        return bundle
