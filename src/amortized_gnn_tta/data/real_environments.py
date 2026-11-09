"""Manifest-driven Arxiv graph contexts and explicit supervision pools.

One full citation graph is reused across perturbations. Distinct corruption
seeds do not constitute independent underlying graphs. Final environments require
an explicit evaluation gate, including when only their features are requested.
"""

import hashlib
import math

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from .ogb import partition_training_labels
from .corruptions import perturb_graph, unfamiliar_perturbation
from .api import DatasetBundle, GraphRecord, graph_fingerprint


def prepare_arxiv_protocol(
    dataset,
    *,
    source_fraction=0.5,
    pool_seed=20281108,
    direction="undirected",
    training_rates=(0.1, 0.3),
    validation_rates=(0.2, 0.4),
):
    if direction not in {"directed", "undirected"}:
        raise ValueError("Unknown direction policy")
    for rates in (training_rates, validation_rates):
        if (
            not rates
            or len(set(rates)) != len(rates)
            or any(
                isinstance(r, bool) or not math.isfinite(r) or not 0 < r < 1
                for r in rates
            )
        ):
            raise ValueError(
                "Corruption rates must be unique values strictly between zero and one"
            )
    graph = dataset.graph
    if direction == "undirected":
        graph = Data(
            x=graph.x,
            edge_index=to_undirected(graph.edge_index, num_nodes=graph.num_nodes),
            num_nodes=graph.num_nodes,
        )
    pools = partition_training_labels(
        dataset, source_fraction=source_fraction, seed=pool_seed
    )
    # A teacher's optional inner validation/test pools stay inside generator labels.
    rng = torch.Generator().manual_seed(pool_seed + 1)
    pieces = {
        k: [] for k in ("generator_train", "generator_stop", "generator_diagnostic")
    }
    for category in range(40):
        indices = pools["generator_train"][
            dataset.labels[pools["generator_train"]] == category
        ]
        indices = indices[torch.randperm(len(indices), generator=rng)]
        count = max(1, len(indices) // 10)
        if len(indices) <= 2 * count:
            raise ValueError("Too few class members for teacher pools")
        pieces["generator_stop"].append(indices[:count])
        pieces["generator_diagnostic"].append(indices[count : 2 * count])
        pieces["generator_train"].append(indices[2 * count :])
    pools.update({k: torch.cat(v).sort().values for k, v in pieces.items()})
    environments = []
    for role, rates in (
        ("meta_train", training_rates),
        ("validation", validation_rates),
    ):
        environments.append(
            dict(
                id=f"{role}-control",
                role=role,
                family="control",
                shift=None,
                severity=0.0,
                feature_rate=0.0,
                edge_rate=0.0,
                feature_seed=701,
                edge_seed=702,
            )
        )
        for index, rate in enumerate(rates):
            for family in ("attribute", "structural", "combined"):
                environments.append(
                    dict(
                        id=f"{role}-{family}-{index}",
                        role=role,
                        family=family,
                        shift=family,
                        severity=rate,
                        seen_severity_in_meta_train=rate in training_rates,
                        feature_rate=rate if family != "structural" else 0.0,
                        edge_rate=rate if family != "attribute" else 0.0,
                        feature_seed=701
                        + 100 * index
                        + (10000 if role == "validation" else 0),
                        edge_seed=702
                        + 100 * index
                        + (10000 if role == "validation" else 0),
                    )
                )
    return dict(
        protocol="arxiv-transductive-v1",
        dataset="ogbn-arxiv",
        direction=direction,
        canonical_graph_sha256=graph_fingerprint(graph),
        official_split_sha256={
            k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
            for k, v in dataset.official_splits.items()
        },
        pool_seed=pool_seed,
        source_fraction=source_fraction,
        source_stopping_fraction=0.1,
        label_pools={k: v.tolist() for k, v in pools.items()},
        environments=environments,
        interpretation="Full-graph transductive controlled perturbations; native test labels are sealed",
    )


class ArxivEnvironments:
    """Reusable loader satisfying the existing meta-training callback interface."""

    def __init__(self, dataset, manifest, *, allow_final=False):
        self.allow_final = allow_final
        self.dataset, self.manifest = dataset, manifest
        self.graph = dataset.graph
        if manifest["direction"] == "undirected":
            self.graph = Data(
                x=self.graph.x,
                edge_index=to_undirected(
                    self.graph.edge_index, num_nodes=self.graph.num_nodes
                ),
                num_nodes=self.graph.num_nodes,
            )
        if graph_fingerprint(self.graph) != manifest["canonical_graph_sha256"]:
            raise ValueError("Arxiv graph identity changed")
        for k, v in dataset.official_splits.items():
            if (
                hashlib.sha256(v.numpy().tobytes()).hexdigest()
                != manifest["official_split_sha256"][k]
            ):
                raise ValueError("Official split changed")
        self.pools = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in manifest["label_pools"].items()
        }
        joined = torch.cat(list(self.pools.values()))
        if joined.unique().numel() != len(joined) or not torch.equal(
            joined.sort().values, dataset.official_splits["train"].sort().values
        ):
            raise ValueError("Pools must partition official train")
        self.environments = {e["id"]: e for e in manifest["environments"]}

    def __call__(self, manifest, environment_id):
        if manifest != self.manifest:
            raise ValueError("Environment manifest differs from loader identity")
        env = self.environments[environment_id]
        if env["role"] == "target_test" and not self.allow_final:
            raise ValueError(
                "Final environment requires separate frozen evaluation protocol"
            )
        if env["role"] not in {"meta_train", "validation", "target_test"}:
            raise ValueError("Invalid environment role")
        if env.get("operation"):
            graph = unfamiliar_perturbation(
                self.graph,
                operation=env["operation"],
                strength=env["severity"],
                seed=env["feature_seed"],
                undirected=self.manifest["direction"] == "undirected",
            )
        else:
            graph = perturb_graph(
                self.graph,
                feature_mask_rate=env["feature_rate"],
                edge_drop_rate=env["edge_rate"],
                feature_seed=env["feature_seed"],
                edge_seed=env["edge_seed"],
                undirected=self.manifest["direction"] == "undirected",
            )
        if env["role"] == "meta_train":
            splits = {
                "train": self.pools["generator_train"],
                "val": self.pools["generator_stop"],
                "test": self.pools["generator_diagnostic"],
            }
        else:
            splits = {
                "eval": self.dataset.official_splits[
                    "test" if env["role"] == "target_test" else "valid"
                ]
            }
        record = GraphRecord(
            graph,
            self.dataset.labels,
            splits,
            env["role"],
            dict(env),
            split_policy="partial",
        )
        bundle = DatasetBundle(
            {environment_id: record},
            tuple(f"category_{i}" for i in range(40)),
            dict(dataset="ogbn-arxiv", protocol=self.manifest["protocol"]),
        )
        bundle.validate()
        return bundle
