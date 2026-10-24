"""Independently implemented balanced two-class Gaussian contextual SBM.

Protocol ``csbm-matcha-code-v1`` follows the sampling distribution in Matcha
revision f0e7104a84d118a8dabb7ad210915c43722b94af, src/dataset/CSBM.py.
No upstream code is copied. Local RNG streams and vectorized row sampling are
intentional engineering differences; exact upstream seeded artifacts are not
reproduced. Edges take O(N²) pair scans but O(N+E) auxiliary storage, never N².
"""

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from .api import DatasetBundle, EnvironmentRole, GraphRecord, ROLES, graph_fingerprint

PROTOCOL = "csbm-matcha-code-v1"
REFERENCE = "https://github.com/baowenxuan/Matcha/blob/f0e7104a84d118a8dabb7ad210915c43722b94af/src/dataset/CSBM.py"


@dataclass(frozen=True)
class CSBMConfig:
    """Explicit edge probabilities and class-center coefficients.

    x_i = (class_mean[y_i] + noise_scale*z_i) / sqrt(num_features),
    where z_i has independent standard-normal coordinates. Class 1 is positive.
    Seeds select local, domain-separated NumPy PCG64 streams, without modifying
    global RNGs. Stream tags distinguish edges/features/splits even for equal seeds.
    """

    num_nodes: int = 1000
    num_features: int = 64
    p_in: float = 0.008
    p_out: float = 0.002
    negative_mean: float = -0.03
    positive_mean: float = 0.03
    noise_scale: float = 1.0
    edge_seed: int = 0
    feature_seed: int = 1
    split_seed: int = 2
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)

    def __post_init__(self) -> None:
        for name in ("num_nodes", "num_features", "edge_seed", "feature_seed", "split_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.num_nodes < 6 or self.num_nodes % 2 or self.num_features < 1:
            raise ValueError("Use an even num_nodes >= 6 and positive num_features")
        if min(self.edge_seed, self.feature_seed, self.split_seed) < 0:
            raise ValueError("Seeds must be nonnegative")
        for value in (self.p_in, self.p_out):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Edge probabilities must be finite and in [0,1]")
        if not all(math.isfinite(x) for x in (self.negative_mean, self.positive_mean, self.noise_scale)):
            raise ValueError("Feature parameters must be finite")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be nonnegative")
        if (len(self.split_ratios) != 3 or
                any(not math.isfinite(x) or x <= 0 for x in self.split_ratios) or
                not math.isclose(sum(self.split_ratios), 1.0, abs_tol=1e-12, rel_tol=0)):
            raise ValueError("Three positive split ratios summing to 1 are required")
        if min(_split_counts(self.num_nodes // 2, self.split_ratios)) < 1:
            raise ValueError("Each class needs at least one node in each source split")

    @property
    def expected_degree(self) -> float:
        """Exact finite-size expectation, excluding self-loops."""
        return (self.num_nodes / 2 - 1) * self.p_in + self.num_nodes / 2 * self.p_out

    @classmethod
    def from_matcha(cls, *, num_nodes: int = 5000, num_features: int = 2000,
                    degree_scale: float = 5.0, homophily_parameter: float = 0.8,
                    **kwargs) -> "CSBMConfig":
        """Translate released-code parameters, not a desired average degree.

        In particular, expected_degree = degree_scale/2 -
        degree_scale*homophily_parameter/num_nodes. The h parameter is not a
        promise about a sampled graph's measured edge/node homophily.
        """
        if (not math.isfinite(degree_scale) or degree_scale < 0 or
                not math.isfinite(homophily_parameter) or not 0 <= homophily_parameter <= 1):
            raise ValueError("Invalid degree_scale or homophily_parameter")
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 6:
            raise ValueError("num_nodes must be an integer >= 6")
        return cls(num_nodes=num_nodes, num_features=num_features,
                   p_in=degree_scale * homophily_parameter / num_nodes,
                   p_out=degree_scale * (1 - homophily_parameter) / num_nodes, **kwargs)


@dataclass(frozen=True)
class CSBMEnvironment:
    """A declared graph identity/role and its complete generation configuration."""

    graph_id: str
    role: EnvironmentRole
    config: CSBMConfig


def _split_counts(n: int, ratios: tuple[float, float, float]) -> np.ndarray:
    """Largest-remainder rounding per class; ties follow train/val/test order."""
    expected = n * np.asarray(ratios)
    counts = np.floor(expected).astype(np.int64)
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:n - int(counts.sum())]] += 1
    return counts


def _source_splits(labels: np.ndarray, config: CSBMConfig) -> dict[str, torch.Tensor]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.split_seed, 2])))
    parts: list[list[np.ndarray]] = [[], [], []]
    counts = _split_counts(config.num_nodes // 2, config.split_ratios)
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        for bucket, indices in zip(parts, np.split(indices, np.cumsum(counts)[:-1])):
            bucket.append(indices)
    return {name: torch.from_numpy(np.sort(np.concatenate(bucket)))
            for name, bucket in zip(("train", "val", "test"), parts)}


def generate_csbm(config: CSBMConfig, *, role: EnvironmentRole = "source_train") -> GraphRecord:
    """Generate a CPU graph without feature normalization or self-loop changes.

    Each unordered pair is sampled once; both directions are then stored. Sampling
    rows with local RNGs avoids the reference's per-pair Python loop, without
    changing the independent Bernoulli edge distribution. Labels retain the
    reference's positive-then-negative storage order; IDs are not model features.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown environment role: {role}")
    n, f = config.num_nodes, config.num_features
    labels = np.concatenate((np.ones(n // 2, dtype=np.int64), np.zeros(n // 2, dtype=np.int64)))
    edge_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.edge_seed, 0])))
    sources, destinations = [], []
    for i in range(n - 1):
        candidates = np.arange(i + 1, n, dtype=np.int64)
        probabilities = np.where(labels[candidates] == labels[i], config.p_in, config.p_out)
        chosen = candidates[edge_rng.random(len(candidates)) < probabilities]
        sources.append(np.full(len(chosen), i, dtype=np.int64))
        destinations.append(chosen)
    src, dst = np.concatenate(sources), np.concatenate(destinations)
    edges = np.stack((np.concatenate((src, dst)), np.concatenate((dst, src))))
    # Canonical row-major ordering makes fingerprints and graph inspections stable.
    edges = edges[:, np.lexsort((edges[1], edges[0]))].copy()
    feature_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.feature_seed, 1])))
    features = feature_rng.standard_normal((n, f), dtype=np.float32)
    features *= config.noise_scale
    means = np.where(labels == 1, config.positive_mean, config.negative_mean)
    features += means[:, None].astype(np.float32)
    features /= math.sqrt(f)
    graph = Data(x=torch.from_numpy(features), edge_index=torch.from_numpy(edges), num_nodes=n)
    splits = (_source_splits(labels, config) if role in {"source_train", "meta_train"}
              else {"eval": torch.arange(n, dtype=torch.long)})
    metadata = {
        "protocol": PROTOCOL, "reference": REFERENCE, "config": asdict(config),
        "software": {"numpy": np.__version__, "torch": str(torch.__version__)},
        "expected_degree": config.expected_degree,
        "matcha_equivalent_parameters": {
            "degree_scale": n * (config.p_in + config.p_out),
            "homophily_parameter": config.p_in / (config.p_in + config.p_out)
            if config.p_in + config.p_out else None,
        },
        "edge_convention": "simple undirected, both directions, no self-loops",
        "graph_context": "full graph; node indices select supervision, not connectivity",
        "rng": "PCG64(SeedSequence([seed, stream_tag])); tags edge=0, feature=1, split=2",
        "fingerprint": graph_fingerprint(graph),
        "preprocessing": [],
    }
    return GraphRecord(graph, torch.from_numpy(labels), splits, role, metadata)


def load_dataset(environments: Sequence[CSBMEnvironment]) -> DatasetBundle:
    """Load independently drawn CSBM environments in memory, with no cache.

    Reusing edge or feature seeds across environments would couple draws and can
    create paired lineages. This initial protocol rejects that; paired-shift
    construction is a distinct, future experiment rather than an implicit mode.
    """
    records: dict[str, GraphRecord] = {}
    edge_seeds, feature_seeds = set(), set()
    for environment in environments:
        if not environment.graph_id or environment.graph_id in records:
            raise ValueError("Graph IDs must be unique and nonempty")
        config = environment.config
        if config.edge_seed in edge_seeds or config.feature_seed in feature_seeds:
            raise ValueError("Independent environments must use distinct edge and feature seeds")
        edge_seeds.add(config.edge_seed)
        feature_seeds.add(config.feature_seed)
        records[environment.graph_id] = generate_csbm(config, role=environment.role)
    bundle = DatasetBundle(records, metadata={
        "dataset": "csbm", "protocol": PROTOCOL,
        "task": "single_label_node_classification", "metric": "accuracy",
        "environment_protocol": "independent_draws",
    })
    bundle.validate()
    return bundle
