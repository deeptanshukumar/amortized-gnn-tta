"""Official Arxiv inputs with immutable split identities and partial label views.

OGB's MIT raw PyG reader is used directly. The published archive is official;
our cache contains plain tensors, avoiding version-sensitive pickled PyG Data.
No features are normalized, edges reversed, or self-loops added by this loader.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

import torch
from torch import Tensor
from torch_geometric.data import Data

from .api import DatasetBundle, GraphRecord

URL = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"


@dataclass
class ArxivDataset:
    graph: Data
    labels: Tensor
    years: Tensor
    official_splits: dict[str, Tensor]
    metadata: dict

    def view(self, role: str, splits: dict[str, Tensor]) -> DatasetBundle:
        """Share graph context while exposing only explicitly selected labels.

        Different views intentionally reference the same graph. Adaptation gets
        only the bundle's cloned allowlisted unlabeled_input, never this object.
        """
        permitted = (
            self.official_splits["train"]
            if role in {"source_train", "meta_train"}
            else self.official_splits["valid"]
            if role == "validation"
            else self.official_splits["test"]
            if role == "target_test"
            else None
        )
        if permitted is None or any(
            not torch.isin(indices, permitted).all() for indices in splits.values()
        ):
            raise ValueError("Label view exceeds the role's official node pool")
        bundle = DatasetBundle(
            {
                "arxiv": GraphRecord(
                    self.graph,
                    self.labels,
                    {k: v.clone() for k, v in splits.items()},
                    role,
                    dict(self.metadata),
                    "partial",
                )
            },
            tuple(f"category_{i}" for i in range(40)),
            dict(self.metadata),
        )
        bundle.validate()
        return bundle


def _payload_sha256(payload):
    """Content identity for cache corruption detection, not publisher authenticity."""
    digest = hashlib.sha256()
    tensors = {name: payload[name] for name in ("x", "edge_index", "years", "labels")}
    tensors.update(
        {f"split_{name}": value for name, value in payload["splits"].items()}
    )
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    digest.update(
        json.dumps(payload["metadata"], sort_keys=True, allow_nan=False).encode()
    )
    return digest.hexdigest()


def load_arxiv(root: Path, *, download: bool = False) -> ArxivDataset:
    """Load native directed Arxiv; explicit download, tensor-only cache, no transforms."""
    root = Path(root)
    cache = root / "arxiv-tensors-v2.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload.get("cache_version") != 2 or payload.get(
            "content_sha256"
        ) != _payload_sha256(payload):
            raise ValueError("Arxiv tensor cache integrity/schema mismatch")
    else:
        raw = root / "arxiv" / "raw"
        if not (raw / "edge.csv.gz").exists():
            if not download:
                raise FileNotFoundError(
                    "Official Arxiv raw files absent; enable download explicitly"
                )
            root.mkdir(parents=True, exist_ok=True)
            if (root / "arxiv").exists():
                raise FileExistsError(
                    "Incomplete Arxiv directory; inspect it before retrying"
                )
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                archive = Path(temporary) / "arxiv.zip"
                urllib.request.urlretrieve(URL, archive)
                hasher = hashlib.sha256()
                with archive.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                with zipfile.ZipFile(archive) as z:
                    destination = Path(temporary) / "extracted"
                    for entry in z.infolist():
                        path = Path(entry.filename)
                        if path.is_absolute() or ".." in path.parts:
                            raise ValueError("Invalid archive member")
                    z.extractall(destination)
                (destination / "arxiv").rename(root / "arxiv")
                (root / "archive.sha256").write_text(digest + "\n")
        import pandas as pd
        from ogb.io.read_graph_pyg import read_graph_pyg

        graph = read_graph_pyg(
            str(raw), add_inverse_edge=False, additional_node_files=["node_year"]
        )[0]
        splits = {
            key: torch.tensor(
                pd.read_csv(
                    root / "arxiv" / "split" / "time" / (name + ".csv.gz"), header=None
                ).values.reshape(-1),
                dtype=torch.long,
            )
            for key, name in (("train", "train"), ("valid", "valid"), ("test", "test"))
        }
        payload = dict(
            x=graph.x.float(),
            edge_index=graph.edge_index.long(),
            years=graph.node_year.reshape(-1).long(),
            labels=torch.tensor(
                pd.read_csv(raw / "node-label.csv.gz", header=None).values.reshape(-1),
                dtype=torch.long,
            ),
            splits=splits,
            metadata=dict(
                dataset="ogbn-arxiv",
                protocol="official-directed-v1",
                origin=URL,
                graph_visibility="full-transductive",
                preprocessing="none",
                archive_sha256=(root / "archive.sha256").read_text().strip()
                if (root / "archive.sha256").exists()
                else None,
            ),
        )
        # Validate before publication, and bind tensor/split/metadata contents.
        validate_arxiv(
            ArxivDataset(
                Data(
                    x=payload["x"],
                    edge_index=payload["edge_index"],
                    num_nodes=len(payload["years"]),
                ),
                payload["labels"],
                payload["years"],
                payload["splits"],
                payload["metadata"],
            )
        )
        payload["cache_version"] = 2
        payload["content_sha256"] = _payload_sha256(payload)
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as f:
            temporary = Path(f.name)
        try:
            torch.save(payload, temporary)
            os.link(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
    graph = Data(
        x=payload["x"],
        edge_index=payload["edge_index"],
        num_nodes=len(payload["years"]),
    )
    dataset = ArxivDataset(
        graph,
        payload["labels"],
        payload["years"],
        payload["splits"],
        payload["metadata"],
    )
    validate_arxiv(dataset)
    return dataset


def validate_arxiv(dataset: ArxivDataset) -> None:
    """Check native dimensions and indices; no final-label statistics/scoring."""
    if dataset.graph.num_nodes != 169343 or dataset.graph.x.shape != (169343, 128):
        raise ValueError("Unexpected official Arxiv feature dimensions")
    if dataset.graph.edge_index.shape != (2, 1166243):
        raise ValueError("Unexpected native directed edge count")
    if (
        dataset.graph.edge_index.dtype != torch.long
        or dataset.graph.x.dtype != torch.float32
        or not torch.isfinite(dataset.graph.x).all()
    ):
        raise ValueError("Invalid input dtype or nonfinite features")
    if (
        dataset.graph.edge_index.min() < 0
        or dataset.graph.edge_index.max() >= dataset.graph.num_nodes
    ):
        raise ValueError("Invalid edge endpoints")
    expected = {"train": 90941, "valid": 29799, "test": 48603}
    for key, count in expected.items():
        indices = dataset.official_splits[key]
        if (
            indices.dtype != torch.long
            or indices.numel() != count
            or indices.unique().numel() != count
        ):
            raise ValueError("Invalid official split")
        if indices.min() < 0 or indices.max() >= dataset.graph.num_nodes:
            raise ValueError("Invalid node index")
        years = dataset.years[indices]
        if not bool(
            (
                (years <= 2017)
                if key == "train"
                else (years == 2018)
                if key == "valid"
                else (years >= 2019)
            ).all()
        ):
            raise ValueError("Official years/splits disagree")
    joined = torch.cat(list(dataset.official_splits.values()))
    if joined.unique().numel() != dataset.graph.num_nodes:
        raise ValueError("Official splits overlap or omit nodes")
    if (
        dataset.labels.shape != (dataset.graph.num_nodes,)
        or dataset.labels.dtype != torch.long
    ):
        raise ValueError("Invalid label shape/dtype")


def partition_training_labels(
    dataset: ArxivDataset,
    *,
    source_fraction: float,
    seed: int,
    stopping_fraction: float = 0.1,
) -> dict[str, Tensor]:
    """Disjoint class-stratified source/generator pools inside official train only.

    Source stopping uses a nested source-only pool. The allocation is explicit;
    2018 and final labels are never inspected to construct these groups.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Pool seed must be an integer in [0,2**32)")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in (source_fraction, stopping_fraction)
    ):
        raise ValueError("Pool fractions must be finite numbers")
    if not 0 < source_fraction < 1 or not 0 < stopping_fraction < 1:
        raise ValueError("Pool fractions must be strictly between zero and one")
    rng = torch.Generator().manual_seed(seed)
    result = {k: [] for k in ("source_train", "source_stop", "generator_train")}
    train = dataset.official_splits["train"]
    labels = dataset.labels[train]
    for category in range(40):
        indices = train[labels == category]
        indices = indices[torch.randperm(len(indices), generator=rng)]
        source_count = int(len(indices) * source_fraction)
        stop_count = max(1, int(source_count * stopping_fraction))
        if source_count <= stop_count or source_count >= len(indices):
            raise ValueError("Allocation cannot represent every class in all pools")
        result["source_stop"].append(indices[:stop_count])
        result["source_train"].append(indices[stop_count:source_count])
        result["generator_train"].append(indices[source_count:])
    return {key: torch.cat(value).sort().values for key, value in result.items()}
