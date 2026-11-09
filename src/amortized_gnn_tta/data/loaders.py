"""Small dataset dispatch boundary shared by experiment orchestration."""

from pathlib import Path


def make_environment_loader(dataset, root, manifest, *, allow_final=False):
    """Return a (manifest, environment_id) -> DatasetBundle callable.

    Keep official downloads explicit in dataset APIs; dispatch never downloads.
    No model-specific preprocessing belongs at this boundary.
    """
    if dataset == "csbm":
        from .environments import load_shift_environment

        def load(current, environment_id):
            env = next(e for e in current["environments"] if e["id"] == environment_id)
            if env["role"] == "target_test" and not allow_final:
                raise ValueError(
                    "Final CSBM environment requires explicit evaluation gate"
                )
            return load_shift_environment(current, environment_id)

        return load
    if dataset == "ogbn-arxiv":
        from .ogb import load_arxiv
        from .real_environments import ArxivEnvironments

        return ArxivEnvironments(
            load_arxiv(Path(root)), manifest, allow_final=allow_final
        )
    if dataset == "ogbn-arxiv-temporal":
        from .ogb import load_arxiv
        from .natural import TemporalEnvironments

        return TemporalEnvironments(
            load_arxiv(Path(root)), manifest, allow_final=allow_final
        )
    if dataset == "pair-align-mag-country":
        from .natural import CountryEnvironments

        return CountryEnvironments(Path(root), manifest, allow_final=allow_final)
    raise ValueError(f"Unsupported research dataset: {dataset}")
