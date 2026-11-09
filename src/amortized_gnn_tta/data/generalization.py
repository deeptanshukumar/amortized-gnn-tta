"""Predeclared held-out Arxiv interventions; no evaluation label access."""

import copy


def arxiv_generalization_manifest(training_manifest):
    """New draws, stronger shifts and new operations on the official test nodes.

    Two independent draws are paired across each feature/edge/combined family.
    Operation strengths have explicit, different units; they are not comparable
    scalar distances across families.
    """
    result = copy.deepcopy(training_manifest)
    environments = []

    def add(name, family, rate, draw, group, operation=None):
        environments.append(
            dict(
                id=name,
                role="target_test",
                family=family,
                shift=None if family == "control" else family,
                severity=rate,
                feature_rate=rate if family in {"attribute", "combined"} else 0.0,
                edge_rate=rate if family in {"structural", "combined"} else 0.0,
                feature_seed=91001 + draw,
                edge_seed=92001 + draw,
                generalization_group=group,
                operation=operation,
            )
        )

    add("test-control", "control", 0.0, 0, "unchanged")
    for draw in (0, 1):
        for group, rates in (("new_draw", (0.1, 0.3)), ("stronger", (0.5, 0.7))):
            for index, rate in enumerate(rates):
                for family in ("attribute", "structural", "combined"):
                    add(
                        f"test-{group}-{family}-{index}-{draw}",
                        family,
                        rate,
                        100 * index + 1000 * draw,
                        group,
                    )
    for operation, family, rate in [
        ("gaussian_features", "novel_attribute", 0.5),
        ("degree_biased_removal", "novel_structural", 0.3),
    ]:
        add("test-" + operation, family, rate, 777, "new_operation", operation)
    result["environments"] = environments
    result["evaluation_protocol"] = "arxiv-generalization-v1"
    result["interpretation"] = (
        "Official 2019–2020 labels; graph and checkpoint holdout, not an independent graph"
    )
    return result
