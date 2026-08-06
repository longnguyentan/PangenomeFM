"""Feature policies for cCRE downstream tasks.

cCRE labels are projected from the GRCh38 path, so reference/path-status fields
can become shortcuts.  Final paper-facing cCRE runs should use the
``leakage_safe`` policy, which removes both SR and is_grch38.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FULL_NODE_FEATURE_NAMES = [
    "log1p_SO",
    "log1p_LN",
    "SR",
    "is_grch38",
    "log1p_degree",
    "orient",
    "component_id",
]

POLICY_TO_KEEP_INDICES = {
    "leakage_safe": [0, 1, 4, 5, 6],
    "legacy_sr": [0, 1, 2, 4, 5, 6],
    "legacy_reference": [0, 1, 2, 3, 4, 5, 6],
}


@dataclass(frozen=True)
class FeaturePolicy:
    name: str
    keep_indices: list[int]
    feature_names: list[str]
    excludes_sr: bool
    excludes_is_grch38: bool


def resolve_feature_policy(name: str) -> FeaturePolicy:
    if name not in POLICY_TO_KEEP_INDICES:
        allowed = ", ".join(sorted(POLICY_TO_KEEP_INDICES))
        raise ValueError(f"Unknown cCRE feature policy {name!r}; expected one of: {allowed}")
    keep = POLICY_TO_KEEP_INDICES[name]
    names = [FULL_NODE_FEATURE_NAMES[i] for i in keep]
    return FeaturePolicy(
        name=name,
        keep_indices=list(keep),
        feature_names=names,
        excludes_sr=2 not in keep,
        excludes_is_grch38=3 not in keep,
    )


def select_node_features(X7: np.ndarray, policy_name: str) -> np.ndarray:
    policy = resolve_feature_policy(policy_name)
    if X7.shape[1] != len(FULL_NODE_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FULL_NODE_FEATURE_NAMES)} raw node features before cCRE "
            f"feature-policy selection, got {X7.shape[1]}."
        )
    return X7[:, policy.keep_indices]

