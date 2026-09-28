"""Persist fitted frozen-embedding probes and verify their serialized predictions."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def persist_probe(model, temperature, threshold, matrix, test, path, metadata):
    """Persist and reload the fitted artifact before it can score external data."""
    import joblib

    bundle = dict(model=model, temperature=temperature, threshold=threshold, **metadata)
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    expected = model.predict_proba(matrix[test])[:, 1]
    observed = restored["model"].predict_proba(matrix[test])[:, 1]
    if not np.array_equal(expected, observed):
        raise ValueError("Persisted probe does not reproduce pre-save predictions exactly")
    return restored



def persist_fitted_probes(fitted: dict, features: dict, directory: Path, metadata: dict) -> dict:
    """Save opt-in fitted probes after the native output directory is created.

    The sklearn pipeline includes its training-fitted scaler. Feature ordering,
    frozen-encoder identity and chromosome partitions travel with every bundle.
    Large feature arrays are used for verification only, never serialized.
    """
    if not fitted:
        return {}
    directory.mkdir(exist_ok=False)
    receipt = {}
    for name, fit in fitted.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid feature-set artifact name")
        path = directory / f"{name}.joblib"
        persist_probe(fit["model"], fit["temperature"], fit["threshold"],
                      features[name], fit["evaluation_mask"], path,
                      {**metadata, **fit["metadata"], "feature_set": name})
        receipt[name] = {"path": str(path.resolve()),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "raw_predictions_exact_after_reload": True}
    return receipt
