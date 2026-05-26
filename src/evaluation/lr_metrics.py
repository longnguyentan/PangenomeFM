from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class TrainResult:
    auc_overall: float
    auc_by_group: pd.DataFrame
    model: LogisticRegression


def fit_lr_auc(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: pd.DataFrame,
) -> TrainResult:
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p))

    # group AUC
    rows = []
    for (sn, closure), idx in groups_test.groupby(
        ["target_sn", "closure"]
    ).groups.items():
        idx = np.array(list(idx), dtype=int)
        if len(np.unique(y_test[idx])) < 2:
            g_auc = float("nan")
        else:
            g_auc = float(roc_auc_score(y_test[idx], p[idx]))
        rows.append(
            {
                "target_sn": sn,
                "closure": closure,
                "n_edges": int(len(idx)),
                "pos_frac": float(y_test[idx].mean()),
                "auc": g_auc,
            }
        )
    auc_by = (
        pd.DataFrame(rows).sort_values(["target_sn", "closure"]).reset_index(drop=True)
    )
    return TrainResult(auc_overall=auc, auc_by_group=auc_by, model=clf)
