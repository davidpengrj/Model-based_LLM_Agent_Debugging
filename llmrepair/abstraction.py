from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .data import read_jsonl


SEMANTIC_COLUMNS = [
    "action_type",
    "progress_state",
    "task_alignment",
    "evidence_quality",
    "unsupported_claim",
    "tool_error",
    "reasoning_error_signal",
]


def profile_rows_to_frame(profile_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in profile_rows:
        profile = row["profile"]
        flat = {
            "log_id": row["log_id"],
            "step_idx": int(row["step_idx"]),
            "agent_name": row["agent_name"],
        }
        flat.update(profile)
        rows.append(flat)
    return pd.DataFrame(rows)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric_cols = [
        "task_alignment",
        "evidence_quality",
        "new_claim",
        "unsupported_claim",
        "tool_use",
        "tool_error",
        "reasoning_error_signal",
    ]
    cat_cols = ["action_type", "progress_state"]
    work = df[numeric_cols + cat_cols].copy()
    for col in numeric_cols:
        work[col] = work[col].astype(int)
    features = pd.get_dummies(work, columns=cat_cols, dtype=float)
    return features.to_numpy(dtype=float), list(features.columns)


def state_stability(assignments: np.ndarray, semantics: pd.DataFrame, k: int) -> dict[str, Any]:
    state_scores = []
    supports = []
    for state_id in range(k):
        subset = semantics[assignments == state_id]
        supports.append(int(len(subset)))
        if subset.empty:
            state_scores.append(0.0)
            continue
        purities = []
        for col in SEMANTIC_COLUMNS:
            counts = subset[col].astype(str).value_counts()
            purities.append(float(counts.iloc[0] / len(subset)))
        state_scores.append(float(np.mean(purities)))
    return {
        "avg_stability": float(np.mean(state_scores)) if state_scores else 0.0,
        "min_stability": float(np.min(state_scores)) if state_scores else 0.0,
        "state_stability": state_scores,
        "supports": supports,
        "min_support": int(min(supports)) if supports else 0,
    }


def select_k_for_agent(
    df: pd.DataFrame,
    *,
    theta: float = 0.72,
    min_support: int = 5,
    min_k: int = 2,
    max_k: int = 15,
    random_state: int = 0,
) -> dict[str, Any]:
    x, feature_names = build_feature_matrix(df)
    n, d = x.shape
    if n < min_support * min_k:
        return {
            "selected_k": 1,
            "reason": "too_few_samples",
            "num_samples": int(n),
            "feature_names": feature_names,
            "candidates": [],
        }

    x_scaled = StandardScaler().fit_transform(x)
    pca_dim = max(1, min(8, d, n - 1))
    x_reduced = PCA(n_components=pca_dim, random_state=random_state).fit_transform(x_scaled)

    max_allowed_k = min(max_k, max(min_k, n // min_support))
    candidates = []
    selected = None
    semantics = df[SEMANTIC_COLUMNS].copy()

    for k in range(min_k, max_allowed_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=random_state,
            n_init=3,
        )
        assignments = gmm.fit_predict(x_reduced)
        metrics = state_stability(assignments, semantics, k)
        row = {
            "k": k,
            "bic": float(gmm.bic(x_reduced)),
            "converged": bool(gmm.converged_),
            **metrics,
        }
        candidates.append(row)
        if selected is None and row["avg_stability"] >= theta and row["min_support"] >= min_support:
            selected = k

    if selected is None:
        best = max(candidates, key=lambda item: (item["avg_stability"], -item["k"]))
        selected = int(best["k"])
        reason = "best_available_below_theta"
    else:
        reason = "smallest_k_meeting_theta"

    return {
        "selected_k": int(selected),
        "reason": reason,
        "num_samples": int(n),
        "pca_dim": int(pca_dim),
        "feature_names": feature_names,
        "candidates": candidates,
    }


def select_k_by_agent(
    profile_path: Path,
    output_path: Path,
    *,
    theta: float = 0.72,
    min_support: int = 5,
) -> dict[str, Any]:
    df = profile_rows_to_frame(read_jsonl(profile_path))
    result = {
        "profile_path": str(profile_path),
        "theta": theta,
        "min_support": min_support,
        "agents": {},
    }
    for agent, agent_df in df.groupby("agent_name"):
        if agent == "human":
            continue
        result["agents"][agent] = select_k_for_agent(
            agent_df.reset_index(drop=True),
            theta=theta,
            min_support=min_support,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
