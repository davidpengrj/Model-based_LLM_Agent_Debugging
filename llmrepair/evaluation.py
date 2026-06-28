from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .abstraction import SEMANTIC_COLUMNS, state_stability
from .data import read_jsonl


NUMERIC_FEATURES = [
    "task_alignment",
    "evidence_quality",
    "new_claim",
    "unsupported_claim",
    "tool_use",
    "tool_error",
    "reasoning_error_signal",
]
CATEGORICAL_FEATURES = ["action_type", "progress_state"]


@dataclass
class AgentStateModel:
    selected_k: int
    reason: str
    num_samples: int
    candidates: list[dict[str, Any]]
    vectorizer: DictVectorizer | None = None
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    gmm: GaussianMixture | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.selected_k <= 1 or self.vectorizer is None or self.gmm is None:
            return np.zeros(len(df), dtype=int)
        x = self.vectorizer.transform(feature_records(df))
        x = self.scaler.transform(x)
        x = self.pca.transform(x)
        return self.gmm.predict(x).astype(int)


class RiskTables:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.tables: dict[str, dict[tuple[Any, ...], list[int]]] = {
            "transition": defaultdict(lambda: [0, 0]),
            "state": defaultdict(lambda: [0, 0]),
            "action": defaultdict(lambda: [0, 0]),
            "agent": defaultdict(lambda: [0, 0]),
        }
        self.base_pos = 0
        self.base_total = 0

    def add(self, row: pd.Series) -> None:
        y = int(bool(row["is_decisive_error"]))
        self.base_pos += y
        self.base_total += 1
        keys = {
            "transition": (
                row["agent_name"],
                int(row["prev_agent_state"]),
                row["action_type"],
                int(row["state"]),
            ),
            "state": (row["agent_name"], int(row["state"])),
            "action": (row["agent_name"], row["action_type"]),
            "agent": (row["agent_name"],),
        }
        for name, key in keys.items():
            self.tables[name][key][0] += y
            self.tables[name][key][1] += 1

    @property
    def base_prob(self) -> float:
        return (self.base_pos + self.alpha) / (self.base_total + 2 * self.alpha)

    def lookup(self, row: pd.Series) -> tuple[float, int, str]:
        keys = [
            (
                "transition",
                (
                    row["agent_name"],
                    int(row["prev_agent_state"]),
                    row["action_type"],
                    int(row["state"]),
                ),
                2,
            ),
            ("state", (row["agent_name"], int(row["state"])), 2),
            ("action", (row["agent_name"], row["action_type"]), 5),
            ("agent", (row["agent_name"],), 5),
        ]
        base = self.base_prob
        for name, key, min_support in keys:
            pos, total = self.tables[name].get(key, [0, 0])
            if total >= min_support:
                prob = (pos + self.alpha) / (total + 2 * self.alpha)
                return float(prob / base), int(total), name
        return 1.0, 0, "base"


def semantic_value(row: dict[str, Any], name: str) -> Any:
    value = row[name]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def feature_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.to_dict("records"):
        record: dict[str, Any] = {}
        for name in NUMERIC_FEATURES:
            record[name] = int(semantic_value(row, name))
        for name in CATEGORICAL_FEATURES:
            record[name] = str(row[name])
        records.append(record)
    return records


def profile_score(row: pd.Series) -> float:
    progress = row["progress_state"]
    progress_penalty = {
        "on_track": 0.0,
        "uncertain": 0.25,
        "stalled": 1.0,
        "off_track": 2.0,
    }.get(progress, 0.25)
    return float(
        (3 - int(row["task_alignment"])) * 2.0
        + (3 - int(row["evidence_quality"])) * 0.75
        + progress_penalty
        + (2.0 if row["reasoning_error_signal"] else 0.0)
        + (1.5 if row["tool_error"] else 0.0)
        + (1.5 if row["unsupported_claim"] else 0.0)
    )


def load_joined_frame(steps_path: Path, profiles_path: Path) -> pd.DataFrame:
    profiles = {}
    for row in read_jsonl(profiles_path):
        profiles[(str(row["log_id"]), int(row["step_idx"]))] = row["profile"]

    rows = []
    for step in read_jsonl(steps_path):
        if step["agent_name"] == "human":
            continue
        key = (str(step["log_id"]), int(step["step_idx"]))
        if key not in profiles:
            raise ValueError(f"Missing profile for {key}")
        row = dict(step)
        row.update(profiles[key])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["step_idx"] = df["step_idx"].astype(int)
    df["mistake_step"] = df["mistake_step"].astype(int)
    df["is_decisive_error"] = df["is_decisive_error"].astype(bool)
    for name in NUMERIC_FEATURES:
        df[name] = df[name].astype(int)
    return df.sort_values(["log_id", "step_idx"]).reset_index(drop=True)


def fit_agent_state_model(
    df: pd.DataFrame,
    *,
    theta: float = 0.72,
    min_support: int = 5,
    min_k: int = 2,
    max_k: int = 15,
    random_state: int = 0,
) -> AgentStateModel:
    n = len(df)
    if n < min_support * min_k:
        return AgentStateModel(1, "too_few_samples", n, [])

    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(feature_records(df))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca_dim = max(1, min(8, x_scaled.shape[1], n - 1))
    pca = PCA(n_components=pca_dim, random_state=random_state)
    x_reduced = pca.fit_transform(x_scaled)

    max_allowed_k = min(max_k, max(min_k, n // min_support))
    semantics = df[SEMANTIC_COLUMNS].copy()
    selected_k = None
    selected_gmm = None
    candidates = []

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
        if selected_k is None and row["avg_stability"] >= theta and row["min_support"] >= min_support:
            selected_k = k
            selected_gmm = gmm

    if selected_k is None:
        best_index, _ = max(
            enumerate(candidates),
            key=lambda item: (item[1]["avg_stability"], -item[1]["k"]),
        )
        selected_k = int(candidates[best_index]["k"])
        selected_gmm = GaussianMixture(
            n_components=selected_k,
            covariance_type="diag",
            random_state=random_state,
            n_init=3,
        ).fit(x_reduced)
        reason = "best_available_below_theta"
    else:
        reason = "smallest_k_meeting_theta"

    return AgentStateModel(
        selected_k=int(selected_k),
        reason=reason,
        num_samples=n,
        candidates=candidates,
        vectorizer=vectorizer,
        scaler=scaler,
        pca=pca,
        gmm=selected_gmm,
    )


def fit_state_models(train_df: pd.DataFrame, random_state: int = 0) -> dict[str, AgentStateModel]:
    models = {}
    for agent, agent_df in train_df.groupby("agent_name"):
        models[agent] = fit_agent_state_model(agent_df.reset_index(drop=True), random_state=random_state)
    return models


def assign_states(df: pd.DataFrame, models: dict[str, AgentStateModel]) -> pd.DataFrame:
    out = df.copy()
    out["state"] = -1
    for agent, agent_df in out.groupby("agent_name"):
        model = models.get(agent)
        if model is None:
            states = np.zeros(len(agent_df), dtype=int)
        else:
            states = model.predict(agent_df)
        out.loc[agent_df.index, "state"] = states

    prev_values = {}
    prev_states = []
    for _, row in out.sort_values(["log_id", "step_idx"]).iterrows():
        agent = row["agent_name"]
        prev_states.append(prev_values.get((row["log_id"], agent), -1))
        prev_values[(row["log_id"], agent)] = int(row["state"])
    out = out.sort_values(["log_id", "step_idx"]).copy()
    out["prev_agent_state"] = prev_states
    return out.reset_index(drop=True)


def build_risk_tables(train_df: pd.DataFrame) -> RiskTables:
    risk = RiskTables()
    usable = train_df[train_df["step_idx"] <= train_df["mistake_step"]]
    for _, row in usable.iterrows():
        risk.add(row)
    return risk


def score_rows(df: pd.DataFrame, risk: RiskTables) -> pd.DataFrame:
    out = df.copy()
    profile_scores = []
    risk_lifts = []
    risk_supports = []
    risk_sources = []
    combined_scores = []
    for _, row in out.iterrows():
        ps = profile_score(row)
        lift, support, source = risk.lookup(row)
        profile_scores.append(ps)
        risk_lifts.append(lift)
        risk_supports.append(support)
        risk_sources.append(source)
        combined_scores.append(ps + 2.0 * math.log(max(lift, 1e-6)))
    out["profile_score"] = profile_scores
    out["risk_lift"] = risk_lifts
    out["risk_support"] = risk_supports
    out["risk_source"] = risk_sources
    out["combined_score"] = combined_scores
    return out


def select_prediction(log_df: pd.DataFrame, mode: str) -> pd.Series:
    ordered = log_df.sort_values("step_idx")
    if mode == "profile_earliest4":
        flagged = ordered[ordered["profile_score"] >= 4.0]
        if not flagged.empty:
            return flagged.iloc[0]
        mode = "profile_argmax"
    if mode == "combined_earliest4":
        flagged = ordered[ordered["combined_score"] >= 4.0]
        if not flagged.empty:
            return flagged.iloc[0]
        mode = "combined_argmax"

    score_col = {
        "profile_argmax": "profile_score",
        "risk_argmax": "risk_lift",
        "combined_argmax": "combined_score",
    }[mode]
    best_value = ordered[score_col].max()
    return ordered[ordered[score_col] == best_value].iloc[0]


def evaluate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(predictions)
    result: dict[str, Any] = {"n": n}
    result["agent_accuracy"] = sum(p["predicted_agent"] == p["true_agent"] for p in predictions) / n
    result["step_accuracy"] = sum(p["predicted_step"] == p["true_step"] for p in predictions) / n
    distances = [abs(p["predicted_step"] - p["true_step"]) for p in predictions]
    result["avg_distance"] = float(np.mean(distances))
    for tol in range(1, 6):
        result[f"step_accuracy_tol_{tol}"] = sum(d <= tol for d in distances) / n
    return result


def sorted_log_ids(df: pd.DataFrame) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        try:
            return int(value), value
        except ValueError:
            return 10**9, value

    return sorted(df["log_id"].astype(str).unique(), key=key)


def run_cv_evaluation(
    steps_path: Path,
    profiles_path: Path,
    output_dir: Path,
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict[str, Any]:
    df = load_joined_frame(steps_path, profiles_path)
    logs = sorted_log_ids(df)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    modes = [
        "profile_argmax",
        "profile_earliest4",
        "risk_argmax",
        "combined_argmax",
        "combined_earliest4",
    ]
    predictions_by_mode = {mode: [] for mode in modes}
    fold_summaries = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(logs), start=1):
        train_logs = {logs[i] for i in train_idx}
        test_logs = {logs[i] for i in test_idx}
        train_df = df[df["log_id"].isin(train_logs)].reset_index(drop=True)
        test_df = df[df["log_id"].isin(test_logs)].reset_index(drop=True)

        models = fit_state_models(train_df, random_state=random_state + fold_idx)
        train_scored = assign_states(train_df, models)
        test_scored = assign_states(test_df, models)
        risk = build_risk_tables(train_scored)
        test_scored = score_rows(test_scored, risk)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_logs": sorted(train_logs),
                "test_logs": sorted(test_logs),
                "base_error_rate": risk.base_prob,
                "state_models": {
                    agent: {
                        "selected_k": model.selected_k,
                        "reason": model.reason,
                        "num_samples": model.num_samples,
                    }
                    for agent, model in models.items()
                },
            }
        )

        for log_id, log_df in test_scored.groupby("log_id"):
            truth = log_df.iloc[0]
            for mode in modes:
                pred = select_prediction(log_df, mode)
                predictions_by_mode[mode].append(
                    {
                        "fold": fold_idx,
                        "log_id": str(log_id),
                        "true_agent": truth["mistake_agent"],
                        "true_step": int(truth["mistake_step"]),
                        "predicted_agent": pred["agent_name"],
                        "predicted_step": int(pred["step_idx"]),
                        "score": float(
                            pred[
                                {
                                    "profile_argmax": "profile_score",
                                    "profile_earliest4": "profile_score",
                                    "risk_argmax": "risk_lift",
                                    "combined_argmax": "combined_score",
                                    "combined_earliest4": "combined_score",
                                }[mode]
                            ]
                        ),
                        "profile_score": float(pred["profile_score"]),
                        "risk_lift": float(pred["risk_lift"]),
                        "risk_source": pred["risk_source"],
                        "risk_support": int(pred["risk_support"]),
                    }
                )

    metrics = {
        mode: evaluate_predictions(preds)
        for mode, preds in predictions_by_mode.items()
    }
    result = {
        "steps_path": str(steps_path),
        "profiles_path": str(profiles_path),
        "n_splits": n_splits,
        "random_state": random_state,
        "metrics": metrics,
        "folds": fold_summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for mode, preds in predictions_by_mode.items():
        path = output_dir / f"predictions_{mode}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in preds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result
