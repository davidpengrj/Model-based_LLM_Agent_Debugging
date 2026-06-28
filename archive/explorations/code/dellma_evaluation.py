from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .data import read_jsonl
from .dellma import FAILURE_MODES, VERBAL_WEIGHTS, belief_probabilities, normalize_likelihood
from .evaluation import (
    build_risk_tables,
    evaluate_predictions,
    fit_state_models,
    load_joined_frame,
    profile_score,
    score_rows,
    select_prediction,
    sorted_log_ids,
)
from .evaluation import assign_states


BASELINE_MODES = [
    "profile_argmax",
    "profile_earliest4",
    "risk_argmax",
    "combined_argmax",
    "combined_earliest4",
]

DELLMA_MODES = [
    "dellma_eu_argmax",
    "dellma_bt_argmax",
    "dellma_hybrid_argmax",
]


def profile_to_mode_probs(profile: dict[str, Any]) -> dict[str, float]:
    probs = profile.get("mode_probabilities")
    if isinstance(probs, dict):
        out = {}
        for mode in FAILURE_MODES:
            try:
                out[mode] = max(0.0, float(probs.get(mode, 0.0)))
            except Exception:
                out[mode] = 0.0
        total = sum(out.values())
        if total > 0:
            return {mode: out[mode] / total for mode in FAILURE_MODES}

    raw_beliefs = profile.get("failure_mode_beliefs", {})
    if not isinstance(raw_beliefs, dict):
        raw_beliefs = {}
    beliefs = {
        mode: normalize_likelihood(raw_beliefs.get(mode, "very unlikely"))
        for mode in FAILURE_MODES
    }
    return belief_probabilities(beliefs)


def load_dellma_frame(
    steps_path: Path,
    profiles_path: Path,
    dellma_profiles_path: Path,
) -> pd.DataFrame:
    df = load_joined_frame(steps_path, profiles_path)
    dellma_profiles = {}
    for row in read_jsonl(dellma_profiles_path):
        dellma_profiles[(str(row["log_id"]), int(row["step_idx"]))] = row["profile"]

    rows = []
    for row in df.to_dict("records"):
        key = (str(row["log_id"]), int(row["step_idx"]))
        if key not in dellma_profiles:
            raise ValueError(f"Missing DeLLMa profile for {key}")
        profile = dellma_profiles[key]
        mode_probs = profile_to_mode_probs(profile)
        row["dellma_top_mode"] = str(profile.get("top_mode", ""))
        if row["dellma_top_mode"] not in FAILURE_MODES:
            row["dellma_top_mode"] = max(mode_probs, key=mode_probs.get)
        row["dellma_severity"] = int(profile.get("severity", 0))
        row["dellma_audit_summary"] = str(profile.get("audit_summary", ""))
        for mode in FAILURE_MODES:
            row[f"mode_prob__{mode}"] = float(mode_probs[mode])
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["log_id", "step_idx"]).reset_index(drop=True)


def row_mode_probs(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return {mode: float(row[f"mode_prob__{mode}"]) for mode in FAILURE_MODES}


class DellmaUtilityTables:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.tables: dict[str, dict[tuple[Any, ...], list[float]]] = {
            "agent_state_mode": defaultdict(lambda: [0.0, 0.0]),
            "agent_action_mode": defaultdict(lambda: [0.0, 0.0]),
            "agent_mode": defaultdict(lambda: [0.0, 0.0]),
            "mode": defaultdict(lambda: [0.0, 0.0]),
        }
        self.base_pos = 0
        self.base_total = 0

    def add(self, row: pd.Series) -> None:
        y = int(bool(row["is_decisive_error"]))
        self.base_pos += y
        self.base_total += 1
        for mode, weight in row_mode_probs(row).items():
            keys = {
                "agent_state_mode": (row["agent_name"], int(row["state"]), mode),
                "agent_action_mode": (row["agent_name"], row["action_type"], mode),
                "agent_mode": (row["agent_name"], mode),
                "mode": (mode,),
            }
            for name, key in keys.items():
                self.tables[name][key][0] += weight * y
                self.tables[name][key][1] += weight

    @property
    def base_prob(self) -> float:
        return (self.base_pos + self.alpha) / (self.base_total + 2 * self.alpha)

    def lookup_mode(self, row: pd.Series, mode: str) -> tuple[float, float, str]:
        keys = [
            ("agent_state_mode", (row["agent_name"], int(row["state"]), mode), 1.5),
            ("agent_action_mode", (row["agent_name"], row["action_type"], mode), 2.0),
            ("agent_mode", (row["agent_name"], mode), 3.0),
            ("mode", (mode,), 3.0),
        ]
        base = self.base_prob
        for name, key, min_support in keys:
            pos, total = self.tables[name].get(key, [0.0, 0.0])
            if total >= min_support:
                prob = (pos + self.alpha) / (total + 2 * self.alpha)
                return math.log(max(prob / base, 1e-6)), float(total), name
        return 0.0, 0.0, "base"

    def score(self, row: pd.Series) -> tuple[float, float, str]:
        score = 0.0
        best_support = 0.0
        best_source = "base"
        best_abs_contribution = -1.0
        for mode, prob in row_mode_probs(row).items():
            utility, support, source = self.lookup_mode(row, mode)
            contribution = prob * utility
            score += contribution
            if abs(contribution) > best_abs_contribution:
                best_abs_contribution = abs(contribution)
                best_support = support
                best_source = source
        return float(score), float(best_support), best_source


def build_dellma_utility_tables(train_df: pd.DataFrame) -> DellmaUtilityTables:
    tables = DellmaUtilityTables()
    for _, row in train_df.iterrows():
        tables.add(row)
    return tables


def dellma_feature_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.to_dict("records"):
        record: dict[str, Any] = {}
        mode_probs = row_mode_probs(row)
        for mode, prob in mode_probs.items():
            record[f"mode_prob={mode}"] = float(prob)
        record["non_benign_mass"] = 1.0 - float(mode_probs["benign_or_on_track"])
        record["severity"] = float(row["dellma_severity"]) / 3.0
        record["profile_score"] = profile_score(pd.Series(row)) / 12.0
        if "risk_lift" in row:
            record["risk_log"] = max(-3.0, min(3.0, math.log(max(float(row["risk_lift"]), 1e-6)))) / 3.0
        if "risk_support" in row:
            record["risk_support_log"] = min(1.0, math.log1p(float(row["risk_support"])) / 6.0)
        if "combined_score" in row:
            record["combined_score"] = max(-2.0, min(2.0, float(row["combined_score"]) / 8.0))
        if "risk_source" in row:
            record["risk_source"] = str(row["risk_source"])
        record["low_task_alignment"] = (3 - int(row["task_alignment"])) / 3.0
        record["low_evidence_quality"] = (3 - int(row["evidence_quality"])) / 3.0
        record["unsupported_claim"] = int(bool(row["unsupported_claim"]))
        record["tool_error"] = int(bool(row["tool_error"]))
        record["reasoning_error_signal"] = int(bool(row["reasoning_error_signal"]))
        denom = max(1, int(row["num_steps"]) - 1)
        record["normalized_step"] = int(row["step_idx"]) / denom
        record["message_len_log"] = min(1.0, math.log1p(len(str(row["content"]))) / 9.0)
        record["agent_name"] = str(row["agent_name"])
        record["action_type"] = str(row["action_type"])
        record["progress_state"] = str(row["progress_state"])
        record["top_mode"] = str(row["dellma_top_mode"])
        record["state"] = f"{row['agent_name']}:{int(row['state'])}"
        records.append(record)
    return records


@dataclass
class PairwiseUtilityModel:
    vectorizer: DictVectorizer
    classifier: LogisticRegression | None
    top_weights: list[dict[str, Any]]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = self.vectorizer.transform(dellma_feature_records(df))
        if self.classifier is None:
            return np.zeros(len(df), dtype=float)
        return self.classifier.decision_function(x)


def fit_pairwise_utility_model(train_df: pd.DataFrame) -> PairwiseUtilityModel:
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(dellma_feature_records(train_df))
    diffs = []
    labels = []
    for _, log_df in train_df.groupby("log_id"):
        positives = log_df[log_df["is_decisive_error"]]
        if positives.empty:
            continue
        pos_idx = int(positives.index[0])
        for other_idx in log_df.index:
            other_idx = int(other_idx)
            if other_idx == pos_idx:
                continue
            diff = x[pos_idx] - x[other_idx]
            diffs.append(diff)
            labels.append(1)
            diffs.append(-diff)
            labels.append(0)

    if not diffs or len(set(labels)) < 2:
        return PairwiseUtilityModel(vectorizer, None, [])

    classifier = LogisticRegression(
        C=0.5,
        fit_intercept=False,
        max_iter=1000,
        solver="liblinear",
    )
    classifier.fit(np.asarray(diffs), np.asarray(labels))
    names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    top = sorted(
        (
            {"feature": str(name), "weight": float(weight)}
            for name, weight in zip(names, weights)
        ),
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )[:25]
    return PairwiseUtilityModel(vectorizer, classifier, top)


def score_dellma_rows(
    df: pd.DataFrame,
    utility_tables: DellmaUtilityTables,
    pairwise_model: PairwiseUtilityModel,
) -> pd.DataFrame:
    out = df.copy()
    eu_scores = []
    eu_supports = []
    eu_sources = []
    for _, row in out.iterrows():
        score, support, source = utility_tables.score(row)
        eu_scores.append(score)
        eu_supports.append(support)
        eu_sources.append(source)
    out["dellma_eu_score"] = eu_scores
    out["dellma_eu_support"] = eu_supports
    out["dellma_eu_source"] = eu_sources
    out["dellma_bt_score"] = pairwise_model.score(out)
    out["dellma_hybrid_score"] = (
        out["combined_score"] + out["dellma_eu_score"] + out["dellma_bt_score"]
    )
    return out


def select_by_score(log_df: pd.DataFrame, score_col: str) -> pd.Series:
    ordered = log_df.sort_values("step_idx")
    best_value = ordered[score_col].max()
    return ordered[ordered[score_col] == best_value].iloc[0]


def prediction_record(
    fold_idx: int,
    log_id: str,
    truth: pd.Series,
    pred: pd.Series,
    score_col: str,
) -> dict[str, Any]:
    return {
        "fold": fold_idx,
        "log_id": str(log_id),
        "true_agent": truth["mistake_agent"],
        "true_step": int(truth["mistake_step"]),
        "predicted_agent": pred["agent_name"],
        "predicted_step": int(pred["step_idx"]),
        "score": float(pred[score_col]),
        "profile_score": float(pred["profile_score"]),
        "risk_lift": float(pred["risk_lift"]),
        "combined_score": float(pred["combined_score"]),
        "dellma_eu_score": float(pred["dellma_eu_score"]),
        "dellma_bt_score": float(pred["dellma_bt_score"]),
        "dellma_hybrid_score": float(pred["dellma_hybrid_score"]),
        "dellma_top_mode": pred["dellma_top_mode"],
        "dellma_eu_source": pred["dellma_eu_source"],
        "dellma_eu_support": float(pred["dellma_eu_support"]),
    }


def run_dellma_cv_evaluation(
    steps_path: Path,
    profiles_path: Path,
    dellma_profiles_path: Path,
    output_dir: Path,
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict[str, Any]:
    df = load_dellma_frame(steps_path, profiles_path, dellma_profiles_path)
    logs = sorted_log_ids(df)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    modes = BASELINE_MODES + DELLMA_MODES
    predictions_by_mode = {mode: [] for mode in modes}
    fold_summaries = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(logs), start=1):
        train_logs = {logs[i] for i in train_idx}
        test_logs = {logs[i] for i in test_idx}
        train_df = df[df["log_id"].isin(train_logs)].reset_index(drop=True)
        test_df = df[df["log_id"].isin(test_logs)].reset_index(drop=True)

        models = fit_state_models(train_df, random_state=random_state + fold_idx)
        train_state = assign_states(train_df, models)
        test_state = assign_states(test_df, models)

        baseline_risk = build_risk_tables(train_state)
        train_state = score_rows(train_state, baseline_risk)
        test_state = score_rows(test_state, baseline_risk)

        utility_tables = build_dellma_utility_tables(train_state)
        pairwise_model = fit_pairwise_utility_model(train_state)
        test_scored = score_dellma_rows(test_state, utility_tables, pairwise_model)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_logs": sorted(train_logs),
                "test_logs": sorted(test_logs),
                "baseline_base_error_rate": baseline_risk.base_prob,
                "dellma_base_error_rate": utility_tables.base_prob,
                "bt_top_weights": pairwise_model.top_weights,
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
            for mode in BASELINE_MODES:
                pred = select_prediction(log_df, mode)
                score_col = {
                    "profile_argmax": "profile_score",
                    "profile_earliest4": "profile_score",
                    "risk_argmax": "risk_lift",
                    "combined_argmax": "combined_score",
                    "combined_earliest4": "combined_score",
                }[mode]
                predictions_by_mode[mode].append(
                    prediction_record(fold_idx, str(log_id), truth, pred, score_col)
                )
            for mode, score_col in {
                "dellma_eu_argmax": "dellma_eu_score",
                "dellma_bt_argmax": "dellma_bt_score",
                "dellma_hybrid_argmax": "dellma_hybrid_score",
            }.items():
                pred = select_by_score(log_df, score_col)
                predictions_by_mode[mode].append(
                    prediction_record(fold_idx, str(log_id), truth, pred, score_col)
                )

    metrics = {
        mode: evaluate_predictions(preds)
        for mode, preds in predictions_by_mode.items()
    }
    result = {
        "steps_path": str(steps_path),
        "profiles_path": str(profiles_path),
        "dellma_profiles_path": str(dellma_profiles_path),
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
