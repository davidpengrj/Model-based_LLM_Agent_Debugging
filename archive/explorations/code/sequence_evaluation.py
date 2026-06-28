from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .evaluation import (
    build_risk_tables,
    evaluate_predictions,
    fit_state_models,
    load_joined_frame,
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

TUNED_MODES = [
    "combined_tuned_step",
    "combined_tuned_tol5",
    "risk_tuned_step",
    "risk_tuned_tol5",
    "hazard_argmax",
    "hazard_tuned_step",
    "hazard_tuned_tol5",
]


def prediction_record(fold_idx: int, log_id: str, truth: pd.Series, pred: pd.Series, score_col: str) -> dict[str, Any]:
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
        "hazard_score": float(pred.get("hazard_score", 0.0)),
        "risk_source": pred["risk_source"],
        "risk_support": int(pred["risk_support"]),
    }


def select_earliest_threshold(log_df: pd.DataFrame, score_col: str, threshold: float, fallback: str = "argmax") -> pd.Series:
    ordered = log_df.sort_values("step_idx")
    flagged = ordered[ordered[score_col] >= threshold]
    if not flagged.empty:
        return flagged.iloc[0]
    if fallback == "last":
        return ordered.iloc[-1]
    best_value = ordered[score_col].max()
    return ordered[ordered[score_col] == best_value].iloc[0]


def evaluate_threshold(df: pd.DataFrame, score_col: str, threshold: float) -> dict[str, Any]:
    preds = []
    for log_id, log_df in df.groupby("log_id"):
        truth = log_df.iloc[0]
        pred = select_earliest_threshold(log_df, score_col, threshold)
        preds.append(
            {
                "true_agent": truth["mistake_agent"],
                "true_step": int(truth["mistake_step"]),
                "predicted_agent": pred["agent_name"],
                "predicted_step": int(pred["step_idx"]),
            }
        )
    return evaluate_predictions(preds)


def choose_threshold(df: pd.DataFrame, score_col: str, objective: str) -> dict[str, Any]:
    values = np.asarray(df[score_col].astype(float))
    if len(values) == 0:
        return {"threshold": 0.0, "metrics": {}}
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    grid = sorted(set(np.linspace(lo, hi, 80).tolist() + np.quantile(values, np.linspace(0.02, 0.98, 49)).tolist()))

    best = None
    for threshold in grid:
        metrics = evaluate_threshold(df, score_col, float(threshold))
        if objective == "step":
            key = (
                metrics["step_accuracy"],
                metrics["step_accuracy_tol_5"],
                metrics["agent_accuracy"],
                -metrics["avg_distance"],
                -threshold,
            )
        elif objective == "tol5":
            key = (
                metrics["step_accuracy_tol_5"],
                metrics["step_accuracy"],
                metrics["agent_accuracy"],
                -metrics["avg_distance"],
                -threshold,
            )
        else:
            raise ValueError(f"Unknown threshold objective: {objective}")
        if best is None or key > best["key"]:
            best = {"threshold": float(threshold), "metrics": metrics, "key": key}
    assert best is not None
    best.pop("key")
    return best


def hazard_feature_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.to_dict("records"):
        record: dict[str, Any] = {}
        profile = float(row["profile_score"])
        risk = float(row["risk_lift"])
        combined = float(row["combined_score"])
        record["profile_score"] = profile / 12.0
        record["risk_log"] = max(-3.0, min(3.0, math.log(max(risk, 1e-6)))) / 3.0
        record["combined_score"] = max(-2.5, min(2.5, combined / 8.0))
        record["risk_support_log"] = min(1.0, math.log1p(float(row["risk_support"])) / 6.0)
        denom = max(1, int(row["num_steps"]) - 1)
        record["normalized_step"] = int(row["step_idx"]) / denom
        record["steps_remaining"] = (denom - int(row["step_idx"])) / denom
        record["message_len_log"] = min(1.0, math.log1p(len(str(row["content"]))) / 9.0)
        record["task_alignment"] = int(row["task_alignment"]) / 3.0
        record["evidence_quality"] = int(row["evidence_quality"]) / 3.0
        record["low_task_alignment"] = (3 - int(row["task_alignment"])) / 3.0
        record["low_evidence_quality"] = (3 - int(row["evidence_quality"])) / 3.0
        record["unsupported_claim"] = int(bool(row["unsupported_claim"]))
        record["tool_error"] = int(bool(row["tool_error"]))
        record["reasoning_error_signal"] = int(bool(row["reasoning_error_signal"]))
        record["new_claim"] = int(bool(row["new_claim"]))
        record["tool_use"] = int(bool(row["tool_use"]))
        record["agent_name"] = str(row["agent_name"])
        record["action_type"] = str(row["action_type"])
        record["progress_state"] = str(row["progress_state"])
        record["risk_source"] = str(row["risk_source"])
        record["state"] = f"{row['agent_name']}:{int(row['state'])}"
        record["prev_state"] = f"{row['agent_name']}:{int(row['prev_agent_state'])}"
        return_record = record
        records.append(return_record)
    return records


class HazardModel:
    def __init__(self, vectorizer: DictVectorizer, classifier: LogisticRegression | None, top_weights: list[dict[str, Any]]) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.top_weights = top_weights

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = self.vectorizer.transform(hazard_feature_records(df))
        if self.classifier is None:
            return np.zeros(len(df), dtype=float)
        return self.classifier.predict_proba(x)[:, 1]


def fit_hazard_model(train_df: pd.DataFrame) -> HazardModel:
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(hazard_feature_records(train_df))
    y = train_df["is_decisive_error"].astype(int).to_numpy()
    if len(set(y.tolist())) < 2:
        return HazardModel(vectorizer, None, [])
    classifier = LogisticRegression(
        C=0.75,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )
    classifier.fit(x, y)
    names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    top = sorted(
        ({"feature": str(name), "weight": float(weight)} for name, weight in zip(names, weights)),
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )[:25]
    return HazardModel(vectorizer, classifier, top)


def score_hazard_rows(df: pd.DataFrame, model: HazardModel) -> pd.DataFrame:
    out = df.copy()
    out["hazard_score"] = model.score(out)
    return out


def run_sequence_cv_evaluation(
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
    modes = BASELINE_MODES + TUNED_MODES
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
        risk = build_risk_tables(train_state)
        train_scored = score_rows(train_state, risk)
        test_scored = score_rows(test_state, risk)

        hazard = fit_hazard_model(train_scored)
        train_scored = score_hazard_rows(train_scored, hazard)
        test_scored = score_hazard_rows(test_scored, hazard)

        threshold_specs = {
            "combined_tuned_step": choose_threshold(train_scored, "combined_score", "step"),
            "combined_tuned_tol5": choose_threshold(train_scored, "combined_score", "tol5"),
            "risk_tuned_step": choose_threshold(train_scored, "risk_lift", "step"),
            "risk_tuned_tol5": choose_threshold(train_scored, "risk_lift", "tol5"),
            "hazard_tuned_step": choose_threshold(train_scored, "hazard_score", "step"),
            "hazard_tuned_tol5": choose_threshold(train_scored, "hazard_score", "tol5"),
        }

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_logs": sorted(train_logs),
                "test_logs": sorted(test_logs),
                "base_error_rate": risk.base_prob,
                "thresholds": threshold_specs,
                "hazard_top_weights": hazard.top_weights,
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
                predictions_by_mode[mode].append(prediction_record(fold_idx, str(log_id), truth, pred, score_col))

            pred = select_prediction(log_df, "combined_argmax")
            predictions_by_mode["hazard_argmax"].append(
                prediction_record(
                    fold_idx,
                    str(log_id),
                    truth,
                    log_df.loc[log_df["hazard_score"].idxmax()],
                    "hazard_score",
                )
            )

            for mode, score_col in {
                "combined_tuned_step": "combined_score",
                "combined_tuned_tol5": "combined_score",
                "risk_tuned_step": "risk_lift",
                "risk_tuned_tol5": "risk_lift",
                "hazard_tuned_step": "hazard_score",
                "hazard_tuned_tol5": "hazard_score",
            }.items():
                threshold = float(threshold_specs[mode]["threshold"])
                pred = select_earliest_threshold(log_df, score_col, threshold)
                predictions_by_mode[mode].append(prediction_record(fold_idx, str(log_id), truth, pred, score_col))

    metrics = {mode: evaluate_predictions(preds) for mode, preds in predictions_by_mode.items()}
    result = {
        "steps_path": str(steps_path),
        "profiles_path": str(profiles_path),
        "n_splits": n_splits,
        "random_state": random_state,
        "metrics": metrics,
        "folds": fold_summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, preds in predictions_by_mode.items():
        path = output_dir / f"predictions_{mode}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in preds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result
