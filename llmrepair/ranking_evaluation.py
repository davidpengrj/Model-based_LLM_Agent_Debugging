from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .evaluation import (
    assign_states,
    build_risk_tables,
    evaluate_predictions,
    fit_state_models,
    load_joined_frame,
    score_rows,
    select_prediction,
    sorted_log_ids,
)


BASELINE_MODES = [
    "profile_argmax",
    "profile_earliest4",
    "risk_argmax",
    "combined_argmax",
    "combined_earliest4",
]

RANKING_MODES = [
    "combined_decay_step",
    "combined_decay_tol5",
    "risk_decay_step",
    "who_then_combined_decay_step",
    "who_then_combined_earliest_step",
    "ranker_argmax",
    "ranker_decay_step",
    "ranker_decay_tol5",
    "ranker_earliest_step",
    "who_then_ranker_decay_step",
    "who_then_ranker_earliest_step",
]

DECAY_GRID = [
    0.0,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.65,
    0.80,
    1.00,
]


def prediction_record(
    fold_idx: int,
    log_id: str,
    truth: pd.Series,
    pred: pd.Series,
    score_col: str,
    *,
    mode: str,
    predicted_by_agent: str | None = None,
) -> dict[str, Any]:
    return {
        "fold": fold_idx,
        "log_id": str(log_id),
        "mode": mode,
        "true_agent": truth["mistake_agent"],
        "true_step": int(truth["mistake_step"]),
        "predicted_agent": pred["agent_name"],
        "predicted_step": int(pred["step_idx"]),
        "predicted_by_agent": predicted_by_agent,
        "score": float(pred[score_col]),
        "profile_score": float(pred["profile_score"]),
        "risk_lift": float(pred["risk_lift"]),
        "combined_score": float(pred["combined_score"]),
        "ranker_score": float(pred.get("ranker_score", 0.0)),
        "risk_source": pred["risk_source"],
        "risk_support": int(pred["risk_support"]),
    }


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    denom = (out["num_steps"].astype(float) - 1.0).clip(lower=1.0)
    step = out["step_idx"].astype(float)
    out["normalized_step"] = step / denom
    out["steps_remaining_norm"] = (denom - step).clip(lower=0.0) / denom
    out["early_exp8"] = np.exp(-step / 8.0)
    out["early_inv_step"] = 1.0 / np.sqrt(step.clip(lower=1.0))
    out["risk_log_score"] = 2.0 * np.log(out["risk_lift"].astype(float).clip(lower=1e-6))
    return out


def objective_key(metrics: dict[str, Any], objective: str, penalty: float | None = None) -> tuple[Any, ...]:
    if objective == "step":
        return (
            metrics["step_accuracy"],
            metrics["step_accuracy_tol_5"],
            metrics["agent_accuracy"],
            -metrics["avg_distance"],
            0.0 if penalty is None else -penalty,
        )
    if objective == "tol5":
        return (
            metrics["step_accuracy_tol_5"],
            metrics["step_accuracy"],
            metrics["agent_accuracy"],
            -metrics["avg_distance"],
            0.0 if penalty is None else -penalty,
        )
    raise ValueError(f"Unknown objective: {objective}")


def select_by_score(log_df: pd.DataFrame, score: pd.Series | np.ndarray) -> pd.Series:
    ordered = log_df.sort_values("step_idx")
    score_series = pd.Series(score, index=log_df.index).loc[ordered.index]
    best_value = score_series.max()
    best_indexes = score_series[score_series == best_value].index
    return ordered.loc[best_indexes].iloc[0]


def select_decayed(
    log_df: pd.DataFrame,
    score_col: str,
    penalty: float,
    *,
    restrict_to_agent: str | None = None,
) -> pd.Series:
    pool = log_df
    if restrict_to_agent is not None:
        agent_pool = log_df[log_df["agent_name"] == restrict_to_agent]
        if not agent_pool.empty:
            pool = agent_pool
    decayed = pool[score_col].astype(float) - penalty * pool["step_idx"].astype(float)
    return select_by_score(pool, decayed)


def select_earliest_threshold(
    log_df: pd.DataFrame,
    score_col: str,
    threshold: float,
    *,
    restrict_to_agent: str | None = None,
) -> pd.Series:
    pool = log_df
    if restrict_to_agent is not None:
        agent_pool = log_df[log_df["agent_name"] == restrict_to_agent]
        if not agent_pool.empty:
            pool = agent_pool
    ordered = pool.sort_values("step_idx")
    flagged = ordered[ordered[score_col].astype(float) >= threshold]
    if not flagged.empty:
        return flagged.iloc[0]
    return select_by_score(pool, pool[score_col].astype(float))


def predict_agent(log_df: pd.DataFrame, score_col: str = "combined_score") -> str:
    pred = select_by_score(log_df, log_df[score_col].astype(float))
    return str(pred["agent_name"])


Selector = Callable[[pd.DataFrame], pd.Series]


def evaluate_selector(df: pd.DataFrame, selector: Selector) -> dict[str, Any]:
    preds = []
    for _, log_df in df.groupby("log_id"):
        truth = log_df.iloc[0]
        pred = selector(log_df)
        preds.append(
            {
                "true_agent": truth["mistake_agent"],
                "true_step": int(truth["mistake_step"]),
                "predicted_agent": pred["agent_name"],
                "predicted_step": int(pred["step_idx"]),
            }
        )
    return evaluate_predictions(preds)


def choose_decay_penalty(
    df: pd.DataFrame,
    score_col: str,
    objective: str,
    *,
    agent_restricted: bool = False,
    agent_score_col: str = "combined_score",
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for penalty in DECAY_GRID:
        def selector(log_df: pd.DataFrame, penalty: float = penalty) -> pd.Series:
            agent = predict_agent(log_df, agent_score_col) if agent_restricted else None
            return select_decayed(log_df, score_col, penalty, restrict_to_agent=agent)

        metrics = evaluate_selector(df, selector)
        key = objective_key(metrics, objective, penalty)
        if best is None or key > best["key"]:
            best = {"penalty": float(penalty), "metrics": metrics, "key": key}
    assert best is not None
    best.pop("key")
    return best


def choose_threshold(
    df: pd.DataFrame,
    score_col: str,
    objective: str,
    *,
    agent_restricted: bool = False,
    agent_score_col: str = "combined_score",
) -> dict[str, Any]:
    values = np.asarray(df[score_col].astype(float))
    if len(values) == 0:
        return {"threshold": 0.0, "metrics": {}}
    grid = sorted(
        set(
            np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), 80).tolist()
            + np.quantile(values, np.linspace(0.02, 0.98, 49)).tolist()
        )
    )

    best: dict[str, Any] | None = None
    for threshold in grid:
        def selector(log_df: pd.DataFrame, threshold: float = float(threshold)) -> pd.Series:
            agent = predict_agent(log_df, agent_score_col) if agent_restricted else None
            return select_earliest_threshold(log_df, score_col, threshold, restrict_to_agent=agent)

        metrics = evaluate_selector(df, selector)
        key = objective_key(metrics, objective, None) + (-float(threshold),)
        if best is None or key > best["key"]:
            best = {"threshold": float(threshold), "metrics": metrics, "key": key}
    assert best is not None
    best.pop("key")
    return best


def ranking_feature_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.to_dict("records"):
        step_idx = max(1, int(row["step_idx"]))
        num_steps = max(2, int(row["num_steps"]))
        normalized_step = step_idx / max(1, num_steps - 1)
        risk_lift = max(float(row["risk_lift"]), 1e-6)
        risk_log = max(-3.0, min(3.0, math.log(risk_lift))) / 3.0
        profile = float(row["profile_score"])
        combined = float(row["combined_score"])
        early_exp = math.exp(-step_idx / 8.0)
        early_inv = 1.0 / math.sqrt(step_idx)
        low_task = (3 - int(row["task_alignment"])) / 3.0
        low_evidence = (3 - int(row["evidence_quality"])) / 3.0

        record: dict[str, Any] = {
            "profile_score": profile / 12.0,
            "risk_log": risk_log,
            "risk_support_log": min(1.0, math.log1p(float(row["risk_support"])) / 6.0),
            "combined_score": max(-2.5, min(2.5, combined / 8.0)),
            "normalized_step": normalized_step,
            "step_idx_scaled": min(1.0, step_idx / 40.0),
            "steps_remaining_norm": max(0.0, (num_steps - 1 - step_idx) / max(1, num_steps - 1)),
            "early_exp8": early_exp,
            "early_inv_step": early_inv,
            "is_first_8_steps": int(step_idx <= 8),
            "is_first_12_steps": int(step_idx <= 12),
            "is_after_midpoint": int(normalized_step >= 0.5),
            "message_len_log": min(1.0, math.log1p(len(str(row["content"]))) / 9.0),
            "low_task_alignment": low_task,
            "low_evidence_quality": low_evidence,
            "unsupported_claim": int(bool(row["unsupported_claim"])),
            "tool_error": int(bool(row["tool_error"])),
            "reasoning_error_signal": int(bool(row["reasoning_error_signal"])),
            "new_claim": int(bool(row["new_claim"])),
            "tool_use": int(bool(row["tool_use"])),
            "profile_x_early": profile / 12.0 * early_exp,
            "risk_x_early": risk_log * early_exp,
            "combined_x_early": max(-2.5, min(2.5, combined / 8.0)) * early_exp,
            "low_task_x_early": low_task * early_exp,
            "low_evidence_x_early": low_evidence * early_exp,
            "agent_name": str(row["agent_name"]),
            "action_type": str(row["action_type"]),
            "progress_state": str(row["progress_state"]),
            "risk_source": str(row["risk_source"]),
            "state": f"{row['agent_name']}:{int(row['state'])}",
            "prev_state": f"{row['agent_name']}:{int(row['prev_agent_state'])}",
            "agent_early": f"{row['agent_name']}:early{int(step_idx <= 12)}",
            "agent_action": f"{row['agent_name']}:{row['action_type']}",
        }
        records.append(record)
    return records


@dataclass
class PairwiseRanker:
    vectorizer: DictVectorizer
    classifier: LogisticRegression | None
    top_weights: list[dict[str, Any]]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = self.vectorizer.transform(ranking_feature_records(df))
        if self.classifier is None:
            return np.zeros(len(df), dtype=float)
        return self.classifier.decision_function(x)


def fit_pairwise_ranker(train_df: pd.DataFrame) -> PairwiseRanker:
    train_df = train_df.reset_index(drop=True)
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(ranking_feature_records(train_df))
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
        return PairwiseRanker(vectorizer, None, [])

    classifier = LogisticRegression(
        C=0.5,
        fit_intercept=False,
        max_iter=1000,
        solver="liblinear",
    )
    classifier.fit(np.asarray(diffs), np.asarray(labels))
    names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    top_weights = sorted(
        (
            {"feature": str(name), "weight": float(weight)}
            for name, weight in zip(names, weights)
        ),
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )[:30]
    return PairwiseRanker(vectorizer, classifier, top_weights)


def score_ranker_rows(df: pd.DataFrame, ranker: PairwiseRanker) -> pd.DataFrame:
    out = df.copy()
    out["ranker_score"] = ranker.score(out)
    return out


def run_ranking_cv_evaluation(
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
    modes = BASELINE_MODES + RANKING_MODES
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
        train_scored = add_position_features(score_rows(train_state, risk))
        test_scored = add_position_features(score_rows(test_state, risk))

        ranker = fit_pairwise_ranker(train_scored)
        train_scored = score_ranker_rows(train_scored, ranker)
        test_scored = score_ranker_rows(test_scored, ranker)

        specs = {
            "combined_decay_step": choose_decay_penalty(train_scored, "combined_score", "step"),
            "combined_decay_tol5": choose_decay_penalty(train_scored, "combined_score", "tol5"),
            "risk_decay_step": choose_decay_penalty(train_scored, "risk_log_score", "step"),
            "who_then_combined_decay_step": choose_decay_penalty(
                train_scored,
                "combined_score",
                "step",
                agent_restricted=True,
            ),
            "who_then_combined_earliest_step": choose_threshold(
                train_scored,
                "combined_score",
                "step",
                agent_restricted=True,
            ),
            "ranker_decay_step": choose_decay_penalty(train_scored, "ranker_score", "step"),
            "ranker_decay_tol5": choose_decay_penalty(train_scored, "ranker_score", "tol5"),
            "ranker_earliest_step": choose_threshold(train_scored, "ranker_score", "step"),
            "who_then_ranker_decay_step": choose_decay_penalty(
                train_scored,
                "ranker_score",
                "step",
                agent_restricted=True,
            ),
            "who_then_ranker_earliest_step": choose_threshold(
                train_scored,
                "ranker_score",
                "step",
                agent_restricted=True,
            ),
        }

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_logs": sorted(train_logs),
                "test_logs": sorted(test_logs),
                "base_error_rate": risk.base_prob,
                "specs": specs,
                "ranker_top_weights": ranker.top_weights,
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
                    prediction_record(fold_idx, str(log_id), truth, pred, score_col, mode=mode)
                )

            combined_step_penalty = float(specs["combined_decay_step"]["penalty"])
            combined_tol5_penalty = float(specs["combined_decay_tol5"]["penalty"])
            risk_step_penalty = float(specs["risk_decay_step"]["penalty"])
            ranker_step_penalty = float(specs["ranker_decay_step"]["penalty"])
            ranker_tol5_penalty = float(specs["ranker_decay_tol5"]["penalty"])
            selected_agent = predict_agent(log_df, "combined_score")

            pred = select_decayed(log_df, "combined_score", combined_step_penalty)
            predictions_by_mode["combined_decay_step"].append(
                prediction_record(fold_idx, str(log_id), truth, pred, "combined_score", mode="combined_decay_step")
            )
            pred = select_decayed(log_df, "combined_score", combined_tol5_penalty)
            predictions_by_mode["combined_decay_tol5"].append(
                prediction_record(fold_idx, str(log_id), truth, pred, "combined_score", mode="combined_decay_tol5")
            )
            pred = select_decayed(log_df, "risk_log_score", risk_step_penalty)
            predictions_by_mode["risk_decay_step"].append(
                prediction_record(fold_idx, str(log_id), truth, pred, "risk_log_score", mode="risk_decay_step")
            )
            pred = select_decayed(
                log_df,
                "combined_score",
                float(specs["who_then_combined_decay_step"]["penalty"]),
                restrict_to_agent=selected_agent,
            )
            predictions_by_mode["who_then_combined_decay_step"].append(
                prediction_record(
                    fold_idx,
                    str(log_id),
                    truth,
                    pred,
                    "combined_score",
                    mode="who_then_combined_decay_step",
                    predicted_by_agent=selected_agent,
                )
            )
            pred = select_earliest_threshold(
                log_df,
                "combined_score",
                float(specs["who_then_combined_earliest_step"]["threshold"]),
                restrict_to_agent=selected_agent,
            )
            predictions_by_mode["who_then_combined_earliest_step"].append(
                prediction_record(
                    fold_idx,
                    str(log_id),
                    truth,
                    pred,
                    "combined_score",
                    mode="who_then_combined_earliest_step",
                    predicted_by_agent=selected_agent,
                )
            )
            pred = select_by_score(log_df, log_df["ranker_score"].astype(float))
            predictions_by_mode["ranker_argmax"].append(
                prediction_record(fold_idx, str(log_id), truth, pred, "ranker_score", mode="ranker_argmax")
            )
            pred = select_decayed(log_df, "ranker_score", ranker_step_penalty)
            predictions_by_mode["ranker_decay_step"].append(
                prediction_record(fold_idx, str(log_id), truth, pred, "ranker_score", mode="ranker_decay_step")
            )
            pred = select_decayed(log_df, "ranker_score", ranker_tol5_penalty)
            predictions_by_mode["ranker_decay_tol5"].append(
                prediction_record(fold_idx, str(log_id), truth, pred, "ranker_score", mode="ranker_decay_tol5")
            )
            pred = select_earliest_threshold(
                log_df,
                "ranker_score",
                float(specs["ranker_earliest_step"]["threshold"]),
            )
            predictions_by_mode["ranker_earliest_step"].append(
                prediction_record(fold_idx, str(log_id), truth, pred, "ranker_score", mode="ranker_earliest_step")
            )
            pred = select_decayed(
                log_df,
                "ranker_score",
                float(specs["who_then_ranker_decay_step"]["penalty"]),
                restrict_to_agent=selected_agent,
            )
            predictions_by_mode["who_then_ranker_decay_step"].append(
                prediction_record(
                    fold_idx,
                    str(log_id),
                    truth,
                    pred,
                    "ranker_score",
                    mode="who_then_ranker_decay_step",
                    predicted_by_agent=selected_agent,
                )
            )
            pred = select_earliest_threshold(
                log_df,
                "ranker_score",
                float(specs["who_then_ranker_earliest_step"]["threshold"]),
                restrict_to_agent=selected_agent,
            )
            predictions_by_mode["who_then_ranker_earliest_step"].append(
                prediction_record(
                    fold_idx,
                    str(log_id),
                    truth,
                    pred,
                    "ranker_score",
                    mode="who_then_ranker_earliest_step",
                    predicted_by_agent=selected_agent,
                )
            )

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
