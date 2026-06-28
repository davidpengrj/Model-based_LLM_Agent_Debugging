from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import KFold

from .data import read_jsonl
from .deepseek import DeepSeekClient, DeepSeekError, extract_json_object
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
from .profiling import short_text
from .sequence_evaluation import (
    choose_threshold,
    fit_hazard_model,
    score_hazard_rows,
    select_earliest_threshold,
)


PROMPT_VERSION = "topk-causal-rerank-v1"


def cache_key(model: str, log_id: str, candidate_steps: list[int], cap: int) -> str:
    payload = {
        "version": PROMPT_VERSION,
        "model": model,
        "log_id": str(log_id),
        "candidate_steps": candidate_steps,
        "cap": cap,
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for row in read_jsonl(path):
        rows[row["cache_key"]] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_priority_items(g: pd.DataFrame, thresholds: dict[str, float]) -> list[tuple[float, int, str]]:
    g = g.sort_values("step_idx")
    available = set(map(int, g["step_idx"]))
    items: list[tuple[float, int, str]] = []

    def add(prio: float, step: int, reason: str, distance: int = 0) -> None:
        if step in available:
            items.append((prio + abs(distance) * 0.35, step, reason))

    reps: list[tuple[int, float, str]] = []
    for mode, prio in [
        ("combined_earliest4", 0.00),
        ("combined_argmax", 0.20),
        ("risk_argmax", 0.30),
        ("profile_earliest4", 0.40),
        ("profile_argmax", 0.60),
    ]:
        reps.append((int(select_prediction(g, mode)["step_idx"]), prio, mode))
    reps.append(
        (
            int(select_earliest_threshold(g, "combined_score", thresholds["combined_tuned_step"])["step_idx"]),
            0.10,
            "combined_tuned_step",
        )
    )
    reps.append(
        (
            int(select_earliest_threshold(g, "combined_score", thresholds["combined_tuned_tol5"])["step_idx"]),
            0.15,
            "combined_tuned_tol5",
        )
    )
    reps.append((int(g.loc[g["hazard_score"].idxmax()]["step_idx"]), 0.50, "hazard_argmax"))

    for step, prio, reason in reps:
        for distance in range(-5, 6):
            add(prio, step + distance, reason, distance)

    for col, k, prio in [
        ("combined_score", 4, 1.0),
        ("risk_lift", 4, 1.2),
        ("profile_score", 3, 1.4),
        ("hazard_score", 3, 1.6),
    ]:
        for rank, (_, row) in enumerate(g.sort_values(col, ascending=False).head(k).iterrows()):
            step = int(row["step_idx"])
            for distance in range(-3, 4):
                add(prio + rank * 0.1, step + distance, col, distance)

    for rank, step in enumerate(sorted(available)[:10]):
        add(0.8 + rank * 0.05, step, "early")
    return items


def select_candidates(g: pd.DataFrame, thresholds: dict[str, float], cap: int) -> pd.DataFrame:
    best: dict[int, tuple[float, str]] = {}
    for prio, step, reason in candidate_priority_items(g, thresholds):
        if step not in best or prio < best[step][0]:
            best[step] = (prio, reason)
    selected_steps = [
        step for step, _ in sorted(best.items(), key=lambda item: (item[1][0], item[0]))[:cap]
    ]
    out = g[g["step_idx"].astype(int).isin(selected_steps)].copy()
    out["candidate_reason"] = out["step_idx"].astype(int).map(lambda step: best[int(step)][1])
    out["candidate_priority"] = out["step_idx"].astype(int).map(lambda step: best[int(step)][0])
    return out.sort_values("step_idx").reset_index(drop=True)


def build_rerank_messages(question: str, candidates: pd.DataFrame) -> list[dict[str, str]]:
    lines = []
    for idx, row in enumerate(candidates.to_dict("records"), start=1):
        lines.append(
            "\n".join(
                [
                    f"candidate_id: {idx}",
                    f"step_idx: {row['step_idx']}",
                    f"agent: {row['agent_name']}",
                    f"candidate_reason: {row.get('candidate_reason', '')}",
                    f"scores: profile={float(row['profile_score']):.2f}, risk={float(row['risk_lift']):.2f}, "
                    f"combined={float(row['combined_score']):.2f}, hazard={float(row.get('hazard_score', 0.0)):.2f}",
                    f"signals: action={row['action_type']}, progress={row['progress_state']}, "
                    f"evidence={row['evidence_quality']}, unsupported={int(bool(row['unsupported_claim']))}, "
                    f"tool_error={int(bool(row['tool_error']))}, reasoning={int(bool(row['reasoning_error_signal']))}",
                    f"message: {short_text(str(row['content']), 180)}",
                ]
            )
        )
    system = (
        "You are a failure-attribution judge for a failed LLM multi-agent trajectory. "
        "Select the candidate step that is the earliest causal decisive error, not merely a later symptom. "
        "Return only valid JSON."
    )
    user = f"""
Task question:
{short_text(question, 1200)}

Candidate steps are listed in chronological order. Choose the candidate that most likely first introduced, accepted,
or committed the decisive error that caused the final task failure. Prefer an earlier causal source over a later visible
wrong final answer. Do not use hidden labels.

Candidates:
{chr(10).join(lines)}

Return exactly one JSON object:
- selected_candidate_id: integer
- responsible_agent: agent name from the selected candidate
- confidence: integer 0..3
- rationale: one short sentence
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_rerank(raw: dict[str, Any], n_candidates: int) -> dict[str, Any]:
    try:
        selected = int(raw.get("selected_candidate_id", 1))
    except Exception:
        selected = 1
    selected = max(1, min(n_candidates, selected))
    try:
        confidence = int(raw.get("confidence", 0))
    except Exception:
        confidence = 0
    confidence = max(0, min(3, confidence))
    return {
        "selected_candidate_id": selected,
        "responsible_agent": str(raw.get("responsible_agent", "")),
        "confidence": confidence,
        "rationale": short_text(str(raw.get("rationale", "")), 500),
    }


def rerank_one(
    client: DeepSeekClient,
    question: str,
    candidates: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    try:
        response = client.chat(
            build_rerank_messages(question, candidates),
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content") or ""
        result = normalize_rerank(extract_json_object(content), len(candidates))
        return "ok", {"result": result, "usage": response.get("usage", {})}
    except (DeepSeekError, ValueError, KeyError) as exc:
        return "error", {"error": str(exc)[:1500]}


def run_rerank_cv(
    steps_path: Path,
    profiles_path: Path,
    output_dir: Path,
    client: DeepSeekClient,
    *,
    cap: int = 32,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict[str, Any]:
    df = load_joined_frame(steps_path, profiles_path)
    logs = sorted_log_ids(df)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cache_path = output_dir / f"rerank_{client.model}_cap{cap}.jsonl"
    error_path = output_dir / f"rerank_{client.model}_cap{cap}.errors"
    cache = load_cache(cache_path)
    predictions = []
    oracle_predictions = []
    fold_summaries = []
    completed = 0
    errors = 0

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
        thresholds = {
            "combined_tuned_step": choose_threshold(train_scored, "combined_score", "step")["threshold"],
            "combined_tuned_tol5": choose_threshold(train_scored, "combined_score", "tol5")["threshold"],
        }

        fold_summaries.append({"fold": fold_idx, "thresholds": thresholds, "test_logs": sorted(test_logs)})

        for log_id, log_df in test_scored.groupby("log_id"):
            log_df = log_df.sort_values("step_idx")
            truth = log_df.iloc[0]
            candidates = select_candidates(log_df, thresholds, cap)
            candidate_steps = [int(step) for step in candidates["step_idx"]]
            key = cache_key(client.model, str(log_id), candidate_steps, cap)
            if key in cache:
                row = cache[key]
            else:
                status, payload = rerank_one(client, str(truth["question"]), candidates)
                row = {
                    "cache_key": key,
                    "prompt_version": PROMPT_VERSION,
                    "model": client.model,
                    "fold": fold_idx,
                    "log_id": str(log_id),
                    "candidate_steps": candidate_steps,
                    "status": status,
                    **payload,
                }
                append_jsonl(cache_path if status == "ok" else error_path, row)
                if status == "ok":
                    cache[key] = row
                else:
                    errors += 1

            if row.get("status") == "ok":
                selected_id = int(row["result"]["selected_candidate_id"])
                pred = candidates.iloc[selected_id - 1]
            else:
                pred = candidates.iloc[0]
            predictions.append(
                {
                    "fold": fold_idx,
                    "log_id": str(log_id),
                    "true_agent": truth["mistake_agent"],
                    "true_step": int(truth["mistake_step"]),
                    "predicted_agent": pred["agent_name"],
                    "predicted_step": int(pred["step_idx"]),
                    "candidate_steps": candidate_steps,
                    "candidate_contains_true": int(truth["mistake_step"]) in candidate_steps,
                    "candidate_min_abs_distance": min(abs(int(step) - int(truth["mistake_step"])) for step in candidate_steps),
                    "confidence": int(row.get("result", {}).get("confidence", 0)),
                    "rationale": row.get("result", {}).get("rationale", ""),
                }
            )

            if int(truth["mistake_step"]) in candidate_steps:
                oracle_step = int(truth["mistake_step"])
            else:
                oracle_step = min(candidate_steps, key=lambda step: abs(step - int(truth["mistake_step"])))
            oracle_row = candidates[candidates["step_idx"].astype(int) == oracle_step].iloc[0]
            oracle_predictions.append(
                {
                    "true_agent": truth["mistake_agent"],
                    "true_step": int(truth["mistake_step"]),
                    "predicted_agent": oracle_row["agent_name"],
                    "predicted_step": int(oracle_row["step_idx"]),
                }
            )
            completed += 1
            if completed % 5 == 0:
                print(f"rerank_completed={completed} errors={errors}", flush=True)

    metrics = {
        "rerank_topk": evaluate_predictions(predictions),
        "candidate_oracle": evaluate_predictions(oracle_predictions),
    }
    result = {
        "steps_path": str(steps_path),
        "profiles_path": str(profiles_path),
        "cache_path": str(cache_path),
        "cap": cap,
        "n_splits": n_splits,
        "random_state": random_state,
        "metrics": metrics,
        "folds": fold_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"metrics_cap{cap}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / f"predictions_rerank_topk_cap{cap}.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result
