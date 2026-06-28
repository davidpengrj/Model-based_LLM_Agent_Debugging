from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .deepseek import DeepSeekClient, DeepSeekError, extract_json_object
from .profiling import short_text


PROMPT_VERSION = "dellma-failure-mode-v1"

FAILURE_MODES = [
    "benign_or_on_track",
    "evidence_gap",
    "unsupported_claim",
    "reasoning_error",
    "tool_or_execution_error",
    "task_drift",
    "premature_or_overconfident_answer",
    "coordination_or_delegation_error",
    "state_tracking_or_memory_error",
]

VERBAL_LIKELIHOODS = [
    "very likely",
    "likely",
    "somewhat likely",
    "somewhat unlikely",
    "unlikely",
    "very unlikely",
]

VERBAL_WEIGHTS = {
    "very likely": 0.90,
    "likely": 0.70,
    "somewhat likely": 0.55,
    "somewhat unlikely": 0.35,
    "unlikely": 0.20,
    "very unlikely": 0.05,
}


def dellma_profile_key(step: dict[str, Any], model: str, window: int) -> str:
    payload = {
        "version": PROMPT_VERSION,
        "model": model,
        "window": window,
        "log_id": step["log_id"],
        "step_idx": step["step_idx"],
        "agent_name": step["agent_name"],
        "question": step["question"],
        "content": step["content"],
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_context(steps: list[dict[str, Any]], index: int, window: int) -> str:
    start = max(0, index - window)
    lines = []
    for row in steps[start:index]:
        lines.append(
            f"step {row['step_idx']} | {row['agent_name']}: {short_text(row['content'], 500)}"
        )
    return "\n".join(lines) if lines else "(no previous context)"


def build_dellma_messages(
    log_steps: list[dict[str, Any]],
    index: int,
    *,
    window: int = 4,
) -> list[dict[str, str]]:
    step = log_steps[index]
    modes = "\n".join(f"- {mode}" for mode in FAILURE_MODES)
    likelihoods = ", ".join(VERBAL_LIKELIHOODS)
    system = (
        "You are forecasting latent failure modes for one step in a failed LLM multi-agent "
        "trajectory. Return only valid JSON. Do not identify the decisive error step, do not "
        "predict the responsible agent, and do not use hidden labels. Treat this as uncertainty "
        "forecasting over possible local failure modes."
    )
    user = f"""
Task question:
{short_text(step['question'], 1200)}

Previous local context:
{build_context(log_steps, index, window)}

Current step:
step_idx: {step['step_idx']}
agent: {step['agent_name']}
raw_agent: {step.get('raw_agent')}
message:
{short_text(step['content'], 3000)}

Failure modes:
{modes}

For this current step, assign a verbal likelihood to every failure mode. Use only:
{likelihoods}

Return exactly one JSON object with these fields:
- failure_mode_beliefs: object whose keys are exactly the failure modes above and whose values are verbal likelihoods
- top_mode: the single most likely failure mode key
- severity: integer 0..3, where 0 means harmless and 3 means severe if this step is wrong
- audit_summary: one short sentence explaining the local uncertainty
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_likelihood(value: Any) -> str:
    text = str(value).strip().lower().replace("-", " ")
    if text in VERBAL_WEIGHTS:
        return text
    for option in VERBAL_LIKELIHOODS:
        if option in text:
            return option
    return "very unlikely"


def coerce_int(value: Any, low: int = 0, high: int = 3) -> int:
    try:
        out = int(value)
    except Exception:
        out = low
    return max(low, min(high, out))


def belief_probabilities(beliefs: dict[str, str]) -> dict[str, float]:
    weights = {mode: VERBAL_WEIGHTS[beliefs[mode]] for mode in FAILURE_MODES}
    total = sum(weights.values())
    if total <= 0:
        return {mode: 1.0 / len(FAILURE_MODES) for mode in FAILURE_MODES}
    return {mode: weights[mode] / total for mode in FAILURE_MODES}


def normalize_dellma_profile(raw: dict[str, Any]) -> dict[str, Any]:
    raw_beliefs = raw.get("failure_mode_beliefs", {})
    if not isinstance(raw_beliefs, dict):
        raw_beliefs = {}

    beliefs = {
        mode: normalize_likelihood(raw_beliefs.get(mode, "very unlikely"))
        for mode in FAILURE_MODES
    }
    top_mode = str(raw.get("top_mode", "")).strip()
    if top_mode not in FAILURE_MODES:
        top_mode = max(belief_probabilities(beliefs), key=belief_probabilities(beliefs).get)

    return {
        "failure_mode_beliefs": beliefs,
        "mode_probabilities": belief_probabilities(beliefs),
        "top_mode": top_mode,
        "severity": coerce_int(raw.get("severity", 0)),
        "audit_summary": short_text(str(raw.get("audit_summary", "")), 500),
    }


def extract_dellma_profile_object(text: str) -> dict[str, Any]:
    try:
        return extract_json_object(text)
    except (json.JSONDecodeError, ValueError):
        pass

    beliefs: dict[str, str] = {}
    for mode in FAILURE_MODES:
        pattern = re.compile(
            rf"['\"]?{re.escape(mode)}['\"]?\s*:\s*['\"]?([^'\",\n}}]+)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            beliefs[mode] = normalize_likelihood(match.group(1))

    if len(beliefs) < max(3, len(FAILURE_MODES) // 2):
        raise ValueError("Could not recover enough failure-mode beliefs from malformed response")

    severity = 0
    severity_match = re.search(r"['\"]?severity['\"]?\s*:\s*['\"]?([0-3])", text, re.IGNORECASE)
    if severity_match:
        severity = int(severity_match.group(1))

    top_mode = ""
    top_match = re.search(
        r"['\"]?top_mode['\"]?\s*:\s*['\"]?([A-Za-z0-9_]+)",
        text,
        re.IGNORECASE,
    )
    if top_match:
        top_mode = top_match.group(1)

    return {
        "failure_mode_beliefs": beliefs,
        "top_mode": top_mode,
        "severity": severity,
        "audit_summary": "Recovered from malformed JSON response.",
    }


def load_dellma_cache(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    for row in read_jsonl(path):
        cache[(row["log_id"], int(row["step_idx"]), row["profile_key"])] = row
    return cache


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def profile_one_dellma(
    client: DeepSeekClient,
    step: dict[str, Any],
    key: str,
    messages: list[dict[str, str]],
    window: int,
) -> tuple[str, dict[str, Any]]:
    try:
        response = client.chat(
            messages,
            max_tokens=2200,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content") or ""
        profile = normalize_dellma_profile(extract_dellma_profile_object(content))
        return (
            "ok",
            {
                "profile_key": key,
                "prompt_version": PROMPT_VERSION,
                "model": client.model,
                "window": window,
                "log_id": step["log_id"],
                "source_file": step["source_file"],
                "step_idx": step["step_idx"],
                "agent_name": step["agent_name"],
                "raw_agent": step.get("raw_agent"),
                "profile": profile,
                "usage": response.get("usage", {}),
            },
        )
    except (DeepSeekError, ValueError, KeyError) as exc:
        return (
            "error",
            {
                "profile_key": key,
                "prompt_version": PROMPT_VERSION,
                "model": client.model,
                "window": window,
                "log_id": step["log_id"],
                "source_file": step["source_file"],
                "step_idx": step["step_idx"],
                "agent_name": step["agent_name"],
                "error": str(exc)[:1500],
            },
        )


def profile_dellma_steps(
    steps: list[dict[str, Any]],
    client: DeepSeekClient,
    output_path: Path,
    *,
    limit: int | None = None,
    progress_every: int = 25,
    workers: int = 1,
    window: int = 4,
) -> tuple[int, int]:
    cache = load_dellma_cache(output_path)
    by_log: dict[str, list[dict[str, Any]]] = {}
    for row in steps:
        by_log.setdefault(row["log_id"], []).append(row)
    for log_rows in by_log.values():
        log_rows.sort(key=lambda item: item["step_idx"])

    attempted = 0
    written = 0
    skipped = 0
    tasks = []
    error_path = output_path.with_suffix(output_path.suffix + ".errors")
    for step in steps:
        if step["agent_name"] == "human":
            continue
        key = dellma_profile_key(step, client.model, window)
        if (step["log_id"], int(step["step_idx"]), key) in cache:
            skipped += 1
            continue
        if limit is not None and attempted >= limit:
            break
        log_steps = by_log[step["log_id"]]
        index = next(i for i, row in enumerate(log_steps) if row["step_idx"] == step["step_idx"])
        messages = build_dellma_messages(log_steps, index, window=window)
        attempted += 1
        tasks.append((step, key, messages))

    if workers <= 1:
        completed = 0
        errors = 0
        for step, key, messages in tasks:
            status, row = profile_one_dellma(client, step, key, messages, window)
            if status == "ok":
                append_jsonl(output_path, row)
                written += 1
            else:
                append_jsonl(error_path, row)
                errors += 1
            completed += 1
            if progress_every and completed % progress_every == 0:
                print(
                    f"completed={completed} written={written} errors={errors} skipped={skipped}",
                    flush=True,
                )
        return attempted, written

    completed = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(profile_one_dellma, client, step, key, messages, window)
            for step, key, messages in tasks
        ]
        for future in as_completed(futures):
            status, row = future.result()
            if status == "ok":
                append_jsonl(output_path, row)
                written += 1
            else:
                append_jsonl(error_path, row)
                errors += 1
            completed += 1
            if progress_every and completed % progress_every == 0:
                print(
                    f"completed={completed} written={written} errors={errors} skipped={skipped}",
                    flush=True,
                )
    return attempted, written
