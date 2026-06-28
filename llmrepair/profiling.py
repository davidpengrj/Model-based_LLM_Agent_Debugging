from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .deepseek import DeepSeekClient, DeepSeekError, extract_json_object


PROMPT_VERSION = "step-profile-v1"

ACTION_TYPES = {
    "plan",
    "delegate",
    "web_search",
    "web_browse",
    "tool_result",
    "file_read",
    "code_execute",
    "reason",
    "answer",
    "termination",
    "other",
}

PROGRESS_STATES = {"on_track", "uncertain", "stalled", "off_track"}


def profile_key(step: dict[str, Any], model: str, ground_truth_mode: str) -> str:
    payload = {
        "version": PROMPT_VERSION,
        "model": model,
        "ground_truth_mode": ground_truth_mode,
        "log_id": step["log_id"],
        "step_idx": step["step_idx"],
        "agent_name": step["agent_name"],
        "content": step["content"],
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def short_text(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 20] + " ...[truncated]"


def build_context(steps: list[dict[str, Any]], index: int, window: int = 4) -> str:
    start = max(0, index - window)
    lines = []
    for row in steps[start:index]:
        lines.append(
            f"step {row['step_idx']} | {row['agent_name']}: {short_text(row['content'], 500)}"
        )
    return "\n".join(lines) if lines else "(no previous context)"


def build_messages(
    log_steps: list[dict[str, Any]],
    index: int,
    *,
    ground_truth_mode: str = "without_ground_truth",
) -> list[dict[str, str]]:
    step = log_steps[index]
    ground_truth_line = ""
    if ground_truth_mode == "with_ground_truth":
        ground_truth_line = f"\nGround truth answer: {short_text(step['ground_truth'], 500)}"

    system = (
        "You are profiling one step in a failed LLM multi-agent trajectory. "
        "Return only valid JSON. Do not identify the decisive error step, "
        "do not predict the responsible agent, and do not use hidden labels."
    )
    user = f"""
Task question: {short_text(step['question'], 1200)}{ground_truth_line}

Previous local context:
{build_context(log_steps, index)}

Current step:
step_idx: {step['step_idx']}
agent: {step['agent_name']}
raw_agent: {step.get('raw_agent')}
message:
{short_text(step['content'], 3000)}

Return exactly one JSON object with these fields:
- action_type: one of {sorted(ACTION_TYPES)}
- agent_intent: short phrase describing what the agent is trying to do
- task_alignment: integer 0..3, where 0 means unrelated and 3 means directly aligned
- evidence_quality: integer 0..3, where 0 means no evidence and 3 means strong verified evidence
- progress_state: one of {sorted(PROGRESS_STATES)}
- new_claim: boolean
- unsupported_claim: boolean
- tool_use: boolean
- tool_error: boolean
- reasoning_error_signal: boolean
- state_summary: one short sentence describing the local state
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def coerce_int(value: Any, low: int = 0, high: int = 3) -> int:
    try:
        out = int(value)
    except Exception:
        out = low
    return max(low, min(high, out))


def normalize_profile(raw: dict[str, Any]) -> dict[str, Any]:
    action_type = str(raw.get("action_type", "other")).strip()
    if action_type not in ACTION_TYPES:
        action_type = "other"
    progress_state = str(raw.get("progress_state", "uncertain")).strip()
    if progress_state not in PROGRESS_STATES:
        progress_state = "uncertain"

    return {
        "action_type": action_type,
        "agent_intent": short_text(str(raw.get("agent_intent", "")), 240),
        "task_alignment": coerce_int(raw.get("task_alignment", 0)),
        "evidence_quality": coerce_int(raw.get("evidence_quality", 0)),
        "progress_state": progress_state,
        "new_claim": coerce_bool(raw.get("new_claim", False)),
        "unsupported_claim": coerce_bool(raw.get("unsupported_claim", False)),
        "tool_use": coerce_bool(raw.get("tool_use", False)),
        "tool_error": coerce_bool(raw.get("tool_error", False)),
        "reasoning_error_signal": coerce_bool(raw.get("reasoning_error_signal", False)),
        "state_summary": short_text(str(raw.get("state_summary", "")), 500),
    }


def load_profile_cache(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    for row in read_jsonl(path):
        cache[(row["log_id"], int(row["step_idx"]), row["profile_key"])] = row
    return cache


def append_profile(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_error(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def profile_one(
    client: DeepSeekClient,
    step: dict[str, Any],
    key: str,
    messages: list[dict[str, str]],
    ground_truth_mode: str,
) -> tuple[str, dict[str, Any]]:
    try:
        response = client.chat(
            messages,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content") or ""
        profile = normalize_profile(extract_json_object(content))
        return (
            "ok",
            {
                "profile_key": key,
                "prompt_version": PROMPT_VERSION,
                "model": client.model,
                "ground_truth_mode": ground_truth_mode,
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
                "model": client.model,
                "ground_truth_mode": ground_truth_mode,
                "log_id": step["log_id"],
                "source_file": step["source_file"],
                "step_idx": step["step_idx"],
                "agent_name": step["agent_name"],
                "error": str(exc)[:1500],
            },
        )


def profile_steps(
    steps: list[dict[str, Any]],
    client: DeepSeekClient,
    output_path: Path,
    *,
    limit: int | None = None,
    ground_truth_mode: str = "without_ground_truth",
    progress_every: int = 25,
    workers: int = 1,
) -> tuple[int, int]:
    cache = load_profile_cache(output_path)
    by_log: dict[str, list[dict[str, Any]]] = {}
    for row in steps:
        by_log.setdefault(row["log_id"], []).append(row)
    for log_rows in by_log.values():
        log_rows.sort(key=lambda item: item["step_idx"])

    attempted = 0
    written = 0
    skipped = 0
    error_path = output_path.with_suffix(output_path.suffix + ".errors")
    tasks = []
    for step in steps:
        if step["agent_name"] == "human":
            continue
        key = profile_key(step, client.model, ground_truth_mode)
        if (step["log_id"], int(step["step_idx"]), key) in cache:
            skipped += 1
            continue
        if limit is not None and attempted >= limit:
            break
        log_steps = by_log[step["log_id"]]
        index = next(i for i, row in enumerate(log_steps) if row["step_idx"] == step["step_idx"])
        messages = build_messages(log_steps, index, ground_truth_mode=ground_truth_mode)
        attempted += 1
        tasks.append((step, key, messages))

    if workers <= 1:
        completed = 0
        errors = 0
        for step, key, messages in tasks:
            status, row = profile_one(client, step, key, messages, ground_truth_mode)
            if status == "ok":
                append_profile(output_path, row)
                written += 1
            else:
                append_error(error_path, row)
                errors += 1
            completed += 1
            if progress_every and completed % progress_every == 0:
                print(
                    f"progress attempted={completed}/{attempted} written={written} errors={errors} skipped_cached={skipped}",
                    flush=True,
                )
        return attempted, written

    completed = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(profile_one, client, step, key, messages, ground_truth_mode)
            for step, key, messages in tasks
        ]
        for future in as_completed(futures):
            status, row = future.result()
            if status == "ok":
                append_profile(output_path, row)
                written += 1
            else:
                append_error(error_path, row)
                errors += 1
            completed += 1
            if progress_every and completed % progress_every == 0:
                print(
                    f"progress attempted={completed}/{attempted} written={written} errors={errors} skipped_cached={skipped}",
                    flush=True,
                )
    return attempted, written
