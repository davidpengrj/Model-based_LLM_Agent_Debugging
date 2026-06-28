from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def normalize_agent(raw: str | None) -> str:
    if not raw:
        return "Unknown"
    text = raw.strip()
    low = text.lower()
    if low == "websurfer" or low == "web surfer":
        return "WebSurfer"
    if low == "websurfer".lower():
        return "WebSurfer"
    if low.startswith("orchestrator"):
        return "Orchestrator"
    if low == "assistant":
        return "Assistant"
    if low == "filesurfer":
        return "FileSurfer"
    if low == "computerterminal" or low == "computer_terminal":
        return "ComputerTerminal"
    if low in {"human", "user"}:
        return "human"
    return text


def load_who_when_logs(dataset_root: Path, subset: str = "Hand-Crafted") -> list[dict[str, Any]]:
    folder = dataset_root / "Who&When" / subset
    if not folder.is_dir():
        raise FileNotFoundError(f"Who&When subset not found: {folder}")

    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 10**9, path.name

    logs = []
    for path in sorted(folder.glob("*.json"), key=sort_key):
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["_source_file"] = path.name
        logs.append(obj)
    return logs


def flatten_logs(logs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for log_idx, log in enumerate(logs):
        source_file = log.get("_source_file") or f"{log_idx + 1}.json"
        log_id = Path(source_file).stem
        history = log["history"]
        mistake_step = int(log["mistake_step"])
        mistake_agent = normalize_agent(log.get("mistake_agent"))
        ground_truth = log.get("groundtruth", log.get("ground_truth", ""))

        for step_idx, message in enumerate(history):
            raw_agent = message.get("name") or message.get("role")
            agent_name = normalize_agent(raw_agent)
            content = message.get("content") or ""
            rows.append(
                {
                    "log_id": log_id,
                    "source_file": source_file,
                    "step_idx": step_idx,
                    "num_steps": len(history),
                    "raw_agent": raw_agent,
                    "agent_name": agent_name,
                    "content": content,
                    "question": log.get("question", ""),
                    "ground_truth": ground_truth,
                    "mistake_agent": mistake_agent,
                    "mistake_step": mistake_step,
                    "is_decisive_error": step_idx == mistake_step,
                    "question_ID": log.get("question_ID", ""),
                }
            )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_handcrafted_steps(dataset_root: Path) -> list[dict[str, Any]]:
    return flatten_logs(load_who_when_logs(dataset_root, "Hand-Crafted"))
