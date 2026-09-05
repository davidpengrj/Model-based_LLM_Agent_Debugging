"""Validated data structures for semantic triples and execution traces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROMPT_VERSION = "famas-canonical-predicate-v2.1"

_IGNORED_TOKENS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "from",
        "with",
        "by",
        "at",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "into",
        "after",
        "before",
        "then",
        "task",
        "system",
        "agent",
        "action",
        "state",
        "provides",
        "provide",
        "shows",
        "show",
        "resulting",
        "result",
        "results",
        "retrieves",
        "retrieve",
        "requests",
        "request",
        "performs",
        "perform",
        "executes",
        "execute",
    ]
)


@dataclass(frozen=True)
class CanonicalTriple:
    """One zero-based ``<Agent, Action, State>`` triple."""

    step: int
    agent: str
    action: str
    state: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ValueError("step must be a non-negative integer")
        for name in ("agent", "action", "state"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class CanonicalExtraction:
    """All triples extracted from one trajectory by one LLM call."""

    trace_id: str
    triples: tuple[CanonicalTriple, ...]
    model: str = "deepseek-v4-flash"
    prompt_version: str = PROMPT_VERSION
    usage: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("trace_id must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"prompt_version must be {PROMPT_VERSION!r}, "
                f"received {self.prompt_version!r}"
            )
        triples = tuple(self.triples)
        if not triples:
            raise ValueError("triples must contain at least one step")
        for expected_step, triple in enumerate(triples):
            if not isinstance(triple, CanonicalTriple):
                raise TypeError("triples must contain only CanonicalTriple values")
            if triple.step != expected_step:
                raise ValueError(
                    "triple steps must be contiguous and zero-based: "
                    f"expected {expected_step}, received {triple.step}"
                )
        if not isinstance(self.usage, dict):
            raise TypeError("usage must be an object")
        object.__setattr__(self, "triples", triples)
        object.__setattr__(self, "usage", dict(self.usage))


@dataclass(frozen=True)
class TraceRecord:
    """One benchmark trajectory; labels are consumed only during training/evaluation."""

    trace_id: str
    source: str
    group_id: str
    question: str
    history: tuple[dict[str, object], ...]
    gold_step: int
    gold_agent: str
    path: Path | None = None

    def __post_init__(self) -> None:
        if not self.trace_id or not self.group_id:
            raise ValueError("trace_id and group_id must be non-empty")
        if not self.history:
            raise ValueError("history must be non-empty")
        if not 0 <= self.gold_step < len(self.history):
            raise ValueError("gold_step must index history")


def load_canonical_cache(cache_dir: Path) -> dict[str, CanonicalExtraction]:
    """Load all strict extraction-cache JSON files in ``cache_dir``."""

    directory = Path(cache_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"canonical cache directory not found: {directory}")
    extractions: dict[str, CanonicalExtraction] = {}
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"cache root must be an object: {path}")
        required = {"trace_id", "model", "prompt_version", "triples", "usage"}
        allowed = required | {"created_unix", "input_fingerprint"}
        if not required <= set(raw) or not set(raw) <= allowed:
            raise ValueError(f"cache fields do not match the contract: {path}")
        raw_triples = raw["triples"]
        if not isinstance(raw_triples, list):
            raise TypeError(f"triples must be a list: {path}")
        triples = []
        for row in raw_triples:
            if not isinstance(row, dict) or set(row) != {
                "step",
                "agent",
                "action",
                "state",
            }:
                raise ValueError(f"invalid triple fields: {path}")
            triples.append(CanonicalTriple(**row))
        extraction = CanonicalExtraction(
            trace_id=raw["trace_id"],
            model=raw["model"],
            prompt_version=raw["prompt_version"],
            triples=tuple(triples),
            usage=raw["usage"],
        )
        if extraction.trace_id in extractions:
            raise ValueError(f"duplicate extraction for {extraction.trace_id!r}")
        extractions[extraction.trace_id] = extraction
    return extractions


def normalize(text: str) -> str:
    """Case-fold text and collapse whitespace."""

    return re.sub(r"\s+", " ", text.strip().casefold())


def normalize_agent(text: str) -> str:
    """Normalize an agent name for evaluation without changing model input."""

    return normalize(text).split("(", 1)[0].strip()


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Return the deterministic lexical view used by the frequency model."""

    tokens: list[str] = []
    for raw in re.findall(r"[a-z0-9]+", normalize(text)):
        if len(raw) < 2 or raw in _IGNORED_TOKENS:
            continue
        token = raw
        for suffix in ("ing", "ied", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) >= len(suffix) + 3:
                token = token[: -len(suffix)]
                break
        if token not in tokens:
            tokens.append(token)
    return tuple(tokens) or ("<EMPTY>",)
