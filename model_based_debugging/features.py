"""Deterministic semantic views used by the model."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from .schema import CanonicalExtraction, CanonicalTriple, lexical_tokens, normalize

ACTION_VIEW_VERSION = "predicate_suffix_anchor_v1"


def action_heads(action: str) -> tuple[str, ...]:
    """Extract ordered predicate heads from a canonical Action string."""

    normalized = normalize(action)
    heads = re.findall(r"(?<![\w.])([a-z_][\w.]*)\s*\(", normalized)
    if not heads:
        heads = re.findall(r"(?:^|;)\s*([a-z_][\w.]*)", normalized)
    return tuple(dict.fromkeys(heads)) or ("<NONE>",)


def _entries(text: str) -> tuple[str, ...]:
    """Split on semicolons outside quotes and bracketed arguments."""

    entries, start, depth, quote, escaped = [], 0, 0, None, False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            if char != "'" or not (index and text[index - 1].isalnum()):
                quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ";" and not depth:
            if text[start:index].strip():
                entries.append(text[start:index].strip())
            start = index + 1
    if text[start:].strip():
        entries.append(text[start:].strip())
    return tuple(entries) or (text,)


def _content_tokens(text: str) -> tuple[str, ...]:
    tokens = lexical_tokens(text)
    digits = tuple(
        dict.fromkeys(re.findall(r"(?<![a-z0-9])[0-9](?![a-z0-9])", normalize(text)))
    )
    if not digits:
        return tokens
    return tuple(token for token in tokens if token != "<EMPTY>") + digits


def _entry_weights(text: str) -> dict[str, float]:
    entries = _entries(text)
    weights = Counter()
    for entry in entries:
        tokens = _content_tokens(entry)
        for token in tokens:
            weights[token] += (1.0 / len(tokens)) / len(entries)
    return dict(weights)


def semantic_term_weights(triple: CanonicalTriple) -> dict[str, dict[str, float]]:
    """Divide each field's weight equally among entries, then among their terms."""

    agent = lexical_tokens(normalize(triple.agent).split("(", 1)[0].strip())
    return {
        "agent_words": {token: 1.0 / len(agent) for token in agent},
        "action_words": _entry_weights(triple.action),
        "state_words": _entry_weights(triple.state),
    }


def operation_anchors(action: str) -> tuple[str, ...]:
    """Return suffix backoffs such as ``propose_code -> code``."""

    anchors: list[str] = []
    heads = action_heads(action)
    existing = frozenset(heads)
    for head in heads:
        _, separator, suffix = head.partition("_")
        if separator and suffix and suffix not in existing and suffix not in anchors:
            anchors.append(suffix)
    return tuple(anchors)


def operation_anchored_triple(triple: CanonicalTriple) -> CanonicalTriple:
    """Append operation anchors to a read-only Action view."""

    anchors = operation_anchors(triple.action)
    action = triple.action
    if anchors:
        action += "; " + "; ".join(f"{anchor}()" for anchor in anchors)
    return CanonicalTriple(
        step=triple.step,
        agent=triple.agent,
        action=action,
        state=triple.state,
    )


def operation_anchored_extraction(
    extraction: CanonicalExtraction,
) -> CanonicalExtraction:
    """Create the Action view used only by relation and state dynamics."""

    if extraction.usage.get("action_view") == ACTION_VIEW_VERSION:
        return extraction
    usage = dict(extraction.usage)
    usage["action_view"] = ACTION_VIEW_VERSION
    return CanonicalExtraction(
        trace_id=extraction.trace_id,
        model=extraction.model,
        prompt_version=extraction.prompt_version,
        triples=tuple(operation_anchored_triple(row) for row in extraction.triples),
        usage=usage,
    )


def state_leaf_fields(state: str) -> tuple[str, ...]:
    """Take the final component of every assignment's left-hand field path."""

    fields = tuple(
        dict.fromkeys(
            match.group(1).casefold().split(".")[-1]
            for match in re.finditer(r"([A-Za-z_][\w.]*)\s*=", state)
        )
    )
    return fields or ("<NONE>",)


def action_state_relation_tokens(triple: CanonicalTriple) -> tuple[str, ...]:
    """Form the Cartesian product of Action heads and State leaf fields."""

    return tuple(
        f"{action}|{field}"
        for action in action_heads(triple.action)
        for field in state_leaf_fields(triple.state)
    )


def action_document(triple: CanonicalTriple) -> str:
    """Build the Action document used by its own TF-IDF vocabulary."""

    return " ".join(
        (
            *(f"action_words__{token}" for token in lexical_tokens(triple.action)),
            *(f"action_head__{token}" for token in action_heads(triple.action)),
        )
    )


def state_document(triple: CanonicalTriple) -> str:
    """Build the State document used by its own TF-IDF vocabulary."""

    return " ".join(f"state_words__{token}" for token in lexical_tokens(triple.state))


def flatten_triples(
    extractions: Sequence[CanonicalExtraction],
) -> tuple[CanonicalTriple, ...]:
    triples = tuple(row for extraction in extractions for row in extraction.triples)
    if not triples:
        raise ValueError("at least one triple is required")
    return triples
