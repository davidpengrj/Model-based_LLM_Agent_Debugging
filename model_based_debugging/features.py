"""Deterministic semantic views used by the model."""

from __future__ import annotations

import re
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


def semantic_components(triple: CanonicalTriple) -> dict[str, tuple[str, ...]]:
    """Split the original triple into equally weighted Agent/Action/State channels."""

    agent = normalize(triple.agent).split("(", 1)[0].strip()
    return {
        "agent_words": lexical_tokens(agent),
        "action_words": lexical_tokens(triple.action),
        "state_words": lexical_tokens(triple.state),
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
