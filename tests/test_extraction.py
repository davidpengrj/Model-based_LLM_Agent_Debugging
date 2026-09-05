from __future__ import annotations

import json

import pytest

from model_based_debugging.extraction import (
    decode_completion,
    system_prompt,
    user_prompt,
)
from model_based_debugging.schema import TraceRecord


def record() -> TraceRecord:
    return TraceRecord(
        trace_id="Hand-Crafted/1",
        source="Hand-Crafted",
        group_id="question-1",
        question="Find the requested value.",
        history=(
            {"name": "Planner", "content": "Search for the value."},
            {"name": "Researcher", "content": "The source reports 42."},
        ),
        gold_step=1,
        gold_agent="Researcher",
    )


def test_prompt_contains_visible_history_but_no_attribution_labels() -> None:
    prompt = user_prompt(record())

    assert "Find the requested value." in prompt
    assert "Search for the value." in prompt
    assert "mistake_step" not in prompt
    assert "mistake_agent" not in prompt
    assert "ground_truth" not in prompt
    assert "Never judge correctness" in system_prompt()


def test_decode_requires_one_aligned_triple_per_source_step() -> None:
    completion = json.dumps(
        {
            "triples": [
                {
                    "step": 0,
                    "agent": "Planner",
                    "action": "request_search(object=value)",
                    "state": "record.observation=not_reported",
                },
                {
                    "step": 1,
                    "agent": "Researcher",
                    "action": "report_value(value=42)",
                    "state": "source.value=42",
                },
            ]
        }
    )

    triples = decode_completion(record(), completion)

    assert len(triples) == 2
    assert triples[1].state == "source.value=42"


def test_decode_rejects_changed_speaker() -> None:
    completion = json.dumps(
        {
            "triples": [
                {
                    "step": 0,
                    "agent": "Other",
                    "action": "request_search()",
                    "state": "record.observation=not_reported",
                },
                {
                    "step": 1,
                    "agent": "Researcher",
                    "action": "report_value(value=42)",
                    "state": "source.value=42",
                },
            ]
        }
    )

    with pytest.raises(ValueError, match="explicit speaker"):
        decode_completion(record(), completion)
