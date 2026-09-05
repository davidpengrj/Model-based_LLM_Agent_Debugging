from __future__ import annotations

import json

import numpy as np

from model_based_debugging.features import (
    action_state_relation_tokens,
    operation_anchored_triple,
    operation_anchors,
    state_leaf_fields,
)
from model_based_debugging.model import (
    ERROR,
    ActionStateRelationEmission,
    FrequencyDTMC,
    ModelBasedDebugger,
    ModelConfig,
    SemanticEmission,
    hierarchical_readout,
)
from model_based_debugging.schema import (
    PROMPT_VERSION,
    CanonicalExtraction,
    CanonicalTriple,
    load_canonical_cache,
)


def triple(step: int, agent: str, action: str, state: str) -> CanonicalTriple:
    return CanonicalTriple(step=step, agent=agent, action=action, state=state)


def extraction(trace_id: str, rows: tuple[CanonicalTriple, ...]) -> CanonicalExtraction:
    return CanonicalExtraction(trace_id=trace_id, triples=rows)


def test_operation_anchor_and_relation_are_deterministic() -> None:
    row = triple(
        0,
        "Coder",
        "propose_code(object=report); request_web_search(query=paper)",
        "record.observation=not_reported; tool.status=ready",
    )

    assert operation_anchors(row.action) == ("code", "web_search")
    assert state_leaf_fields(row.state) == ("observation", "status")

    anchored = operation_anchored_triple(row)
    assert anchored.action.endswith("code(); web_search()")
    assert action_state_relation_tokens(anchored) == (
        "propose_code|observation",
        "propose_code|status",
        "request_web_search|observation",
        "request_web_search|status",
        "code|observation",
        "code|status",
        "web_search|observation",
        "web_search|status",
    )


def test_semantic_group_weight_gives_each_group_equal_total_mass() -> None:
    sequences = (
        (
            triple(0, "Alpha", "find_item()", "record.status=bad"),
            triple(1, "Alpha", "report_item()", "record.status=ok"),
        ),
        (
            triple(0, "Alpha", "find_item()", "record.status=bad"),
            triple(1, "Alpha", "report_item()", "record.status=ok"),
        ),
        (
            triple(0, "Gamma", "find_value()", "record.status=bad"),
            triple(1, "Gamma", "report_value()", "record.status=ok"),
        ),
    )
    emission = SemanticEmission(alpha=0.075).fit(
        sequences,
        (0, 0, 0),
        group_ids=("duplicate", "duplicate", "single"),
    )

    assert emission.counts[(ERROR, "agent_words")]["alpha"] == 1.0
    assert emission.counts[(ERROR, "agent_words")]["gamma"] == 1.0


def test_relation_score_is_gated_when_any_pair_is_unseen() -> None:
    training = (
        (
            triple(0, "A", "search_web()", "record.observation=missing"),
            triple(1, "A", "report_answer()", "record.answer=ready"),
        ),
    )
    relation = ActionStateRelationEmission(alpha=0.4).fit(
        training, (0,), group_ids=("g",)
    )
    values, seen = relation.score(
        (
            training[0][0],
            triple(1, "A", "execute_code()", "record.output=ready"),
        )
    )

    assert values.shape == (2,)
    assert seen.tolist() == [True, False]


def test_soft_dtmc_counts_apply_group_weight_and_normalize() -> None:
    first = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    second = np.asarray([[0.6, 0.4], [0.1, 0.9]])
    third = np.asarray([[0.2, 0.8], [0.9, 0.1]])
    model = FrequencyDTMC(2, alpha=0.1).fit(
        (first, second, third), group_ids=("duplicate", "duplicate", "single")
    )

    expected_initial = 0.5 * first[0] + 0.5 * second[0] + third[0]
    expected_transition = (
        0.5 * np.outer(first[0], first[1])
        + 0.5 * np.outer(second[0], second[1])
        + np.outer(third[0], third[1])
    )
    np.testing.assert_allclose(model.initial_counts, expected_initial)
    np.testing.assert_allclose(model.transition_counts, expected_transition)
    np.testing.assert_allclose(model.initial_probability.sum(), 1.0)
    np.testing.assert_allclose(model.transition_probability.sum(axis=1), 1.0)


def test_hierarchical_readout_selects_who_then_when() -> None:
    rows = (
        triple(0, "A", "a()", "x.y=1"),
        triple(1, "B", "b()", "x.y=1"),
        triple(2, "B", "c()", "x.y=1"),
        triple(3, "A", "d()", "x.y=1"),
    )
    probability = np.asarray([0.15, 0.35, 0.40, 0.10])
    surprise = np.asarray([100.0, 1.0, 2.0, 100.0])

    agent, step, masses, evidence = hierarchical_readout(probability, surprise, rows)

    assert agent == "B"
    assert step == 2
    assert masses == {"A": 0.25, "B": 0.75}
    np.testing.assert_allclose(evidence, probability * surprise)


def test_end_to_end_model_returns_an_auditable_prediction() -> None:
    actions = (
        "plan_task(object=report)",
        "search_web(query=paper)",
        "propose_code(object=table)",
        "report_result(object=answer)",
    )
    states = (
        "record.plan=ready",
        "record.observation=found",
        "record.code=present",
        "record.answer=reported",
    )
    agents = ("Planner", "Researcher", "Coder", "Planner")
    train = tuple(
        extraction(
            f"trace-{index}",
            tuple(
                triple(
                    step,
                    agents[step],
                    actions[step],
                    states[(step + index) % len(states)],
                )
                for step in range(4)
            ),
        )
        for index in range(4)
    )
    config = ModelConfig(
        pca_dimensions=2,
        state_count=2,
        tfidf_min_df=1,
        gmm_n_init=1,
    )
    model = ModelBasedDebugger(config).fit(
        train,
        (1, 2, 1, 2),
        group_ids=("g0", "g1", "g2", "g3"),
    )

    prediction = model.predict(train[0])

    assert prediction.predicted_agent in agents
    assert 0 <= prediction.predicted_step < 4
    assert (
        train[0].triples[prediction.predicted_step].agent == prediction.predicted_agent
    )
    np.testing.assert_allclose(prediction.content_probability.sum(), 1.0)
    np.testing.assert_allclose(prediction.state_membership.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        prediction.step_evidence,
        prediction.content_probability * prediction.transition_surprise,
    )


def test_cache_loader_enforces_canonical_contract(tmp_path) -> None:
    payload = {
        "trace_id": "Hand-Crafted/1",
        "model": "deepseek-v4-flash",
        "prompt_version": PROMPT_VERSION,
        "triples": [
            {
                "step": 0,
                "agent": "Planner",
                "action": "plan_task()",
                "state": "record.plan=ready",
            }
        ],
        "usage": {},
    }
    (tmp_path / "one.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_canonical_cache(tmp_path)

    assert loaded["Hand-Crafted/1"].triples[0].agent == "Planner"
