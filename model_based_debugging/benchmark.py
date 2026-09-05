"""Who&When loading, leakage-safe grouped splits, and stability evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .extraction import visible_history
from .model import ModelBasedDebugger, ModelConfig
from .schema import CanonicalExtraction, TraceRecord, normalize_agent

DEFAULT_SPLIT_SEEDS = tuple(range(20))
_ACCURACY_METRICS = (
    "when_accuracy",
    "who_accuracy",
    "joint_accuracy",
    "within_1_accuracy",
    "within_2_accuracy",
    "within_3_accuracy",
    "within_5_accuracy",
)


@dataclass(frozen=True)
class HistorySafeSplit:
    """One grouped holdout after visible-history clone quarantine."""

    development_records: tuple[TraceRecord, ...]
    test_records: tuple[TraceRecord, ...]
    quarantine_records: tuple[TraceRecord, ...]
    effective_group_ids: tuple[str, ...]
    question_group_overlap: tuple[str, ...]
    visible_history_overlap: tuple[str, ...]


def load_who_when(dataset: Path) -> list[TraceRecord]:
    """Load the original ``Who&When/<subset>/*.json`` layout."""

    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Who&When dataset not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Who&When dataset must be a directory: {path}")
    dataset_root = path / "Who&When" if (path / "Who&When").is_dir() else path
    json_paths = tuple(
        item
        for subset in ("Algorithm-Generated", "Hand-Crafted")
        for item in (dataset_root / subset).glob("*.json")
    )
    if json_paths:
        return _load_json_files(json_paths)
    raise FileNotFoundError(f"no Who&When/<subset>/*.json records found under {path}")


def grouped_split(
    records: Sequence[TraceRecord],
    *,
    seed: int,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict[str, tuple[TraceRecord, ...]]:
    """Split question groups so repeated tasks never cross partitions."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0 < train_fraction < 1 or not 0 <= calibration_fraction < 1:
        raise ValueError("split fractions are invalid")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must sum to less than one")
    rows = tuple(records)
    if not rows:
        raise ValueError("records must be non-empty")
    group_ids = sorted({row.group_id for row in rows})
    random.Random(seed).shuffle(group_ids)
    train_end = round(len(group_ids) * train_fraction)
    calibration_end = train_end + round(len(group_ids) * calibration_fraction)
    partition = {
        group_id: (
            "train"
            if index < train_end
            else "calibration"
            if index < calibration_end
            else "test"
        )
        for index, group_id in enumerate(group_ids)
    }
    return {
        name: tuple(row for row in rows if partition[row.group_id] == name)
        for name in ("train", "calibration", "test")
    }


def history_safe_split(
    records: Sequence[TraceRecord],
    *,
    seed: int,
) -> HistorySafeSplit:
    """Quarantine development traces cloned into the grouped test partition."""

    rows = tuple(records)
    split = grouped_split(rows, seed=seed)
    original_development = split["train"] + split["calibration"]
    test = split["test"]
    development_ids = {row.trace_id for row in original_development}
    test_ids = {row.trace_id for row in test}
    if len({row.trace_id for row in rows}) != len(rows):
        raise ValueError("trace IDs must be unique")

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            lower, upper = sorted((left_root, right_root))
            parent[upper] = lower

    fingerprints = tuple(visible_history_fingerprint(row) for row in rows)
    by_group: dict[str, list[int]] = {}
    by_history: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_group.setdefault(row.group_id, []).append(index)
        by_history.setdefault(fingerprints[index], []).append(index)
    for buckets in (by_group, by_history):
        for indices in buckets.values():
            for index in indices[1:]:
                union(indices[0], index)

    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    component_ids = {
        root: "history-component:"
        + hashlib.sha256(
            json.dumps(
                sorted(rows[index].trace_id for index in indices),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for root, indices in components.items()
    }
    cross_boundary_roots = {
        root
        for root, indices in components.items()
        if any(rows[index].trace_id in development_ids for index in indices)
        and any(rows[index].trace_id in test_ids for index in indices)
    }
    index_by_id = {row.trace_id: index for index, row in enumerate(rows)}
    quarantine_ids = {
        row.trace_id
        for row in original_development
        if find(index_by_id[row.trace_id]) in cross_boundary_roots
    }
    development = tuple(
        row for row in original_development if row.trace_id not in quarantine_ids
    )
    quarantine = tuple(
        row for row in original_development if row.trace_id in quarantine_ids
    )
    effective_groups = tuple(
        component_ids[find(index_by_id[row.trace_id])] for row in development
    )
    question_overlap = tuple(
        sorted({row.group_id for row in development} & {row.group_id for row in test})
    )
    history_overlap = tuple(
        sorted(
            {visible_history_fingerprint(row) for row in development}
            & {visible_history_fingerprint(row) for row in test}
        )
    )
    if question_overlap or history_overlap:
        raise RuntimeError("history-safe grouped split leaked evaluation data")
    return HistorySafeSplit(
        development_records=development,
        test_records=test,
        quarantine_records=quarantine,
        effective_group_ids=effective_groups,
        question_group_overlap=question_overlap,
        visible_history_overlap=history_overlap,
    )


def visible_history_fingerprint(record: TraceRecord) -> str:
    """Hash exactly the label-free history consumed by the extractor."""

    serialized = json.dumps(
        visible_history(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def evaluate_seed(
    records: Sequence[TraceRecord],
    extraction_cache: Mapping[str, CanonicalExtraction],
    *,
    split_seed: int,
    config: ModelConfig | None = None,
) -> dict[str, object]:
    """Fit on one history-safe development partition and score its test set."""

    partition = history_safe_split(records, seed=split_seed)
    development = partition.development_records
    test = partition.test_records
    development_extractions = tuple(
        _aligned_extraction(row, extraction_cache) for row in development
    )
    test_extractions = tuple(_aligned_extraction(row, extraction_cache) for row in test)
    model = ModelBasedDebugger(config or ModelConfig()).fit(
        development_extractions,
        tuple(row.gold_step for row in development),
        group_ids=partition.effective_group_ids,
    )

    predictions: list[dict[str, object]] = []
    for record, extraction in zip(test, test_extractions):
        prediction = model.predict(extraction)
        step_error = abs(prediction.predicted_step - record.gold_step)
        who_correct = normalize_agent(prediction.predicted_agent) == normalize_agent(
            record.gold_agent
        )
        when_correct = prediction.predicted_step == record.gold_step
        predictions.append(
            {
                "trace_id": record.trace_id,
                "group_id": record.group_id,
                "source": record.source,
                "gold_agent": record.gold_agent,
                "predicted_agent": prediction.predicted_agent,
                "gold_step": record.gold_step,
                "predicted_step": prediction.predicted_step,
                "who_correct": who_correct,
                "when_correct": when_correct,
                "joint_correct": who_correct and when_correct,
                "absolute_step_error": step_error,
                "within_1": step_error <= 1,
                "within_2": step_error <= 2,
                "within_3": step_error <= 3,
                "within_5": step_error <= 5,
                "selected_agent_mass": prediction.agent_masses[
                    prediction.predicted_agent
                ],
                "selected_step_evidence": float(
                    prediction.step_evidence[prediction.predicted_step]
                ),
                "content_probability_sum_error": abs(
                    math.fsum(prediction.content_probability) - 1.0
                ),
                "selected_step_belongs_to_selected_agent": (
                    extraction.triples[prediction.predicted_step].agent
                    == prediction.predicted_agent
                ),
                "relation_seen_fraction": float(prediction.relation_seen.mean()),
                "semantic_oov_fraction": prediction.semantic_oov_fraction,
                "state_zero_vector_fraction": prediction.state_zero_vector_fraction,
            }
        )

    return {
        "split_seed": split_seed,
        "original_development_records": (
            len(partition.development_records) + len(partition.quarantine_records)
        ),
        "effective_development_records": len(partition.development_records),
        "effective_development_groups": len(set(partition.effective_group_ids)),
        "quarantine_records": len(partition.quarantine_records),
        "quarantine_trace_ids": sorted(
            row.trace_id for row in partition.quarantine_records
        ),
        "test_records": len(test),
        "test_groups": len({row.group_id for row in test}),
        "test_trace_ids": sorted(row.trace_id for row in test),
        "question_group_overlap": list(partition.question_group_overlap),
        "visible_history_overlap": list(partition.visible_history_overlap),
        "maximum_content_probability_sum_error": max(
            float(row["content_probability_sum_error"]) for row in predictions
        ),
        "all_selected_steps_belong_to_selected_agents": all(
            bool(row["selected_step_belongs_to_selected_agent"]) for row in predictions
        ),
        "metrics": accuracy_metrics(predictions),
        "metrics_by_source": {
            source: accuracy_metrics(
                tuple(row for row in predictions if row["source"] == source)
            )
            for source in sorted({str(row["source"]) for row in predictions})
        },
        "predictions": predictions,
    }


def run_stability(
    records: Sequence[TraceRecord],
    extraction_cache: Mapping[str, CanonicalExtraction],
    *,
    seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    config: ModelConfig | None = None,
) -> dict[str, object]:
    """Repeat the grouped holdout with equal weight for each split seed."""

    selected_seeds = tuple(seeds)
    if not selected_seeds:
        raise ValueError("at least one split seed is required")
    if len(set(selected_seeds)) != len(selected_seeds) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in selected_seeds
    ):
        raise ValueError("split seeds must be unique non-negative integers")
    reports = [
        evaluate_seed(
            records,
            extraction_cache,
            split_seed=seed,
            config=config,
        )
        for seed in selected_seeds
    ]
    all_predictions = [
        row
        for report in reports
        for row in report["predictions"]  # type: ignore[union-attr]
    ]
    metrics = {
        metric: aggregate_values(
            [float(report["metrics"][metric]) for report in reports]  # type: ignore[index]
        )
        for metric in _ACCURACY_METRICS
    }
    return {
        "method": "Model-based LLM Agent Debugging",
        "protocol": {
            "splitter": "question-grouped 60/20/20 split",
            "split_seeds": list(selected_seeds),
            "backend_fit": "train+calibration after visible-history quarantine",
            "training_group_weight": "1 / effective group size",
            "standard_deviation": "population SD across split-level accuracies",
            "equal_seed_weighting": True,
            "overlap_warning": (
                "test partitions overlap across seeds; mean and SD measure split "
                "sensitivity, not independent trials"
            ),
        },
        "dataset_records": len(records),
        "seeds": {"values": list(selected_seeds), "count": len(selected_seeds)},
        "metrics": metrics,
        "micro_metrics_over_overlapping_appearances": accuracy_metrics(all_predictions),
        "test_partition_reuse": _test_partition_reuse(reports),
        "leakage_and_readout_audit": {
            "all_question_group_overlaps_zero": all(
                not report["question_group_overlap"] for report in reports
            ),
            "all_visible_history_overlaps_zero": all(
                not report["visible_history_overlap"] for report in reports
            ),
            "all_content_probabilities_normalized": all(
                float(report["maximum_content_probability_sum_error"]) <= 1e-12
                for report in reports
            ),
            "all_selected_steps_belong_to_selected_agents": all(
                bool(report["all_selected_steps_belong_to_selected_agents"])
                for report in reports
            ),
        },
        "per_seed": reports,
    }


def accuracy_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    """Compute WHO, WHEN, joint, and step-tolerance accuracies."""

    count = len(rows)
    if not count:
        return {
            "count": 0,
            "when_correct": 0,
            "who_correct": 0,
            "joint_correct": 0,
            **{metric: 0.0 for metric in _ACCURACY_METRICS},
            "mean_absolute_step_error": 0.0,
        }
    return {
        "count": count,
        "when_correct": sum(bool(row["when_correct"]) for row in rows),
        "who_correct": sum(bool(row["who_correct"]) for row in rows),
        "joint_correct": sum(bool(row["joint_correct"]) for row in rows),
        "when_accuracy": sum(bool(row["when_correct"]) for row in rows) / count,
        "who_accuracy": sum(bool(row["who_correct"]) for row in rows) / count,
        "joint_accuracy": sum(bool(row["joint_correct"]) for row in rows) / count,
        **{
            f"within_{radius}_accuracy": sum(
                bool(row[f"within_{radius}"]) for row in rows
            )
            / count
            for radius in (1, 2, 3, 5)
        },
        "mean_absolute_step_error": math.fsum(
            float(row["absolute_step_error"]) for row in rows
        )
        / count,
    }


def aggregate_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot aggregate an empty sequence")
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _aligned_extraction(
    record: TraceRecord,
    cache: Mapping[str, CanonicalExtraction],
) -> CanonicalExtraction:
    try:
        extraction = cache[record.trace_id]
    except KeyError as error:
        raise KeyError(f"missing extraction for {record.trace_id!r}") from error
    if extraction.trace_id != record.trace_id:
        raise ValueError(f"cache trace ID mismatch for {record.trace_id!r}")
    if len(extraction.triples) != len(record.history):
        raise ValueError(f"cache step count mismatch for {record.trace_id!r}")
    speakers = tuple(row["speaker"] for row in visible_history(record))
    if tuple(row.agent for row in extraction.triples) != speakers:
        raise ValueError(f"cache speaker alignment mismatch for {record.trace_id!r}")
    return extraction


def _test_partition_reuse(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    appearances: dict[str, int] = {}
    for report in reports:
        for trace_id in report["test_trace_ids"]:  # type: ignore[union-attr]
            key = str(trace_id)
            appearances[key] = appearances.get(key, 0) + 1
    return {
        "total_test_predictions": sum(appearances.values()),
        "unique_test_records": len(appearances),
        "minimum_test_appearances_per_record": min(appearances.values()),
        "maximum_test_appearances_per_record": max(appearances.values()),
        "mean_test_appearances_per_record": statistics.fmean(appearances.values()),
    }


def _load_json_files(paths: Sequence[Path]) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    for path in sorted(paths):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"dataset record root must be an object: {path}")
        records.append(_record_from_mapping(raw, path.parent.name, path.stem, path))
    return records


def _record_from_mapping(
    raw: Mapping[str, object],
    source: str,
    local_id: str,
    path: Path,
) -> TraceRecord:
    required = {"history", "question", "question_ID", "mistake_agent", "mistake_step"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"dataset record is missing {sorted(missing)}: {path}")
    history_value = raw["history"]
    if hasattr(history_value, "tolist"):
        history_value = history_value.tolist()  # type: ignore[union-attr]
    if not isinstance(history_value, (list, tuple)):
        raise TypeError(f"history must be a sequence: {path}")
    history: list[dict[str, object]] = []
    for step, item in enumerate(history_value):
        if not isinstance(item, Mapping):
            raise TypeError(f"history step {step} must be an object: {path}")
        history.append(dict(item))
    return TraceRecord(
        trace_id=f"{source}/{local_id}",
        source=source,
        group_id=str(raw["question_ID"]),
        question=str(raw["question"]),
        history=tuple(history),
        gold_step=int(raw["mistake_step"]),
        gold_agent=str(raw["mistake_agent"]),
        path=path,
    )
