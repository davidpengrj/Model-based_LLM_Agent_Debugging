"""Interpretable frequency, clustering, DTMC, and hierarchical readout."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

from .features import (
    action_document,
    action_state_relation_tokens,
    flatten_triples,
    operation_anchored_extraction,
    semantic_components,
    state_document,
)
from .schema import CanonicalExtraction, CanonicalTriple

NORMAL = 0
ERROR = 1


@dataclass(frozen=True)
class ModelConfig:
    """Frozen configuration used by the reported method."""

    semantic_alpha: float = 0.075
    relation_alpha: float = 0.4
    pca_dimensions: int = 32
    state_count: int = 16
    max_features: int = 8192
    tfidf_min_df: int = 2
    gmm_n_init: int = 10
    gmm_reg_covar: float = 1e-4
    gmm_max_iter: int = 300
    transition_alpha: float = 0.1
    abstraction_seed: int = 17

    def __post_init__(self) -> None:
        positive = {
            "semantic_alpha": self.semantic_alpha,
            "relation_alpha": self.relation_alpha,
            "pca_dimensions": self.pca_dimensions,
            "state_count": self.state_count,
            "max_features": self.max_features,
            "tfidf_min_df": self.tfidf_min_df,
            "gmm_n_init": self.gmm_n_init,
            "gmm_reg_covar": self.gmm_reg_covar,
            "gmm_max_iter": self.gmm_max_iter,
            "transition_alpha": self.transition_alpha,
        }
        if any(isinstance(value, bool) or value <= 0 for value in positive.values()):
            raise ValueError("all model size and smoothing parameters must be positive")
        if self.state_count < 2:
            raise ValueError("state_count must be at least two")


@dataclass(frozen=True)
class SemanticScores:
    values: np.ndarray
    channel_values: dict[str, np.ndarray]
    oov_fraction: float


class SemanticEmission:
    """Grouped E/N likelihood ratios over three equal semantic channels."""

    channels = ("agent_words", "action_words", "state_words")

    def __init__(self, *, alpha: float) -> None:
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        self.alpha = float(alpha)
        self.counts: dict[tuple[int, str], Counter[str]] = {}
        self.totals: Counter[tuple[int, str]] = Counter()
        self.vocabularies: dict[str, frozenset[str]] = {}
        self._fitted = False

    def fit(
        self,
        sequences: Sequence[Sequence[CanonicalTriple]],
        error_steps: Sequence[int],
        *,
        group_ids: Sequence[str],
    ) -> SemanticEmission:
        _validate_training_inputs(sequences, error_steps, group_ids)
        group_sizes = Counter(group_ids)
        counts: defaultdict[tuple[int, str], Counter[str]] = defaultdict(Counter)
        totals: Counter[tuple[int, str]] = Counter()
        vocabulary: defaultdict[str, set[str]] = defaultdict(set)

        for sequence, error_step, group_id in zip(sequences, error_steps, group_ids):
            weight = 1.0 / group_sizes[group_id]
            for step, triple in enumerate(sequence):
                label = ERROR if step == int(error_step) else NORMAL
                for channel, raw_tokens in semantic_components(triple).items():
                    tokens = tuple(dict.fromkeys(raw_tokens))
                    token_weight = weight / len(tokens)
                    vocabulary[channel].update(tokens)
                    for token in tokens:
                        counts[(label, channel)][token] += token_weight
                        totals[(label, channel)] += token_weight

        self.counts = {
            (label, channel): Counter(counts[(label, channel)])
            for label in (NORMAL, ERROR)
            for channel in self.channels
        }
        self.totals = totals
        self.vocabularies = {
            channel: frozenset(vocabulary[channel]) for channel in self.channels
        }
        self._fitted = True
        return self

    def score(self, sequence: Sequence[CanonicalTriple]) -> SemanticScores:
        if not self._fitted:
            raise RuntimeError("fit must be called before score")
        triples = tuple(sequence)
        if not triples:
            raise ValueError("prediction trace must be non-empty")
        by_channel: dict[str, list[float]] = {channel: [] for channel in self.channels}
        token_count = 0
        oov_count = 0
        for triple in triples:
            components = semantic_components(triple)
            for channel in self.channels:
                tokens = tuple(dict.fromkeys(components[channel]))
                ratios = [
                    self._token_log_probability(ERROR, channel, token)
                    - self._token_log_probability(NORMAL, channel, token)
                    for token in tokens
                ]
                by_channel[channel].append(math.fsum(ratios) / len(ratios))
                token_count += len(tokens)
                oov_count += sum(
                    token not in self.vocabularies[channel] for token in tokens
                )
        arrays = {
            channel: np.asarray(values, dtype=np.float64)
            for channel, values in by_channel.items()
        }
        total = np.sum(np.stack(tuple(arrays.values()), axis=0), axis=0)
        return SemanticScores(
            values=total,
            channel_values=arrays,
            oov_fraction=oov_count / token_count,
        )

    def _token_log_probability(self, label: int, channel: str, token: str) -> float:
        vocabulary_size = len(self.vocabularies[channel]) + 1
        return math.log(self.counts[(label, channel)][token] + self.alpha) - math.log(
            self.totals[(label, channel)] + self.alpha * vocabulary_size
        )


class ActionStateRelationEmission:
    """Grouped E/N likelihood for Action-head × State-leaf relation tokens."""

    def __init__(self, *, alpha: float) -> None:
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        self.alpha = float(alpha)
        self.counts = {NORMAL: Counter(), ERROR: Counter()}
        self.totals: Counter[int] = Counter()
        self.vocabulary: frozenset[str] = frozenset()
        self._fitted = False

    def fit(
        self,
        sequences: Sequence[Sequence[CanonicalTriple]],
        error_steps: Sequence[int],
        *,
        group_ids: Sequence[str],
    ) -> ActionStateRelationEmission:
        _validate_training_inputs(sequences, error_steps, group_ids)
        group_sizes = Counter(group_ids)
        vocabulary: set[str] = set()
        for sequence, error_step, group_id in zip(sequences, error_steps, group_ids):
            weight = 1.0 / group_sizes[group_id]
            for step, triple in enumerate(sequence):
                label = ERROR if step == int(error_step) else NORMAL
                tokens = action_state_relation_tokens(triple)
                token_weight = weight / len(tokens)
                vocabulary.update(tokens)
                for token in tokens:
                    self.counts[label][token] += token_weight
                    self.totals[label] += token_weight
        self.vocabulary = frozenset(vocabulary)
        self._fitted = True
        return self

    def score(
        self, sequence: Sequence[CanonicalTriple]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("fit must be called before score")
        vocabulary_size = len(self.vocabulary) + 1
        values: list[float] = []
        seen: list[bool] = []
        for triple in sequence:
            tokens = action_state_relation_tokens(triple)
            class_logs = {}
            for label in (NORMAL, ERROR):
                class_logs[label] = math.fsum(
                    math.log(self.counts[label][token] + self.alpha)
                    - math.log(self.totals[label] + self.alpha * vocabulary_size)
                    for token in tokens
                ) / len(tokens)
            values.append(class_logs[ERROR] - class_logs[NORMAL])
            seen.append(all(token in self.vocabulary for token in tokens))
        return np.asarray(values, dtype=np.float64), np.asarray(seen, dtype=bool)


@dataclass(frozen=True)
class AbstractedTrace:
    responsibilities: np.ndarray
    zero_vector_fraction: float


class ActionStateAbstraction:
    """Separate Action/State TF-IDF, joint PCA, then one diagonal GMM."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.vectorizers = {
            name: TfidfVectorizer(
                lowercase=False,
                token_pattern=r"(?u)\b[\w.<>]+\b",
                min_df=config.tfidf_min_df,
                max_features=config.max_features,
                sublinear_tf=True,
                norm="l2",
                dtype=np.float64,
            )
            for name in ("action", "state")
        }
        self.pca: PCA | None = None
        self.gmm: GaussianMixture | None = None
        self.feature_counts: dict[str, int] = {}
        self._fitted = False

    @property
    def state_count(self) -> int:
        return self.config.state_count

    def fit(self, extractions: Sequence[CanonicalExtraction]) -> ActionStateAbstraction:
        triples = flatten_triples(extractions)
        blocks: list[np.ndarray] = []
        for name in ("action", "state"):
            documents = tuple(_document(name, triple) for triple in triples)
            features = self.vectorizers[name].fit_transform(documents).toarray()
            self.feature_counts[name] = int(features.shape[1])
            blocks.append(features)
        combined = np.concatenate(blocks, axis=1)
        dimensions = min(
            self.config.pca_dimensions,
            combined.shape[0] - 1,
            combined.shape[1],
        )
        if dimensions < 1:
            raise ValueError("Action/State text has insufficient rank for PCA")
        if self.state_count > len(combined):
            raise ValueError("GMM state_count exceeds the number of training steps")
        self.pca = PCA(
            n_components=dimensions,
            svd_solver="randomized",
            random_state=self.config.abstraction_seed,
        )
        projected = self.pca.fit_transform(combined)
        self.gmm = GaussianMixture(
            n_components=self.state_count,
            covariance_type="diag",
            reg_covar=self.config.gmm_reg_covar,
            n_init=self.config.gmm_n_init,
            max_iter=self.config.gmm_max_iter,
            random_state=self.config.abstraction_seed,
        ).fit(projected)
        self._fitted = True
        return self

    def transform(self, extraction: CanonicalExtraction) -> AbstractedTrace:
        if not self._fitted or self.pca is None or self.gmm is None:
            raise RuntimeError("fit must be called before transform")
        blocks: list[np.ndarray] = []
        any_nonzero = np.zeros(len(extraction.triples), dtype=bool)
        for name in ("action", "state"):
            documents = tuple(_document(name, row) for row in extraction.triples)
            tfidf = self.vectorizers[name].transform(documents)
            any_nonzero |= np.asarray(tfidf.getnnz(axis=1)) > 0
            blocks.append(tfidf.toarray())
        projected = self.pca.transform(np.concatenate(blocks, axis=1))
        return AbstractedTrace(
            responsibilities=self.gmm.predict_proba(projected),
            zero_vector_fraction=float(np.mean(~any_nonzero)),
        )


class FrequencyDTMC:
    """First-order DTMC estimated from group-weighted soft state counts."""

    def __init__(self, state_count: int, *, alpha: float) -> None:
        if state_count < 2 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("state_count must be >=2 and alpha must be positive")
        self.state_count = int(state_count)
        self.alpha = float(alpha)
        self.initial_counts = np.empty(0, dtype=np.float64)
        self.transition_counts = np.empty((0, 0), dtype=np.float64)
        self.initial_probability = np.empty(0, dtype=np.float64)
        self.transition_probability = np.empty((0, 0), dtype=np.float64)
        self._fitted = False

    def fit(
        self, sequences: Sequence[np.ndarray], *, group_ids: Sequence[str]
    ) -> FrequencyDTMC:
        if len(sequences) != len(group_ids) or not sequences:
            raise ValueError("one group ID is required for every training trace")
        group_sizes = Counter(group_ids)
        initial = np.zeros(self.state_count, dtype=np.float64)
        transitions = np.zeros((self.state_count, self.state_count), dtype=np.float64)
        for sequence, group_id in zip(sequences, group_ids):
            q = _validate_responsibilities(sequence, self.state_count)
            weight = 1.0 / group_sizes[group_id]
            initial += weight * q[0]
            if len(q) > 1:
                transitions += weight * q[:-1].T @ q[1:]
        self.initial_counts = initial
        self.transition_counts = transitions
        self.initial_probability = (initial + self.alpha) / (
            initial.sum() + self.alpha * self.state_count
        )
        smoothed = transitions + self.alpha
        self.transition_probability = smoothed / smoothed.sum(axis=1, keepdims=True)
        self._fitted = True
        return self

    def score_transition_surprise(
        self, responsibilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("fit must be called before scoring")
        q = _validate_responsibilities(responsibilities, self.state_count)
        conformance = np.empty(len(q), dtype=np.float64)
        conformance[0] = float(q[0] @ self.initial_probability)
        for step in range(1, len(q)):
            conformance[step] = float(
                q[step - 1] @ self.transition_probability @ q[step]
            )
        return conformance, -np.log(conformance)


@dataclass(frozen=True)
class Prediction:
    """Auditable WHO and WHEN output for one trajectory."""

    predicted_agent: str
    predicted_step: int
    agent_masses: dict[str, float]
    semantic_score: np.ndarray
    semantic_channel_scores: dict[str, np.ndarray]
    relation_score: np.ndarray
    relation_seen: np.ndarray
    content_probability: np.ndarray
    state_membership: np.ndarray
    transition_surprise: np.ndarray
    step_evidence: np.ndarray
    semantic_oov_fraction: float
    state_zero_vector_fraction: float


class ModelBasedDebugger:
    """Complete Model-based LLM Agent Debugging estimator."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.semantic = SemanticEmission(alpha=self.config.semantic_alpha)
        self.relation = ActionStateRelationEmission(alpha=self.config.relation_alpha)
        self.abstraction = ActionStateAbstraction(self.config)
        self.dtmc = FrequencyDTMC(
            self.config.state_count, alpha=self.config.transition_alpha
        )
        self._fitted = False

    def fit(
        self,
        extractions: Sequence[CanonicalExtraction],
        error_steps: Sequence[int],
        *,
        group_ids: Sequence[str],
    ) -> ModelBasedDebugger:
        if not (len(extractions) == len(error_steps) == len(group_ids)):
            raise ValueError("training inputs must have equal lengths")
        if not extractions:
            raise ValueError("at least one training trace is required")
        sequences = tuple(item.triples for item in extractions)
        self.semantic.fit(sequences, error_steps, group_ids=group_ids)
        action_views = tuple(
            operation_anchored_extraction(item) for item in extractions
        )
        self.relation.fit(
            tuple(item.triples for item in action_views),
            error_steps,
            group_ids=group_ids,
        )
        self.abstraction.fit(action_views)
        abstracted = tuple(self.abstraction.transform(item) for item in action_views)
        self.dtmc.fit(
            tuple(item.responsibilities for item in abstracted),
            group_ids=group_ids,
        )
        self._fitted = True
        return self

    def predict(self, extraction: CanonicalExtraction) -> Prediction:
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        semantic = self.semantic.score(extraction.triples)
        action_view = operation_anchored_extraction(extraction)
        relation_score, relation_seen = self.relation.score(action_view.triples)
        content_score = semantic.values + np.where(relation_seen, relation_score, 0.0)
        content_probability = softmax(content_score)
        abstracted = self.abstraction.transform(action_view)
        _, surprise = self.dtmc.score_transition_surprise(abstracted.responsibilities)
        predicted_agent, predicted_step, masses, evidence = hierarchical_readout(
            content_probability,
            surprise,
            extraction.triples,
        )
        return Prediction(
            predicted_agent=predicted_agent,
            predicted_step=predicted_step,
            agent_masses=masses,
            semantic_score=semantic.values,
            semantic_channel_scores=semantic.channel_values,
            relation_score=relation_score,
            relation_seen=relation_seen,
            content_probability=content_probability,
            state_membership=abstracted.responsibilities,
            transition_surprise=surprise,
            step_evidence=evidence,
            semantic_oov_fraction=semantic.oov_fraction,
            state_zero_vector_fraction=abstracted.zero_vector_fraction,
        )


def hierarchical_readout(
    content_probability: np.ndarray,
    transition_surprise: np.ndarray,
    triples: Sequence[CanonicalTriple],
) -> tuple[str, int, dict[str, float], np.ndarray]:
    """Select WHO by content mass, then WHEN by content × transition surprise."""

    triples = tuple(triples)
    probability = np.asarray(content_probability, dtype=np.float64)
    surprise = np.asarray(transition_surprise, dtype=np.float64)
    if (
        not triples
        or probability.shape != (len(triples),)
        or surprise.shape != probability.shape
    ):
        raise ValueError("readout inputs must contain one value per step")
    if np.any(probability < 0) or not np.isfinite(probability).all():
        raise ValueError("content probabilities must be finite and non-negative")
    if not math.isclose(math.fsum(probability), 1.0, abs_tol=1e-12):
        raise ValueError("content probabilities must sum to one")
    if np.any(surprise < 0) or not np.isfinite(surprise).all():
        raise ValueError("transition surprise must be finite and non-negative")

    indices: defaultdict[str, list[int]] = defaultdict(list)
    for step, triple in enumerate(triples):
        indices[triple.agent].append(step)
    masses = {
        agent: float(probability[np.asarray(steps, dtype=np.int64)].sum())
        for agent, steps in indices.items()
    }
    predicted_agent = max(masses, key=masses.__getitem__)
    evidence = probability * surprise
    predicted_step = max(indices[predicted_agent], key=evidence.__getitem__)
    return predicted_agent, int(predicted_step), masses, evidence


def softmax(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("softmax requires a non-empty finite vector")
    shifted = scores - scores.max()
    result = np.exp(shifted)
    return result / result.sum()


def _document(name: str, triple: CanonicalTriple) -> str:
    if name == "action":
        return action_document(triple)
    if name == "state":
        return state_document(triple)
    raise ValueError(f"unknown document block: {name!r}")


def _validate_training_inputs(
    sequences: Sequence[Sequence[CanonicalTriple]],
    error_steps: Sequence[int],
    group_ids: Sequence[str],
) -> None:
    if not (len(sequences) == len(error_steps) == len(group_ids)):
        raise ValueError("training inputs must have equal lengths")
    if not sequences:
        raise ValueError("at least one training trace is required")
    for sequence, error_step, group_id in zip(sequences, error_steps, group_ids):
        if not sequence or not 0 <= int(error_step) < len(sequence):
            raise ValueError("each error step must index a non-empty trace")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group IDs must be non-empty strings")


def _validate_responsibilities(values: np.ndarray, state_count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != state_count or not len(result):
        raise ValueError("responsibilities have the wrong shape")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("responsibilities must be finite and non-negative")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("responsibility rows must sum to one")
    return result
