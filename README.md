# Model-based LLM Agent Debugging

This repository contains an interpretable method for locating **WHO** caused a
multi-agent task failure and **WHEN** the decisive error occurred. The method
uses one LLM call per trajectory to compile every log step into a canonical
`<Agent, Action, State>` triple. All later stages are explicit frequency counts,
TF-IDF/PCA/GMM state abstraction, and a first-order discrete-time Markov chain
(DTMC). There is no replay, reranking, second LLM judge, or hidden correction
rule.

## Method

For a trajectory with steps `t = 1, ..., L`:

1. **One semantic extraction call.** The complete visible trajectory is sent to
   the LLM once. It returns exactly one source-grounded `<Agent, Action, State>`
   triple for every original step. The prompt never receives the gold error step
   or responsible agent.

2. **Three-channel semantic evidence.** Agent, Action, and State are tokenized
   separately. During training, the gold error step is `E` and every other step
   is `N`. For duplicated task group `g`, each trajectory receives weight
   `w_i = 1 / n_g`. Within a step and channel, this weight is divided equally
   among its unique tokens. Lidstone-smoothed `E` versus `N` log-likelihood
   ratios are averaged inside each channel and then summed across the three
   equally weighted channels.

3. **Action-State relation.** Each canonical Action predicate head is paired
   with the final field name on the left side of every State assignment. For
   example, `propose_code(...)` and `record.observation=not_reported` produce
   `propose_code|observation`. A deterministic suffix anchor also adds the
   coarser operation `code()` to a read-only Action view, producing
   `code|observation`. The original triple is unchanged. A relation score is
   used only when every relation token for that step was seen in training.

4. **Abstract execution states.** The anchored Action and full State use
   independent TF-IDF vocabularies. Their matrices are concatenated, projected
   by one joint PCA, and assigned soft membership over GMM states. Agent identity
   is excluded from this branch.

5. **Frequency DTMC.** Soft GMM memberships estimate the initial distribution
   `pi` and the `K x K` transition matrix `T` using the same group weights. For a
   test step, incoming conformance is
   `c_1 = q_1^T pi` and `c_t = q_(t-1)^T T q_t`; transition surprise is
   `h_t = -log(c_t)`.

6. **WHO, then WHEN.** Let `p_t` be the softmax of semantic plus seen-relation
   evidence. WHO is the agent with the largest total content mass:
   `a* = argmax_a sum_{t: agent_t=a} p_t`. WHEN is selected only among that
   agent's steps: `t* = argmax_{t: agent_t=a*} p_t h_t`.

The frozen configuration is in [`configs/default.json`](configs/default.json).
The implementation keeps every intermediate score in the returned `Prediction`
object so an individual decision can be audited step by step.

## Repository layout

```text
model_based_debugging/
  schema.py       strict trajectory and triple contracts
  extraction.py   one-call semantic compiler and cache
  features.py     tokens, operation anchors, and relation construction
  model.py        semantic counts, TF-IDF/PCA/GMM, DTMC, and readout
  benchmark.py    leakage-safe Who&When split and evaluation protocol
  cli.py          command-line interface
configs/default.json
scripts/run_model_based_debugging.py
tests/
```

The older `llmrepair/`, archived experiments, and their artifacts are retained
only as repository history. They are not imported by the current method.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest
```

## Data and extraction cache

The evaluator expects the Who&When layout below:

```text
DATASET_ROOT/
  Hand-Crafted/*.json
  Algorithm-Generated/*.json
```

If `DATASET_ROOT/Who&When/` exists, the loader detects it automatically. Each
record must contain `history`, `question`, `question_ID`, `mistake_step`, and
`mistake_agent`.

Create canonical triples with one API call per trajectory:

```bash
export DEEPSEEK_API_KEY="<your-key>"
model-based-debugging extract \
  --dataset DATASET_ROOT \
  --cache cache/canonical_v2_1 \
  --workers 3
```

The API key is read only from the environment and is never stored in the cache.
Extraction output is validated for exact step count, order, speaker identity,
and JSON fields before it is accepted.

Evaluate the frozen method over split seeds 0 through 19:

```bash
model-based-debugging evaluate \
  --dataset DATASET_ROOT \
  --cache cache/canonical_v2_1 \
  --seeds 0-19 \
  --output results/who_when_20_splits.local.json
```

## Reported split stability

On 184 Who&When trajectories, using question-grouped 80/20 partitions and
split seeds 0–19:

| Metric | Mean ± population SD |
|---|---:|
| WHEN accuracy | 37.09% ± 5.79% |
| WHO accuracy | 65.95% ± 6.25% |
| Joint WHO & WHEN | 36.15% ± 5.69% |
| Step accuracy within ±1 | 42.47% ± 4.88% |

These 20 test partitions overlap. The mean and standard deviation therefore
measure sensitivity to the grouped split; they are not 20 independent trials.
The machine-readable summary is in
[`results/who_when_20_split_summary.json`](results/who_when_20_split_summary.json).

## Fixed defaults

- semantic smoothing: `0.075`
- relation smoothing: `0.4`
- PCA dimensions: `32`
- diagonal GMM states: `16`
- GMM initializations: `10`
- DTMC smoothing: `0.1`
- abstraction seed: `17`

Hyperparameters can be changed through `ModelConfig` for ablation studies, but
the reported method uses the values above.
