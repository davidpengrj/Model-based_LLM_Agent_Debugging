# Archived Explorations

These branches were tested during development but are not part of the current main pipeline.
They are kept here for reproducibility and possible negative-ablation reporting.

## DeLLMa-inspired failure-mode utility

Code:

- `code/dellma.py`
- `code/dellma_evaluation.py`

Artifacts:

- `artifacts/dellma/`
- `artifacts/eval/dellma_without_gt/`

Outcome on hand-crafted split:

- best DeLLMa variant: `dellma_hybrid_argmax`
- agent accuracy: 63.79%
- step accuracy: 17.24%
- tol5 accuracy: 39.66%

This did not beat the stronger semantic-risk baselines or the current ranking decoder.

## Sequence/hazard tuning

Code:

- `code/sequence_evaluation.py`

Artifacts:

- `artifacts/eval/sequence_without_gt/`

Outcome:

- useful as an intermediate experiment for threshold tuning;
- superseded by `llmrepair/ranking_evaluation.py`, which keeps the useful ranking/decoding ideas in the main pipeline.

## Top-k LLM reranking

Code:

- `code/rerank.py`

Artifacts:

- `artifacts/rerank/`

Outcome:

- candidate oracle was high, but DeepSeek reranking did not improve final step attribution;
- expensive relative to the current zero-inference ranking decoder.

## Restore Notes

These files were moved out of the `llmrepair` package, so the archived code is not wired into the current CLI.
To rerun one of these branches, move the corresponding files back into `llmrepair/` and restore the matching CLI commands from version control history.
