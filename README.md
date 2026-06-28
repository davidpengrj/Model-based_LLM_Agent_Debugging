# Model-based LLM Agent Debugging

This workspace contains a prototype for LLM-based multi-agent failure attribution on the Who&When benchmark.

The current pipeline is:

1. flatten Who&When trajectories into step-level rows;
2. use DeepSeek v4 Flash for local-window semantic profiling of each step;
3. build RNNRepair-style semantic state abstractions per agent;
4. train influence/risk tables with 5-fold log-level cross validation;
5. evaluate agent-level and step-level failure attribution.

The current ranking branch tests Claude's suggested early-error fixes:

1. tune position-decay scores on each training fold;
2. use a WHO-then-WHEN decoder that first selects the responsible agent, then searches only that agent's steps;
3. train a pairwise per-log ranker from semantic/risk/position features.

Main entry point:

```bash
python scripts/llmrepair_pipeline.py --help
```

Current hand-crafted baseline outputs are under `artifacts/eval/without_gt/`.

Ranking command:

```bash
python scripts/llmrepair_pipeline.py evaluate-ranking
```

Current ranking outputs:

- hand-crafted: `artifacts/eval/ranking_without_gt/`
- algorithm-generated: `artifacts/eval/ranking_alg_generated_without_gt/`
- all logs: `artifacts/eval/ranking_all_without_gt/`

Algorithm-generated split:

```bash
python scripts/llmrepair_pipeline.py prepare-steps \
  --subset Algorithm-Generated \
  --output artifacts/algorithm_generated_steps.jsonl

python scripts/llmrepair_pipeline.py profile-steps \
  --subset Algorithm-Generated \
  --steps artifacts/algorithm_generated_steps.jsonl \
  --output artifacts/profiles/deepseek-v4-flash_alg_generated_without_gt.jsonl

python scripts/llmrepair_pipeline.py evaluate-ranking \
  --steps artifacts/algorithm_generated_steps.jsonl \
  --profiles artifacts/profiles/deepseek-v4-flash_alg_generated_without_gt.jsonl \
  --output-dir artifacts/eval/ranking_alg_generated_without_gt
```

Archived negative explorations are kept under `archive/explorations/`.
