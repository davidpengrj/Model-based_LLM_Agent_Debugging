# Model-based LLM Agent Debugging

This workspace contains a prototype for LLM-based multi-agent failure attribution on the Who&When benchmark.

The current pipeline is:

1. flatten Who&When trajectories into step-level rows;
2. use DeepSeek v4 Flash for local-window semantic profiling of each step;
3. build RNNRepair-style semantic state abstractions per agent;
4. train influence/risk tables with 5-fold log-level cross validation;
5. evaluate agent-level and step-level failure attribution.

Main entry point:

```bash
python scripts/llmrepair_pipeline.py --help
```

Current hand-crafted evaluation outputs are under `artifacts/eval/without_gt/`.
