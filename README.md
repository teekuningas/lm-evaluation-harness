# Finnish LM Evaluation Harness

Fork of [LumiOpen/lm-evaluation-harness](https://github.com/LumiOpen/lm-evaluation-harness) adapted to evaluate both non-thinking and thinking models via OpenAI-compatible APIs (`/v1/chat/completions`).

## Quick Start

```bash
# Quick test: FIN-bench v1, 3 tasks, 5 samples each (15 total)
./run_benchmark.sh gpt-oss-120b

# Full FIN-bench v1: All 12 tasks with at max 15 subtasks for each
LIMIT=15 BENCHMARK=v1 SUBSET=full ./run_benchmark.sh gpt-oss-120b

# Compare results across models
./aggregate_results.sh
```

Results saved in `./results/` directory.

## Upstreams

- **LumiOpen/lm-evaluation-harness**: Source of Finnish benchmarks (FIN-bench v1, v2)
- **EleutherAI/lm-evaluation-harness**: Original evaluation framework

