# Finnish LM Evaluation Harness

Fork of [LumiOpen/lm-evaluation-harness](https://github.com/LumiOpen/lm-evaluation-harness) adapted to evaluate both non-thinking and thinking models via OpenAI-compatible APIs (`/v1/chat/completions`).

## Some results

[https://github.com/teekuningas/lm-evaluation-harness/wiki/test-results-for-local-models](https://github.com/teekuningas/lm-evaluation-harness/wiki/test-results-for-local-models)

## To run

```bash
BENCHMARK=v2 LIMIT=15 ./run_benchmark.sh gpt-oss-120b

# Compare results across models
./aggregate_results.sh
```

Results saved in `./results/` directory.

## Upstreams

- **LumiOpen/lm-evaluation-harness**: Source of Finnish benchmarks (FIN-bench v1, v2)
- **EleutherAI/lm-evaluation-harness**: Original evaluation framework

