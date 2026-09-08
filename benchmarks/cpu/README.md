# CPU Benchmarks

This directory stores reproducible RockSoul Mind CPU baseline evidence.

## GitHub-hosted reference baseline

`github-actions-baseline.json` was produced by the `phase0-model-smoke` workflow on run `34214001131`, head SHA `6179212331659253b7e8f8845f175ab27025ab6e`.

The run successfully:

- installed the pinned CPU runtime
- downloaded and loaded `jingyaogong/minimind-3`
- ran deterministic CPU generation
- started the RockSoul OpenAI-compatible API
- returned HTTP 200 from `POST /v1/chat/completions`

Reference result with 2 PyTorch CPU threads:

- parameters: 63.912M
- model load: 3.9872 s
- approximate TTFT: 0.0549 s
- generation: 45.5754 tokens/s for the 8-token smoke run
- RSS after model load: 583.61 MB
- final RSS: 596.80 MB
- incremental RSS across model load: 295.48 MB

These numbers are **reference evidence only**. GitHub-hosted runner hardware is not the production target, and short 8-token throughput should not be treated as a final capacity benchmark.

## Production baseline

Production/server results must be stored separately and include:

- CPU model
- physical/logical core count
- total RAM
- OS/kernel
- Python/PyTorch/runtime versions
- thread count
- model/checkpoint identifier
- context size
- prompt token count
- generated token count
- load time
- TTFT
- tokens/sec
- RSS/peak RSS

Do not overwrite the reference baseline with production numbers.
