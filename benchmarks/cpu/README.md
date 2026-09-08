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

A second full smoke run produced 38.2343 tokens/s for the same short 8-token benchmark, demonstrating that such tiny runs are sensitive to hosted-runner variance. Treat these values as smoke/reference evidence, not production throughput claims.

## GitHub-hosted context-memory reference

`github-actions-context-baseline.json` was produced by workflow run `34214186582`, head SHA `b3868d12236623ebebdc2bf67f9801d7a0281322`. Every context size was measured in a fresh child process.

| Context tokens | RSS after load | RSS after generation | Peak RSS | One-token generation |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 546.64 MB | 566.47 MB | 668.04 MB | 0.1341 s |
| 512 | 546.66 MB | 608.59 MB | 667.94 MB | 0.5005 s |
| 1024 | 546.66 MB | 618.10 MB | 698.46 MB | 1.2001 s |
| 2048 | 546.81 MB | 662.90 MB | 992.18 MB | 3.0062 s |

The 2048-token result shows why context length must be benchmarked separately from model-weight memory: the model remains small, but attention/cache/runtime allocations can push peak process memory close to 1 GB in FP32 PyTorch.

These numbers are **reference evidence only**. GitHub-hosted runner hardware is not the production target.

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

Use `runtime/cpu/system_info.py` to capture target-server metadata. Do not overwrite the reference baselines with production numbers.
