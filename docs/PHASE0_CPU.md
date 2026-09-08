# Phase 0 — CPU Baseline

Phase 0 proves the unmodified MiniMind-3 model can be run reproducibly on CPU before any ID+EN or domain training begins.

## Clean local environment

Python 3.10 is the reference runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r runtime/cpu/requirements.txt
```

## Core benchmark

```bash
python runtime/cpu/benchmark.py \
  --model jingyaogong/minimind-3 \
  --max-new-tokens 64 \
  --output benchmarks/cpu/local.json
```

The core benchmark records:

- exact model identifier
- Python and PyTorch versions
- CPU thread count
- parameter count
- prompt and generated token counts
- model load time
- approximate first-token latency
- generation tokens/sec
- process RSS before and after model loading

The first execution downloads the model from Hugging Face and is therefore not representative of steady-state startup if network transfer time dominates. Keep the Hugging Face cache between runs.

## Context-memory benchmark

Run each context size in an isolated child process so allocator state from a previous measurement cannot contaminate the next one:

```bash
python runtime/cpu/context_benchmark.py \
  --model jingyaogong/minimind-3 \
  --contexts 128 512 1024 2048 \
  --output benchmarks/cpu/context-local.json
```

For each context size the result records RSS after model loading, RSS after one-token generation, peak RSS and generation latency.

## OpenAI-compatible API

```bash
python runtime/cpu/serve.py \
  --model jingyaogong/minimind-3 \
  --threads 4
```

Then from another shell:

```bash
python runtime/cpu/smoke_api.py
```

Endpoint:

```text
POST /v1/chat/completions
```

## Docker

From the repository root:

```bash
docker compose -f deployment/cpu/docker-compose.yml up --build
```

The named `rocksoul_hf_cache` volume preserves downloaded model files across container rebuilds.

## Reproducibility rules

1. Record CPU model, core/thread count and total RAM with every benchmark result.
2. Use the same prompt and `max_new_tokens` when comparing runs.
3. Run at least three measured passes after model download and warm-up.
4. Run context-memory measurements in isolated processes.
5. Do not compare PyTorch FP32 results directly with later GGUF quantized results without labeling runtime and quantization.
6. Keep ID+EN/domain training out of Phase 0 results.

## Exit criteria

Phase 0 can close when:

- clean CPU inference succeeds
- API smoke test succeeds
- benchmark result from the target CPU server is committed
- context-memory baseline from the target CPU server is committed
- tool-call parser tests pass
- CPU CI passes
- baseline RAM, TTFT and tokens/sec are known

GGUF/llama.cpp work starts only after this PyTorch CPU baseline is captured.
