# RockSoul Mind Roadmap

RockSoul Mind is developed from the MiniMind codebase while keeping the upstream-compatible baseline intact.

## Branch policy

- `master`: upstream-compatible baseline. Keep suitable for syncing with `jingyaogong/minimind`.
- `dev`: RockSoul integration branch.
- `feature/*`: isolated experiments and implementation work.

Current upstream baseline: `6fc918beb68a0d8c40452338df6319fe168014ba`.

## Phase 0 — Foundation & CPU Baseline

Goal: prove the original MiniMind-3 64M runtime on CPU before changing language, tokenizer, architecture, or domain behavior.

### Implementation status

- [x] Keep upstream-compatible `master` and isolate RockSoul work on `dev`.
- [x] Add a dedicated FP32 CPU runtime without modifying MiniMind core inference behavior.
- [x] Add deterministic CPU benchmark for load time, RSS, TTFT and tokens/sec.
- [x] Add isolated context-memory benchmark.
- [x] Add OpenAI-compatible API smoke client.
- [x] Add CPU Dockerfile and Docker Compose deployment.
- [x] Define tool-call contract v0 and parser unit tests.
- [x] Add lightweight CI for syntax, tests and Compose validation.
- [x] Confirm clean dependency installation and MiniMind-3 download/model load on GitHub-hosted CPU runner.
- [ ] Complete real-model inference + API smoke after compatibility fixes.
- [ ] Capture benchmark results on the target production CPU server.
- [ ] Capture context-memory results on the target production CPU server.

### Required work

- Reproduce clean CPU inference.
- Record hardware and software metadata.
- Measure process RSS/RAM, model load time, first-token latency, generation tokens/sec, and context-memory growth.
- Validate the existing OpenAI-compatible API path.
- Add deterministic smoke tests for inference and structured output.
- Define a stable tool-call schema.
- Establish a CPU-oriented export/runtime path after PyTorch baseline measurements are captured.

### Definition of done

- CPU inference works from a clean environment.
- Baseline results are reproducible and checked into the repository.
- OpenAI-compatible API works locally.
- Tool-call contract is documented and validated by tests.
- No ID/EN or domain training starts before the baseline is recorded.

## Phase 1 — ID+EN Adaptation

Target Indonesian-heavy bilingual foundation while retaining English technical comprehension.

Initial target mix:

- 70% Indonesian
- 30% English

Evaluate tokenizer efficiency, perplexity/loss, general instruction retention, technical English retention, and catastrophic forgetting.

## Phase 2 — Instruction & Structured Output

Train and evaluate:

- Indonesian instruction following
- concise English instruction following
- JSON/structured output
- extraction
- classification
- deterministic tool arguments

## Phase 3 — Tool Calling, Web & Scraping Controller

RockSoul Mind acts as planner/router; programmatic tools perform external work.

Initial tool families:

- `web.search`
- `web.fetch`
- `browser.open`
- `browser.extract`
- `scraper.extract`

Scraping/browser execution stays programmatic. The model selects tools and arguments rather than parsing arbitrarily large raw HTML directly.

## Phase 4 — Domain Specialization

Candidate domains:

- ERP / Odoo
- ISP operations
- RADIUS
- MikroTik / routing
- FTTH / OLT
- PostgreSQL / API operations

Domain models should descend from the validated ID+EN checkpoint rather than replace the bilingual foundation.

## Training strategy

Production target is CPU-only. GPU is treated as an intermittent training resource, not a deployment requirement.

Recommended lifecycle:

1. Develop and evaluate locally on CPU.
2. Prepare deterministic datasets and training configs.
3. Use temporary/free GPU infrastructure for SFT/LoRA when needed.
4. Publish/version checkpoints.
5. Export/quantize for CPU deployment.
6. Re-run the same benchmark suite before promotion.

## Guardrails

- Do not enlarge the model before baseline evidence shows it is necessary.
- Do not mix language adaptation, domain specialization, tool calling, and architecture changes in one experiment.
- Keep benchmarks deterministic and compare exact checkpoint/config hashes.
- Prefer programmatic tools for browsing/scraping/database/network actions; use the LLM for planning, selection, extraction, and bounded reasoning.
- Preserve upstream license and attribution requirements.
