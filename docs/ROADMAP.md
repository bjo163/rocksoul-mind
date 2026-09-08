# RockSoul Mind Roadmap

RockSoul Mind is developed from the MiniMind codebase while keeping the upstream-compatible baseline intact.

## Branch policy

- `master`: upstream-compatible baseline. Keep suitable for syncing with `jingyaogong/minimind`.
- `dev`: RockSoul integration branch.
- `feature/*`: isolated experiments and implementation work.

Current upstream baseline: `6fc918beb68a0d8c40452338df6319fe168014ba`.

## Phase 0 — Foundation & CPU Baseline

Status: **complete enough to proceed**. Target-server benchmarking is now optional rather than a gate.

Completed:

- upstream-compatible `master` and isolated `dev`
- dedicated CPU runtime
- OpenAI-compatible API smoke path
- Docker/Compose CPU deployment
- tool-call contract v0 and parser tests
- deterministic CPU benchmark utilities
- GitHub-hosted real-model CPU inference evidence
- GitHub-hosted context-memory evidence
- target-server benchmark tooling retained for optional later use

The project proceeds without requiring a production CPU benchmark before language training.

## Phase 1 — ID+EN Adaptation

Status: **active**.

Goal: create an Indonesian-heavy bilingual foundation while retaining useful English technical comprehension.

### Language target

- 70% Indonesian
- 30% English

### Strategy

- continued pretraining from the released MiniMind-3 dense `pretrain_768.pth`
- do not train from zero
- keep architecture unchanged
- keep tokenizer unchanged
- do not mix ERP/ISP/domain/tool-call data yet

### Data

- Indonesian: `HuggingFaceFW/fineweb-2`, config `ind_Latn`
- English: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`
- token-aware chunking
- deterministic train/eval/test split
- 20M-token pilot first
- scale to 100M+ tokens only after the pilot is evaluated

### Implementation status

- [x] Define ID+EN corpus sources and attribution.
- [x] Add deterministic streaming corpus builder.
- [x] Add verified/pinned base checkpoint downloader.
- [x] Add single-GPU continued-pretraining launcher.
- [x] Add bilingual base-vs-adapted perplexity evaluation.
- [x] Add Kaggle training notebook.
- [ ] Build the 20M-token pilot corpus on GPU training environment.
- [ ] Run continued pretraining.
- [ ] Record Indonesian/English holdout deltas.
- [ ] Promote or revise the language mix based on evidence.
- [ ] Version the accepted `rocksoul_iden_pretrain_768.pth` checkpoint.

Detailed procedure: `docs/PHASE1_IDEN.md`.

## Phase 2 — Instruction & Structured Output

After Phase 1 checkpoint promotion, train and evaluate:

- Indonesian instruction following
- concise English instruction following
- JSON/structured output
- extraction
- classification
- deterministic tool arguments
- tool-selection behavior

The instruction dataset should descend from the accepted ID+EN checkpoint, not from the original MiniMind checkpoint.

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

Domain models should descend from the validated ID+EN + instruction checkpoint rather than replace the bilingual foundation.

## Training strategy

Production target is CPU-only. GPU is an intermittent training resource, not a deployment requirement.

Lifecycle:

1. Build deterministic datasets/configs.
2. Train using temporary/free GPU infrastructure.
3. Evaluate exact checkpoint against the previous stage.
4. Version accepted checkpoints and manifests.
5. Export/quantize for CPU deployment after model behavior is accepted.

## Guardrails

- Do not enlarge the model before evidence shows 64M is insufficient.
- Do not change tokenizer during Phase 1.
- Do not mix language adaptation, domain specialization, tool calling, and architecture changes in one experiment.
- Keep dataset sources, ratios, seeds, checkpoint hashes and evaluation results reproducible.
- Prefer programmatic tools for browsing/scraping/database/network actions; use the LLM for planning, selection, extraction and bounded reasoning.
- Preserve upstream license and attribution requirements.
