# Phase 1 — RockSoul ID+EN Language Adaptation

Phase 1 adapts the MiniMind-3 dense 64M pretraining checkpoint toward an Indonesian-heavy bilingual foundation while preserving useful English technical comprehension.

## Strategy

Do **continued pretraining**, not training from zero.

- Base checkpoint: `jingyaogong/minimind-3-pytorch/pretrain_768.pth`
- Base revision: `6e6b53a5361ab41b6376f93e9f0749d5eff3110e`
- Verified SHA256: `0191958e7a96ef7cd1e443d379dc2ebea90cfea72cc4450fbcb4a7b87ff945f6`
- Architecture: unchanged dense MiniMind-3, hidden size 768, 8 layers
- Tokenizer: unchanged MiniMind tokenizer
- Initial language target: 70% Indonesian / 30% English

The tokenizer is deliberately kept unchanged so language adaptation can be measured independently from architecture/tokenizer changes.

## Corpus sources

### Indonesian

- Dataset: `HuggingFaceFW/fineweb-2`
- Config: `ind_Latn`
- License: ODC-By 1.0
- Source data is filtered/deduplicated CommonCrawl-derived text.

### English

- Dataset: `HuggingFaceFW/fineweb-edu`
- Config: `sample-10BT`
- License: ODC-By 1.0
- Selected as the English component to retain strong general/technical/educational text quality without letting English dominate the adaptation corpus.

Dataset use must retain the required attribution and comply with the applicable CommonCrawl terms.

## Dataset stages

The builder streams data instead of downloading the complete corpora.

It performs:

1. source-specific streaming
2. deterministic buffered shuffle
3. basic text normalization
4. minimum-document filtering
5. token-aware chunking
6. exact SHA256 deduplication per language stream
7. deterministic hash split into train/eval/test
8. independent token budgets for Indonesian and English
9. deterministic 7:3 merge for the training file
10. manifest generation

Default pilot size is 20M training tokens. This is intentionally a pilot, not the final language budget.

Expected output:

```text
dataset/rocksoul_iden/
├── pretrain_iden_train.jsonl
├── train_id.jsonl
├── train_en.jsonl
├── eval_id.jsonl
├── eval_en.jsonl
├── test_id.jsonl
├── test_en.jsonl
└── manifest.json
```

## Prepare environment

From repository root:

```bash
pip install -r requirements.txt
pip install -r phase1/requirements.txt
```

## Download verified base checkpoint

```bash
python phase1/download_base.py
```

This writes the verified base weight to:

```text
out/pretrain_768.pth
```

## Build 20M-token pilot corpus

```bash
python phase1/prepare_iden.py \
  --target-train-tokens 20000000 \
  --id-ratio 0.70
```

For a larger promotion candidate, increase only the token budget, for example:

```bash
python phase1/prepare_iden.py \
  --target-train-tokens 100000000 \
  --id-ratio 0.70
```

Do not change the language ratio, tokenizer, architecture and training hyperparameters in the same experiment.

## Train

The default launcher is tuned as a conservative single-GPU continued-pretraining pilot:

```bash
bash phase1/train_iden.sh
```

Defaults:

- 1 epoch
- batch size 16
- gradient accumulation 8
- sequence length 384
- learning rate `1e-4`
- float16
- seed 163
- output weight `out/rocksoul_iden_pretrain_768.pth`

All settings can be overridden with environment variables.

Example:

```bash
BATCH_SIZE=8 ACCUMULATION_STEPS=16 bash phase1/train_iden.sh
```

## Evaluate language adaptation

```bash
python phase1/eval_iden.py
```

The evaluator compares the untouched base checkpoint against the adapted checkpoint on separate Indonesian and English holdouts and writes:

```text
phase1/results/iden_eval.json
```

Primary signals:

- Indonesian loss/perplexity should materially improve.
- English loss/perplexity should remain stable enough that technical English comprehension is not catastrophically forgotten.

For the first pilot, record the deltas before setting a strict promotion threshold. After at least two reproducible runs, freeze an acceptance threshold rather than selecting one retroactively.

## Phase 1 exit criteria

Phase 1 is complete when:

- corpus manifest records the exact source/config/ratio/seed
- base checkpoint hash is verified
- continued-pretraining run completes
- adapted checkpoint is versioned
- Indonesian and English holdout results are recorded
- Indonesian quality improves without unacceptable English regression
- no domain-specific ERP/ISP/tool-call corpus has been mixed into this phase

After this gate, proceed to Phase 2 instruction/structured-output tuning.
