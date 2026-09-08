# RockSoul ID+EN Corpus

This directory is generated locally or on Kaggle by `phase1/prepare_iden.py` and should not contain a manually assembled opaque corpus.

## Sources

### Indonesian

- `HuggingFaceFW/fineweb-2`
- config: `ind_Latn`
- license: ODC-By 1.0
- derived from filtered/deduplicated CommonCrawl data

### English

- `HuggingFaceFW/fineweb-edu`
- config: `sample-10BT`
- license: ODC-By 1.0
- derived from CommonCrawl/FineWeb and filtered for educational quality

Use of these datasets is subject to their dataset licenses and applicable CommonCrawl terms. Retain attribution when redistributing derived data or publishing a model/data card that relies on this corpus.

## Default mix

- 70% Indonesian training-token budget
- 30% English training-token budget
- tokenizer-aware chunks
- deterministic seed `163`
- deterministic train/eval/test hash split

Generated dataset files are intentionally not committed by default. Commit only manifests, evaluation summaries, and small reproducibility metadata unless a separate data-release decision is made.
