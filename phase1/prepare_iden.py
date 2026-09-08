import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

SOURCES = {
    "id": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "ind_Latn",
        "license": "ODC-By-1.0",
    },
    "en": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "license": "ODC-By-1.0",
    },
}


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_bucket(text: str) -> str:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if value < 980:
        return "train"
    if value < 990:
        return "eval"
    return "test"


def token_chunks(tokenizer, text: str, chunk_tokens: int):
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    for start in range(0, len(token_ids), chunk_tokens):
        chunk = token_ids[start : start + chunk_tokens]
        if len(chunk) < max(64, chunk_tokens // 4):
            continue
        decoded = tokenizer.decode(chunk, skip_special_tokens=True).strip()
        if decoded:
            yield decoded, len(chunk)


def stream_language(lang, tokenizer, train_budget, output_dir, chunk_tokens, seed, shuffle_buffer):
    source = SOURCES[lang]
    stream = load_dataset(
        source["dataset"],
        source["config"],
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)

    files = {
        split: open(output_dir / f"{split}_{lang}.jsonl", "w", encoding="utf-8")
        for split in ("train", "eval", "test")
    }
    seen = set()
    stats = defaultdict(int)

    try:
        for row in stream:
            text = normalize_text(str(row.get("text", "")))
            if len(text) < 200:
                continue
            for chunk_text, tokens in token_chunks(tokenizer, text, chunk_tokens):
                digest = hashlib.sha256(chunk_text.encode("utf-8")).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                split = split_bucket(chunk_text)
                json.dump({"text": chunk_text}, files[split], ensure_ascii=False)
                files[split].write("\n")
                stats[f"{split}_chunks"] += 1
                stats[f"{split}_tokens"] += tokens
                if stats["train_tokens"] >= train_budget:
                    return dict(stats)
    finally:
        for handle in files.values():
            handle.close()

    return dict(stats)


def merge_train(output_dir: Path, seed: int):
    # Interleave deterministically without loading the whole corpus into memory.
    paths = {lang: output_dir / f"train_{lang}.jsonl" for lang in ("id", "en")}
    handles = {lang: open(path, "r", encoding="utf-8") for lang, path in paths.items()}
    output = output_dir / "pretrain_iden_train.jsonl"
    counters = {"id": 0, "en": 0}
    # 7:3 cycle preserves the intended bilingual mixture at sample level;
    # token budgets are enforced separately before this merge.
    cycle = ["id"] * 7 + ["en"] * 3
    offset = seed % len(cycle)
    cycle = cycle[offset:] + cycle[:offset]

    try:
        with open(output, "w", encoding="utf-8") as out:
            exhausted = set()
            while len(exhausted) < 2:
                progressed = False
                for lang in cycle:
                    if lang in exhausted:
                        continue
                    line = handles[lang].readline()
                    if not line:
                        exhausted.add(lang)
                        continue
                    out.write(line)
                    counters[lang] += 1
                    progressed = True
                if not progressed:
                    break
    finally:
        for handle in handles.values():
            handle.close()
    return counters


def main():
    parser = argparse.ArgumentParser(description="Build RockSoul Mind Indonesian-English continued-pretraining corpus")
    parser.add_argument("--target-train-tokens", type=int, default=20_000_000)
    parser.add_argument("--id-ratio", type=float, default=0.70)
    parser.add_argument("--chunk-tokens", type=int, default=360)
    parser.add_argument("--tokenizer", default="model")
    parser.add_argument("--output-dir", default="dataset/rocksoul_iden")
    parser.add_argument("--seed", type=int, default=163)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    args = parser.parse_args()

    if not 0.0 < args.id_ratio < 1.0:
        raise SystemExit("--id-ratio must be between 0 and 1")
    if args.target_train_tokens < 100_000:
        raise SystemExit("--target-train-tokens is too small")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    budgets = {
        "id": int(args.target_train_tokens * args.id_ratio),
        "en": args.target_train_tokens - int(args.target_train_tokens * args.id_ratio),
    }
    stats = {}
    for index, lang in enumerate(("id", "en")):
        stats[lang] = stream_language(
            lang,
            tokenizer,
            budgets[lang],
            output_dir,
            args.chunk_tokens,
            args.seed + index,
            args.shuffle_buffer,
        )

    merged = merge_train(output_dir, args.seed)
    manifest = {
        "schema_version": 1,
        "target_train_tokens": args.target_train_tokens,
        "target_ratio": {"id": args.id_ratio, "en": 1.0 - args.id_ratio},
        "chunk_tokens": args.chunk_tokens,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "sources": SOURCES,
        "language_stats": stats,
        "merged_train_samples": merged,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
