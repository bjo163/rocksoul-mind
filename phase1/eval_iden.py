import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_model(weight_path: str, device: str):
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(config)
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def evaluate(model, tokenizer, path: str, device: str, max_seq_len: int, batch_size: int, max_batches: int):
    dataset = PretrainDataset(path, tokenizer, max_length=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    weighted_loss = 0.0
    valid_tokens = 0
    batches = 0

    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            result = model(input_ids, labels=labels)
            count = int((labels != -100).sum().item())
            weighted_loss += float(result.loss.item()) * count
            valid_tokens += count
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break

    loss = weighted_loss / max(valid_tokens, 1)
    return {
        "loss": round(loss, 6),
        "perplexity": round(math.exp(min(loss, 20.0)), 6),
        "valid_tokens": valid_tokens,
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs RockSoul ID-EN continued-pretraining checkpoint")
    parser.add_argument("--base-weight", default="out/pretrain_768.pth")
    parser.add_argument("--adapted-weight", default="out/rocksoul_iden_pretrain_768.pth")
    parser.add_argument("--data-dir", default="dataset/rocksoul_iden")
    parser.add_argument("--tokenizer", default="model")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output", default="phase1/results/iden_eval.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_dir = Path(args.data_dir)
    datasets = {
        "id": str(data_dir / "eval_id.jsonl"),
        "en": str(data_dir / "eval_en.jsonl"),
    }

    results = {}
    for name, weight in (("base", args.base_weight), ("adapted", args.adapted_weight)):
        model = load_model(weight, args.device)
        results[name] = {
            lang: evaluate(model, tokenizer, path, args.device, args.max_seq_len, args.batch_size, args.max_batches)
            for lang, path in datasets.items()
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_id = results["base"]["id"]["perplexity"]
    adapted_id = results["adapted"]["id"]["perplexity"]
    base_en = results["base"]["en"]["perplexity"]
    adapted_en = results["adapted"]["en"]["perplexity"]
    results["delta"] = {
        "id_perplexity_pct": round((adapted_id / base_id - 1.0) * 100.0, 3),
        "en_perplexity_pct": round((adapted_en / base_en - 1.0) * 100.0, 3),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
