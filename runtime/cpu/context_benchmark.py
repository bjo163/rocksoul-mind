import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 ** 2)
    return value / 1024


def safe_token_id(tokenizer):
    special = set(tokenizer.all_special_ids)
    for token_id in range(tokenizer.vocab_size):
        if token_id not in special:
            return token_id
    return 0


def run_single(args):
    torch.set_num_threads(max(args.threads, 1))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval().to("cpu")

    loaded_rss = rss_mb()
    token_id = safe_token_id(tokenizer)
    input_ids = torch.full((1, args.single_context), token_id, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    started = time.perf_counter()
    with torch.inference_mode():
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started

    result = {
        "context_tokens": args.single_context,
        "rss_after_load_mb": round(loaded_rss, 2),
        "rss_after_generation_mb": round(rss_mb(), 2),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "generation_seconds": round(elapsed, 4),
    }
    print(json.dumps(result, ensure_ascii=False))


def run_parent(args):
    rows = []
    script = os.path.abspath(__file__)
    for context in args.contexts:
        command = [
            sys.executable,
            script,
            "--model",
            args.model,
            "--threads",
            str(args.threads),
            "--single-context",
            str(context),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        rows.append(json.loads(lines[-1]))

    result = {
        "model": args.model,
        "device": "cpu",
        "dtype": "float32",
        "contexts": rows,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


def main():
    parser = argparse.ArgumentParser(description="RockSoul Mind CPU context-memory benchmark")
    parser.add_argument("--model", default="jingyaogong/minimind-3")
    parser.add_argument("--threads", type=int, default=max(os.cpu_count() or 1, 1))
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--single-context", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.single_context > 0:
        run_single(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
