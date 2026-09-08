import argparse
import json
import os
import platform
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def build_prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return text


def generate(model, tokenizer, prompt, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
    }
    if inputs["attention_mask"] is None:
        inputs.pop("attention_mask")

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    generated = max(output.shape[-1] - encoded["input_ids"].shape[-1], 0)
    return elapsed, generated


def main():
    parser = argparse.ArgumentParser(description="RockSoul Mind CPU inference benchmark")
    parser.add_argument("--model", default="jingyaogong/minimind-3")
    parser.add_argument("--prompt", default="Jelaskan secara singkat apa itu jaringan komputer.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=max(os.cpu_count() or 1, 1))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    torch.set_num_threads(max(args.threads, 1))
    before_load = rss_mb()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval().to("cpu")
    load_seconds = time.perf_counter() - load_started
    after_load = rss_mb()

    prompt = build_prompt(tokenizer, args.prompt)
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_tokens = int(encoded["input_ids"].shape[-1])

    generate(model, tokenizer, prompt, 1)
    ttft_seconds, _ = generate(model, tokenizer, prompt, 1)
    run_seconds, generated_tokens = generate(model, tokenizer, prompt, args.max_new_tokens)
    final_rss = rss_mb()

    parameter_count = sum(p.numel() for p in model.parameters())
    result = {
        "model": args.model,
        "device": "cpu",
        "dtype": "float32",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "parameter_count": parameter_count,
        "parameter_millions": round(parameter_count / 1_000_000, 3),
        "prompt_tokens": prompt_tokens,
        "requested_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "load_seconds": round(load_seconds, 4),
        "ttft_seconds": round(ttft_seconds, 4),
        "generation_seconds": round(run_seconds, 4),
        "tokens_per_second": round(generated_tokens / run_seconds, 4) if run_seconds else 0.0,
        "rss_before_load_mb": round(before_load, 2),
        "rss_after_load_mb": round(after_load, 2),
        "rss_final_mb": round(final_rss, 2),
        "rss_model_delta_mb": round(after_load - before_load, 2),
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
