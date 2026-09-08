import argparse
import os
import sys
from pathlib import Path

import torch
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.serve_openai_api as upstream_api


def main():
    parser = argparse.ArgumentParser(description="RockSoul Mind OpenAI-compatible CPU server")
    parser.add_argument("--model", default="jingyaogong/minimind-3")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--threads", type=int, default=max(os.cpu_count() or 1, 1))
    args = parser.parse_args()

    torch.set_num_threads(max(args.threads, 1))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval().to("cpu")

    upstream_api.device = "cpu"
    upstream_api.model = model
    upstream_api.tokenizer = tokenizer

    parameters = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f"RockSoul Mind CPU ready: {parameters:.2f}M parameters, threads={torch.get_num_threads()}")
    uvicorn.run(upstream_api.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
