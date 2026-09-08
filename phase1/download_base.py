import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "jingyaogong/minimind-3-pytorch"
FILENAME = "pretrain_768.pth"
EXPECTED_SHA256 = "0191958e7a96ef7cd1e443d379dc2ebea90cfea72cc4450fbcb4a7b87ff945f6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Download verified MiniMind-3 dense pretrain checkpoint")
    parser.add_argument("--output-dir", default="out")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            revision=args.revision,
            local_dir=out_dir,
        )
    )
    actual = sha256(downloaded)
    if actual != EXPECTED_SHA256:
        raise SystemExit(f"Checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
    print(f"Verified: {downloaded}")
    print(f"SHA256: {actual}")


if __name__ == "__main__":
    main()
