import argparse
import json
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description="Smoke-test RockSoul Mind OpenAI-compatible API")
    parser.add_argument("--url", default="http://127.0.0.1:8998/v1/chat/completions")
    parser.add_argument("--model", default="rocksoul-mind")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Balas dengan kata OK."}],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 8,
        "stream": False,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"API smoke test failed: {exc}") from exc

    choices = data.get("choices") or []
    if not choices or "message" not in choices[0]:
        raise SystemExit(f"Invalid OpenAI-compatible response: {json.dumps(data, ensure_ascii=False)}")

    print(json.dumps({"status": "ok", "response": data}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
