import json
import re
import time


def parse_response(text: str):
    reasoning_content = None
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ""

    tool_calls = []
    for i, match in enumerate(re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)):
        try:
            call = json.loads(match.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        tool_calls.append(
            {
                "id": f"call_{int(time.time())}_{i}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                },
            }
        )

    if tool_calls:
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)

    return text.strip(), reasoning_content, tool_calls or None
