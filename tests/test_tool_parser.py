import json
import unittest

from rocksoul.tool_parser import parse_response


class ToolParserTests(unittest.TestCase):
    def test_plain_text(self):
        content, reasoning, tool_calls = parse_response("halo")
        self.assertEqual(content, "halo")
        self.assertIsNone(reasoning)
        self.assertIsNone(tool_calls)

    def test_reasoning_and_tool_call(self):
        text = (
            "<think>cek kode pelanggan</think>"
            "<tool_call>{\"name\":\"subscriber.get\",\"arguments\":{\"subscriber_code\":\"T-000029\"}}</tool_call>"
        )
        content, reasoning, tool_calls = parse_response(text)
        self.assertEqual(content, "")
        self.assertEqual(reasoning, "cek kode pelanggan")
        self.assertEqual(tool_calls[0]["function"]["name"], "subscriber.get")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"subscriber_code": "T-000029"},
        )

    def test_invalid_tool_json_is_ignored(self):
        content, reasoning, tool_calls = parse_response("before <tool_call>{bad json}</tool_call> after")
        self.assertIn("before", content)
        self.assertIsNone(reasoning)
        self.assertIsNone(tool_calls)


if __name__ == "__main__":
    unittest.main()
