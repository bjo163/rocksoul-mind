# RockSoul Tool-Call Contract v0

RockSoul Mind uses OpenAI-style function definitions externally while preserving MiniMind's native `<tool_call>` training format internally.

## Native model output

```text
<tool_call>{"name":"subscriber.get","arguments":{"subscriber_code":"T-000029"}}</tool_call>
```

## API-normalized output

```json
{
  "id": "call_<generated-id>",
  "type": "function",
  "function": {
    "name": "subscriber.get",
    "arguments": "{\"subscriber_code\":\"T-000029\"}"
  }
}
```

## Rules

- `name` must be a registered tool name.
- `arguments` must be a JSON object.
- Do not invent optional arguments when the user did not provide them and the tool can operate without them.
- Required missing arguments should result in clarification rather than fabricated values.
- Tool execution is always performed by programmatic runtime code, never by the model itself.
- External web, browser, database and network side effects require the runtime to enforce authorization, validation, timeout and rate limits.

## Initial naming convention

Tool names use lowercase dotted namespaces:

```text
web.search
web.fetch
browser.open
browser.extract
scraper.extract
subscriber.get
radius.session.get
router.ping
```

## Evaluation dimensions

Every tool-calling checkpoint will be scored independently on:

1. valid tool-call syntax
2. correct tool selection
3. valid JSON arguments
4. argument extraction accuracy
5. unsupported-tool hallucination rate
6. unnecessary-tool-call rate
7. missing-required-argument behavior

Phase 0 only validates parsing and API normalization. Accuracy targets belong to the later tool-calling training phase.
