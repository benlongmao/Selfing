# Claude + Aiberm tool-call compatibility notes

## Symptoms

- Plain chat works.
- As soon as **tool calls** or heartbeat / scheduler follow-up turns run, intermittent or steady **`400`** responses appear; the UI may show a generic **API request format error** toast (older locale builds may still show a non-English label).
- Often the first model turn succeeds; the next hop inside the tool loop fails.

## Root cause (summary)

This is primarily **request-body validation failures** in the OpenAI Chat Completions compatibility layer when mapping to Claude—not “the model got worse.”

Common triggers:

1. Multiple **`system`** messages (especially after tool loops or compression inject extra system lines).
2. **`assistant`** messages with `tool_calls` where `content` is empty or rejected by the gateway.
3. Last message is **`tool`**, while the compat layer expects a **`user`**-style continuation context at the end.
4. `tool_calls.function.arguments` is not a **parseable JSON string** (type drift or half-structured text).
5. Duplicate tool names in definitions, or incomplete JSON Schema (strict gateways reject).

## Fixes implemented (code)

### 1) Message normalization

- **File:** `backend/chat_message_builder.py`
- **Highlights:**
  - Merge duplicate system: `collapse_duplicate_system_messages`
  - Normalize roles: `normalize_openai_compatible_messages`
  - Force `tool_calls.arguments` to valid JSON strings; fall back to `"{}"` on parse errors.

### 2) Anthropic OpenAI-compat shim (via vLLM / Aiberm)

- **File:** `backend/chat_message_builder.py`
- **Highlights:**
  - `anthropic_vllm_openai_message_shim`
  - Handles `assistant` + `tool_calls` + empty `content`.
  - If the tail is `tool`, append a protocol-style **`user`** placeholder to reduce gateway rejections.

### 3) Request build + tool definition hardening

- **File:** `backend/chat_service.py`
- **Highlights:**
  - On vLLM paths, omit sampling fields incompatible with the configured gateway.
  - `sanitize_openai_tools_for_strict_providers`: fill `parameters.required`, normalize `type` / `properties`, dedupe tool names.

### 4) Tool loop robustness

- **File:** `backend/chat_tool_runner.py`
- **Highlights:**
  - Broader exception handling around tool argument parsing so type errors do not kill the loop.
  - Tool-budget notices use **`user`**-role protocol messages instead of injecting mid-loop **`system`**.

## Debugging and regression

### Debug flag (redacted)

- **Config:** `models.vllm.debug_payload_shape`
- **Effect:** logs a **shape summary** of the payload (role sequence, tool names, argument types)—not message bodies.
- **Usage:** enable while diagnosing `400`s; disable in steady state to reduce log noise.

### Regression checklist

1. Chat with **no** tools (baseline).
2. Single tool, **no** arguments (e.g. `get_current_time`).
3. Single tool **with** arguments (e.g. `read_file`).
4. **Three or more** tools chained in one session.
5. Tool loop interleaved with heartbeat / scheduler system messages.
6. Long-context compression, then continue tool calls.
7. Run once with `anthropic/...` and once with a non-Anthropic model id.

**Pass criteria:**

- No `400` “API request format error” responses from the gateway.
- Tool chains progress.
- Logs show `tool_calls.arguments` as parseable JSON strings.
- Role ordering does not show broken tail shapes.

## Runtime recommendations

- Prefer keeping:
  - `omit_sampling_for_anthropic_vllm: true`
  - `omit_openai_penalties: true`
  - `anthropic_openai_compat_shim: true`
- On new `400`s, turn on `debug_payload_shape`, capture `Error detail` and `PAYLOAD-SHAPE-ON-ERROR`, then narrow down the offending message or tool definition.
