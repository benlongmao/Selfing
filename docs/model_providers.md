# Model provider configuration

> This expands the “supported model providers” section from [README](../README.md).

The LLM stack supports four integration styles, selected with `system.model_provider` in `config/settings.yaml` (overridden by `MODEL_PROVIDER` in `.env` when set).

---

## DeepSeek API (default)

```bash
# .env
DEEPSEEK_API_KEY=sk-...
MODEL_PROVIDER=deepseek_api
```

```yaml
# config/settings.yaml
system:
  model_provider: deepseek_api
models:
  deepseek:
    model_id: deepseek-reasoner      # reasoning model (main chat)
    model_id_lite: deepseek-chat     # lighter model (side tasks)
```

> Main chat defaults to `deepseek-reasoner` (chain-of-thought style). In that mode, sampling parameters such as `temperature` are **not** sent to the API (provider restriction); `z_self` and prompt text steer behavior instead.

---

## Claude / Anthropic API

```bash
# .env
CLAUDE_API_KEY=sk-ant-...
MODEL_PROVIDER=claude_api
```

```yaml
# config/settings.yaml
system:
  model_provider: claude_api
models:
  claude:
    model_id: claude-opus-4-5        # main chat model
    model_id_lite: claude-haiku-4-5  # lighter side tasks
    thinking_enabled: false          # enable Extended Thinking
    thinking_budget_tokens: 10000    # budget when thinking_enabled=true
```

> Anthropic’s native API (`/v1/messages`) is **not** OpenAI-compatible. The runtime uses `backend/claude_adapter.py` for bidirectional conversion: system prompt extraction, `tool_use` shape mapping, and `thinking` → `reasoning_content` where applicable.  
> Extended Thinking requires a model that supports it (e.g. `claude-3-7-sonnet-20250219` or newer thinking-capable IDs).

---

## OpenAI API (GPT family)

Internally the stack speaks **OpenAI-compatible** `/v1/chat/completions`, so GPT endpoints can be wired through the **`vllm`** provider without protocol changes:

```bash
# .env
VLLM_BASE_URL=https://api.openai.com/v1
VLLM_API_KEY=sk-...
MODEL_ID=gpt-4o
MODEL_PROVIDER=vllm
```

> OpenAI responses omit `reasoning_content` (empty string is fine). There is no DeepSeek-style “thinking mode” branch; steering uses `temperature` / `top_p` and prompt-side state.

---

## Local vLLM (or any OpenAI-compatible server)

```bash
# .env
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=      # empty is OK for local
MODEL_ID=your-model-name
MODEL_PROVIDER=vllm
```

> Works with any service that implements `/v1/chat/completions`: vLLM, Ollama in OpenAI mode, LM Studio, self-hosted OpenRouter-style gateways, etc.
