# Contributing

Thanks for helping improve this repository.

## Workflow

1. Open an issue for larger changes (especially anything touching `z_self` sampling paths, L0/L1 rules, or memory schemas).
2. Keep pull requests focused: one logical change per PR is easier to review.
3. Run Python checks locally:

```bash
cd /path/to/s-main
export PYTHONPATH=.
python3 -m py_compile backend/user_fact_capture.py backend/unified_memory.py
python3 -m unittest backend.test_memory_helpers -v
python3 scripts/memory_eval_smoke.py
```

## Areas that welcome work

- Reliability and tests around chat, memory, and reflection pipelines.
- Additional LLM provider adapters (OpenAI-compatible, Anthropic, etc.).
- Documentation and operational runbooks.

## Language note

This fork targets an **English-first** runtime (default embedder: `BAAI/bge-small-en-v1.5`). If you change prompt-facing strings or stored persona text, keep them aligned with the embedder language to avoid weak retrieval.

Policy and scope: [`docs/LOCALE_EN.md`](docs/LOCALE_EN.md). After large UI merges, run `python3 scripts/apply_locale_en_index.py` and re-scan `frontend/index.html` for any remaining CJK in user-visible strings.
