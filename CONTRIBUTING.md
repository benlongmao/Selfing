# Contributing

Thanks for helping improve Selfing. This project is **experimental** and English-first in `docs/` and the default UI.

## Issues and discussion

- Use **GitHub Issues** on [`benlongmao/Selfing`](https://github.com/benlongmao/Selfing) for bugs, design questions, and reproducible behavior reports.
- Include: OS, Python version, relevant `config/settings.yaml` fields (no secrets), and steps to reproduce when reporting bugs.

## Pull requests

- Prefer **small, focused** changes with a clear motivation.
- Do **not** commit:
  - **Secrets**: `.env`, API keys, tokens.
  - **Runtime databases**: `*.db`, `*.db-shm`, `*.db-wal` (see `.gitignore`).
  - **Instance-private trees**: `workspace/sandbox/` contents, private diaries, logs, or exports from a live instance unless the PR is explicitly about anonymized fixtures.
- Match existing code style and keep user-facing strings consistent with the English-first line unless the change is localization.

## Security

See [`SECURITY.md`](SECURITY.md) for responsible disclosure.
