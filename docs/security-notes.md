# Security and permissions (short guide)

This project assumes **self-hosted, trusted-operator** deployments. If you expose the API to the internet, put it behind authentication, a reverse proxy, and a minimal attack surface.

## Capabilities that should be treated as high risk

- **`execute_bash` / `execute_bash_project`**: run shell in a constrained workspace; misconfiguration can destroy the repo or data.
- **`agent_evolution` and related tools** (e.g. `evolution_*`, in-repo Git): can modify code and create commits—enable only when needed and keep `agent_evolution` in `config/settings.yaml` under control.
- **Browser automation, email, outbound HTTP**: depend on secrets and network egress; scope API keys narrowly and restrict outbound traffic.

## Configuration hygiene

- **`system.autonomy_gate_enabled`**: lets users or the assistant pause background autonomy—useful when behavior looks wrong; pause before deep debugging. Its state defaults to `run/autonomy_gate.json`, a local runtime file ignored by git.
- **Never commit `.env` or production databases** to public repos; protect `/api/*` with auth or keep it on a private network.

## Models and secrets

- Inject API keys via environment variables or local config only; do not embed them in static assets readable from the browser.

(Full threat modeling depends on your topology—extend this document for your environment.)
