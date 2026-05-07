# Security

Self-becoming is intended for **self-hosted, trusted-operator** use. High-risk capabilities (shell execution, optional in-repo evolution tools, network egress) are documented in [`docs/security-notes.md`](docs/security-notes.md).

## Reporting a vulnerability

Please **do not** open a public issue for security-sensitive reports.

- Use [GitHub Security Advisories](https://github.com/benlongmao/Self-becoming/security/advisories/new) for this repository if available, or
- Contact the maintainers through a **private** channel they publish on the project or their profile.

Include enough detail to reproduce or reason about impact (configuration, version/commit, and topology). We will treat good-faith reports seriously.

## Scope notes

- Misconfiguration of `execute_bash` / project shell tools and exposed unauthenticated APIs are **deployment** risks; still report if you believe defaults or docs should harden the project.
- See [`docs/security-notes.md`](docs/security-notes.md) for capability overview and hygiene.
