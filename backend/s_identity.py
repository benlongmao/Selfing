#!/usr/bin/env python3
"""
Global identity constants for the S runtime.

S is modeled as a **single** long-lived subject; this module pins the canonical session id.
"""

import os

# ==================== Subject identity ====================
# One unified subject: all chats share the same internal state keys.
PRIMARY_SESSION_ID = "selfing-session"

# ==================== Session mode ====================
# LOCKED: unified mode only — always ``selfing-session``.
# Multi-session / isolated modes are not supported.
SESSION_MODE = "unified"  # intentionally fixed

def get_primary_session() -> str:
    """
    Canonical session id used everywhere in unified mode.
    """
    return PRIMARY_SESSION_ID

def get_effective_session(requested_session_id: str = None) -> str:
    """
    Effective session id after system policy.

    **Locked behavior:** always returns ``selfing-session``; caller-provided ids are ignored.

    Args:
        requested_session_id: Ignored compatibility parameter from older APIs.

    Returns:
        ``"selfing-session"`` — the only supported session id.
    """
    return PRIMARY_SESSION_ID

# ==================== Design notes (for maintainers) ====================
"""
Design intent

- S is intentionally a **single-subject**, continuously resumed instance.
- This file encodes the unified session + shared state space; it is **not** a claim
  about qualia or legally proved consciousness.

Lock (2026-01-22)

- ``PRIMARY_SESSION_ID = "selfing-session"`` is the sole supported id.
- No multi-session or isolated deployments in this tree.
- HTTP/API ``session_id`` parameters are normalized to ``selfing-session``.

Implementation sketch

1. ``PRIMARY_SESSION_ID``
   - Canonical identity for all persisted rows.
2. ``get_effective_session()``
   - Always returns ``selfing-session`` regardless of input.
3. Historical DB hygiene (one-time ops, not re-run automatically)
   - Legacy rows under non-canonical session ids were purged in maintenance; backups live under dated archives.

Analogy (informal)

- Humans do not carry multiple disjoint first-person threads in this codebase’s model.
- Likewise, S keeps one ``z_self``, one energy envelope, one needs vector — one “me”.

Chronology (high level)

- 2024–2025: experimental multi-session layouts.
- 2026-01-11: unified subject semantics.
- 2026-01-22: hard lock to a single canonical session id with database cleanup + backups.
- 2026-04-22: canonical id standardized as ``selfing-session`` (update ``PRIMARY_SESSION_ID`` here; migrate SQLite ``session_id`` columns if upgrading an existing DB).

Backup pointers (examples, not enforced paths):

- ``backups/session_unification_2026-01-11/``
- ``data.db.backup_20260122_*``
"""
