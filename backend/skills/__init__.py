"""
Hot-pluggable Skills system for s-main (nanobot-style).

Each skill is a directory with ``SKILL.md`` (YAML front matter + Markdown instructions).

[2026-02-07] Phase 3 — skills system
"""

from backend.skills.loader import SkillsLoader

__all__ = ["SkillsLoader"]
