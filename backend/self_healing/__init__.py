"""
Self-healing loop for the agent (detect, diagnose, propose/fix code issues).

Modules:
- ``healing_loop.py`` — orchestration
- ``code_reviewer.py`` — lightweight detection (static + log tail)
- ``auto_fixer.py`` — LLM-generated patches (approval-gated)
- ``diagnosis_engine.py`` — root-cause analysis via configured LLM client

[2026-01-29] Initial self-healing package.
"""

from .healing_loop import SelfHealingSystem
from .code_reviewer import CodeReviewer
from .auto_fixer import AutoFixer
from .diagnosis_engine import DiagnosisEngine

__all__ = [
    "SelfHealingSystem",
    "CodeReviewer",
    "AutoFixer",
    "DiagnosisEngine",
]
