#!/usr/bin/env python3
"""
Diagnosis engine: infer root cause and fix strategy for a detected ``Issue``.

Uses the project ``SimpleLLMClient`` (or injected client) with a structured JSON reply.

[2026-01-29] Created.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    """Single detected problem attached to a file path."""
    file_path: str
    description: str
    line_number: Optional[int] = None
    severity: str = "medium"  # low, medium, high, critical
    error_log: Optional[str] = None
    code_context: Optional[str] = None


@dataclass
class Diagnosis:
    """Structured diagnosis returned to ``AutoFixer``."""
    issue: Issue
    root_cause: str
    problem_type: str  # syntax_error, logic_error, performance_issue, security_vulnerability
    fix_approach: str
    estimated_difficulty: str  # easy, medium, hard
    is_fixable: bool
    confidence: float  # 0.0-1.0


class DiagnosisEngine:
    """LLM-backed diagnosis from code + optional stack/log context."""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional pre-built client; otherwise ``SimpleLLMClient``.
        """
        if llm_client is None:
            from backend.simple_llm_client import SimpleLLMClient
            self.llm = SimpleLLMClient()
        else:
            self.llm = llm_client

        logger.info("DiagnosisEngine initialized")

    def diagnose(self, issue: Issue) -> Diagnosis:
        """Run diagnosis pipeline for ``issue``."""
        logger.info(f"Diagnosing issue in {issue.file_path}")

        try:
            if not issue.code_context:
                issue.code_context = self._read_code_context(issue.file_path)

            prompt = self._build_diagnosis_prompt(issue)

            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500,
            )

            if not response["success"]:
                logger.error("Diagnosis LLM call failed")
                return self._create_fallback_diagnosis(issue)

            diagnosis = self._parse_diagnosis_response(issue, response["content"])

            logger.info(
                f"Diagnosis complete: {diagnosis.problem_type} (confidence: {diagnosis.confidence})"
            )
            return diagnosis

        except Exception as e:
            logger.error(f"Diagnosis error: {e}", exc_info=True)
            return self._create_fallback_diagnosis(issue)

    def _read_code_context(self, file_path: str) -> str:
        """Load full file text for prompting."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""

    def _build_diagnosis_prompt(self, issue: Issue) -> str:
        """Build English-first instructions; JSON keys must match the parser below."""

        error_section = ""
        if issue.error_log:
            error_section = f"""
[Error log]
{issue.error_log}
"""

        line_section = ""
        if issue.line_number:
            line_section = f"(near line {issue.line_number})"

        prompt = f"""You are a senior Python engineer. Diagnose the following defect.

[Description]
{issue.description}

[File] {issue.file_path} {line_section}
```python
{issue.code_context[:3000]}
```
{error_section}

[Tasks]
1. Choose **problem_type**:
   - syntax_error
   - logic_error
   - performance_issue
   - security_vulnerability

2. **root_cause** — why this happens (concise but concrete).

3. **fix_approach** — how to fix it safely.

4. Choose **difficulty**:
   - easy — about 5 minutes
   - medium — about 30 minutes
   - hard — more than one hour

5. **is_fixable** — true or false (whether an automated patch is realistic).

6. **confidence** — float 0.0–1.0 for this diagnosis.

Reply with **JSON only**, no markdown fences, matching exactly:
{{
    "problem_type": "logic_error",
    "root_cause": "...",
    "fix_approach": "...",
    "difficulty": "medium",
    "is_fixable": true,
    "confidence": 0.85
}}
"""
        return prompt

    def _parse_diagnosis_response(self, issue: Issue, response_text: str) -> Diagnosis:
        """Parse JSON blob from model output."""
        import json
        import re

        try:
            json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = self._fallback_parse(response_text)

            return Diagnosis(
                issue=issue,
                root_cause=data.get("root_cause", "Unknown root cause"),
                problem_type=data.get("problem_type", "logic_error"),
                fix_approach=data.get("fix_approach", "Manual review required"),
                estimated_difficulty=data.get("difficulty", "medium"),
                is_fixable=data.get("is_fixable", True),
                confidence=float(data.get("confidence", 0.5)),
            )

        except Exception as e:
            logger.error(f"Failed to parse diagnosis response: {e}")
            return self._create_fallback_diagnosis(issue)

    def _fallback_parse(self, text: str) -> Dict:
        """Heuristic typing when JSON is malformed (bilingual cues)."""
        data = {
            "problem_type": "logic_error",
            "root_cause": "Needs deeper analysis",
            "fix_approach": "Manual review required",
            "difficulty": "medium",
            "is_fixable": True,
            "confidence": 0.3,
        }

        if "语法错误" in text or "syntax" in text.lower():
            data["problem_type"] = "syntax_error"
        elif "性能" in text or "performance" in text.lower():
            data["problem_type"] = "performance_issue"
        elif "安全" in text or "security" in text.lower():
            data["problem_type"] = "security_vulnerability"

        return data

    def _create_fallback_diagnosis(self, issue: Issue) -> Diagnosis:
        """Safe default when the model or parser fails."""
        return Diagnosis(
            issue=issue,
            root_cause="Diagnosis failed; manual inspection required",
            problem_type="logic_error",
            fix_approach="Review the file and logs manually",
            estimated_difficulty="medium",
            is_fixable=False,
            confidence=0.0,
        )
