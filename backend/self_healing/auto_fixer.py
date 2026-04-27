#!/usr/bin/env python3
"""
Auto-fixer: LLM-generated full-file patches with backup + compile check + approval gate.

[2026-01-29] Created.
"""

import os
import shutil
import logging
from typing import Optional
from dataclasses import dataclass

from .diagnosis_engine import Diagnosis

logger = logging.getLogger(__name__)


@dataclass
class Fix:
    """One proposed patch bundle."""
    diagnosis: Diagnosis
    original_code: str
    fixed_code: str
    backup_path: Optional[str] = None
    applied: bool = False
    verified: bool = False


class AutoFixer:
    """Generate Python file rewrites from ``Diagnosis`` and apply when approved."""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional client; otherwise ``SimpleLLMClient``.
        """
        if llm_client is None:
            from backend.simple_llm_client import SimpleLLMClient

            self.llm = SimpleLLMClient()
        else:
            self.llm = llm_client

        logger.info("AutoFixer initialized")

    def generate_fix(self, diagnosis: Diagnosis) -> Optional[Fix]:
        """Return a ``Fix`` or ``None`` when generation/validation fails."""
        if not diagnosis.is_fixable:
            logger.warning(f"Issue is not auto-fixable: {diagnosis.issue.file_path}")
            return None

        logger.info(f"Generating fix for {diagnosis.issue.file_path}")

        try:
            original_code = self._read_file(diagnosis.issue.file_path)
            if not original_code:
                logger.error(f"Failed to read file: {diagnosis.issue.file_path}")
                return None

            prompt = self._build_fix_prompt(diagnosis, original_code)

            response = self.llm.call(
                prompt=prompt,
                temperature=0.2,
                max_tokens=3000,
            )

            if not response["success"]:
                logger.error("Fix LLM call failed")
                return None

            fixed_code = self._extract_code(response["content"])

            if not self._basic_validation(original_code, fixed_code):
                logger.error("Fixed code failed basic validation")
                return None

            fix = Fix(
                diagnosis=diagnosis,
                original_code=original_code,
                fixed_code=fixed_code,
            )

            logger.info("Fix generated successfully")
            return fix

        except Exception as e:
            logger.error(f"Fix generation error: {e}", exc_info=True)
            return None

    def apply_fix(self, fix: Fix, require_approval: bool = True) -> bool:
        """
        Backup, write, verify. When ``require_approval`` is True, writes go through
        ``_request_approval`` which currently **always** records a DB row and returns False
        so no disk write happens without external approval workflow.
        """
        file_path = fix.diagnosis.issue.file_path
        logger.info(f"Applying fix to {file_path}")

        try:
            if require_approval:
                approved = self._request_approval(fix)
                if not approved:
                    logger.info("Fix rejected or pending approval")
                    return False

            backup_path = self._backup_file(file_path)
            if not backup_path:
                logger.error("Failed to backup file")
                return False
            fix.backup_path = backup_path

            if not self._write_file(file_path, fix.fixed_code):
                logger.error("Failed to write fixed code")
                self._rollback(file_path, backup_path)
                return False

            fix.applied = True

            if self._verify_fix(fix):
                fix.verified = True
                logger.info(f"Fix applied and verified: {file_path}")
                return True

            logger.warning("Fix verification failed, rolling back")
            self._rollback(file_path, backup_path)
            return False

        except Exception as e:
            logger.error(f"Apply fix error: {e}", exc_info=True)
            if fix.backup_path:
                self._rollback(file_path, fix.backup_path)
            return False

    def _read_file(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""

    def _write_file(self, file_path: str, content: str) -> bool:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write {file_path}: {e}")
            return False

    def _backup_file(self, file_path: str) -> Optional[str]:
        try:
            backup_path = f"{file_path}.backup_self_healing"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up to {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to backup {file_path}: {e}")
            return None

    def _rollback(self, file_path: str, backup_path: str) -> bool:
        try:
            shutil.copy2(backup_path, file_path)
            logger.info(f"Rolled back {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback {file_path}: {e}")
            return False

    def _build_fix_prompt(self, diagnosis: Diagnosis, original_code: str) -> str:
        return f"""You are a senior Python maintainer applying a minimal surgical fix.

[Diagnosis]
problem_type: {diagnosis.problem_type}
root_cause: {diagnosis.root_cause}
fix_approach: {diagnosis.fix_approach}

[Original file] {diagnosis.issue.file_path}
```python
{original_code[:4000]}
```

[Task]
Output the **complete fixed Python file** only (no markdown fences, no prose).

Rules:
1. Match existing style (indentation, naming).
2. Fix only what is required — no drive-by refactors.
3. Add a short end-of-line comment on touched lines, e.g. ``# [self-healing YYYY-MM-DD] reason``.
4. The file must remain syntactically valid.
5. Preserve imports and public defs unless the diagnosis explicitly requires changes.
"""

    def _extract_code(self, response_text: str) -> str:
        import re

        code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
        if code_match:
            return code_match.group(1)

        lines = response_text.strip().split("\n")
        code_lines = [
            line for line in lines if not line.startswith("#") or "import" in line
        ]

        if code_lines:
            return "\n".join(code_lines)

        return response_text

    def _basic_validation(self, original_code: str, fixed_code: str) -> bool:
        if not fixed_code or len(fixed_code) < 10:
            logger.error("Fixed code is too short")
            return False

        if original_code == fixed_code:
            logger.warning("Fixed code is identical to original")
            return False

        try:
            compile(fixed_code, "<string>", "exec")
        except SyntaxError as e:
            logger.error(f"Fixed code has syntax error: {e}")
            return False

        original_imports = len(
            [l for l in original_code.split("\n") if l.strip().startswith("import")]
        )
        fixed_imports = len(
            [l for l in fixed_code.split("\n") if l.strip().startswith("import")]
        )

        if original_imports > 0 and fixed_imports == 0:
            logger.error("Fixed code lost all imports")
            return False

        return True

    def _verify_fix(self, fix: Fix) -> bool:
        """Re-compile on disk for ``.py`` targets."""
        file_path = fix.diagnosis.issue.file_path

        if file_path.endswith(".py"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                compile(code, file_path, "exec")
                logger.info("Fixed code compiles successfully")
                return True
            except Exception as e:
                logger.error(f"Fixed code failed to compile: {e}")
                return False

        return True

    def _request_approval(self, fix: Fix) -> bool:
        """
        Record intent in ``self_healing_requests`` and **deny** immediate apply.

        [2026-02-22] Wired to approval table; [2026-03-13] all difficulty levels stay pending
        until an operator approves out-of-band.
        """
        import sqlite3
        import json
        from datetime import datetime

        file_path = fix.diagnosis.issue.file_path
        difficulty = fix.diagnosis.estimated_difficulty

        fix_summary = {
            "file": file_path,
            "issue": fix.diagnosis.issue.description,
            "problem_type": fix.diagnosis.problem_type,
            "difficulty": difficulty,
            "confidence": fix.diagnosis.confidence,
            "fix_approach": fix.diagnosis.fix_approach,
            "lines_changed": abs(
                len(fix.fixed_code.split("\n")) - len(fix.original_code.split("\n"))
            ),
            "timestamp": datetime.now().isoformat(),
        }

        try:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data.db")
            db_path = os.path.abspath(db_path)

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS self_healing_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL,
                        difficulty TEXT NOT NULL,
                        fix_summary TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        approved_by TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        reviewed_at TEXT
                    )
                """
                )

                conn.execute(
                    """
                    INSERT INTO self_healing_requests (file_path, difficulty, fix_summary, status)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        file_path,
                        difficulty,
                        json.dumps(fix_summary, ensure_ascii=False),
                        "pending",
                    ),
                )
                conn.commit()

                logger.info(
                    f"[SELF-HEAL] Recorded fix request for {file_path} (difficulty: {difficulty})"
                )
        except Exception as e:
            logger.warning(f"Failed to record fix request: {e}")

        logger.warning(
            f"[SELF-HEAL] Fix requires approval (difficulty: {difficulty}) file={file_path} "
            f"issue={fix.diagnosis.issue.description}"
        )
        logger.warning("[SELF-HEAL] Inspect self_healing_requests table to approve")
        return False
