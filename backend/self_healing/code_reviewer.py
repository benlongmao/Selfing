#!/usr/bin/env python3
"""
Code reviewer: shallow detection pass (static tool + recent ERROR log lines).

[2026-01-29] Created — complements LLM diagnosis in ``diagnosis_engine``.
"""

import os
import logging
from typing import List
from pathlib import Path

from .diagnosis_engine import Issue

logger = logging.getLogger(__name__)


class CodeReviewer:
    """
    Collect candidate ``Issue`` rows before diagnosis.

    Steps:
    1. Static scan via ``CodeAnalysisTool`` when importable
    2. Tail scan of backend logs for ERROR/CRITICAL lines
    3. Performance hooks reserved (currently empty)
    """

    def __init__(self, project_root: str = None):
        """Args: ``project_root`` — repo root; defaults to ``PROJECT_ROOT``."""
        if project_root is None:
            from backend.project_paths import PROJECT_ROOT

            project_root = PROJECT_ROOT

        self.project_root = Path(project_root)
        self.backend_dir = self.project_root / "backend"
        self.logs_dir = self.project_root / "logs"

        try:
            from backend.tools.code_analysis import CodeAnalysisTool

            self.code_analyzer = CodeAnalysisTool()
            logger.info("CodeReviewer initialized with CodeAnalysisTool")
        except ImportError:
            self.code_analyzer = None
            logger.warning("CodeAnalysisTool not available, using basic analysis")

    def detect_issues(self) -> List[Issue]:
        """Return merged static + log-derived issues (bounded work)."""
        logger.info("Starting code review")

        issues: List[Issue] = []

        issues.extend(self._static_analysis())
        issues.extend(self._analyze_error_logs())
        issues.extend(self._analyze_performance())

        logger.info(f"Code review complete: found {len(issues)} issues")
        return issues

    def _static_analysis(self) -> List[Issue]:
        """Run CodeAnalysisTool on a capped list of ``backend/**/*.py`` files."""
        issues: List[Issue] = []

        if not self.code_analyzer:
            logger.warning("CodeAnalysisTool not available, skipping static analysis")
            return issues

        try:
            python_files = list(self.backend_dir.rglob("*.py"))

            for py_file in python_files[:10]:
                if "__pycache__" in str(py_file):
                    continue

                try:
                    result = self.code_analyzer.analyze_python_file(str(py_file))

                    if hasattr(result, "issues") and result.issues:
                        for issue_data in result.issues:
                            issues.append(
                                Issue(
                                    file_path=str(py_file),
                                    description=issue_data.get("description", "Unknown issue"),
                                    line_number=issue_data.get("line", None),
                                    severity=issue_data.get("severity", "medium"),
                                )
                            )

                except Exception as e:
                    logger.debug(f"Failed to analyze {py_file}: {e}")

        except Exception as e:
            logger.error(f"Static analysis error: {e}", exc_info=True)

        return issues

    def _analyze_error_logs(self) -> List[Issue]:
        """Scan last 1000 lines of known log files for ERROR/CRITICAL."""
        issues: List[Issue] = []

        try:
            log_files = [
                self.logs_dir / "backend.log",
                self.logs_dir / "backend_restart.log",
            ]

            for log_file in log_files:
                if not log_file.exists():
                    continue

                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()[-1000:]

                    for i, line in enumerate(lines):
                        if "ERROR" in line or "CRITICAL" in line:
                            issue = self._parse_error_line(line, lines[i : i + 5])
                            if issue:
                                issues.append(issue)

                except Exception as e:
                    logger.debug(f"Failed to read {log_file}: {e}")

        except Exception as e:
            logger.error(f"Log analysis error: {e}", exc_info=True)

        return issues

    def _parse_error_line(self, error_line: str, context_lines: List[str]) -> Issue:
        """Build an ``Issue`` from a log line + a few following lines."""
        import re

        file_match = re.search(r'File "([^"]+)"', error_line)
        if file_match:
            file_path = file_match.group(1)
        else:
            file_path = "unknown"

        error_context = "".join(context_lines)

        return Issue(
            file_path=file_path,
            description=f"Runtime error: {error_line.strip()}",
            severity="high",
            error_log=error_context[:500],
        )

    def _analyze_performance(self) -> List[Issue]:
        """Placeholder for future slow-query / memory / hot-loop detectors."""
        return []

    def review_specific_file(self, file_path: str) -> List[Issue]:
        """Static review for one path."""
        issues: List[Issue] = []

        if not self.code_analyzer:
            return issues

        try:
            result = self.code_analyzer.analyze_python_file(file_path)

            if hasattr(result, "issues") and result.issues:
                for issue_data in result.issues:
                    issues.append(
                        Issue(
                            file_path=file_path,
                            description=issue_data.get("description", "Unknown issue"),
                            line_number=issue_data.get("line", None),
                            severity=issue_data.get("severity", "medium"),
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to review {file_path}: {e}")

        return issues
