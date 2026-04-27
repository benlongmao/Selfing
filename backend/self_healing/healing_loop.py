#!/usr/bin/env python3
"""
Self-healing loop orchestrator (detect → diagnose → generate fix → approval-gated apply).

[2026-01-29] Created.
"""

import time
import logging
import numpy as np
from typing import Dict, Optional
from datetime import datetime

from .code_reviewer import CodeReviewer
from .diagnosis_engine import DiagnosisEngine, Issue
from .auto_fixer import AutoFixer

logger = logging.getLogger(__name__)


class SelfHealingSystem:
    """
    End-to-end loop:
    1. Detect (``CodeReviewer``)
    2. Diagnose (``DiagnosisEngine``)
    3. Generate patch (``AutoFixer``)
    4. Apply (currently blocked on approval)
    5. Verify when applied
    6. Optional ``z_self`` nudge + diary note on success
    """

    def __init__(self, project_root: str = None):
        """Args: ``project_root`` — repo root; defaults to ``PROJECT_ROOT``."""
        if project_root is None:
            from backend.project_paths import PROJECT_ROOT

            project_root = PROJECT_ROOT

        self.project_root = project_root

        self.code_reviewer = CodeReviewer(project_root)
        self.diagnosis_engine = DiagnosisEngine()
        self.auto_fixer = AutoFixer()

        self.stats = {
            "total_scans": 0,
            "issues_detected": 0,
            "issues_diagnosed": 0,
            "fixes_generated": 0,
            "fixes_applied": 0,
            "fixes_verified": 0,
            "fixes_failed": 0,
        }

        logger.info("SelfHealingSystem initialized")

    def healing_loop(
        self, max_fixes: int = 5, session_id: Optional[str] = None, self_model=None
    ) -> Dict:
        """
        Args:
            max_fixes: Cap how many issues to process this pass.
            session_id: When provided with ``self_model``, bumps achievement slice lightly.
            self_model: Optional ``SelfModel`` for vector persistence.

        Returns:
            Summary dict with counts and per-issue detail rows.
        """
        logger.info("=" * 60)
        logger.info("Starting Self-Healing Loop")
        logger.info("=" * 60)

        self.stats["total_scans"] += 1
        start_time = time.time()

        result = {
            "success": False,
            "issues_found": 0,
            "fixes_attempted": 0,
            "fixes_successful": 0,
            "fixes_failed": 0,
            "details": [],
        }

        try:
            logger.info("Phase 1: Detecting issues...")
            issues = self.code_reviewer.detect_issues()

            if not issues:
                logger.info("No issues detected")
                result["success"] = True
                return result

            logger.info(f"Found {len(issues)} issues")
            result["issues_found"] = len(issues)
            self.stats["issues_detected"] += len(issues)

            for i, issue in enumerate(issues[:max_fixes]):
                logger.info(
                    f"\n--- Processing issue {i + 1}/{min(len(issues), max_fixes)} ---"
                )
                logger.info(f"File: {issue.file_path}")
                logger.info(f"Description: {issue.description}")

                issue_result = self._process_issue(issue)
                result["details"].append(issue_result)

                if issue_result["fixed"]:
                    result["fixes_successful"] += 1
                else:
                    result["fixes_failed"] += 1

            result["fixes_attempted"] = min(len(issues), max_fixes)
            result["success"] = result["fixes_successful"] > 0

        except Exception as e:
            logger.error(f"Healing loop error: {e}", exc_info=True)
            result["error"] = str(e)

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info(f"Self-Healing Loop complete ({elapsed:.1f}s)")
        logger.info(f"   Issues found: {result['issues_found']}")
        logger.info(f"   Fixes attempted: {result['fixes_attempted']}")
        logger.info(f"   Fixes successful: {result['fixes_successful']}")
        logger.info(f"   Fixes failed: {result['fixes_failed']}")
        logger.info("=" * 60)

        if result["fixes_successful"] > 0:
            self._record_growth(result, session_id=session_id, self_model=self_model)

        return result

    def _process_issue(self, issue: Issue) -> Dict:
        issue_result = {
            "issue": issue.description,
            "file": issue.file_path,
            "diagnosed": False,
            "fix_generated": False,
            "fixed": False,
            "verified": False,
            "error": None,
        }

        try:
            logger.info("Phase 2: Diagnosing issue...")
            diagnosis = self.diagnosis_engine.diagnose(issue)
            self.stats["issues_diagnosed"] += 1
            issue_result["diagnosed"] = True

            logger.info(f"   Problem type: {diagnosis.problem_type}")
            logger.info(f"   Root cause: {diagnosis.root_cause[:100]}...")
            logger.info(f"   Fixable: {diagnosis.is_fixable}")
            logger.info(f"   Confidence: {diagnosis.confidence:.2f}")

            if not diagnosis.is_fixable or diagnosis.confidence < 0.5:
                logger.warning("Issue is not auto-fixable or confidence too low")
                issue_result["error"] = "Not auto-fixable"
                return issue_result

            logger.info("Phase 3: Generating fix...")
            fix = self.auto_fixer.generate_fix(diagnosis)

            if not fix:
                logger.error("Failed to generate fix")
                self.stats["fixes_failed"] += 1
                issue_result["error"] = "Fix generation failed"
                return issue_result

            self.stats["fixes_generated"] += 1
            issue_result["fix_generated"] = True
            logger.info("Fix generated")

            logger.info("Phase 4: Applying fix...")
            success = self.auto_fixer.apply_fix(fix, require_approval=True)

            if success:
                self.stats["fixes_applied"] += 1
                self.stats["fixes_verified"] += 1
                issue_result["fixed"] = True
                issue_result["verified"] = fix.verified
                logger.info("Fix applied and verified")
            else:
                self.stats["fixes_failed"] += 1
                issue_result["error"] = "Fix application failed"
                logger.error("Fix application failed")

        except Exception as e:
            logger.error(f"Error processing issue: {e}", exc_info=True)
            issue_result["error"] = str(e)
            self.stats["fixes_failed"] += 1

        return issue_result

    def _record_growth(self, result: Dict, session_id: Optional[str] = None, self_model=None):
        """Light ``z_self`` bump + English diary stub under sandbox diaries."""
        try:
            if session_id and self_model:
                try:
                    z_self = self_model.get_z_self(session_id)
                    if z_self is None:
                        logger.warning("z_self is None; skipping vector update")
                    elif z_self.shape[0] >= 128:
                        from backend.self_model import MOTIVATION_SUBSPACE_DIMS

                        if "achievement" in MOTIVATION_SUBSPACE_DIMS:
                            a0, a1 = MOTIVATION_SUBSPACE_DIMS["achievement"]
                            if a1 <= z_self.shape[0]:
                                inc = 0.1 * result["fixes_successful"]
                                z_self[a0:a1] = np.clip(z_self[a0:a1] + inc, 0.0, 1.0)
                        self_model._save_z_self(session_id, z_self)
                        logger.info(
                            f"Updated z_self (128-d): achievement +{0.1 * result['fixes_successful']:.3f}"
                        )
                    else:
                        logger.warning(
                            f"z_self too small (got {z_self.shape[0]}, need >= 128); skipping update"
                        )
                except Exception as e:
                    logger.warning(f"Failed to update z_self: {e}")
            else:
                logger.debug("session_id or self_model missing; skipping z_self update")

            diary_entry = f"""
## Self-healing run

I ran an internal scan and repair pass:
- Potential issues flagged: {result["issues_found"]}
- Successful fixes (post-approval path): {result["fixes_successful"]}

This is deliberate self-maintenance: detect, diagnose, propose, and only mutate disk
after human approval in ``self_healing_requests``.

Growth hint: +{0.05 * result["fixes_successful"]:.2f} (narrative only; vector bump logged above)

[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]
"""

            diary_file = (
                f"{self.project_root}/workspace/sandbox/diaries/"
                f"self_healing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            with open(diary_file, "w", encoding="utf-8") as f:
                f.write(diary_entry)

            logger.info(f"Growth recorded to diary: {diary_file}")

        except Exception as e:
            logger.error(f"Failed to record growth: {e}")

    def get_stats(self) -> Dict:
        return self.stats.copy()

    def review_file(self, file_path: str) -> Dict:
        """Run ``review_specific_file`` and wrap as a small JSON-friendly dict."""
        logger.info(f"Reviewing file: {file_path}")

        issues = self.code_reviewer.review_specific_file(file_path)

        if not issues:
            return {"success": True, "issues_found": 0, "message": "No issues found"}

        return {
            "success": True,
            "issues_found": len(issues),
            "issues": [
                {
                    "description": issue.description,
                    "line": issue.line_number,
                    "severity": issue.severity,
                }
                for issue in issues
            ],
        }


def run_self_healing(project_root: str = None, max_fixes: int = 5) -> Dict:
    """Convenience wrapper around ``SelfHealingSystem.healing_loop``."""
    system = SelfHealingSystem(project_root)
    return system.healing_loop(max_fixes)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = run_self_healing()
    print("\n" + "=" * 60)
    print("Self-Healing Loop Result:")
    print(f"  Success: {result['success']}")
    print(f"  Issues Found: {result['issues_found']}")
    print(f"  Fixes Successful: {result['fixes_successful']}")
    print("=" * 60)
