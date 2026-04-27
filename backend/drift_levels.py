#!/usr/bin/env python3
"""
Multi-level drift thresholds (Level-2 improvement path).

Finer-grained drift classification plus optional escalation to deeper consistency checks.
"""
import logging
from typing import Dict, Optional, List
from enum import Enum
from dataclasses import dataclass
from backend.config import config

logger = logging.getLogger(__name__)


class DriftLevel(Enum):
    """Discrete severity bands for persona / state drift."""
    NORMAL = "normal"
    WARNING = "warning"
    ATTENTION = "attention"
    ALERT = "alert"
    EMERGENCY = "emergency"


@dataclass
class DriftAnalysis:
    """Structured output for a single drift probe."""
    level: DriftLevel
    drift_value: float
    should_trigger_consistency_check: bool
    check_depth: str  # "light", "normal", "deep"
    reason: str
    cumulative_drift: Optional[float] = None
    trend: Optional[str] = None  # "stable", "rising", "falling"


class MultiLevelDriftAnalyzer:
    """Tracks short-window drift samples and maps them to operator-facing severity."""

    def __init__(self):
        self.threshold_normal = config.get("parameters.thresholds.drift_normal", 0.05)
        self.threshold_warning = config.get("parameters.thresholds.drift_warning", 0.10)
        self.threshold_attention = config.get("parameters.thresholds.drift_attention", 0.15)
        self.threshold_alert = config.get("parameters.thresholds.drift_alert", 0.25)

        self.cumulative_window = config.get("parameters.thresholds.drift_cumulative_window", 10)
        self.cumulative_threshold = config.get("parameters.thresholds.drift_cumulative_threshold", 1.0)

        self.drift_history: Dict[str, List[float]] = {}

        logger.info(
            f"MultiLevelDriftAnalyzer initialized: "
            f"normal={self.threshold_normal}, warning={self.threshold_warning}, "
            f"attention={self.threshold_attention}, alert={self.threshold_alert}"
        )

    def analyze(self, session_id: str, current_drift: float) -> DriftAnalysis:
        """
        Classify ``current_drift`` for ``session_id`` and optionally request a consistency pass.

        Args:
            session_id: logical session key
            current_drift: latest drift scalar (cosine distance or similar)

        Returns:
            ``DriftAnalysis`` with recommended check depth + human-readable ``reason``.
        """
        if session_id not in self.drift_history:
            self.drift_history[session_id] = []
        self.drift_history[session_id].append(current_drift)

        if len(self.drift_history[session_id]) > self.cumulative_window:
            self.drift_history[session_id] = self.drift_history[session_id][-self.cumulative_window:]

        cumulative_drift = sum(self.drift_history[session_id])

        trend = self._analyze_trend(session_id)

        if current_drift >= self.threshold_alert:
            level = DriftLevel.ALERT
            should_check = True
            check_depth = "deep"
            reason = (
                f"Drift {current_drift:.3f} exceeds alert threshold {self.threshold_alert}"
            )

        elif current_drift >= self.threshold_attention:
            level = DriftLevel.ATTENTION
            should_check = True
            check_depth = "normal"
            reason = (
                f"Drift {current_drift:.3f} exceeds attention threshold {self.threshold_attention}"
            )

        elif current_drift >= self.threshold_warning:
            level = DriftLevel.WARNING
            should_check = (trend == "rising")
            check_depth = "light"
            reason = (
                f"Drift {current_drift:.3f} reached warning threshold {self.threshold_warning}"
                + ("; trend rising" if trend == "rising" else "")
            )

        elif current_drift >= self.threshold_normal:
            level = DriftLevel.NORMAL
            should_check = False
            check_depth = "none"
            reason = "Drift within expected band"

        else:
            level = DriftLevel.NORMAL
            should_check = False
            check_depth = "none"
            reason = "State stable"

        if cumulative_drift > self.cumulative_threshold:
            level = DriftLevel.ALERT
            should_check = True
            check_depth = "deep"
            reason = (
                f"Cumulative drift {cumulative_drift:.3f} exceeds {self.cumulative_threshold} "
                f"(last {len(self.drift_history[session_id])} ticks)"
            )
            logger.warning(f"[CUMULATIVE DRIFT DETECTED] {reason} | session={session_id}")

        if current_drift >= 0.40:
            level = DriftLevel.EMERGENCY
            should_check = True
            check_depth = "deep"
            reason = (
                f"Emergency: drift {current_drift:.3f} is extremely high (possible prompt tampering)"
            )
            logger.error(f"[EMERGENCY DRIFT] {reason} | session={session_id}")

        return DriftAnalysis(
            level=level,
            drift_value=current_drift,
            should_trigger_consistency_check=should_check,
            check_depth=check_depth,
            reason=reason,
            cumulative_drift=cumulative_drift,
            trend=trend
        )

    def _analyze_trend(self, session_id: str) -> str:
        """
        Compare the trailing mean against the previous window.

        Returns:
            ``"stable"``, ``"rising"``, or ``"falling"``.
        """
        history = self.drift_history.get(session_id, [])
        if len(history) < 3:
            return "stable"

        recent_avg = sum(history[-3:]) / 3
        prev_avg = sum(history[-6:-3]) / 3 if len(history) >= 6 else sum(history[:-3]) / len(history[:-3]) if len(history) > 3 else recent_avg

        diff = recent_avg - prev_avg

        if diff > 0.02:
            return "rising"
        elif diff < -0.02:
            return "falling"
        else:
            return "stable"

    def reset_history(self, session_id: str):
        """Drop buffered drift samples (e.g., after a rollback)."""
        if session_id in self.drift_history:
            self.drift_history[session_id] = []
            logger.info(f"Drift history reset for session: {session_id}")

    def get_statistics(self, session_id: str) -> Dict:
        """Return simple descriptive stats for the rolling buffer."""
        history = self.drift_history.get(session_id, [])
        if not history:
            return {
                "count": 0,
                "avg": 0.0,
                "max": 0.0,
                "min": 0.0,
                "trend": "stable"
            }

        return {
            "count": len(history),
            "avg": sum(history) / len(history),
            "max": max(history),
            "min": min(history),
            "cumulative": sum(history),
            "trend": self._analyze_trend(session_id)
        }


_analyzer_instance: Optional[MultiLevelDriftAnalyzer] = None


def get_drift_analyzer() -> MultiLevelDriftAnalyzer:
    """Process-wide singleton drift analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = MultiLevelDriftAnalyzer()
    return _analyzer_instance
