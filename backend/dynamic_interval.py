#!/usr/bin/env python3
"""
Dynamic scheduler interval helper.

Chooses how long to wait before the next background check based on ``z_self``
somatic bands and high-level ``needs`` scalars, with hysteresis-friendly bounds.
"""

import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DynamicIntervalCalculator:
    """Maps urgency (0-1) to a wait time between ``min_interval`` and ``max_interval`` seconds."""

    def __init__(self):
        self.min_interval = 60      # floor: 1 minute under crisis
        self.max_interval = 300   # ceiling: 5 minutes when calm

        self.thresholds = {
            "critical": 0.8,
            "urgent": 0.6,
            "moderate": 0.4,
            "normal": 0.2,
            "idle": 0.0,
        }

    def calculate_interval(
        self,
        session_id: str,
        z_self: np.ndarray,
        needs: Dict
    ) -> float:
        """
        Return the recommended delay (seconds) before the next housekeeping pass.

        Args:
            session_id: logical session key (for logs)
            z_self: flattened self-state vector (expects energy/pain slots used below)
            needs: ``novelty`` / ``clarity`` / ``connection`` demand scalars

        Returns:
            Floating seconds in ``[min_interval, max_interval]``.
        """
        try:
            energy = float(z_self[66]) if len(z_self) > 66 else 100
            pain = float(z_self[70]) if len(z_self) > 70 else 0

            novelty = needs.get("novelty", 0.5)
            clarity = needs.get("clarity", 0.5)
            connection = needs.get("connection", 0.5)

            urgency = self._calculate_urgency(
                energy, pain, novelty, clarity, connection
            )

            interval = self._urgency_to_interval(urgency)

            logger.debug(
                f"[DynamicInterval] {session_id}: "
                f"urgency={urgency:.2f}, interval={interval}s"
            )

            return interval

        except Exception as e:
            logger.warning(f"[DynamicInterval] Error calculating interval: {e}")
            return self.max_interval

    def _calculate_urgency(
        self,
        energy: float,
        pain: float,
        novelty: float,
        clarity: float,
        connection: float
    ) -> float:
        """
        Collapse heterogeneous signals into a single urgency scalar.

        Signals:
        - Pain dominates when high.
        - Energy tail risks (very low or very high) add urgency.
        - Novelty / clarity / connection extremes add moderate pressure.
        """
        urgency_scores = []

        if pain > 0.7:
            urgency_scores.append(1.0)
        elif pain > 0.5:
            urgency_scores.append(0.8)
        elif pain > 0.3:
            urgency_scores.append(0.5)

        if energy < 20:
            urgency_scores.append(0.9)
        elif energy > 95:
            urgency_scores.append(0.6)
        elif energy < 40:
            urgency_scores.append(0.5)

        if novelty > 0.9:
            urgency_scores.append(0.7)
        elif novelty < 0.1:
            urgency_scores.append(0.4)

        if clarity < 0.2:
            urgency_scores.append(0.6)
        elif clarity < 0.4:
            urgency_scores.append(0.4)

        if connection < 0.2:
            urgency_scores.append(0.5)
        elif connection < 0.3:
            urgency_scores.append(0.3)

        if not urgency_scores:
            return 0.0

        max_urgency = max(urgency_scores)

        if len(urgency_scores) >= 3:
            avg_urgency = sum(urgency_scores) / len(urgency_scores)
            max_urgency = min(1.0, max_urgency * 0.7 + avg_urgency * 0.3)

        return max_urgency

    def _urgency_to_interval(self, urgency: float) -> float:
        """
        Piecewise mapping from urgency to seconds.

        Bands (seconds):
        - ``>= critical`` → ``min_interval`` (60s)
        - ``>= urgent`` → 90s
        - ``>= moderate`` → 120s
        - ``>= normal`` → 180s
        - else interpolate toward ``max_interval`` (300s)
        """
        if urgency >= self.thresholds["critical"]:
            return self.min_interval

        elif urgency >= self.thresholds["urgent"]:
            return 90

        elif urgency >= self.thresholds["moderate"]:
            return 120

        elif urgency >= self.thresholds["normal"]:
            return 180

        else:
            return self.max_interval - (urgency / 0.2) * 60

    def get_interval_description(self, interval: float) -> str:
        """Human-readable label for dashboards / operator logs."""
        if interval <= 60:
            return "Critical cadence (~1 minute)"
        elif interval <= 90:
            return "Urgent cadence (~1.5 minutes)"
        elif interval <= 120:
            return "Elevated cadence (~2 minutes)"
        elif interval <= 180:
            return "Watch cadence (~3 minutes)"
        elif interval <= 240:
            return "Relaxed cadence (~4 minutes)"
        else:
            return "Default cadence (~5 minutes)"


_global_calculator: Optional[DynamicIntervalCalculator] = None


def get_global_calculator() -> DynamicIntervalCalculator:
    """Singleton accessor used by background loops."""
    global _global_calculator
    if _global_calculator is None:
        _global_calculator = DynamicIntervalCalculator()
        logger.info("[DynamicInterval] Initialized global calculator")
    return _global_calculator
