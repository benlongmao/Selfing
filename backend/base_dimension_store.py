#!/usr/bin/env python3
"""
Base class for vector-backed “dimension” pattern stores.

Shares scoring, capacity enforcement, and archival hooks between emotion and motivation stores.
"""
import os
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple, TypeVar, Generic
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
import math
import logging

try:
    from backend.embedder import get_embedder
except Exception as e:
    import logging
    logging.warning(f"Failed to import real embedder: {e}, using fallback")
    from backend.embedder_fallback import get_embedder_fallback as get_embedder

logger = logging.getLogger(__name__)

# Generic row types (pattern payload vs aggregate state)
PatternType = TypeVar('PatternType')
StateType = TypeVar('StateType')


class BaseDimensionStore(ABC, Generic[PatternType, StateType]):
    """Abstract store for learned patterns with embedding-backed retrieval."""
    
    def __init__(
        self,
        db_path: str,
        table_name: str,
        pattern_prefix: str,
        max_patterns: int,
        replacement_threshold: float,
        stale_days: int
    ):
        """
        Args:
            db_path: SQLite path
            table_name: Concrete patterns table (e.g. ``emotion_patterns``)
            pattern_prefix: Primary-key prefix for ids originating here
            max_patterns: Hard cap on concurrently active rows
            replacement_threshold: Minimum intensity delta required to evict a row
            stale_days: ``last_seen_at`` horizon for “obviously stale” eviction bias
        """
        self.db_path = db_path
        self.table_name = table_name
        self.pattern_prefix = pattern_prefix
        self.max_patterns = max_patterns
        self.replacement_threshold = replacement_threshold
        self.stale_days = stale_days
        self.embedder = get_embedder()
        
        # Dynamic intensity weights (mirrors Rules dimension scoring philosophy)
        self.DYNAMIC_INTENSITY_BASE_WEIGHT = 0.3  # prior belief in the seeded intensity
        self.DYNAMIC_INTENSITY_EVIDENCE_WEIGHT = 0.4  # log-scaled evidence mass
        self.DYNAMIC_INTENSITY_RECENCY_WEIGHT = 0.2  # exponential decay since last activation
        self.DYNAMIC_INTENSITY_CONTEXT_WEIGHT = 0.1  # reserved for contextual reranking
        self.EVIDENCE_DECAY_HALF_LIFE_DAYS = 30.0  # days for recency half-life
    
    @abstractmethod
    def _ensure_tables(self) -> None:
        """Create schema if missing (subclass)."""
        pass
    
    @abstractmethod
    def _count_active_patterns(self, conn: sqlite3.Connection) -> int:
        """Count active rows (subclass-specific filters)."""
        pass
    
    @abstractmethod
    def _archive_pattern(self, conn: sqlite3.Connection, pattern_id: str) -> None:
        """Move / soft-delete a pattern row (subclass)."""
        pass
    
    def _calculate_dynamic_intensity(
        self,
        base_intensity: float,
        evidence_count: int,
        last_seen_at: Optional[str],
        created_at: Optional[str],
        pattern_id: str = ""
    ) -> float:
        """
        Blend prior intensity, evidence mass, recency, and a reserved context slot.

        Formula (weights configured on ``self``)::

            dynamic = base*w_b + evidence*w_e + recency*w_r + context*w_c

        Args:
            base_intensity: Seeded importance in ``[0, 1]``
            evidence_count: Number of supporting observations
            last_seen_at: ISO8601 last activation timestamp
            created_at: ISO8601 creation timestamp (reserved for future use)
            pattern_id: Used to detect the pinned core row (``is_core`` + ``locked``)

        Returns:
            Clamped dynamic intensity in ``[0, 1]``
        """
        now = datetime.now(timezone.utc)
        
        # Evidence strength via sublinear log scaling
        if evidence_count > 0:
            max_evidence = 100.0
            evidence_strength = min(1.0, math.log(1 + evidence_count) / math.log(1 + max_evidence))
        else:
            evidence_strength = 0.0
        
        # Recency decay from last_seen_at
        if last_seen_at:
            try:
                last_seen = datetime.fromisoformat(last_seen_at.replace('Z', '+00:00'))
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)
                days_since_last_use = (now - last_seen).total_seconds() / 86400.0
                recency_factor = math.exp(-days_since_last_use / self.EVIDENCE_DECAY_HALF_LIFE_DAYS)
            except (ValueError, AttributeError) as e:
                logger.debug(f"Failed to parse last_seen_at '{last_seen_at}': {e}")
                recency_factor = 0.5
            except Exception as e:
                logger.warning(f"Unexpected error parsing last_seen_at '{last_seen_at}': {e}")
                recency_factor = 0.5
        else:
            recency_factor = 0.3
        
        # Contextual relevance (placeholder until wired to live context)
        context_relevance = 0.5
        
        # Pin the canonical core row at maximum dynamic intensity
        if pattern_id and pattern_id.startswith(self.pattern_prefix):
            # Confirm both ``is_core`` and ``locked`` flags in SQLite
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        f"SELECT is_core, locked FROM {self.table_name} WHERE id = ?",
                        (pattern_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0] == 1 and row[1] == 1:
                        return 1.0  # pinned core occupies top of [0, 1] range
            except sqlite3.Error as e:
                logger.debug(f"Failed to check core pattern: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error checking core pattern: {e}")
        
        # Weighted blend then clamp
        dynamic_intensity = (
            base_intensity * self.DYNAMIC_INTENSITY_BASE_WEIGHT +
            evidence_strength * self.DYNAMIC_INTENSITY_EVIDENCE_WEIGHT +
            recency_factor * self.DYNAMIC_INTENSITY_RECENCY_WEIGHT +
            context_relevance * self.DYNAMIC_INTENSITY_CONTEXT_WEIGHT
        )
        # Clamp to [0, 1]
        dynamic_intensity = max(0.0, min(1.0, dynamic_intensity))
        
        return dynamic_intensity
    
    def _ensure_capacity(self, new_intensity: float) -> None:
        """Evict lowest-value row when at capacity unless newcomer is clearly stronger."""
        with sqlite3.connect(self.db_path) as conn:
            current = self._count_active_patterns(conn)
            if current < self.max_patterns:
                return
            
            candidate = self._select_replacement_candidate(conn)
            if not candidate:
                raise RuntimeError(f"{self.table_name} pattern limit reached and no replacement candidate found.")
            
            # Compare proposed row against weakest survivor
            candidate_dynamic_intensity = candidate["dynamic_intensity"]
            if new_intensity <= candidate_dynamic_intensity + self.replacement_threshold:
                raise RuntimeError(
                    f"{self.table_name} pattern limit reached ({current} >= {self.max_patterns}) "
                    f"and new pattern intensity ({new_intensity:.3f}) not significantly higher than "
                    f"candidate dynamic intensity ({candidate_dynamic_intensity:.3f})."
                )
            
            logger.info(
                f"{self.table_name} limit reached. Archiving pattern {candidate['id']} "
                f"(base_intensity={candidate['intensity']:.3f}, dynamic_intensity={candidate_dynamic_intensity:.3f}) "
                f"to insert new pattern (intensity={new_intensity:.3f})."
            )
            self._archive_pattern(conn, candidate["id"])
    
    def _select_replacement_candidate(self, conn: sqlite3.Connection) -> Optional[Dict]:
        """
        Prefer stale unlocked rows; otherwise pick lowest dynamic intensity among unlockeds.
        """
        stale_cutoff = (datetime.now(timezone.utc) - timedelta(days=self.stale_days)).isoformat()
        
        # Pass 1: anything not touched since ``stale_days``
        cur = conn.execute(
            f"""
            SELECT id, intensity, evidence_count, last_seen_at, created_at
            FROM {self.table_name}
            WHERE status='active' AND locked=0 AND last_seen_at <= ?
            """,
            (stale_cutoff,)
        )
        candidates = cur.fetchall()
        
        # Pass 2: entire unlocked active pool
        if not candidates:
            cur = conn.execute(
                f"""
                SELECT id, intensity, evidence_count, last_seen_at, created_at
                FROM {self.table_name}
                WHERE status='active' AND locked=0
                """
            )
            candidates = cur.fetchall()
        
        if not candidates:
            return None
        
        # Score each candidate; evict the weakest dynamic intensity
        best_candidate = None
        lowest_dynamic_intensity = float('inf')
        
        for row in candidates:
            pattern_id = row[0]
            base_intensity = float(row[1])
            evidence_count = row[2]
            last_seen_at = row[3]
            created_at = row[4]
            
            # Dynamic score for fair comparison across evidence ages
            dynamic_intensity = self._calculate_dynamic_intensity(
                base_intensity=base_intensity,
                evidence_count=evidence_count,
                last_seen_at=last_seen_at,
                created_at=created_at,
                pattern_id=pattern_id
            )
            
            # Track minimum
            if dynamic_intensity < lowest_dynamic_intensity:
                lowest_dynamic_intensity = dynamic_intensity
                best_candidate = {
                    "id": pattern_id,
                    "intensity": base_intensity,
                    "dynamic_intensity": dynamic_intensity,
                    "evidence_count": evidence_count,
                    "last_seen_at": last_seen_at,
                    "created_at": created_at
                }
        
        return best_candidate

