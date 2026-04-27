#!/usr/bin/env python3
"""
In-process cache for ``(z_self, needs)`` tuples to cut duplicate SQLite reads during autonomy loops.

Design:
1. Short TTL (default 60s) avoids serving stale state for long stretches.
2. ``tick + updated_at`` hash catches out-of-band writes without loading full vectors.
3. Explicit invalidation hooks keep correctness when callers mutate ``self_state``.
"""

import time
import logging
import sqlite3
import numpy as np
import json
from typing import Optional, Dict, Tuple
from threading import Lock

logger = logging.getLogger(__name__)


class StateCache:
    """Thread-safe session → latent snapshot cache."""

    def __init__(self, ttl: int = 60):
        """
        Args:
            ttl: seconds before an entry is considered expired.
        """
        self.ttl = ttl
        self.cache: Dict[str, Tuple[np.ndarray, Dict, float, str]] = {}
        self.lock = Lock()

        self.stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "expirations": 0
        }

    def get(self, session_id: str, db_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Returns:
            ``(z_self, needs)`` or ``None`` when miss / expired / hash mismatch.
        """
        with self.lock:
            if session_id not in self.cache:
                self.stats["misses"] += 1
                return None

            z_self, needs, timestamp, cached_hash = self.cache[session_id]
            current_time = time.time()

            if current_time - timestamp > self.ttl:
                del self.cache[session_id]
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                logger.debug(f"[StateCache] Expired cache for {session_id}")
                return None

            current_hash = self._get_state_hash(session_id, db_path)
            if current_hash != cached_hash:
                del self.cache[session_id]
                self.stats["invalidations"] += 1
                self.stats["misses"] += 1
                logger.debug(f"[StateCache] Invalidated cache for {session_id} (hash mismatch)")
                return None

            self.stats["hits"] += 1
            logger.debug(f"[StateCache] Cache hit for {session_id}")
            return (z_self.copy(), needs.copy())

    def set(
        self,
        session_id: str,
        z_self: np.ndarray,
        needs: Dict,
        db_path: str
    ):
        """Store a defensive copy plus the current DB fingerprint."""
        with self.lock:
            state_hash = self._get_state_hash(session_id, db_path)
            timestamp = time.time()
            self.cache[session_id] = (
                z_self.copy(),
                needs.copy(),
                timestamp,
                state_hash
            )
            logger.debug(f"[StateCache] Cached state for {session_id}")

    def invalidate(self, session_id: str):
        """
        Drop one session after explicit mutations (``_save_z_self``, tick, chat turn, …).
        """
        with self.lock:
            if session_id in self.cache:
                del self.cache[session_id]
                self.stats["invalidations"] += 1
                logger.debug(f"[StateCache] Manually invalidated {session_id}")

    def clear(self):
        """Flush every cached session."""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"[StateCache] Cleared {count} cached states")

    def get_stats(self) -> Dict:
        """Lightweight counters for diagnostics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (
                self.stats["hits"] / total_requests
                if total_requests > 0
                else 0
            )

            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": f"{hit_rate:.2%}",
                "invalidations": self.stats["invalidations"],
                "expirations": self.stats["expirations"],
                "total_requests": total_requests,
                "cached_sessions": len(self.cache)
            }

    def _get_state_hash(self, session_id: str, db_path: str) -> str:
        """Cheap fingerprint from ``self_state.tick`` + ``updated_at``."""
        try:
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cur = conn.execute(
                    "SELECT tick, updated_at FROM self_state WHERE session_id = ?",
                    (session_id,)
                )
                row = cur.fetchone()
                if row:
                    return f"{row[0]}_{row[1]}"
                return ""
        except Exception as e:
            logger.warning(f"[StateCache] Failed to get state hash: {e}")
            return ""

    def cleanup_expired(self):
        """Sweep TTL-expired rows (optional periodic housekeeping)."""
        with self.lock:
            current_time = time.time()
            expired_sessions = [
                sid for sid, (_, _, ts, _) in self.cache.items()
                if current_time - ts > self.ttl
            ]

            for sid in expired_sessions:
                del self.cache[sid]
                self.stats["expirations"] += 1

            if expired_sessions:
                logger.debug(
                    f"[StateCache] Cleaned up {len(expired_sessions)} expired caches"
                )


_global_cache: Optional[StateCache] = None


def get_global_cache(ttl: int = 60) -> StateCache:
    """Process-wide singleton."""
    global _global_cache
    if _global_cache is None:
        _global_cache = StateCache(ttl=ttl)
        logger.info(f"[StateCache] Initialized global cache (TTL={ttl}s)")
    return _global_cache


def invalidate_session_cache(session_id: str):
    """
    Convenience hook after:
    - ``SelfModel._save_z_self``
    - ``SelfTick.trigger``
    - end-of-turn chat persistence
    """
    cache = get_global_cache()
    cache.invalidate(session_id)
