"""
z_self Observer — passive telemetry.

After each chat turn, records z_self snapshots and simple LLM-output statistics
for later calibration (e.g. regressing z_self → surface features).

Stored per row:
  - z_self_before / z_self_after: full vectors before/after the turn
  - output_features: numeric features parsed from the assistant reply
  - context: session metadata (session_id, tick, timestamp, …)

SQLite table `z_self_observations`, ~2KB per row.

[2026-04-08] introduced.
"""
import json
import logging
import sqlite3
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ZSelfObserver:
    """Passive logging of z_self and reply-side features; failures never break chat."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS z_self_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        tick INTEGER DEFAULT 0,
                        timestamp REAL NOT NULL,
                        z_self_before BLOB,
                        z_self_after BLOB,
                        output_features TEXT,
                        context TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_zso_session
                    ON z_self_observations(session_id, timestamp)
                """)
        except Exception as e:
            logger.warning(f"ZSelfObserver table init failed: {e}")

    def record(
        self,
        session_id: str,
        z_self_before: Optional[np.ndarray],
        z_self_after: Optional[np.ndarray],
        llm_response: str = "",
        tick: int = 0,
        extra_context: Optional[Dict] = None,
    ) -> bool:
        """
        Persist one observation after a turn. Exceptions are swallowed so chat never fails.
        """
        try:
            features = self._extract_output_features(llm_response)

            before_blob = z_self_before.tobytes() if z_self_before is not None else None
            after_blob = z_self_after.tobytes() if z_self_after is not None else None

            context = {
                "tick": tick,
                **(extra_context or {}),
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO z_self_observations
                       (session_id, tick, timestamp, z_self_before, z_self_after,
                        output_features, context)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        tick,
                        time.time(),
                        before_blob,
                        after_blob,
                        json.dumps(features, ensure_ascii=False),
                        json.dumps(context, ensure_ascii=False),
                    ),
                )
            return True
        except Exception as e:
            logger.debug(f"ZSelfObserver.record failed (silent): {e}")
            return False

    @staticmethod
    def _extract_output_features(text: str) -> Dict[str, float]:
        """
        Lightweight numeric features from the assistant text (for z_self → behavior analysis).
        """
        if not text:
            return {}

        char_count = len(text)

        # Bilingual: models may hedge in Chinese or English
        hedging_markers = [
            "可能", "也许", "或许", "大概", "不确定", "似乎", "看起来",
            "maybe", "perhaps", "might", "could", "possibly", "uncertain", "seems", "appears",
            "probably", "presumably", "likely", "unlikely", "I think", "I guess", "not sure",
            "not certain", "approximately", "roughly",
        ]
        hedging_count = sum(text.count(m) for m in hedging_markers)

        question_count = text.count("？") + text.count("?")

        code_blocks = text.count("```")
        has_code = code_blocks >= 2

        list_markers = text.count("\n- ") + text.count("\n* ") + text.count("\n1.")
        has_structure = list_markers >= 2

        sentences = [s for s in text.replace("。", ".|").replace("！", "!|").replace("？", "?|").split("|") if s.strip()]
        sentence_count = max(1, len(sentences))
        avg_sentence_len = char_count / sentence_count

        return {
            "char_count": float(char_count),
            "sentence_count": float(sentence_count),
            "avg_sentence_len": float(round(avg_sentence_len, 1)),
            "hedging_count": float(hedging_count),
            "hedging_ratio": float(round(hedging_count / max(1, sentence_count), 3)),
            "question_count": float(question_count),
            "has_code": float(has_code),
            "has_structure": float(has_structure),
            "code_block_count": float(code_blocks // 2),
        }

    def get_observations(
        self, session_id: str, limit: int = 100
    ) -> list:
        """Return recent observations for a session (analysis / debugging)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """SELECT tick, timestamp, z_self_before, z_self_after,
                              output_features, context
                       FROM z_self_observations
                       WHERE session_id = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (session_id, limit),
                ).fetchall()

            results = []
            for row in rows:
                entry = {
                    "tick": row[0],
                    "timestamp": row[1],
                    "z_self_before": np.frombuffer(row[2], dtype=np.float32) if row[2] else None,
                    "z_self_after": np.frombuffer(row[3], dtype=np.float32) if row[3] else None,
                    "output_features": json.loads(row[4]) if row[4] else {},
                    "context": json.loads(row[5]) if row[5] else {},
                }
                results.append(entry)
            return results
        except Exception as e:
            logger.debug(f"ZSelfObserver.get_observations failed: {e}")
            return []

    def count_observations(self, session_id: Optional[str] = None) -> int:
        """Count rows, optionally filtered by session_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if session_id:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM z_self_observations WHERE session_id=?",
                        (session_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM z_self_observations"
                    ).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0
