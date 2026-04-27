#!/usr/bin/env python3
"""
Unit tests for ``memory_enhancer`` (stdlib ``unittest``; no pytest required).
"""

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone

from backend.memory_enhancer import MemoryEnhancer, init_memory_enhancer
from backend.s_identity import PRIMARY_SESSION_ID


def _make_enhancer_with_one_turn():
    td = tempfile.TemporaryDirectory()
    db_path = os.path.join(td.name, "case.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE chat_turns (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                turn_index INTEGER,
                user_input TEXT,
                assistant_output TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chat_turns (id, session_id, turn_index, user_input, assistant_output, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "tid-ingest-1",
                PRIMARY_SESSION_ID,
                1,
                "Who are you? I want to understand your identity, sense of self, and consciousness. "
                "Between us I need trust that you will stay with me.",
                "I reflect on identity and existence, and I will remember the boundaries we discuss together.",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    return td, MemoryEnhancer(db_path)


class TestMemoryEnhancer(unittest.TestCase):
    """Smoke tests for ``MemoryEnhancer``."""

    def test_init(self):
        enhancer = MemoryEnhancer()
        self.assertIsNotNone(enhancer)
        self.assertIsNotNone(enhancer.engine)
        self.assertIsNotNone(enhancer.importance_evaluator)

    def test_background_process(self):
        enhancer = MemoryEnhancer()
        result = enhancer.background_process()
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "completed")
        self.assertIn("memories_loaded", result)
        self.assertIn("memories_cleaned", result)
        self.assertIn("due_count", result)

    def test_get_statistics(self):
        enhancer = MemoryEnhancer()
        stats = enhancer.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_memories", stats)
        self.assertIn("due_memories", stats)
        self.assertIn("avg_importance", stats)
        self.assertIn("last_review", stats)

    def test_init_memory_enhancer(self):
        enhancer = init_memory_enhancer()
        self.assertIsNotNone(enhancer)
        self.assertIsInstance(enhancer, MemoryEnhancer)

    def test_daily_memory_review_task(self):
        enhancer = MemoryEnhancer()
        result = enhancer.daily_memory_review_task()
        self.assertIsInstance(result, dict)
        self.assertIn("reviewed_count", result)
        self.assertIn("successful_reviews", result)
        self.assertIn("failed_reviews", result)
        self.assertIn("ingested_new", result)

    def test_add_memory_to_spaced_repetition(self):
        enhancer = MemoryEnhancer()
        self.assertTrue(hasattr(enhancer, "add_memory_to_spaced_repetition"))

    def test_review_due_memories(self):
        enhancer = MemoryEnhancer()
        self.assertTrue(hasattr(enhancer, "review_due_memories"))


class TestMemoryEnhancerChatIntegration(unittest.TestCase):
    """Isolated DB + one ``chat_turns`` row: scan salience and daily ingest dedup."""

    def setUp(self):
        self._td, self.enhancer = _make_enhancer_with_one_turn()

    def tearDown(self):
        self._td.cleanup()

    def test_load_high_importance_from_chat_turns(self):
        mems = self.enhancer.load_high_importance_memories(
            min_score=0.35, limit=10, max_age_days=7
        )
        self.assertGreaterEqual(len(mems), 1)
        self.assertEqual(mems[0].source_type, "conversation")
        self.assertEqual(mems[0].source_id, "tid-ingest-1")

    def test_daily_ingest_dedup(self):
        r1 = self.enhancer.daily_memory_review_task(PRIMARY_SESSION_ID)
        self.assertGreaterEqual(r1.get("ingested_new", 0), 1)
        r2 = self.enhancer.daily_memory_review_task(PRIMARY_SESSION_ID)
        self.assertEqual(r2.get("ingested_new", 0), 0)


if __name__ == "__main__":
    unittest.main()
