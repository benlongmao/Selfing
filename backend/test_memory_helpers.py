#!/usr/bin/env python3
"""Unit tests for memory helpers (user_fact_capture + unified_memory prompt tags)."""
import os
import sqlite3
import tempfile
import unittest

from backend.user_fact_capture import (
    apply_user_fact_capture,
    format_user_profile_block_for_prompt,
    parse_user_fact_message,
    _merge_facts,
)
from backend.unified_memory import UnifiedMemoryCandidate, _iso_date_prefix, _source_tag_for_prompt
from backend.memory_salience import (
    explicit_fact_salience,
    compute_biography_salience_and_class,
    trivial_retrieval_penalty,
)


class TestMemorySalience(unittest.TestCase):
    def test_explicit_beats_auto_narrative_ceiling(self):
        floor = explicit_fact_salience(mention_hit_count=1)
        boosted = explicit_fact_salience(mention_hit_count=3)
        self.assertGreater(floor, 0.85)
        self.assertGreaterEqual(boosted, floor)

    def test_trivial_class_low(self):
        ss, mc = compute_biography_salience_and_class(
            significance=0.4,
            emotional_intensity=0.1,
            identity_relevance=0.1,
            relationship_depth=0.1,
            memory_type="episodic",
        )
        self.assertEqual(mc, "trivial")
        self.assertLess(ss, 0.4)

    def test_trivial_penalty_positive(self):
        self.assertGreater(trivial_retrieval_penalty(), 0)


class TestUserFactCapture(unittest.TestCase):
    def test_remember_lines(self):
        n, facts = parse_user_fact_message("请记住：A\n请记住：B")
        self.assertIsNone(n)
        self.assertEqual(facts, ["A", "B"])

    def test_remember_lines_english(self):
        n, facts = parse_user_fact_message("Please remember: A\nPlease remember: B")
        self.assertIsNone(n)
        self.assertEqual(facts, ["A", "B"])

    def test_call_me(self):
        n, facts = parse_user_fact_message("请叫我老李")
        self.assertEqual(n, "老李")
        self.assertEqual(facts, [])

    def test_call_me_english(self):
        n, facts = parse_user_fact_message("call me Alex")
        self.assertEqual(n, "Alex")
        self.assertEqual(facts, [])

    def test_merge_dedup(self):
        m = _merge_facts("a\nb", ["a", "c"], 100)
        self.assertIn("a", m.split("\n"))
        self.assertIn("c", m.split("\n"))

    def test_format_user_profile_block(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.execute(
                "CREATE TABLE user_profiles (session_id TEXT PRIMARY KEY, name TEXT, facts TEXT, last_seen TEXT)"
            )
            conn.execute(
                "INSERT INTO user_profiles VALUES ('s1','Alex','Please remember: smoke test', '')"
            )
            conn.commit()
            conn.close()
            b = format_user_profile_block_for_prompt(path, "s1")
            self.assertIn("[USER profile]", b)
            self.assertIn("Alex", b)
        finally:
            os.unlink(path)

    def test_remember_three_times_boosts_salience(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            msg = "Please remember: doorplate test 999"
            apply_user_fact_capture(path, "s3", msg, turn_index=1)
            apply_user_fact_capture(path, "s3", msg, turn_index=2)
            apply_user_fact_capture(path, "s3", msg, turn_index=3)
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT salience_score FROM user_stated_facts WHERE session_id='s3' "
                "AND kind='fact_line' LIMIT 1"
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertGreaterEqual(float(row[0]), explicit_fact_salience(mention_hit_count=3))
        finally:
            os.unlink(path)

    def test_apply_creates_stated_facts_with_turn(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            r = apply_user_fact_capture(
                path, "sx", "Please remember: door 302", turn_index=7
            )
            self.assertTrue(r.get("updated"))
            conn = sqlite3.connect(path)
            n = conn.execute(
                "SELECT COUNT(*) FROM user_stated_facts WHERE session_id='sx'"
            ).fetchone()[0]
            self.assertGreaterEqual(n, 1)
            tid = conn.execute(
                "SELECT turn_index FROM user_stated_facts WHERE session_id='sx' LIMIT 1"
            ).fetchone()[0]
            self.assertEqual(tid, 7)
            conn.close()
        finally:
            os.unlink(path)


class TestUnifiedMemoryTags(unittest.TestCase):
    def test_iso_date_prefix(self):
        self.assertEqual(_iso_date_prefix("2026-04-16T09:50:35+00:00"), "2026-04-16")
        self.assertEqual(_iso_date_prefix(""), "")

    def test_source_tag_self_biography(self):
        c = UnifiedMemoryCandidate(
            memory_key="x",
            session_id="s",
            memory_type="episodic",
            source_table="self_biography",
            source_id="1",
            content="hello",
            created_at="2026-01-02T12:00:00Z",
        )
        self.assertIn("memoir", _source_tag_for_prompt(c))
        self.assertIn("2026-01-02", _source_tag_for_prompt(c))

    def test_user_profiles_no_extra_tag(self):
        c = UnifiedMemoryCandidate(
            memory_key="u",
            session_id="s",
            memory_type="identity",
            source_table="user_profiles",
            source_id="s",
            content="Display name: A",
            created_at="2026-01-01T00:00:00Z",
        )
        self.assertEqual(_source_tag_for_prompt(c), "")


if __name__ == "__main__":
    unittest.main()
