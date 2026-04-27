#!/usr/bin/env python3
"""unified_scheduler: enqueue dedupe, priority ordering, autonomy_resume enqueue (no executor thread)."""
import os
import tempfile
import unittest
from unittest.mock import patch

import backend.unified_scheduler as unified_scheduler
from backend.unified_scheduler import (
    PRIORITY_HEARTBEAT,
    PRIORITY_IDLE_PULSE,
    UnifiedScheduler,
    enqueue_autonomy_resume_check,
)


class TestUnifiedSchedulerEnqueue(unittest.TestCase):
    def setUp(self):
        self._fd, self._db = tempfile.mkstemp(suffix=".db")
        os.close(self._fd)
        self.sched = UnifiedScheduler(self._db)

    def tearDown(self):
        try:
            os.unlink(self._db)
        except OSError:
            pass

    def test_enqueue_duplicate_task_id_skipped(self):
        ok1 = self.sched.enqueue(
            PRIORITY_HEARTBEAT, "heartbeat", "h-1", "prompt", "demo-session"
        )
        ok2 = self.sched.enqueue(
            PRIORITY_HEARTBEAT, "heartbeat", "h-1", "prompt", "demo-session"
        )
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertTrue(self.sched.is_queued("h-1"))
        self.assertEqual(self.sched.queue_size, 1)

    def test_dequeue_priority_smaller_number_first(self):
        """heapq: lower numeric priority value is dequeued first."""
        self.sched.enqueue(
            PRIORITY_IDLE_PULSE, "idle", "i-1", "idle prompt", "demo-session"
        )
        self.sched.enqueue(
            PRIORITY_HEARTBEAT, "hb", "h-1", "hb prompt", "demo-session"
        )
        first = self.sched._dequeue()
        self.assertIsNotNone(first)
        self.assertEqual(first.task_id, "h-1")
        self.assertEqual(first.priority, PRIORITY_HEARTBEAT)


class TestEnqueueAutonomyResumeCheck(unittest.TestCase):
    def test_returns_false_when_no_global_scheduler(self):
        old = unified_scheduler._scheduler_instance
        try:
            unified_scheduler._scheduler_instance = None
            self.assertFalse(enqueue_autonomy_resume_check("demo-session"))
        finally:
            unified_scheduler._scheduler_instance = old

    def test_enqueues_when_scheduler_present(self):
        old = unified_scheduler._scheduler_instance
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            unified_scheduler.init_scheduler(path)
            ok = enqueue_autonomy_resume_check("demo-session")
            self.assertTrue(ok)
            sched = unified_scheduler.get_scheduler()
            self.assertIsNotNone(sched)
            self.assertGreater(sched.queue_size, 0)
        finally:
            unified_scheduler._scheduler_instance = old
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_same_second_calls_are_deduped_by_identical_task_id(self):
        old = unified_scheduler._scheduler_instance
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            unified_scheduler.init_scheduler(path)
            with patch("backend.unified_scheduler.time.time", return_value=1713340000.1):
                ok1 = enqueue_autonomy_resume_check("demo-session")
                ok2 = enqueue_autonomy_resume_check("demo-session")
            sched = unified_scheduler.get_scheduler()
            self.assertIsNotNone(sched)
            self.assertTrue(ok1)
            self.assertFalse(ok2)
            self.assertEqual(sched.queue_size, 1)
        finally:
            unified_scheduler._scheduler_instance = old
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_cross_second_calls_enqueue_twice_with_different_task_ids(self):
        old = unified_scheduler._scheduler_instance
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            unified_scheduler.init_scheduler(path)
            with patch("backend.unified_scheduler.time.time", return_value=1713340000.1):
                ok1 = enqueue_autonomy_resume_check("demo-session")
            with patch("backend.unified_scheduler.time.time", return_value=1713340001.1):
                ok2 = enqueue_autonomy_resume_check("demo-session")
            sched = unified_scheduler.get_scheduler()
            self.assertIsNotNone(sched)
            self.assertTrue(ok1)
            self.assertTrue(ok2)
            self.assertEqual(sched.queue_size, 2)
            task_ids = {item.task_id for item in sched._queue}
            self.assertEqual(len(task_ids), 2)
        finally:
            unified_scheduler._scheduler_instance = old
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
