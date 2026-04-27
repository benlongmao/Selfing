#!/usr/bin/env python3
"""autonomy_gate: assistant-side pause markers mirror user commands."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from backend import autonomy_gate


class TestAssistantAutonomyMarkers(unittest.TestCase):
    def test_pause_own_line_triggers(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_assistant"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=lambda k, d=None: True if "agent_markers" in k else d):
                r = autonomy_gate.apply_assistant_autonomy_markers(
                    "好的。\n\n[S44_PAUSE]\n", "demo-session"
                )
                self.assertEqual(r, "paused")
                m_set.assert_called_once_with(True, "demo-session")

    def test_prose_mention_pause_not_trigger(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_assistant"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=lambda k, d=None: True if "agent_markers" in k else d):
                r = autonomy_gate.apply_assistant_autonomy_markers(
                    "不要误用 [S44_PAUSE] 在句中。", "demo-session"
                )
                self.assertIsNone(r)
                m_set.assert_not_called()

    def test_autonomy_resume_inline_in_sentence(self):
        """[S44_AUTONOMY_RESUME] does not need to be on its own line."""
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_assistant"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=lambda k, d=None: True if "agent_markers" in k else d):
                r = autonomy_gate.apply_assistant_autonomy_markers(
                    "好的，现在执行 `[S44_AUTONOMY_RESUME]` 以恢复推送。", "demo-session"
                )
                self.assertEqual(r, "resumed")
                m_set.assert_called_once_with(False, "demo-session")

    def test_pause_wins_over_resume_same_message(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_assistant"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=lambda k, d=None: True if "agent_markers" in k else d):
                r = autonomy_gate.apply_assistant_autonomy_markers(
                    "[S44_AUTONOMY_RESUME]\n[S44_PAUSE]\n", "demo-session"
                )
                self.assertEqual(r, "paused")
                m_set.assert_called_once_with(True, "demo-session")


class TestUserAutonomyCommands(unittest.TestCase):
    def _cfg_user_on(self, k: str, d=None):
        if "user_text_commands" in k:
            return True
        if k == "system.autonomy_gate_enabled":
            return True
        return d

    def test_user_negated_pause_not_trigger(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "你不要停止自主行动，我们继续聊。"
                )
                self.assertIsNone(r)
                m_set.assert_not_called()

    def test_user_negated_resume_not_trigger(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "别恢复自主执行，先停着。"
                )
                self.assertIsNone(r)
                m_set.assert_not_called()

    def test_user_quoted_pause_phrase_not_trigger(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "我说的是‘停止自主行动’这几个字，不是现在执行。"
                )
                self.assertIsNone(r)
                m_set.assert_not_called()

    def test_user_quoted_resume_phrase_not_trigger(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "我刚才说的是‘恢复自主行动’这几个字，不是现在执行。"
                )
                self.assertIsNone(r)
                m_set.assert_not_called()

    def test_user_explain_resume_token_not_trigger(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "请解释 [S44_AUTONOMY_RESUME] 的作用。"
                )
                self.assertIsNone(r)
                m_set.assert_not_called()

    def test_user_inline_resume_command_still_triggers(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "好的，现在请执行 [S44_AUTONOMY_RESUME]。"
                )
                self.assertEqual(r, "resumed")
                m_set.assert_called_once_with(False, "demo-session")

    def test_user_start_autonomy_synonym_resumes(self):
        """The Chinese start-autonomy phrase is equivalent to resume."""
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "开始自主行动"
                )
                self.assertEqual(r, "resumed")
                m_set.assert_called_once_with(False, "demo-session")

    def test_user_plain_s44_autonomy_resume_triggers(self):
        """A bare S44_AUTONOMY_RESUME token resumes autonomy."""
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "好了，S44_AUTONOMY_RESUME，继续。"
                )
                self.assertEqual(r, "resumed")
                m_set.assert_called_once_with(False, "demo-session")

    def test_user_cn_bracket_autonomy_resume_triggers(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "【S44_AUTONOMY_RESUME】"
                )
                self.assertEqual(r, "resumed")
                m_set.assert_called_once_with(False, "demo-session")

    def test_user_negated_start_autonomy_not_trigger(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "不要开始自主行动，先停着。"
                )
                self.assertIsNone(r)
                m_set.assert_not_called()

    def test_user_pause_then_resume_last_wins(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "停止自主行动。恢复自主行动。"
                )
                self.assertEqual(r, "resumed")
                m_set.assert_called_once_with(False, "demo-session")

    def test_user_resume_then_pause_last_wins(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=True), patch.object(
            autonomy_gate, "set_autonomous_pause_from_user"
        ) as m_set:
            with patch.object(autonomy_gate.config, "get", side_effect=self._cfg_user_on):
                r = autonomy_gate.apply_user_autonomy_command_from_text(
                    "demo-session", "恢复自主行动。停止自主行动。"
                )
                self.assertEqual(r, "paused")
                m_set.assert_called_once_with(True, "demo-session")


class TestAutonomyPauseStateFile(unittest.TestCase):
    """State-file behavior for is_autonomous_execution_paused."""

    def test_default_state_path_uses_run_directory(self):
        with patch.object(autonomy_gate.config, "get", side_effect=lambda k, d=None: d), \
             patch.object(autonomy_gate, "get_project_root", return_value="/tmp/s-main"):
            self.assertEqual(
                autonomy_gate._state_path(),
                os.path.abspath("/tmp/s-main/run/autonomy_gate.json"),
            )

    def test_configured_relative_state_path_uses_project_root(self):
        def cfg(key, default=None):
            if key == "system.autonomy_gate_state_path":
                return "var/autonomy_gate.json"
            return default

        with patch.object(autonomy_gate.config, "get", side_effect=cfg), \
             patch.object(autonomy_gate, "get_project_root", return_value="/tmp/s-main"):
            self.assertEqual(
                autonomy_gate._state_path(),
                os.path.abspath("/tmp/s-main/var/autonomy_gate.json"),
            )

    def test_paused_when_state_file_says_true(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"paused": True, "updated_at": 1.0, "reason": "test"},
                    f,
                )
            with patch.object(autonomy_gate, "_state_path", return_value=path), patch.object(
                autonomy_gate, "gate_enabled", return_value=True
            ):
                self.assertTrue(
                    autonomy_gate.is_autonomous_execution_paused("demo-session")
                )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_not_paused_when_gate_disabled(self):
        with patch.object(autonomy_gate, "gate_enabled", return_value=False):
            self.assertFalse(
                autonomy_gate.is_autonomous_execution_paused("demo-session")
            )

    def test_string_false_paused_not_treated_as_true(self):
        """A hand-edited string value of "false" must not pause autonomy."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"paused": "false", "updated_at": 1.0, "reason": ""},
                    f,
                )
            with patch.object(autonomy_gate, "_state_path", return_value=path), patch.object(
                autonomy_gate, "gate_enabled", return_value=True
            ):
                self.assertFalse(
                    autonomy_gate.is_autonomous_execution_paused("demo-session")
                )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
