#!/usr/bin/env python3
"""chat_service：覆盖“用户恢复自主 → 立即入队 autonomy_resume 检查”的桥接逻辑。"""
import unittest
from unittest.mock import patch

from backend.chat_service import ChatService


class _Mode:
    def __init__(self, value: str):
        self.value = value


class _FakeExistentialMode:
    SOLITARY = _Mode("solitary")
    RESTING = _Mode("resting")


class _FakeExistential:
    def get_current_mode(self, session_id):
        return _FakeExistentialMode.RESTING, "test-rest"

    def get_mode_influence(self, current_mode):
        return {"suggestion": "resting"}

    def check_solitude_expired(self, session_id):
        return False


class _FakeDailyNarrativeGenerator:
    def check_and_generate_daily_narrative(self, session_id):
        return None


class TestChatServiceAutonomyResumeKickoff(unittest.TestCase):
    def _make_service(self):
        svc = ChatService.__new__(ChatService)
        svc.db_path = ":memory:"
        svc.session_history = {
            "selfing-session": [{"role": "user", "content": "hi"}]
        }
        svc.sensory_buffer = None
        svc.notification_queue = None
        svc.self_model = None
        return svc

    def test_resume_command_queues_immediate_kickoff(self):
        svc = self._make_service()
        with patch("backend.chat_service.get_effective_session", side_effect=lambda s: s), \
             patch(
                 "backend.autonomy_gate.apply_user_autonomy_command_from_text",
                 return_value="resumed",
             ) as m_gate, \
             patch(
                 "backend.unified_scheduler.enqueue_autonomy_resume_check",
                 return_value=True,
             ) as m_enqueue, \
             patch(
                 "backend.daily_narrative.get_daily_narrative_generator",
                 return_value=_FakeDailyNarrativeGenerator(),
             ), \
             patch("backend.existential_state.ExistentialMode", _FakeExistentialMode), \
             patch(
                 "backend.existential_state.get_existential_state",
                 return_value=_FakeExistential(),
             ):
            result = svc.chat("恢复自主行动", session_id="selfing-session")

        m_gate.assert_called_once_with("selfing-session", "恢复自主行动")
        m_enqueue.assert_called_once_with(session_id="selfing-session")
        self.assertEqual(result["response"], "resting")
        self.assertEqual(result["existential_mode"], "resting")
        self.assertEqual(result["mode_reason"], "test-rest")

    def test_non_resume_command_does_not_queue(self):
        svc = self._make_service()
        with patch("backend.chat_service.get_effective_session", side_effect=lambda s: s), \
             patch(
                 "backend.autonomy_gate.apply_user_autonomy_command_from_text",
                 return_value=None,
             ) as m_gate, \
             patch(
                 "backend.unified_scheduler.enqueue_autonomy_resume_check",
                 return_value=True,
             ) as m_enqueue, \
             patch(
                 "backend.daily_narrative.get_daily_narrative_generator",
                 return_value=_FakeDailyNarrativeGenerator(),
             ), \
             patch("backend.existential_state.ExistentialMode", _FakeExistentialMode), \
             patch(
                 "backend.existential_state.get_existential_state",
                 return_value=_FakeExistential(),
             ):
            svc.chat("只是普通聊天", session_id="selfing-session")

        m_gate.assert_called_once_with("selfing-session", "只是普通聊天")
        m_enqueue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
