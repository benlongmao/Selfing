#!/usr/bin/env python3
"""Regression tests for multi-turn entry, completion, and stop heuristics (``intent_markers``)."""
import unittest

from backend.intent_markers import (
    S44_CONTINUE_LITERAL,
    S44_COMPLETE_LITERAL,
    has_scheduler_continue_marker,
    should_block_multi_turn_before_loop,
    loop_has_stop_intent,
    explain_loop_stop_match,
    loop_has_complete_intent,
    basic_multiturn_has_complete,
    should_allow_implicit_continue_fallback,
)


class TestIntentMarkers(unittest.TestCase):
    def test_scheduler_continue_false_when_continue_only_in_prose(self):
        """Prose that merely mentions emitting S44_CONTINUE must not trigger scheduler continue."""
        ans = "需要将 x 接入 y，并在返回 True 时输出" + S44_CONTINUE_LITERAL + "。"
        self.assertFalse(has_scheduler_continue_marker(ans))
        ans2 = (
            "需要将 x 接入 y，并在返回 True 时输出"
            + S44_CONTINUE_LITERAL
            + "。\n\n"
            + S44_COMPLETE_LITERAL
        )
        self.assertFalse(has_scheduler_continue_marker(ans2))

    def test_scheduler_continue_true_on_own_line(self):
        ans = "小结如下。\n\n" + S44_CONTINUE_LITERAL
        self.assertTrue(has_scheduler_continue_marker(ans))

    def test_scheduler_continue_false_when_complete_even_with_line_marker(self):
        ans = S44_CONTINUE_LITERAL + "\n" + S44_COMPLETE_LITERAL
        self.assertFalse(has_scheduler_continue_marker(ans))

    def test_scheduler_continue_false_when_pause_line_present(self):
        ans = "停一下。\n[S44_PAUSE]\n" + S44_CONTINUE_LITERAL
        self.assertFalse(has_scheduler_continue_marker(ans))

    def test_entry_weak_summary_with_continue_allowed(self):
        vis = "分析如下。\n综上所述，第一点成立。\n" + S44_CONTINUE_LITERAL
        full = vis
        self.assertFalse(should_block_multi_turn_before_loop(vis, full))

    def test_entry_weak_summary_without_continue_still_allowed(self):
        """Weak wrap-up without bracketed stop should not hard-block multi-turn entry."""
        vis = "综上所述，结论如此。"
        self.assertFalse(should_block_multi_turn_before_loop(vis, vis))

    def test_entry_hard_stop_blocks_even_with_continue(self):
        vis = "我不再继续。\n" + S44_CONTINUE_LITERAL
        self.assertTrue(should_block_multi_turn_before_loop(vis, vis))

    def test_entry_complete_bracket_blocks(self):
        full = "done " + S44_COMPLETE_LITERAL
        self.assertTrue(should_block_multi_turn_before_loop(full, full))

    def test_loop_complete_overridden_by_continue(self):
        vis = "综上所述。\n" + S44_CONTINUE_LITERAL
        full = vis
        self.assertFalse(
            loop_has_complete_intent(vis, full, (S44_COMPLETE_LITERAL,))
        )

    def test_loop_complete_weak_when_no_continue(self):
        vis = "综上所述，完毕。"
        self.assertTrue(
            loop_has_complete_intent(vis, vis, (S44_COMPLETE_LITERAL,))
        )

    def test_loop_complete_explicit_bracket(self):
        vis = "x\n" + S44_COMPLETE_LITERAL
        full = vis + "\nreasoning_tail"
        self.assertTrue(
            loop_has_complete_intent(vis, full, (S44_COMPLETE_LITERAL,))
        )

    def test_loop_complete_bracket_only_in_reasoning_not_counted(self):
        vis = "我继续补充。"
        full = vis + "\n[S44_COMPLETE]"
        self.assertFalse(
            loop_has_complete_intent(vis, full, (S44_COMPLETE_LITERAL,))
        )

    def test_loop_stop_hard_natural(self):
        vis = "深度休眠"
        self.assertTrue(loop_has_stop_intent(vis, (S44_COMPLETE_LITERAL,)))

    def test_loop_stop_not_triggered_by_weak_only(self):
        vis = "综上所述，小结。"
        self.assertFalse(loop_has_stop_intent(vis, ()))

    def test_loop_stop_bracket_not_in_reasoning_tail_only(self):
        """Bracket stop token only in reasoning tail must not count as visible stop."""
        vis = "续写第二节。"
        self.assertFalse(loop_has_stop_intent(vis, (S44_COMPLETE_LITERAL,)))

    def test_explain_loop_stop_match(self):
        vis = "能否澄清一下需求？" + S44_COMPLETE_LITERAL
        self.assertIn("bracket:", explain_loop_stop_match(vis, (S44_COMPLETE_LITERAL,)))

    def test_basic_multiturn_same_as_loop_complete(self):
        vis = "综上所述"
        full = vis
        self.assertTrue(basic_multiturn_has_complete(vis, full))
        full2 = vis + "\n" + S44_CONTINUE_LITERAL
        self.assertFalse(basic_multiturn_has_complete(vis, full2))

    def test_implicit_continue_fallback_allows_task_tail_signal(self):
        vis = (
            "我先完成第一部分：整理现有证明思路与反例边界。\n"
            "接下来我要把第二部分的参数化构造补完，并给出可运行的验证脚本。"
        )
        allowed, reason = should_allow_implicit_continue_fallback(vis, vis)
        self.assertTrue(allowed)
        self.assertIn("implicit:", reason)

    def test_implicit_continue_fallback_blocks_short_chatty_line(self):
        vis = "哈哈，我继续。"
        allowed, reason = should_allow_implicit_continue_fallback(vis, vis)
        self.assertFalse(allowed)
        self.assertEqual(reason, "too_short")

    def test_implicit_continue_fallback_blocks_when_awaiting_user(self):
        vis = (
            "我已经把前两种方案都梳理出来了。下一步我会继续把第三种方案展开。"
            "你希望我先写代码还是先写设计文档？"
        )
        allowed, reason = should_allow_implicit_continue_fallback(vis, vis)
        self.assertFalse(allowed)
        self.assertEqual(reason, "awaiting_user")


if __name__ == "__main__":
    unittest.main()
