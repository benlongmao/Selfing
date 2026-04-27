#!/usr/bin/env python3
"""response_verifier: receipt append uses visible text only (not <thought> claims)."""
import unittest

from backend.response_verifier import enforce_receipts


class TestReceiptEnforceVisibleOnly(unittest.TestCase):
    def test_no_append_when_claim_only_in_thought(self):
        text = "<thought>我调用了 list_files 查看目录。</thought>"
        rids = ["rct_" + "a" * 32, "rct_" + "b" * 32]
        new_text, modified = enforce_receipts(text, rids)
        self.assertFalse(modified)
        self.assertEqual(new_text, text)

    def test_no_append_when_claim_only_in_意识流_line(self):
        """CN stream-of-consciousness prefix is non-visible; receipts must not attach from it alone."""
        text = "[意识流]: 我读取了文件，内容如下。\n"
        rids = ["rct_" + "f" * 32]
        new_text, modified = enforce_receipts(text, rids)
        self.assertFalse(modified)
        self.assertEqual(new_text, text)

    def test_append_when_visible_claims_tool(self):
        text = "我读取了文件，内容如下。\nfoo"
        rids = ["rct_" + "c" * 32]
        new_text, modified = enforce_receipts(text, rids)
        self.assertTrue(modified)
        self.assertIn("Receipts:", new_text)
        self.assertIn("rct_", new_text)

    def test_no_append_when_already_has_rct(self):
        text = "完成。rct_" + "d" * 32
        rids = ["rct_" + "e" * 32]
        new_text, modified = enforce_receipts(text, rids)
        self.assertFalse(modified)


if __name__ == "__main__":
    unittest.main()
