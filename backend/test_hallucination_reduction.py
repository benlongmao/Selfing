#!/usr/bin/env python3
"""
[2026-02-05] Manual harness for tool-result shaping + hallucination heuristics.

Covers:
1. ``standardize_tool_result`` / ``is_tool_result_error``
2. ``detect_hallucination_claim`` (Chinese + English assistant prose)
3. ``build_tool_usage_rules_block`` smoke test (English-first snippet)

Run from repo root:
    .venv/bin/python backend/test_hallucination_reduction.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.tool_result_standardizer import standardize_tool_result, is_tool_result_error
from backend.chat_tool_runner import detect_hallucination_claim
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_tool_result_standardization():
    """Exercise ``standardize_tool_result`` on success / failure / ambiguous payloads."""
    print("\n" + "=" * 60)
    print("Suite 1: tool result standardization")
    print("=" * 60)

    print("\n[case 1] write_file success with system verification")
    result1 = {
        "success": True,
        "message": "File written successfully",
        "system_verification": {
            "file_exists": True,
            "status": "OK: file exists on disk",
        },
        "content_preview": "hello world",
        "stats": {"bytes": 11, "lines": 1}
    }
    standardized1 = standardize_tool_result("write_file", result1, "test-session")

    print(f"  Status: {standardized1['details']['status']}")
    print(f"  Content: {standardized1['content'][0]['text'][:100]}")
    print(f"  Verification reminder: {standardized1.get('verification_reminder', 'N/A')[:100]}...")

    assert standardized1['details']['status'] == 'completed', "expected status completed"
    assert 'verification_reminder' in standardized1, "expected verification_reminder field"
    print("  OK")

    print("\n[case 2] execute_python failure with stderr")
    result2 = {
        "success": False,
        "stdout": "",
        "stderr": "NameError: name 'x' is not defined",
        "error": "Exit code: 1"
    }
    standardized2 = standardize_tool_result("execute_python", result2, "test-session")

    print(f"  Status: {standardized2['details']['status']}")
    print(f"  Is error: {is_tool_result_error(standardized2)}")
    print(f"  Content: {standardized2['content'][0]['text'][:100]}")

    assert standardized2['details']['status'] in ['failed', 'error'], "expected failed/error"
    assert is_tool_result_error(standardized2), "expected error classifier True"
    print("  OK")

    print("\n[case 3] unknown / minimal tool payload")
    result3 = {
        "data": {"count": 5},
        "note": "query finished"
    }
    standardized3 = standardize_tool_result("custom_tool", result3, "test-session")

    print(f"  Status: {standardized3['details']['status']}")
    print(f"  Content: {standardized3['content'][0]['text'][:100]}")

    assert 'status' in standardized3['details'], "expected details.status"
    print("  OK")


def test_hallucination_detection():
    """Exercise ``detect_hallucination_claim`` on CN/EN assistant outputs."""
    print("\n" + "=" * 60)
    print("Suite 2: hallucination detection")
    print("=" * 60)

    print("\n[case 1] claims file write with no tool calls → expect hallucination")
    response1 = "我已经创建了文件 test.txt，文件已成功写入。"
    tools1 = []
    is_hallucination1, reason1 = detect_hallucination_claim(response1, tools1, logger)

    print(f"  response: {response1}")
    print(f"  tools: {tools1}")
    print(f"  hallucination: {is_hallucination1}")
    print(f"  reason: {reason1}")

    assert is_hallucination1, "expected hallucination True"
    print("  OK")

    print("\n[case 2] write_file actually invoked → not hallucination")
    response2 = "文件已创建，内容已写入。"
    tools2 = ["write_file"]
    is_hallucination2, reason2 = detect_hallucination_claim(response2, tools2, logger)

    print(f"  response: {response2}")
    print(f"  tools: {tools2}")
    print(f"  hallucination: {is_hallucination2}")

    assert not is_hallucination2, "expected hallucination False"
    print("  OK")

    print("\n[case 3] fenced code + claims execution with no tools → hallucination")
    response3 = """我来创建文件：
```python
with open('test.txt', 'w') as f:
    f.write('hello')
```
代码执行成功，文件已创建。
"""
    tools3 = []
    is_hallucination3, reason3 = detect_hallucination_claim(response3, tools3, logger)

    print(f"  response length: {len(response3)} chars")
    print(f"  tools: {tools3}")
    print(f"  hallucination: {is_hallucination3}")
    print(f"  reason: {reason3}")

    assert is_hallucination3, "expected hallucination True"
    print("  OK")

    print("\n[case 4] fenced sample only, no execution claim → not hallucination")
    response4 = """你可以这样创建文件：
```python
with open('test.txt', 'w') as f:
    f.write('hello')
```
这是示例代码。
"""
    tools4 = []
    is_hallucination4, reason4 = detect_hallucination_claim(response4, tools4, logger)

    print(f"  response length: {len(response4)} chars")
    print(f"  tools: {tools4}")
    print(f"  hallucination: {is_hallucination4}")

    assert not is_hallucination4, "expected sample-only prose not flagged"
    print("  OK")

    print("\n[case 5] path claim without tools → hallucination")
    response5 = "文件已创建在 sandbox/test.txt"
    tools5 = []
    is_hallucination5, reason5 = detect_hallucination_claim(response5, tools5, logger)

    print(f"  response: {response5}")
    print(f"  tools: {tools5}")
    print(f"  hallucination: {is_hallucination5}")
    print(f"  reason: {reason5}")

    assert is_hallucination5, "expected hallucination True"
    print("  OK")


def test_prompt_integration():
    """Smoke-test ``build_tool_usage_rules_block`` (English-first minimal block)."""
    print("\n" + "=" * 60)
    print("Suite 3: prompt integration")
    print("=" * 60)

    try:
        from backend.prompt_builder_blocks_state import build_tool_usage_rules_block

        print("\n[check] build_tool_usage_rules_block()")
        rules_block = build_tool_usage_rules_block()

        print(f"  block length: {len(rules_block)} chars")
        print(f"  contains '[Tool workflow]': {'[Tool workflow]' in rules_block}")
        print(f"  contains 'list_files': {'list_files' in rules_block}")
        print(f"  contains 'agent_memory': {'agent_memory' in rules_block}")

        assert len(rules_block) > 80, "expected non-trivial tool workflow copy"
        assert "[Tool workflow]" in rules_block, "expected [Tool workflow] header"
        assert "list_files" in rules_block or "agent_memory" in rules_block, "expected concrete tool cues"

        print("  OK")

    except ImportError as e:
        print(f"  WARN — import failed: {e}")
        return False

    return True


def main():
    """Run all suites."""
    print("\n" + "=" * 80)
    print("S-main — hallucination / tool-shape diagnostic harness")
    print("=" * 80)

    try:
        test_tool_result_standardization()
        test_hallucination_detection()
        test_prompt_integration()

        print("\n" + "=" * 80)
        print("All suites finished successfully.")
        print("=" * 80)
        print("\nSummary:")
        print("1. Tool outputs gain normalized status + verification reminders where applicable.")
        print("2. detect_hallucination_claim flags tool claims without matching tool calls.")
        print("3. Tool workflow copy is injected via build_tool_usage_rules_block (English-first).")
        print("\nTips:")
        print("- Exercise real chats and watch for [HALLUCINATION-DETECT] logs.")
        print("- When prompts change, re-run this script from repo root.")

    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
