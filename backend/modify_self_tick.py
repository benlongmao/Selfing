#!/usr/bin/env python3
"""
Legacy one-shot patcher for ``backend/self_tick.py``.

Inserts an S44-style HEARTBEAT pending-task scan immediately after
``energy = self_model.get_energy(session_id)``. The same logic already lives in
``self_tick.py`` in the main tree; **do not re-run** unless you know the target
file lacks that block (re-running can duplicate code).

Run from the **repository root** so ``backend/self_tick.py`` resolves correctly.
"""

import re


def main():
    with open("backend/self_tick.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Anchor: line that reads current energy for the tick.
    pattern = r"(energy = self_model\.get_energy\(session_id\)\s*\n)"

    # Injected block mirrors bilingual HEARTBEAT headers + English operator logs.
    insert_code = '''    # [S44] Scan HEARTBEAT.md for pending tasks after energy read
    try:
        workspace_root = config.get("paths.workspace_root", "workspace/sandbox")
        heartbeat_path = os.path.join(workspace_root, "HEARTBEAT.md")
        if os.path.exists(heartbeat_path):
            with open(heartbeat_path, 'r', encoding='utf-8') as f:
                content = f.read()
                import re
                pending_section = re.search(
                    r'- \\*\\*(?:待办任务|Pending tasks)\\*\\*(.*?)(?=\\n##|\\Z)',
                    content,
                    re.DOTALL,
                )
                if pending_section:
                    pending_text = pending_section.group(1)
                    pending_count = pending_text.count('- ')
                    logger.info(
                        f"📋 Found {pending_count} pending task(s) in HEARTBEAT; consider S44_continue or tick reminder."
                    )
    except Exception as e:
        logger.debug(f"HEARTBEAT pending-task scan failed: {e}")
'''

    def replace_func(match):
        raw = match.group(0)
        indent = raw[: len(raw) - len(raw.lstrip())]
        lines = insert_code.split("\n")
        indented_lines = [indent + line if line.strip() else "" for line in lines]
        return match.group(0) + "\n".join(indented_lines)

    new_content = re.sub(pattern, replace_func, content)

    if new_content != content:
        with open("backend/self_tick.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated backend/self_tick.py")
    else:
        print("No matching anchor line; file unchanged")


if __name__ == "__main__":
    main()
