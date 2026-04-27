#!/usr/bin/env python3
"""
One-off repair for historical ``self_tick.py`` layout issues:

1. Align the comment + ``try:`` that followed a specific ``logger.info`` (same indent as that log line).
2. Split a combined ``logger.debug(...) energy = ...`` line so ``energy`` sits outside the ``except`` body.

Supports both **legacy Chinese** debug text and the current **English** heartbeat message.
"""
import re

def main():
    filepath = 'backend/self_tick.py'
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Fix 1: line numbers are 1-based; list indices are 0-based (legacy snapshot).
    line_idx_183 = 182  # logger.info
    line_idx_184 = 183  # comment
    line_idx_185 = 184  # try:

    indent_183 = len(lines[line_idx_183]) - len(lines[line_idx_183].lstrip())
    indent_184 = len(lines[line_idx_184]) - len(lines[line_idx_184].lstrip())
    indent_185 = len(lines[line_idx_185]) - len(lines[line_idx_185].lstrip())

    print(f"Line 183 indent={indent_183}, text={lines[line_idx_183].rstrip()}")
    print(f"Line 184 indent={indent_184}, text={lines[line_idx_184].rstrip()}")
    print(f"Line 185 indent={indent_185}, text={lines[line_idx_185].rstrip()}")

    if indent_184 == indent_183 + 4:
        lines[line_idx_184] = ' ' * indent_183 + lines[line_idx_184].lstrip()
        print(f"Normalized line 184 indent to {indent_183}")
    if indent_185 == indent_183 + 4:
        lines[line_idx_185] = ' ' * indent_183 + lines[line_idx_185].lstrip()
        print(f"Normalized line 185 indent to {indent_183}")

    # Fix 2: locate a fused debug+energy line (content evolved; match CN or EN).
    split_regexes = (
        r'^( *)(logger\.debug\(f"检查待办任务失败: .*?"\))( +)(energy = self_model\.get_energy\(session_id\))',
        r'^( *)(logger\.debug\(f"HEARTBEAT pending-task scan failed: .*?"\))( +)(energy = self_model\.get_energy\(session_id\))',
    )

    for i, line in enumerate(lines):
        legacy = '检查待办任务失败' in line
        modern = 'pending-task scan failed' in line.lower() and 'energy = self_model.get_energy' in line
        if not (legacy or modern) or 'energy = self_model.get_energy' not in line:
            continue

        print(f"Candidate fused line at {i + 1}")
        matched = None
        for rx in split_regexes:
            matched = re.search(rx, line)
            if matched:
                break
        if matched:
            indent = matched.group(1)
            debug_stmt = matched.group(2)
            energy_stmt = matched.group(4)
            for j in range(i - 1, max(-1, i - 10), -1):
                if 'except Exception as e:' in lines[j]:
                    except_indent = len(lines[j]) - len(lines[j].lstrip())
                    print(f"Matched ``except`` at line {j + 1}, indent={except_indent}")
                    new_debug_indent = except_indent + 4
                    new_energy_indent = except_indent
                    new_debug_line = ' ' * new_debug_indent + debug_stmt + '\n'
                    new_energy_line = ' ' * new_energy_indent + energy_stmt + '\n'
                    lines[i] = new_debug_line + new_energy_line
                    print(f"Split into:\n  {new_debug_line.rstrip()}\n  {new_energy_line.rstrip()}")
                    break
        else:
            print("Regex split failed; attempting naive split before ``energy =``")
            parts = line.split('energy = self_model.get_energy')
            if len(parts) == 2:
                debug_part = parts[0].rstrip()
                energy_part = 'energy = self_model.get_energy' + parts[1]
                debug_indent = len(debug_part) - len(debug_part.lstrip())
                energy_indent = max(0, debug_indent - 4)
                new_debug_line = debug_part + '\n'
                new_energy_line = ' ' * energy_indent + energy_part.lstrip()
                lines[i] = new_debug_line + new_energy_line
                print(f"Naive split:\n  {new_debug_line.rstrip()}\n  {new_energy_line.rstrip()}")
        break

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Done. Wrote {filepath}")

if __name__ == '__main__':
    main()
