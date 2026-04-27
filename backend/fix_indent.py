#!/usr/bin/env python3
"""
One-off repair script: realign a mis-indented idle/todo scan block in ``backend/self_tick.py``.

The regex targets **legacy** log/comment text that may no longer exist in current ``self_tick.py``;
keep the pattern unchanged if you are patching an older checkout.
"""

import re

def main():
    with open('backend/self_tick.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Locate the block starting at the legacy logger.info line (legacy CN log text) through the following ``try:``.
    # Be careful not to match unrelated ``except`` handlers elsewhere in the file.
    pattern = r'(\s*)logger\.info\(f"[S44进化] 检测到空闲状态，正在检查待办任务\.\.\."\)\s*\n\s*# \[S44进化增强\] 实际检查待办任务\s*\n\s*try:\s*\n'

    match = re.search(pattern, content)
    if not match:
        print("Target code block not found (file may already be fixed or log text changed).")
        return

    indent = match.group(1)  # leading whitespace before logger.info
    print(f"Matched indent prefix: {repr(indent)} (length {len(indent)})")

    try_pos = content.find("try:", match.end())
    if try_pos == -1:
        print("Could not find ``try:`` after the matched region.")
        return

    except_pattern = re.compile(r'\s*except Exception as e:\s*\n')
    except_match = except_pattern.search(content, try_pos)
    if not except_match:
        print("Could not find ``except Exception as e:`` after ``try:``.")
        return

    except_end = except_match.end()

    original_block = content[try_pos:except_end]
    print("Original block:")
    print(original_block)

    lines = original_block.split('\n')
    if lines[0].strip() == 'try:':
        try_indent = lines[0][:len(lines[0]) - len(lines[0].lstrip())]
        print(f"Current ``try:`` indent: {repr(try_indent)} (length {len(try_indent)})")
        offset = len(try_indent) - len(indent)
        print(f"Indent delta (try vs logger): {offset}")

        new_lines = []
        for line in lines:
            if line.strip() == '':
                new_lines.append('')
                continue
            current_indent = line[:len(line) - len(line.lstrip())]
            if len(current_indent) >= len(try_indent):
                new_indent_len = len(current_indent) - offset
                if new_indent_len < 0:
                    new_indent_len = 0
                new_indent = ' ' * new_indent_len
                new_line = new_indent + line.lstrip()
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        new_block = '\n'.join(new_lines)
        print("Rewritten block:")
        print(new_block)

        new_content = content[:try_pos] + new_block + content[except_end:]

        with open('backend/self_tick.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Write complete.")
    else:
        print("First line of extracted region is not ``try:``; aborting.")
        return

if __name__ == '__main__':
    main()
