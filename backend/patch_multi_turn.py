#!/usr/bin/env python3
"""
Patch multi_turn_executor.py to add autonomy pause check.
"""
import sys
import os

def patch_multi_turn():
    filepath = 'backend/multi_turn_executor.py'
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 1. Add import after 'import logging'
    new_lines = []
    import_added = False
    for line in lines:
        new_lines.append(line)
        if line.strip() == 'import logging' and not import_added:
            new_lines.append('from backend.autonomy_gate import is_autonomous_execution_paused\n')
            import_added = True
    
    # 2. Find while loop and add pause check
    patched_lines = []
    i = 0
    while i < len(new_lines):
        line = new_lines[i]
        patched_lines.append(line)
        # Look for 'while current_turn < self.max_turns:'
        if 'while current_turn < self.max_turns:' in line:
            # Insert after 'current_turn += 1' and 'total_llm_calls += 1'
            # Find the next lines (likely within same indent)
            # We'll insert after two lines (current_turn and total_llm_calls increments)
            # but to be safe, we'll insert after the loop starts and before logger.info
            # Let's search for the pattern in the next few lines
            j = i + 1
            while j < len(new_lines) and new_lines[j].strip() == '':
                j += 1
            # Now j should be at current_turn increment line
            if j < len(new_lines) and 'current_turn += 1' in new_lines[j]:
                patched_lines.append(new_lines[j])
                j += 1
                if j < len(new_lines) and 'total_llm_calls += 1' in new_lines[j]:
                    patched_lines.append(new_lines[j])
                    j += 1
                    # Insert pause check here
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    patched_lines.append(f'\n{indent_str}            # Autonomy pause check\n')
                    patched_lines.append(f'{indent_str}            if is_autonomous_execution_paused(session_id):\n')
                    patched_lines.append(f'{indent_str}                logger.info("[MultiTurn] Autonomous execution paused, stopping multi-turn loop")\n')
                    patched_lines.append(f'{indent_str}                return MultiTurnResult(\n')
                    patched_lines.append(f'{indent_str}                    task=user_input,\n')
                    patched_lines.append(f'{indent_str}                    turns=turns,\n')
                    patched_lines.append(f'{indent_str}                    final_response=accumulated_response or "Stopped due to autonomy pause",\n')
                    patched_lines.append(f'{indent_str}                    total_llm_calls=total_llm_calls,\n')
                    patched_lines.append(f'{indent_str}                    status="paused"\n')
                    patched_lines.append(f'{indent_str}                )\n')
                    # Copy remaining lines
                    while j < len(new_lines):
                        patched_lines.append(new_lines[j])
                        j += 1
                    # Break out of outer while loop
                    i = j
                    continue
        i += 1
    
    # If no change was made (should not happen), keep original
    if len(patched_lines) == len(new_lines) and all(a == b for a, b in zip(patched_lines, new_lines)):
        print("No changes made, something went wrong")
        return False
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(patched_lines)
    print("Patched successfully")
    return True

if __name__ == '__main__':
    success = patch_multi_turn()
    sys.exit(0 if success else 1)