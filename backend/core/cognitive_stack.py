"""
Bounded metacognitive stack for optional reflection scaffolds in prompts.

Snapshots are kept in-process only (no SQLite persistence in this module).
"""
import logging
from typing import List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CognitiveState:
    level: int  # metacognitive depth (0: baseline, 1: meta-observe, 2: deep reflection)
    content: str  # short note captured for this frame
    timestamp: str

class CognitiveStack:
    """
    Bounded stack of recent metacognitive snapshots.

    Used to thread light recursion hints into prompts without persisting to SQLite yet.
    """
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.stack: List[CognitiveState] = []
        self.max_depth = 3

    def push(self, content: str, level: int = 0):
        """Append a new cognitive snapshot."""
        state = CognitiveState(
            level=level,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        self.stack.append(state)
        if len(self.stack) > 10:
            self.stack.pop(0)

    def get_recent_states(self, limit: int = 3) -> List[CognitiveState]:
        return self.stack[-limit:]

    def generate_recursion_prompt(self, current_context: str) -> str:
        """
        Build a gentle, non-technical reflection scaffold (English, model-facing).
        """
        return f"""
[Reflection depth — optional inner pass]
Move through these layers at your own pace:

1. **Direct contact**: Facing “{current_context}”, what is my first felt response?
2. **Empathic widen**: What need or fear might sit under that response—for me and for the other?
3. **Values check**: Does the move I want to make still feel kind and proportionate?
4. **Integration**: Let the answer read as one honest paragraph, not a checklist.
"""
