import os
import json
import random
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class EndogenousSystem:
    """
    Lightweight endogenous motivation / urge generator.

    Tracks boredom and curiosity per session and occasionally emits short urge lines
    (already English) for the planner or prompt injector.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.state_file = os.path.join(data_dir, "endogenous_state.json")
        self.states = self._load_states()
        
        self.boredom_threshold = 0.6
        self.curiosity_threshold = 0.5
        self.urge_decay = 0.1  # per-update decay constant (reserved for future use)
        
    def _load_states(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load endogenous states: {e}")
                return {}
        return {}

    def _save_states(self):
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.states, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save endogenous states: {e}")

    def _get_session_state(self, session_id: str) -> Dict:
        if session_id not in self.states:
            self.states[session_id] = {
                "boredom": 0.0,
                "curiosity": 0.2,
                "urges": [],
                "last_topic": "",
                "topic_repeat_count": 0,
                "unresolved_questions": []
            }
        return self.states[session_id]

    def update(self, session_id: str, user_input: str) -> List[str]:
        """
        Update boredom/curiosity heuristics from ``user_input`` and return fresh urge strings.
        """
        state = self._get_session_state(session_id)
        
        # 1) Boredom — very short or duplicated input increases it
        if len(user_input) < 5:
            state["boredom"] = min(1.0, state["boredom"] + 0.15)
        elif user_input == state.get("last_input", ""):
            state["boredom"] = min(1.0, state["boredom"] + 0.3)
        else:
            state["boredom"] = max(0.0, state["boredom"] - 0.2)
            
        state["last_input"] = user_input
        
        # 2) Curiosity — question-shaped cues (bilingual substrings + decay / rare spikes)
        triggers_cn = ["为什么", "怎么", "如果不", "假设", "?"]
        triggers_en = ["why", "how", "what if", "if ", "if?", "assume", "hypothetically"]
        ui_low = user_input.lower()
        if any(t in user_input for t in triggers_cn) or any(t in ui_low for t in triggers_en):
            state["curiosity"] = min(1.0, state["curiosity"] + 0.1)
        else:
            state["curiosity"] = max(0.0, state["curiosity"] - 0.05)
            
        if random.random() < 0.05:
            state["curiosity"] = min(1.0, state["curiosity"] + 0.3)
            logger.info(f"Session {session_id}: Spontaneous curiosity spike!")

        new_urges = []
        
        if state["boredom"] > self.boredom_threshold:
            urgency = "Strong" if state["boredom"] > 0.8 else "Mild"
            new_urges.append(
                f"[{urgency} boredom] The thread feels repetitive; I want to shift topic or wind down the exchange."
            )

        if state["curiosity"] > self.curiosity_threshold:
            new_urges.append(
                "[Exploration urge] I'm curious about deeper reasons behind the current topic and want more context."
            )

        if random.random() < 0.02:
            new_urges.append(
                "[Sudden association] An unrelated metaphor popped into mind and I'd like to share it."
            )

        state["urges"] = new_urges
        
        self._save_states()
        return new_urges

    def get_urges(self, session_id: str) -> List[str]:
        return self.states.get(session_id, {}).get("urges", [])

