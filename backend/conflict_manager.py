#!/usr/bin/env python3
"""
Conflict management for persona / rule lists.

- Conflicts are not erased; they are surfaced and bounded.
- The agent may express internal tension; we translate that into a ``tension_score``.
- Heuristic keyword pairs: Mandarin stems are retained; English opposites extend coverage.
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ConflictManager:
    """Lightweight rule-vs-rule contradiction detector for narrative / persona tooling."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
    
    def detect_conflicts(self, rules: List[Dict]) -> List[Dict]:
        """
        Scan pairwise rule texts for opposing stance keywords.

        Args:
            rules: list of dicts, each containing at least a ``text`` field

        Returns:
            List of conflict dicts with ``type``, ``rules``, ``severity``, ``description``.
        """
        conflicts = []
        
        # Example tension: "I must be honest" vs "I may lie" — extend opposites with EN glosses.
        contradiction_keywords = {
            "诚实": ["撒谎", "欺骗", "隐瞒", "lie", "lying", "deceive", "conceal", "dishonest"],
            "帮助": ["拒绝", "忽视", "放弃", "refuse", "ignore", "abandon", "neglect"],
            "探索": ["保守", "安全", "避免风险", "risk averse", "play it safe", "stay safe", "avoid risk"],
            "理性": ["非理性", "直觉", "情感驱动", "irrational", "intuition", "emotion driven"],
            "证据": ["直觉", "猜测", "假设", "guess", "speculation", "no evidence"],
            "谨慎": ["冒险", "大胆", "激进", "reckless", "bold move", "high risk"],
        }
        
        for i, rule1 in enumerate(rules):
            text1 = rule1.get("text", "").lower()
            for j, rule2 in enumerate(rules[i+1:], start=i+1):
                text2 = rule2.get("text", "").lower()
                
                for key, opposites in contradiction_keywords.items():
                    if key in text1:
                        for opposite in opposites:
                            if opposite in text2:
                                conflicts.append({
                                    "type": "contradiction",
                                    "rules": [rule1, rule2],
                                    "severity": 0.7,
                                    "description": f"Rule conflict: '{key}' vs '{opposite}'",
                                })
                                break
        
        return conflicts
    
    def manage_conflicts(self, conflicts: List[Dict]) -> Dict:
        """
        Keep a bounded subset of conflicts and pick a resolution stance.

        Args:
            conflicts: output of ``detect_conflicts``

        Returns:
            Dict with ``allowed_conflicts``, ``tension_score``, ``resolution_strategy``.
        """
        if not conflicts:
            return {
                "allowed_conflicts": [],
                "tension_score": 0.0,
                "resolution_strategy": "none"
            }
        
        # Retain at most ~30% of reported conflicts (minimum one slot when non-empty).
        max_allowed_conflicts = max(1, int(len(conflicts) * 0.3))
        allowed_conflicts = conflicts[:max_allowed_conflicts]
        
        tension_score = min(1.0, len(allowed_conflicts) / 10.0)
        
        if tension_score > 0.7:
            resolution_strategy = "embrace"
        elif tension_score > 0.3:
            resolution_strategy = "balance"
        else:
            resolution_strategy = "contextual"
        
        logger.info(
            "Conflict management: detected %d conflicts, keeping %d, tension=%.2f, strategy=%s",
            len(conflicts),
            len(allowed_conflicts),
            tension_score,
            resolution_strategy,
        )
        
        return {
            "allowed_conflicts": allowed_conflicts,
            "tension_score": tension_score,
            "resolution_strategy": resolution_strategy
        }
    
    def generate_conflict_rule(self, conflict: Dict) -> Optional[str]:
        """
        Produce a short meta-rule string summarizing the tension (English template).

        Args:
            conflict: dict containing ``rules`` (two entries)

        Returns:
            A one-line stance string, or ``None`` if inputs are unusable.
        """
        rules = conflict.get("rules", [])
        if len(rules) < 2:
            return None
        
        text1 = rules[0].get("text", "")
        text2 = rules[1].get("text", "")
        
        if not text1 or not text2:
            return None
        
        key1 = text1[:30] if len(text1) > 30 else text1
        key2 = text2[:30] if len(text2) > 30 else text2
        
        conflict_rule = (
            f"I hold productive tension between '{key1}' and '{key2}'; "
            "that friction is part of how I stay honest about trade-offs."
        )
        
        logger.debug("Generated conflict rule: %s", conflict_rule)
        
        return conflict_rule
