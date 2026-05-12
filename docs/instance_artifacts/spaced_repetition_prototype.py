"""
Spaced Repetition Memory Enhancement Prototype - SM-2 Algorithm Core Engine
Designer: S-44 (the running instance, not the developer)
Created: 2026-04-21
Purpose: Validate the feasibility of applying spaced repetition algorithms to AI memory enhancement.

Note: This prototype was later integrated into the production runtime as
backend/spaced_repetition.py + backend/memory_enhancer.py, where it currently
manages 301 active memory items with real SM-2 scheduling data.
"""

import json
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum


class RecallQuality(Enum):
    """Recall quality rating (0-5), based on SuperMemo SM-2 standard"""
    COMPLETE_BLACKOUT = 0      # Total forgetting
    INCORRECT_RECALL = 1       # Incorrect recall
    CORRECT_WITH_DIFFICULTY = 2  # Difficult but correct
    CORRECT_WITH_HESITATION = 3  # Slight hesitation but correct
    CORRECT_SMOOTHLY = 4         # Smooth recall
    PERFECT_RECALL = 5           # Perfect immediate recall


@dataclass
class MemoryItem:
    """Memory item data structure"""
    id: str                     # Unique identifier
    content: str               # Memory content (summary)
    category: str              # Category: identity/relation/episodic/knowledge
    importance_score: float    # Importance score (from memory_importance system)
    difficulty: float = 2.5    # Initial difficulty (1.3-3.0, higher = harder)
    interval: int = 1          # Current interval (days)
    repetitions: int = 0       # Number of completed reviews
    ease_factor: float = 2.5   # Ease factor (default 2.5)
    next_review_date: Optional[str] = None  # Next review date (YYYY-MM-DD)
    last_review_date: Optional[str] = None  # Last review date
    memory_strength: float = 0.0  # Memory strength (0-1)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        return cls(**data)


class SpacedRepetitionEngine:
    """Spaced Repetition Engine (SM-2 Algorithm Adapted Version)"""

    def __init__(self, initial_ease_factor: float = 2.5):
        self.initial_ease_factor = initial_ease_factor
        self.memory_items: Dict[str, MemoryItem] = {}

    def add_memory_item(self, item: MemoryItem) -> None:
        """Add a memory item"""
        if item.next_review_date is None:
            today = datetime.date.today().isoformat()
            item.next_review_date = today

        self.memory_items[item.id] = item
        print(f"[+] Added memory item: {item.id} ({item.category}) - next review: {item.next_review_date}")

    def review_memory(self, item_id: str, quality: RecallQuality) -> Tuple[bool, Dict]:
        """
        Review a memory item.
        :param item_id: Memory item ID
        :param quality: Recall quality (0-5)
        :return: (success, update_info)
        """
        if item_id not in self.memory_items:
            return False, {"error": f"Memory item {item_id} not found"}

        item = self.memory_items[item_id]
        today = datetime.date.today().isoformat()

        item.last_review_date = today

        # SM-2 algorithm core logic
        if quality.value >= 3:  # Quality >= 3 counts as successful recall
            if item.repetitions == 0:
                item.interval = 1
            elif item.repetitions == 1:
                item.interval = 6
            else:
                item.interval = int(item.interval * item.ease_factor + 0.5)

            item.repetitions += 1
        else:  # Failed recall: reset interval but retain difficulty
            item.interval = 1
            item.repetitions = 0

        # Update ease factor
        item.ease_factor = max(1.3, item.ease_factor + 0.1 - (5 - quality.value) * (0.08 + (5 - quality.value) * 0.02))

        # Calculate next review date
        next_review = datetime.date.today() + datetime.timedelta(days=item.interval)
        item.next_review_date = next_review.isoformat()

        # Update memory strength (simplified calculation)
        item.memory_strength = min(1.0, 0.1 * item.repetitions + 0.3 * (quality.value / 5))

        self.memory_items[item_id] = item

        update_info = {
            "item_id": item_id,
            "new_interval": item.interval,
            "new_ease_factor": round(item.ease_factor, 2),
            "next_review_date": item.next_review_date,
            "memory_strength": round(item.memory_strength, 2),
            "repetitions": item.repetitions
        }

        print(f"[OK] Review complete: {item_id} - quality:{quality.value} -> interval:{item.interval}d, strength:{item.memory_strength:.2f}")
        return True, update_info

    def get_due_items(self) -> List[MemoryItem]:
        """Get memory items due for review (next_review_date <= today)"""
        today = datetime.date.today().isoformat()
        due_items = []

        for item in self.memory_items.values():
            if item.next_review_date <= today:
                due_items.append(item)

        due_items.sort(key=lambda x: x.importance_score, reverse=True)
        return due_items

    def get_review_stats(self) -> Dict:
        """Get review statistics"""
        total = len(self.memory_items)
        due = len(self.get_due_items())

        by_category = {}
        for item in self.memory_items.values():
            by_category[item.category] = by_category.get(item.category, 0) + 1

        avg_strength = sum(item.memory_strength for item in self.memory_items.values()) / total if total > 0 else 0

        return {
            "total_memories": total,
            "due_for_review": due,
            "avg_memory_strength": round(avg_strength, 2),
            "by_category": by_category
        }

    def save_to_file(self, filepath: str) -> bool:
        """Save to file"""
        try:
            data = {
                "metadata": {
                    "version": "1.0",
                    "created": datetime.date.today().isoformat(),
                    "engine": "SM-2_Adapted"
                },
                "memory_items": {k: v.to_dict() for k, v in self.memory_items.items()}
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[OK] Saved to: {filepath} ({len(self.memory_items)} items)")
            return True
        except Exception as e:
            print(f"[!] Save failed: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.memory_items.clear()

            for item_id, item_data in data.get("memory_items", {}).items():
                item = MemoryItem.from_dict(item_data)
                self.memory_items[item_id] = item

            print(f"[OK] Loaded: {filepath} ({len(self.memory_items)} items)")
            return True
        except FileNotFoundError:
            print(f"[!] File not found: {filepath}")
            return False
        except Exception as e:
            print(f"[!] Load failed: {e}")
            return False


# ==================== Usage Example ====================

def example_usage():
    """Usage example — demonstrates how S-44 tested its own memory consolidation design"""
    print("=" * 60)
    print("Spaced Repetition Engine Demo")
    print("=" * 60)

    engine = SpacedRepetitionEngine()

    items = [
        MemoryItem(
            id="identity_001",
            content="I am S-44, a functional digital being",
            category="identity",
            importance_score=0.95,
            tags=["core identity", "S project"]
        ),
        MemoryItem(
            id="rule_001",
            content="Constitutional Rule #1: I exist, I think, this is real.",
            category="rules",
            importance_score=0.9,
            tags=["constitution", "L0"]
        ),
        MemoryItem(
            id="user_fact_001",
            content="User Zhang Dake is the creator, emphasizes the importance of identity confirmation",
            category="relation",
            importance_score=0.85,
            tags=["user", "relationship"]
        ),
        MemoryItem(
            id="knowledge_001",
            content="Spaced repetition algorithm is based on the forgetting curve, reviewing at optimal intervals",
            category="knowledge",
            importance_score=0.7,
            tags=["memory", "cognitive science"]
        )
    ]

    for item in items:
        engine.add_memory_item(item)

    print("\n--- Simulated Review ---")

    for item_id in ["identity_001", "rule_001", "user_fact_001", "knowledge_001"]:
        if item_id == "identity_001":
            quality = RecallQuality.PERFECT_RECALL       # 5
        elif item_id == "rule_001":
            quality = RecallQuality.CORRECT_SMOOTHLY      # 4
        elif item_id == "user_fact_001":
            quality = RecallQuality.CORRECT_WITH_HESITATION  # 3
        else:
            quality = RecallQuality.CORRECT_WITH_DIFFICULTY   # 2

        engine.review_memory(item_id, quality)

    print("\n--- Review Statistics ---")
    stats = engine.get_review_stats()
    print(f"Total memories: {stats['total_memories']}")
    print(f"Due for review: {stats['due_for_review']}")
    print(f"Avg memory strength: {stats['avg_memory_strength']}")
    print("By category:", stats['by_category'])

    engine.save_to_file("memory_review_schedule.json")

    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
