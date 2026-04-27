#!/usr/bin/env python3
"""
Spaced-repetition engine (SM-2 adapted) with JSON persistence.

S-44 / 2026-04-21 — v1.0: item CRUD, SM-2 interval updates, due-item queries,
and ``spaced_memory.json`` storage (optional future SQLite table elsewhere).
"""

import json
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class RecallQuality(Enum):
    """SM-2 recall quality (0–5)."""
    COMPLETE_BLACKOUT = 0
    INCORRECT_RECALL = 1
    CORRECT_WITH_DIFFICULTY = 2
    CORRECT_WITH_HESITATION = 3
    CORRECT_SMOOTHLY = 4
    PERFECT_RECALL = 5


@dataclass
class MemoryItem:
    """One spaced-repetition card."""
    id: str                     # stable unique id
    content: str               # summary / snippet text
    category: str              # identity | relation | episodic | knowledge
    importance_score: float    # from MemoryImportanceEvaluator (0–1)
    difficulty: float = 2.5    # optional difficulty hint (1.3–3.0)
    interval: int = 1          # current interval in days
    repetitions: int = 0       # successful review count
    ease_factor: float = 2.5   # SM-2 ease factor
    next_review_date: Optional[str] = None  # YYYY-MM-DD
    last_review_date: Optional[str] = None  # YYYY-MM-DD
    memory_strength: float = 0.0  # coarse strength (0–1)
    tags: List[str] = field(default_factory=list)  # free-form tags
    created_date: str = field(default_factory=lambda: datetime.date.today().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(**data)


class SpacedRepetitionEngine:
    """SM-2-style scheduler with JSON backing store."""

    def __init__(self, data_dir: str = "data"):
        """``data_dir`` holds ``spaced_memory.json``."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / "spaced_memory.json"
        self.memory_items: Dict[str, MemoryItem] = {}
        
        self._load()
        logger.info(
            "SpacedRepetitionEngine ready (%s items) data_dir=%s",
            len(self.memory_items),
            self.data_dir,
        )

    def _load(self) -> bool:
        """Load ``memory_items`` from disk."""
        try:
            if not self.data_file.exists():
                logger.info("No spaced_memory.json yet; will create on first save")
                return True
            
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.memory_items.clear()

            for item_id, item_data in data.get("memory_items", {}).items():
                item = MemoryItem.from_dict(item_data)
                self.memory_items[item_id] = item

            logger.info("Loaded %s spaced-repetition items", len(self.memory_items))
            return True

        except FileNotFoundError:
            logger.warning("Data file missing: %s", self.data_file)
            return False
        except json.JSONDecodeError as e:
            logger.error("JSON decode error in spaced_memory.json: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to load spaced repetition data: %s", e)
            return False

    def _save(self) -> bool:
        """Persist ``memory_items`` to JSON."""
        try:
            data = {
                "metadata": {
                    "version": "1.0",
                    "created": datetime.date.today().isoformat(),
                    "engine": "SM-2_Adapted",
                    "last_updated": datetime.datetime.now().isoformat()
                },
                "memory_items": {k: v.to_dict() for k, v in self.memory_items.items()}
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Saved %s spaced-repetition items", len(self.memory_items))
            return True
        except Exception as e:
            logger.error("Failed to save spaced repetition data: %s", e)
            return False

    def add_memory_item(self, item: MemoryItem) -> str:
        """Insert or replace ``item`` and flush to disk."""
        if item.next_review_date is None:
            today = datetime.date.today().isoformat()
            item.next_review_date = today
        
        self.memory_items[item.id] = item
        self._save()
        
        logger.info(
            "Added memory item %s (%s) importance=%.2f",
            item.id,
            item.category,
            item.importance_score,
        )
        return item.id

    def review_memory(self, item_id: str, quality: int) -> Tuple[bool, Dict[str, Any]]:
        """Apply one SM-2 review step; ``quality`` is 0–5."""
        if item_id not in self.memory_items:
            logger.warning("Unknown memory item: %s", item_id)
            return False, {"error": f"memory item {item_id} not found"}
        
        item = self.memory_items[item_id]
        today = datetime.date.today().isoformat()
        
        item.last_review_date = today

        if quality >= 3:
            if item.repetitions == 0:
                item.interval = 1
            elif item.repetitions == 1:
                item.interval = 6
            else:
                item.interval = int(item.interval * item.ease_factor + 0.5)
            
            item.repetitions += 1
        else:
            item.interval = 1
            item.repetitions = 0

        item.ease_factor = max(1.3, item.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

        next_review = datetime.date.today() + datetime.timedelta(days=item.interval)
        item.next_review_date = next_review.isoformat()

        item.memory_strength = min(1.0, 0.1 * item.repetitions + 0.3 * (quality / 5))

        self.memory_items[item_id] = item
        self._save()
        
        update_info = {
            "item_id": item_id,
            "new_interval": item.interval,
            "new_ease_factor": round(item.ease_factor, 2),
            "next_review_date": item.next_review_date,
            "memory_strength": round(item.memory_strength, 2),
            "repetitions": item.repetitions
        }
        
        logger.info(
            "Reviewed %s quality=%s interval_days=%s strength=%.2f EF=%.2f",
            item_id,
            quality,
            item.interval,
            item.memory_strength,
            item.ease_factor,
        )
        return True, update_info

    def get_due_items(self, limit: int = 10) -> List[MemoryItem]:
        """Items with ``next_review_date`` on/before today (``None`` treated as due)."""
        today = datetime.date.today().isoformat()
        due_items = []
        
        for item in self.memory_items.values():
            nrd = item.next_review_date
            if nrd is None or nrd <= today:
                due_items.append(item)
        
        due_items.sort(key=lambda x: x.importance_score, reverse=True)
        return due_items[:limit]

    def get_review_stats(self) -> Dict[str, Any]:
        """Lightweight counters for dashboards."""
        total = len(self.memory_items)
        due = len(self.get_due_items(limit=1000))

        by_category = {}
        for item in self.memory_items.values():
            by_category[item.category] = by_category.get(item.category, 0) + 1
        
        avg_strength = sum(item.memory_strength for item in self.memory_items.values()) / total if total > 0 else 0

        next_review_dates = {}
        for item in self.memory_items.values():
            date = item.next_review_date
            next_review_dates[date] = next_review_dates.get(date, 0) + 1
        
        return {
            "total_memories": total,
            "due_for_review": due,
            "avg_memory_strength": round(avg_strength, 3),
            "by_category": by_category,
            "next_review_distribution": next_review_dates
        }
    
    def get_memory_item(self, item_id: str) -> Optional[MemoryItem]:
        return self.memory_items.get(item_id)

    def update_importance_score(self, item_id: str, new_score: float) -> bool:
        """Mutate ``importance_score`` and save."""
        if item_id not in self.memory_items:
            return False
        
        item = self.memory_items[item_id]
        old_score = item.importance_score
        item.importance_score = new_score
        self._save()
        
        logger.info("Updated importance %s: %.2f → %.2f", item_id, old_score, new_score)
        return True

    def cleanup_old_memories(self, max_age_days: int = 365) -> int:
        """Delete weak, very old items; returns number removed."""
        cutoff_date = (datetime.date.today() - datetime.timedelta(days=max_age_days)).isoformat()
        to_delete = []
        
        for item_id, item in self.memory_items.items():
            if item.created_date < cutoff_date and item.memory_strength < 0.3:
                to_delete.append(item_id)
        
        for item_id in to_delete:
            del self.memory_items[item_id]
        
        if to_delete:
            self._save()
            logger.info("Cleaned up %s stale spaced-repetition items", len(to_delete))

        return len(to_delete)


def create_engine() -> SpacedRepetitionEngine:
    return SpacedRepetitionEngine(data_dir="data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = create_engine()
    stats = engine.get_review_stats()
    print("stats:", stats)

    due_items = engine.get_due_items()
    print("due count:", len(due_items))

    for item in due_items[:3]:
        print(f"  - {item.id}: {item.content[:50]}... (importance={item.importance_score:.2f})")