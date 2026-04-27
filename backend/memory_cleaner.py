#!/usr/bin/env python3
"""
Memory cleaner: hard-delete archived persona rules with backups.

Integrates ``BackupManager`` so rows are exported before ``DELETE``.

**Scope vs ``db_cleanup`` (P3):**
- **This module** only touches ``persona_items``: tiered retention for ``archived`` / ``suppressed``
  rows, with an audit trail in ``deleted_memories_log``.
- **``backend/db_cleanup.py``** owns periodic pruning for ``chat_turns``, ``sensory_memory_temp``,
  ``prompt_logs``, ``events`` (and writes chat summaries into ``self_biography``). The two paths
  do not overlap on the same tables.
"""
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from backend.backup_manager import BackupManager
from backend.config import config

logger = logging.getLogger(__name__)

class MemoryCleaner:
    """Hard-delete helper for ``persona_items`` with tiered retention."""

    # Tiered protection metadata (``name`` is surfaced on backup HTTP helpers).
    PROTECTION_LEVELS = {
        "ABSOLUTE": {
            "name": "Absolute (never delete)",
            "retention_days": float('inf'),
            "priority": 0
        },
        "CRITICAL": {
            "name": "Critical (5 years)",
            "retention_days": 1825,  # 5 years
            "priority": 1
        },
        "HIGH": {
            "name": "High (2 years)",
            "retention_days": 730,  # 2 years
            "priority": 2
        },
        "MEDIUM": {
            "name": "Medium (1 year)",
            "retention_days": 365,  # 1 year
            "priority": 3
        },
        "LOW": {
            "name": "Low (6 months)",
            "retention_days": 180,  # 6 months
            "priority": 4
        },
        "DISPOSABLE": {
            "name": "Disposable (90 days)",
            "retention_days": 90,  # 3 months
            "priority": 5
        }
    }
    
    def __init__(self, db_path: str, backup_manager: Optional[BackupManager] = None):
        self.db_path = db_path
        self.backup_manager = backup_manager or BackupManager(db_path)
        self._init_tables()
        logger.info("MemoryCleaner initialized")
    
    def _init_tables(self):
        """Create ``deleted_memories_log`` if missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deleted_memories_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id TEXT,
                    deleted_at TEXT,
                    protection_level TEXT,
                    retention_days INTEGER,
                    reason TEXT,
                    backup_file TEXT
                )
            """)
            conn.commit()
    
    def get_protection_level(self, item: Dict) -> str:
        """
        Map a ``persona_items`` row to the strictest applicable protection tier.

        Rules are evaluated from strongest to weakest.
        """
        if item.get("locked") == 1:
            return "ABSOLUTE"
        
        if item.get("is_core") == 1 and item.get("importance", 0) > 0.9:
            return "ABSOLUTE"
        
        anchors = config.get("system.identity_anchors", []) or []
        if anchors:
            text = item.get("text", "")
            if any(kw in text for kw in anchors):
                return "ABSOLUTE"
        
        if item.get("is_core") == 1:
            return "CRITICAL"
        
        if item.get("importance", 0) >= 0.8 or item.get("evidence_count", 0) > 10:
            return "CRITICAL"
        
        if item.get("importance", 0) >= 0.6 or item.get("evidence_count", 0) > 5:
            return "HIGH"
        
        if item.get("score", 0) > 0.5:
            return "HIGH"
        
        if item.get("importance", 0) >= 0.4 or item.get("score", 0) > 0.3:
            return "MEDIUM"
        
        if item.get("importance", 0) >= 0.2 or item.get("score", 0) > 0.2:
            return "LOW"
        
        return "DISPOSABLE"
    
    def safe_cleanup(
        self,
        dry_run: bool = True,
        max_delete: int = 1000
    ) -> Dict:
        """
        Analyze archived persona rows, optionally delete them after backup.

        Args:
            dry_run: When ``True``, only count / preview rows—no deletes or VACUUM.
            max_delete: Safety cap on rows removed in one invocation.

        Returns:
            Summary dict with ``analyzed``, ``backed_up``, ``deleted``, ``retained``,
            ``by_level``, optional ``deleted_items`` preview, and ``freed_mb``.
        """
        logger.info(f"Starting safe cleanup (dry_run={dry_run}, max_delete={max_delete})")
        
        results = {
            "analyzed": 0,
            "backed_up": 0,
            "deleted": 0,
            "retained": 0,
            "by_level": {},
            "deleted_items": [],
            "freed_mb": 0.0
        }
        
        candidates = self._get_cleanup_candidates()
        results["analyzed"] = len(candidates)
        
        if not candidates:
            logger.info("No candidates for cleanup")
            return results
        
        classified = self._classify_by_protection(candidates)

        now = datetime.now(timezone.utc)
        to_delete = []
        
        for level in sorted(self.PROTECTION_LEVELS.keys(), 
                           key=lambda x: self.PROTECTION_LEVELS[x]["priority"], 
                           reverse=True):
            
            items = classified.get(level, [])
            retention_days = self.PROTECTION_LEVELS[level]["retention_days"]
            
            deleted_in_level = 0
            retained_in_level = 0
            
            for item in items:
                last_seen = item.get("last_seen_at")
                if not last_seen:
                    retained_in_level += 1
                    continue
                
                try:
                    archived_at = datetime.fromisoformat(last_seen)
                    days_since_archived = (now - archived_at).days
                except Exception:
                    retained_in_level += 1
                    continue

                if days_since_archived > retention_days:
                    to_delete.append({
                        "item": item,
                        "level": level,
                        "days_archived": days_since_archived
                    })
                    deleted_in_level += 1
                else:
                    retained_in_level += 1
            
            results["by_level"][level] = {
                "total": len(items),
                "deleted": deleted_in_level,
                "retained": retained_in_level,
                "retention_policy": (
                    f"{int(retention_days)} days"
                    if retention_days != float("inf")
                    else "never delete"
                ),
            }

        if len(to_delete) > max_delete:
            logger.warning(f"Too many items to delete ({len(to_delete)}), limiting to {max_delete}")
            to_delete = to_delete[:max_delete]
        
        if to_delete:
            if dry_run:
                logger.info(f"[DRY RUN] Would delete {len(to_delete)} items")
                results["deleted"] = len(to_delete)
                results["deleted_items"] = [
                    {
                        "id": d["item"]["id"],
                        "text": d["item"].get("text", "")[:50],
                        "level": d["level"],
                        "days_archived": d["days_archived"]
                    }
                    for d in to_delete[:10]  # preview at most 10 ids
                ]
            else:
                deleted_items_for_backup = [d["item"] for d in to_delete]

                backup_result = self.backup_manager.backup_deleted_memories(
                    deleted_items_for_backup
                )
                results["backed_up"] = backup_result.get("count", 0)

                for delete_info in to_delete:
                    item = delete_info["item"]
                    self._hard_delete(
                        item["id"],
                        delete_info["level"],
                        delete_info["days_archived"],
                        backup_result.get("file")
                    )
                
                results["deleted"] = len(to_delete)
                results["deleted_items"] = [
                    {
                        "id": d["item"]["id"],
                        "text": d["item"].get("text", "")[:50],
                        "level": d["level"]
                    }
                    for d in to_delete[:10]
                ]
                
                results["freed_mb"] = self._vacuum_database()
        
        results["retained"] = results["analyzed"] - results["deleted"]
        
        logger.info(
            f"Cleanup completed: analyzed={results['analyzed']}, "
            f"deleted={results['deleted']}, retained={results['retained']}, "
            f"freed={results['freed_mb']:.2f}MB"
        )
        
        return results
    
    def _get_cleanup_candidates(self) -> List[Dict]:
        """Return ``archived`` / ``suppressed`` persona rows eligible for review."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT 
                    id, text, score, importance, novelty,
                    evidence_count, is_core, locked, status,
                    last_seen_at, created_at
                FROM persona_items
                WHERE status IN ('archived', 'suppressed')
                ORDER BY last_seen_at ASC
            """)
            return [dict(row) for row in cur.fetchall()]
    
    def _classify_by_protection(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """Bucket rows by ``get_protection_level`` outcome."""
        classified = {level: [] for level in self.PROTECTION_LEVELS.keys()}
        
        for item in items:
            level = self.get_protection_level(item)
            classified[level].append(item)
        
        return classified
    
    def _hard_delete(
        self, 
        item_id: str, 
        protection_level: str,
        retention_days: int,
        backup_file: Optional[str]
    ):
        """Delete a row after writing ``deleted_memories_log``."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO deleted_memories_log 
                    (original_id, deleted_at, protection_level, retention_days, reason, backup_file)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                item_id,
                datetime.now(timezone.utc).isoformat(),
                protection_level,
                retention_days,
                "auto_cleanup",
                backup_file
            ))

            conn.execute("DELETE FROM persona_items WHERE id = ?", (item_id,))
            conn.commit()
        
        logger.debug(f"Hard deleted: {item_id} (level={protection_level})")
    
    def _vacuum_database(self) -> float:
        """Run ``VACUUM`` and return estimated MiB reclaimed."""
        try:
            import os
            size_before = os.path.getsize(self.db_path)
            
            logger.info("Running VACUUM to reclaim space...")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
            
            size_after = os.path.getsize(self.db_path)
            freed_mb = (size_before - size_after) / 1024 / 1024
            
            logger.info(f"VACUUM completed: freed {freed_mb:.2f} MB")
            return freed_mb
        except Exception as e:
            logger.error(f"VACUUM failed: {e}")
            return 0.0
    
    def get_cleanup_stats(self) -> Dict:
        """Summaries for dashboards / backup tooling."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM persona_items
                GROUP BY status
            """)
            stats["by_status"] = {row["status"]: row["count"] for row in cur.fetchall()}

            now = datetime.now(timezone.utc)
            for level, config in self.PROTECTION_LEVELS.items():
                days = config["retention_days"]
                if days == float('inf'):
                    stats[f"cleanable_{level}"] = 0
                    continue
                
                cutoff = (now - timedelta(days=days)).isoformat()
                cur = conn.execute("""
                    SELECT COUNT(*) as count FROM persona_items
                    WHERE status IN ('archived', 'suppressed')
                    AND last_seen_at < ?
                """, (cutoff,))
                
                stats[f"cleanable_{level}"] = cur.fetchone()["count"]

            import os
            stats["db_file_size_mb"] = os.path.getsize(self.db_path) / 1024 / 1024

            cur = conn.execute("""
                SELECT COUNT(*) as total_deleted,
                       MIN(deleted_at) as first_deletion,
                       MAX(deleted_at) as last_deletion
                FROM deleted_memories_log
            """)
            row = cur.fetchone()
            stats["deletion_history"] = {
                "total_deleted": row["total_deleted"],
                "first_deletion": row["first_deletion"],
                "last_deletion": row["last_deletion"]
            }
            
            return stats

