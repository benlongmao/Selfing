#!/usr/bin/env python3
"""HTTP routes for backups and memory cleanup."""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from backend.backup_manager import BackupManager
from backend.memory_cleaner import MemoryCleaner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backup", tags=["backup"])

# Globals injected when chat_service starts backup routes
backup_manager: Optional[BackupManager] = None
memory_cleaner: Optional[MemoryCleaner] = None

def init_backup_routes(db_path: str):
    """Wire BackupManager + MemoryCleaner for this process."""
    global backup_manager, memory_cleaner
    backup_manager = BackupManager(db_path)
    memory_cleaner = MemoryCleaner(db_path, backup_manager)
    logger.info("Backup routes initialized")

@router.get("/stats")
def get_backup_stats():
    """Backup and cleanup counters."""
    if not backup_manager:
        raise HTTPException(status_code=503, detail="Backup manager not initialized")
    
    backup_stats = backup_manager.get_backup_stats()
    cleanup_stats = memory_cleaner.get_cleanup_stats()
    
    return {
        "backup": backup_stats,
        "cleanup": cleanup_stats
    }

@router.post("/incremental")
def create_incremental_backup():
    """Run an incremental backup."""
    if not backup_manager:
        raise HTTPException(status_code=503, detail="Backup manager not initialized")
    
    try:
        result = backup_manager.incremental_backup()
        if result.get("file"):
            return {
                "success": True,
                "file": result["file"],
                "size_mb": result["size_mb"]
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error")
            }
    except Exception as e:
        logger.error(f"Incremental backup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monthly")
def create_monthly_backup():
    """Run a monthly full backup."""
    if not backup_manager:
        raise HTTPException(status_code=503, detail="Backup manager not initialized")
    
    try:
        result = backup_manager.monthly_backup()
        if result.get("file"):
            return {
                "success": True,
                "file": result["file"],
                "size_mb": result["size_mb"],
                "skipped": result.get("skipped", False)
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error")
            }
    except Exception as e:
        logger.error(f"Monthly backup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
def cleanup_memories(
    dry_run: bool = Query(True, description="If true, analyze only; do not delete"),
    max_delete: int = Query(1000, description="Max rows to delete in one run")
):
    """Run safe memory cleanup."""
    if not memory_cleaner:
        raise HTTPException(status_code=503, detail="Memory cleaner not initialized")
    
    try:
        result = memory_cleaner.safe_cleanup(dry_run=dry_run, max_delete=max_delete)
        return {
            "success": True,
            "dry_run": dry_run,
            "analyzed": result["analyzed"],
            "deleted": result["deleted"],
            "retained": result["retained"],
            "backed_up": result.get("backed_up", 0),
            "freed_mb": result["freed_mb"],
            "by_level": result["by_level"],
            "deleted_items_preview": result.get("deleted_items", [])
        }
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup-old-backups")
def cleanup_old_backups():
    """Delete old backup files on disk."""
    if not backup_manager:
        raise HTTPException(status_code=503, detail="Backup manager not initialized")
    
    try:
        result = backup_manager.cleanup_old_backups()
        return {
            "success": True,
            "deleted_memories": result["deleted_memories"],
            "incremental": result["incremental"],
            "monthly": result["monthly"],
            "freed_mb": result["freed_mb"]
        }
    except Exception as e:
        logger.error(f"Cleanup old backups failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/restore/deleted/{backup_date}/{item_id}")
def restore_deleted_memory(backup_date: str, item_id: str):
    """Restore one memory row from a deletion backup."""
    if not backup_manager:
        raise HTTPException(status_code=503, detail="Backup manager not initialized")
    
    try:
        item = backup_manager.restore_deleted_memory(backup_date, item_id)
        if item:
            return {
                "success": True,
                "item": item
            }
        else:
            raise HTTPException(status_code=404, detail="Memory not found in backup")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/protection-level/{item_id}")
def get_memory_protection_level(item_id: str):
    """Return cleanup protection tier for a persona_items row."""
    if not memory_cleaner:
        raise HTTPException(status_code=503, detail="Memory cleaner not initialized")
    
    try:
        import sqlite3
        with sqlite3.connect(memory_cleaner.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM persona_items WHERE id = ?",
                (item_id,)
            )
            row = cur.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            item = dict(row)
            level = memory_cleaner.get_protection_level(item)
            policy = memory_cleaner.PROTECTION_LEVELS[level]
            
            return {
                "item_id": item_id,
                "protection_level": level,
                "retention_days": policy["retention_days"],
                "policy_name": policy["name"],
                "item": {
                    "text": item.get("text", "")[:100],
                    "importance": item.get("importance"),
                    "is_core": item.get("is_core"),
                    "locked": item.get("locked"),
                    "status": item.get("status")
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get protection level failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

