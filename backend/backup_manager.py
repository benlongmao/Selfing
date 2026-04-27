#!/usr/bin/env python3
"""
Backup manager: incremental SQLite snapshots plus lightweight deleted-memory JSONL.
"""
import sqlite3
import os
import gzip
import shutil
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class BackupManager:
    """Filesystem layout under ``backup_dir`` for three retention classes."""
    
    def __init__(
        self, 
        db_path: str, 
        backup_dir: str = "backups",
        config: Optional[Dict] = None
    ):
        self.db_path = db_path
        self.backup_dir = backup_dir
        
        # Default retention knobs (caller may override via ``config``)
        self.config = config or {
            "deleted_memories": {
                "enabled": True,
                "include_embedding": False,
                "retention_days": 365
            },
            "incremental": {
                "enabled": True,
                "retention_days": 7
            },
            "monthly": {
                "enabled": True,
                "retention_months": 12
            }
        }
        
        # Ensure layout exists
        self.deleted_dir = os.path.join(backup_dir, "deleted_memories")
        self.incremental_dir = os.path.join(backup_dir, "incremental")
        self.monthly_dir = os.path.join(backup_dir, "monthly")
        
        for directory in [self.deleted_dir, self.incremental_dir, self.monthly_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"BackupManager initialized: {backup_dir}")
    
    # --- Deleted-memory JSONL (lightweight) ---
    
    def backup_deleted_memories(
        self, 
        deleted_items: List[Dict],
        include_embedding: bool = None
    ) -> Dict:
        """
        Append deleted memory rows to a daily JSONL file.

        Args:
            deleted_items: Records removed from the primary store
            include_embedding: When False, strip ``embedding`` tensors to save space

        Returns:
            ``{"file", "count", "size_kb"}`` metadata (``file`` may be None)
        """
        if not self.config["deleted_memories"]["enabled"]:
            return {"file": None, "count": 0, "size_kb": 0}
        
        if include_embedding is None:
            include_embedding = self.config["deleted_memories"]["include_embedding"]
        
        if not deleted_items:
            return {"file": None, "count": 0, "size_kb": 0}
        
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        backup_file = os.path.join(self.deleted_dir, f"deleted_{date_str}.jsonl")
        
        try:
            with open(backup_file, 'a', encoding='utf-8') as f:
                for item in deleted_items:
                    # Optionally strip embeddings
                    backup_item = {
                        k: v for k, v in item.items()
                        if k != 'embedding' or include_embedding
                    }
                    
                    backup_entry = {
                        "deleted_at": datetime.now(timezone.utc).isoformat(),
                        "item": backup_item
                    }
                    
                    json.dump(backup_entry, f, ensure_ascii=False)
                    f.write('\n')
            
            size_kb = os.path.getsize(backup_file) / 1024
            logger.info(
                f"✅ Backed up {len(deleted_items)} deleted memories: "
                f"{backup_file} ({size_kb:.2f} KB)"
            )
            
            return {
                "file": backup_file,
                "count": len(deleted_items),
                "size_kb": size_kb
            }
        
        except Exception as e:
            logger.error(f"Failed to backup deleted memories: {e}", exc_info=True)
            return {"file": None, "count": 0, "size_kb": 0, "error": str(e)}
    
    def restore_deleted_memory(self, backup_date: str, item_id: str) -> Optional[Dict]:
        """
        Scan ``deleted_{backup_date}.jsonl`` for a matching ``item.id``.

        Args:
            backup_date: YYYYMMDD suffix in the filename
            item_id: Primary key inside the archived JSON payload

        Returns:
            Restored dict or ``None``
        """
        backup_file = os.path.join(self.deleted_dir, f"deleted_{backup_date}.jsonl")
        
        if not os.path.exists(backup_file):
            logger.error(f"Backup file not found: {backup_file}")
            return None
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data["item"]["id"] == item_id:
                        logger.info(f"Found deleted memory: {item_id}")
                        return data["item"]
            
            logger.warning(f"Memory {item_id} not found in {backup_file}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to restore deleted memory: {e}", exc_info=True)
            return None
    
    # --- Incremental SQLite backup ---
    
    def incremental_backup(self) -> Dict:
        """
        Hot-copy ``db_path`` through ``sqlite3.Connection.backup``, gzip result.

        Returns:
            ``{"file", "size_mb", "compressed"}`` (``file`` may be None when disabled)
        """
        if not self.config["incremental"]["enabled"]:
            return {"file": None, "size_mb": 0, "compressed": False}
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.incremental_dir, f"inc_{timestamp}.db")
        
        try:
            # Online backup API (page iterator keeps call short)
            src = sqlite3.connect(self.db_path)
            dst = sqlite3.connect(backup_file)
            
            with dst:
                src.backup(dst, pages=100, progress=self._backup_progress)
            
            src.close()
            dst.close()
            
            # Compress to .gz
            compressed_file = f"{backup_file}.gz"
            with open(backup_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb', compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Drop plaintext temp once gzip succeeds
            os.remove(backup_file)
            
            size_mb = os.path.getsize(compressed_file) / 1024 / 1024
            logger.info(f"✅ Incremental backup: {compressed_file} ({size_mb:.2f} MB)")
            
            return {
                "file": compressed_file,
                "size_mb": size_mb,
                "compressed": True
            }
        
        except Exception as e:
            logger.error(f"Failed to create incremental backup: {e}", exc_info=True)
            return {"file": None, "size_mb": 0, "compressed": False, "error": str(e)}
    
    def _backup_progress(self, status, remaining, total):
        """SQLite backup API progress hook."""
        if remaining == 0:
            logger.debug(f"Backup progress: {total} pages copied")
    
    def restore_incremental_backup(self, backup_file: str, target_db: str) -> bool:
        """
        Decompress ``backup_file`` (.db.gz) into ``target_db`` with integrity check.

        Args:
            backup_file: Path to gzipped SQLite file
            target_db: Live database path to replace

        Returns:
            ``True`` on success
        """
        try:
            # Gunzip to temp path
            temp_db = f"{target_db}.temp"
            with gzip.open(backup_file, 'rb') as f_in:
                with open(temp_db, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # ``PRAGMA integrity_check`` before swap
            conn = sqlite3.connect(temp_db)
            conn.execute("PRAGMA integrity_check")
            conn.close()
            
            # Atomic-ish swap: rename old DB aside, promote temp
            if os.path.exists(target_db):
                os.rename(target_db, f"{target_db}.old")
            os.rename(temp_db, target_db)
            
            logger.info(f"✅ Restored database from: {backup_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}", exc_info=True)
            if os.path.exists(temp_db):
                os.remove(temp_db)
            return False
    
    # --- Monthly full snapshot ---
    
    def monthly_backup(self) -> Dict:
        """
        One gzip per calendar month (``monthly_YYYYMM.db.gz``).

        Returns:
            ``{"file", "size_mb"}`` plus ``skipped`` when file already exists
        """
        if not self.config["monthly"]["enabled"]:
            return {"file": None, "size_mb": 0}
        
        month_str = datetime.now(timezone.utc).strftime("%Y%m")
        backup_file = os.path.join(self.monthly_dir, f"monthly_{month_str}.db.gz")
        
        # Skip if this month already captured
        if os.path.exists(backup_file):
            logger.info(f"Monthly backup already exists: {backup_file}")
            size_mb = os.path.getsize(backup_file) / 1024 / 1024
            return {"file": backup_file, "size_mb": size_mb, "skipped": True}
        
        try:
            # Stream-compress primary DB file
            with open(self.db_path, 'rb') as f_in:
                with gzip.open(backup_file, 'wb', compresslevel=9) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            size_mb = os.path.getsize(backup_file) / 1024 / 1024
            logger.info(f"✅ Monthly backup: {backup_file} ({size_mb:.2f} MB)")
            
            return {
                "file": backup_file,
                "size_mb": size_mb
            }
        
        except Exception as e:
            logger.error(f"Failed to create monthly backup: {e}", exc_info=True)
            return {"file": None, "size_mb": 0, "error": str(e)}
    
    # --- Retention cleanup ---
    
    def cleanup_old_backups(self) -> Dict:
        """
        Delete files older than per-class retention windows.

        Returns:
            Per-bucket delete counts plus ``freed_mb`` aggregate
        """
        now = datetime.now(timezone.utc)
        freed_bytes = 0
        counts = {
            "deleted_memories": 0,
            "incremental": 0,
            "monthly": 0
        }
        
        # JSONL tombstones
        if self.config["deleted_memories"]["enabled"]:
            cutoff = now - timedelta(days=self.config["deleted_memories"]["retention_days"])
            freed, count = self._cleanup_directory(
                self.deleted_dir, 
                cutoff, 
                "deleted_*.jsonl"
            )
            freed_bytes += freed
            counts["deleted_memories"] = count
        
        # Incremental gzip chain
        if self.config["incremental"]["enabled"]:
            cutoff = now - timedelta(days=self.config["incremental"]["retention_days"])
            freed, count = self._cleanup_directory(
                self.incremental_dir,
                cutoff,
                "inc_*.db.gz"
            )
            freed_bytes += freed
            counts["incremental"] = count
        
        # Monthly archives (approx month = 30d for cutoff math)
        if self.config["monthly"]["enabled"]:
            cutoff = now - timedelta(days=self.config["monthly"]["retention_months"] * 30)
            freed, count = self._cleanup_directory(
                self.monthly_dir,
                cutoff,
                "monthly_*.db.gz"
            )
            freed_bytes += freed
            counts["monthly"] = count
        
        freed_mb = freed_bytes / 1024 / 1024
        logger.info(f"✅ Cleanup: freed {freed_mb:.2f} MB, deleted {sum(counts.values())} files")
        
        return {
            **counts,
            "freed_mb": freed_mb
        }
    
    def _cleanup_directory(self, directory: str, cutoff: datetime, pattern: str) -> tuple:
        """Delete files under ``directory`` matching ``glob`` older than ``cutoff``."""
        freed_bytes = 0
        count = 0
        
        for file in Path(directory).glob(pattern):
            try:
                mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    size = file.stat().st_size
                    file.unlink()
                    freed_bytes += size
                    count += 1
                    logger.debug(f"Deleted old backup: {file}")
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {e}")
        
        return freed_bytes, count
    
    # --- Operator stats ---
    
    def get_backup_stats(self) -> Dict:
        """Aggregate sizes/counts for dashboards."""
        stats = {}
        
        # Deleted-memory lane
        deleted_files = list(Path(self.deleted_dir).glob("deleted_*.jsonl"))
        stats["deleted_memories"] = {
            "count": len(deleted_files),
            "size_mb": sum(f.stat().st_size for f in deleted_files) / 1024 / 1024,
            "oldest": min((f.stat().st_mtime for f in deleted_files), default=0),
            "newest": max((f.stat().st_mtime for f in deleted_files), default=0)
        }
        
        # Incremental lane
        inc_files = list(Path(self.incremental_dir).glob("inc_*.db.gz"))
        stats["incremental"] = {
            "count": len(inc_files),
            "size_mb": sum(f.stat().st_size for f in inc_files) / 1024 / 1024,
            "oldest": min((f.stat().st_mtime for f in inc_files), default=0),
            "newest": max((f.stat().st_mtime for f in inc_files), default=0)
        }
        
        # Monthly lane
        monthly_files = list(Path(self.monthly_dir).glob("monthly_*.db.gz"))
        stats["monthly"] = {
            "count": len(monthly_files),
            "size_mb": sum(f.stat().st_size for f in monthly_files) / 1024 / 1024,
            "oldest": min((f.stat().st_mtime for f in monthly_files), default=0),
            "newest": max((f.stat().st_mtime for f in monthly_files), default=0)
        }
        
        # Grand total
        stats["total_size_mb"] = (
            stats["deleted_memories"]["size_mb"] +
            stats["incremental"]["size_mb"] +
            stats["monthly"]["size_mb"]
        )
        
        return stats

