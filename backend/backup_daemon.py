#!/usr/bin/env python3
"""
Backup and cleanup daemon.

Runs scheduled incremental / monthly backups plus memory and disk retention sweeps.
"""
import os
import sys
import time
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when launched as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.backup_manager import BackupManager
from backend.memory_cleaner import MemoryCleaner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/backup_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupDaemon:
    """Long-running worker for backups and retention cleanup."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.backup_manager = BackupManager(db_path)
        self.memory_cleaner = MemoryCleaner(db_path, self.backup_manager)
        self.running = False
        
        # Intervals in seconds
        self.config = {
            "incremental_interval": 86400,  # daily
            "monthly_check_interval": 3600,  # hourly check for monthly snapshot
            "cleanup_interval": 604800,  # weekly memory cleanup
            "cleanup_old_backups_interval": 86400,  # daily prune of old backup files
        }

        # Last successful run timestamps (epoch seconds)
        self.last_run = {
            "incremental": 0,
            "monthly": 0,
            "cleanup": 0,
            "cleanup_old_backups": 0
        }
        
        logger.info("BackupDaemon initialized")
    
    def start(self):
        """Main loop until SIGINT/SIGTERM."""
        self.running = True

        # Graceful shutdown hooks
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("🚀 BackupDaemon started")
        
        # Run one monthly eligibility check immediately on startup
        self._run_monthly_backup()
        
        try:
            while self.running:
                now = time.time()
                
                # Incremental DB backup
                if now - self.last_run["incremental"] >= self.config["incremental_interval"]:
                    self._run_incremental_backup()
                    self.last_run["incremental"] = now
                
                # Monthly full backup (idempotent per month)
                if now - self.last_run["monthly"] >= self.config["monthly_check_interval"]:
                    self._run_monthly_backup()
                    self.last_run["monthly"] = now
                
                # Archived memory GC
                if now - self.last_run["cleanup"] >= self.config["cleanup_interval"]:
                    self._run_memory_cleanup()
                    self.last_run["cleanup"] = now
                
                # Drop backup files past retention
                if now - self.last_run["cleanup_old_backups"] >= self.config["cleanup_old_backups_interval"]:
                    self._run_cleanup_old_backups()
                    self.last_run["cleanup_old_backups"] = now
                
                # Wake once per minute
                time.sleep(60)
        
        except Exception as e:
            logger.error(f"BackupDaemon crashed: {e}", exc_info=True)
        finally:
            logger.info("BackupDaemon stopped")
    
    def _run_incremental_backup(self):
        """SQLite incremental backup to compressed artifact."""
        try:
            logger.info("📦 Running incremental backup...")
            result = self.backup_manager.incremental_backup()
            
            if result.get("file"):
                logger.info(
                    f"✅ Incremental backup completed: "
                    f"{result['file']} ({result['size_mb']:.2f} MB)"
                )
            else:
                logger.warning("⚠️ Incremental backup failed or disabled")
        
        except Exception as e:
            logger.error(f"❌ Incremental backup error: {e}", exc_info=True)
    
    def _run_monthly_backup(self):
        """Monthly compressed snapshot if missing for current month."""
        try:
            logger.info("📦 Checking monthly backup...")
            result = self.backup_manager.monthly_backup()
            
            if result.get("file"):
                if result.get("skipped"):
                    logger.info(f"ℹ️ Monthly backup already exists for this month")
                else:
                    logger.info(
                        f"✅ Monthly backup completed: "
                        f"{result['file']} ({result['size_mb']:.2f} MB)"
                    )
            else:
                logger.warning("⚠️ Monthly backup failed or disabled")
        
        except Exception as e:
            logger.error(f"❌ Monthly backup error: {e}", exc_info=True)
    
    def _run_memory_cleanup(self):
        """Safe memory cleaner with dry-run first."""
        try:
            logger.info("🧹 Running memory cleanup...")
            
            # Dry-run for visibility
            dry_result = self.memory_cleaner.safe_cleanup(dry_run=True)
            logger.info(
                f"📊 Cleanup analysis: "
                f"analyzed={dry_result['analyzed']}, "
                f"would_delete={dry_result['deleted']}, "
                f"retained={dry_result['retained']}"
            )
            
            # Apply deletes only when the plan is non-empty
            if dry_result['deleted'] > 0:
                result = self.memory_cleaner.safe_cleanup(dry_run=False, max_delete=1000)
                logger.info(
                    f"✅ Memory cleanup completed: "
                    f"deleted={result['deleted']}, "
                    f"backed_up={result['backed_up']}, "
                    f"freed={result['freed_mb']:.2f}MB"
                )
            else:
                logger.info("ℹ️ No memories to cleanup")
        
        except Exception as e:
            logger.error(f"❌ Memory cleanup error: {e}", exc_info=True)
    
    def _run_cleanup_old_backups(self):
        """Retention pass on on-disk backup trees."""
        try:
            logger.info("🧹 Cleaning up old backups...")
            result = self.backup_manager.cleanup_old_backups()
            
            if result["freed_mb"] > 0:
                logger.info(
                    f"✅ Old backups cleaned: "
                    f"freed={result['freed_mb']:.2f}MB, "
                    f"deleted_memories={result['deleted_memories']}, "
                    f"incremental={result['incremental']}, "
                    f"monthly={result['monthly']}"
                )
            else:
                logger.info("ℹ️ No old backups to clean")
        
        except Exception as e:
            logger.error(f"❌ Cleanup old backups error: {e}", exc_info=True)
    
    def _signal_handler(self, signum, frame):
        """Flip ``running`` so the main loop exits."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def status(self) -> dict:
        """Snapshot counters for operators / health checks."""
        stats = self.backup_manager.get_backup_stats()
        cleanup_stats = self.memory_cleaner.get_cleanup_stats()
        
        return {
            "running": self.running,
            "last_run": self.last_run,
            "backup_stats": stats,
            "cleanup_stats": cleanup_stats
        }

def main():
    """CLI entry: ``DB_PATH`` overrides default ``data.db``."""
    db_path = os.environ.get("DB_PATH", "data.db")

    os.makedirs("logs", exist_ok=True)

    daemon = BackupDaemon(db_path)
    daemon.start()

if __name__ == "__main__":
    main()

