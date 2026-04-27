#!/usr/bin/env python3
"""
Enhanced file management for S: batch moves, duplicate detection, and date-based archiving.

Designed for tidying the agent workspace (sandbox).
"""
import os
import hashlib
import shutil
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class FileManagerTool:
    """Enhanced file management utilities for the sandbox."""

    def __init__(self, sandbox_dir: str = "."):
        """
        Initialize the file manager.

        [2026-02-28] Option A: the agent's world is ``workspace/sandbox``.
        All read/write/move/delete/analyze operations stay inside the sandbox.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.sandbox_dir = os.path.abspath(os.path.join(project_root, "workspace/sandbox"))
        
        if not os.path.exists(self.sandbox_dir):
            os.makedirs(self.sandbox_dir, exist_ok=True)
            
        logger.info(f"[FileManagerTool] Agent workspace: {self.sandbox_dir}")
    
    def _is_safe_path(self, path: str) -> bool:
        """Return True if ``path`` resolves inside the sandbox."""
        abs_path = os.path.abspath(os.path.join(self.sandbox_dir, path))
        return abs_path.startswith(self.sandbox_dir)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Compute MD5 hash of file contents."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash {filepath}: {e}")
            return ""
    
    # ==================== 1. Batch file move ====================

    def batch_move_files(
        self, 
        file_list: List[str], 
        target_dir: str,
        create_dir: bool = True
    ) -> Dict[str, Any]:
        """
        Move multiple files into a target directory (paths relative to sandbox).

        Args:
            file_list: Relative file paths under the sandbox.
            target_dir: Target directory relative to the sandbox.
            create_dir: If True, create the target directory when missing.
        """
        if not self._is_safe_path(target_dir):
            return {"error": "Target directory is outside the sandbox"}

        target_path = os.path.join(self.sandbox_dir, target_dir)

        # Ensure target directory exists
        if create_dir and not os.path.exists(target_path):
            try:
                os.makedirs(target_path, exist_ok=True)
            except Exception as e:
                return {"error": f"Could not create target directory: {e}"}
        
        results = {
            "success": [],
            "failed": [],
            "skipped": []
        }
        
        for file_rel in file_list:
            if not self._is_safe_path(file_rel):
                results["failed"].append({
                    "file": file_rel,
                    "reason": "Path is outside the sandbox"
                })
                continue
            
            src_path = os.path.join(self.sandbox_dir, file_rel)
            
            if not os.path.exists(src_path):
                results["skipped"].append({
                    "file": file_rel,
                    "reason": "File does not exist"
                })
                continue

            if os.path.isdir(src_path):
                results["skipped"].append({
                    "file": file_rel,
                    "reason": "Path is a directory, not a file"
                })
                continue

            # Destination filename (may change on collision)
            filename = os.path.basename(file_rel)
            dest_path = os.path.join(target_path, filename)

            # Resolve name collisions
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    filename = f"{base}_{counter}{ext}"
                    dest_path = os.path.join(target_path, filename)
                    counter += 1
            
            try:
                shutil.move(src_path, dest_path)
                results["success"].append({
                    "from": file_rel,
                    "to": os.path.relpath(dest_path, self.sandbox_dir)
                })
            except Exception as e:
                results["failed"].append({
                    "file": file_rel,
                    "reason": str(e)
                })
        
        return {
            "status": "completed",
            "moved": len(results["success"]),
            "failed": len(results["failed"]),
            "skipped": len(results["skipped"]),
            "details": results
        }
    
    # ==================== 2. Duplicate file detection ====================

    def detect_duplicate_files(
        self, 
        directory: str = "",
        min_size: int = 100,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find duplicate files by content hash.

        Args:
            directory: Scope relative to sandbox; empty string means whole sandbox.
            min_size: Minimum file size in bytes; smaller files are skipped.
            extensions: Optional whitelist (e.g. [".py", ".md"]).
        """
        target_dir = os.path.join(self.sandbox_dir, directory) if directory else self.sandbox_dir

        if not self._is_safe_path(directory):
            return {"error": "Directory is outside the sandbox"}

        if not os.path.exists(target_dir):
            return {"error": f"Directory does not exist: {directory}"}

        # hash -> list of relative paths
        hash_map: Dict[str, List[str]] = defaultdict(list)
        scanned = 0
        skipped = 0
        
        for root, _, files in os.walk(target_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                # Extension filter
                if extensions:
                    if not any(filename.endswith(ext) for ext in extensions):
                        skipped += 1
                        continue
                
                # Size filter
                try:
                    size = os.path.getsize(filepath)
                    if size < min_size:
                        skipped += 1
                        continue
                except:
                    skipped += 1
                    continue
                
                file_hash = self._get_file_hash(filepath)
                if file_hash:
                    rel_path = os.path.relpath(filepath, self.sandbox_dir)
                    hash_map[file_hash].append(rel_path)
                    scanned += 1
        
        # Build duplicate groups
        duplicates = []
        total_duplicate_files = 0

        for file_hash, paths in hash_map.items():
            if len(paths) > 1:
                duplicates.append({
                    "hash": file_hash[:8],  # short hash for display
                    "count": len(paths),
                    "files": sorted(paths),
                    "size_bytes": os.path.getsize(os.path.join(self.sandbox_dir, paths[0]))
                })
                total_duplicate_files += len(paths) - 1  # extras beyond one keeper

        # Sort by group size (largest first)
        duplicates.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "status": "completed",
            "scanned_files": scanned,
            "skipped_files": skipped,
            "duplicate_groups": len(duplicates),
            "total_duplicates": total_duplicate_files,
            "duplicates": duplicates[:50],  # cap at 50 groups in response
            "note": (
                "Showing first 50 duplicate groups only"
                if len(duplicates) > 50
                else "Showing all duplicate groups"
            ),
        }

    # ==================== 3. Date-based archive ====================

    def archive_by_date(
        self,
        source_pattern: str,
        archive_base: str = "archives",
        date_format: str = "%Y-%m",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Archive files under ``archive_base/<date>/`` using each file's mtime.

        Args:
            source_pattern: Source directory or file path (e.g. ``autonomous_diaries/``).
            archive_base: Root folder for archives (e.g. ``archives``).
            date_format: ``strftime`` format (default ``%Y-%m``).
            dry_run: If True, only report planned moves.
        """
        if not self._is_safe_path(source_pattern) or not self._is_safe_path(archive_base):
            return {"error": "Path is outside the sandbox"}

        source_path = os.path.join(self.sandbox_dir, source_pattern)

        if not os.path.exists(source_path):
            return {"error": f"Source path does not exist: {source_pattern}"}

        # Collect files
        files_to_archive = []
        
        if os.path.isfile(source_path):
            files_to_archive.append(source_path)
        else:
            for root, _, files in os.walk(source_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    files_to_archive.append(filepath)
        
        # Group by date string
        date_groups: Dict[str, List[str]] = defaultdict(list)
        
        for filepath in files_to_archive:
            try:
                mtime = os.path.getmtime(filepath)
                date_str = datetime.fromtimestamp(mtime).strftime(date_format)
                rel_path = os.path.relpath(filepath, self.sandbox_dir)
                date_groups[date_str].append(rel_path)
            except Exception as e:
                logger.warning(f"Could not read mtime for {filepath}: {e}")
        
        results = {
            "moved": [],
            "failed": [],
            "would_move": []  # populated when dry_run
        }
        
        for date_str, file_list in date_groups.items():
            target_dir = os.path.join(self.sandbox_dir, archive_base, date_str)
            
            if not dry_run:
                os.makedirs(target_dir, exist_ok=True)
            
            for file_rel in file_list:
                src_path = os.path.join(self.sandbox_dir, file_rel)
                filename = os.path.basename(file_rel)
                dest_path = os.path.join(target_dir, filename)
                
                # Resolve name collisions in target folder
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        filename = f"{base}_{counter}{ext}"
                        dest_path = os.path.join(target_dir, filename)
                        counter += 1
                
                dest_rel = os.path.relpath(dest_path, self.sandbox_dir)
                
                if dry_run:
                    results["would_move"].append({
                        "from": file_rel,
                        "to": dest_rel,
                        "date": date_str
                    })
                else:
                    try:
                        shutil.move(src_path, dest_path)
                        results["moved"].append({
                            "from": file_rel,
                            "to": dest_rel,
                            "date": date_str
                        })
                    except Exception as e:
                        results["failed"].append({
                            "file": file_rel,
                            "reason": str(e)
                        })
        
        return {
            "status": "completed" if not dry_run else "dry_run",
            "total_files": len(files_to_archive),
            "date_groups": len(date_groups),
            "moved": len(results["moved"]) if not dry_run else 0,
            "would_move": len(results["would_move"]) if dry_run else 0,
            "failed": len(results["failed"]),
            "details": results,
            "note": (
                "Dry run: no files were moved"
                if dry_run
                else "Files were moved"
            ),
        }

    # ==================== 4. Remove duplicates (keep one) ====================

    def remove_duplicates(
        self,
        duplicate_group: List[str],
        keep_first: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Delete redundant copies in a duplicate group, keeping one file.

        Args:
            duplicate_group: Relative paths in the sandbox.
            keep_first: If True, keep the first path; else keep the newest by mtime.
            dry_run: If True, only report deletions.
        """
        if len(duplicate_group) < 2:
            return {"error": "Need at least two paths to remove duplicates"}

        # All paths must stay inside sandbox
        for file_rel in duplicate_group:
            if not self._is_safe_path(file_rel):
                return {"error": f"File is outside the sandbox: {file_rel}"}

        # Choose keeper
        if keep_first:
            keep_file = duplicate_group[0]
            remove_files = duplicate_group[1:]
        else:
            # Newest mtime wins
            files_with_time = []
            for file_rel in duplicate_group:
                filepath = os.path.join(self.sandbox_dir, file_rel)
                if os.path.exists(filepath):
                    mtime = os.path.getmtime(filepath)
                    files_with_time.append((file_rel, mtime))
            
            files_with_time.sort(key=lambda x: x[1], reverse=True)
            keep_file = files_with_time[0][0]
            remove_files = [f[0] for f in files_with_time[1:]]
        
        results = {
            "kept": keep_file,
            "removed": [],
            "would_remove": [],
            "failed": []
        }
        
        for file_rel in remove_files:
            filepath = os.path.join(self.sandbox_dir, file_rel)
            
            if not os.path.exists(filepath):
                results["failed"].append({
                    "file": file_rel,
                    "reason": "File does not exist"
                })
                continue

            if dry_run:
                results["would_remove"].append(file_rel)
            else:
                try:
                    os.remove(filepath)
                    results["removed"].append(file_rel)
                except Exception as e:
                    results["failed"].append({
                        "file": file_rel,
                        "reason": str(e)
                    })
        
        return {
            "status": "completed" if not dry_run else "dry_run",
            "kept_file": keep_file,
            "removed_count": len(results["removed"]) if not dry_run else 0,
            "would_remove_count": len(results["would_remove"]) if dry_run else 0,
            "failed_count": len(results["failed"]),
            "details": results,
            "note": (
                "Dry run: no files were deleted"
                if dry_run
                else "Duplicate files were deleted"
            ),
        }

    # ==================== 5. Workspace statistics ====================

    def analyze_workspace(
        self,
        directory: str = "",
        group_by: str = "extension"
    ) -> Dict[str, Any]:
        """
        Summarize file distribution in the sandbox (or a subdirectory).

        Args:
            directory: Scope; empty means whole sandbox.
            group_by: One of ``extension``, ``directory``, or ``size``.
        """
        target_dir = os.path.join(self.sandbox_dir, directory) if directory else self.sandbox_dir

        if not self._is_safe_path(directory):
            return {"error": "Directory is outside the sandbox"}

        if not os.path.exists(target_dir):
            return {"error": f"Directory does not exist: {directory}"}
        
        stats = {
            "total_files": 0,
            "total_size": 0,
            "by_extension": defaultdict(lambda: {"count": 0, "size": 0}),
            "by_directory": defaultdict(lambda: {"count": 0, "size": 0}),
            "by_size_range": {
                "0-1KB": 0,
                "1KB-10KB": 0,
                "10KB-100KB": 0,
                "100KB-1MB": 0,
                "1MB+": 0
            }
        }
        
        for root, _, files in os.walk(target_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                try:
                    size = os.path.getsize(filepath)
                    stats["total_files"] += 1
                    stats["total_size"] += size
                    
                    # By extension
                    ext = os.path.splitext(filename)[1] or "(no extension)"
                    stats["by_extension"][ext]["count"] += 1
                    stats["by_extension"][ext]["size"] += size
                    
                    # By parent directory
                    rel_dir = os.path.relpath(root, self.sandbox_dir)
                    stats["by_directory"][rel_dir]["count"] += 1
                    stats["by_directory"][rel_dir]["size"] += size
                    
                    # By size bucket
                    if size < 1024:
                        stats["by_size_range"]["0-1KB"] += 1
                    elif size < 10 * 1024:
                        stats["by_size_range"]["1KB-10KB"] += 1
                    elif size < 100 * 1024:
                        stats["by_size_range"]["10KB-100KB"] += 1
                    elif size < 1024 * 1024:
                        stats["by_size_range"]["100KB-1MB"] += 1
                    else:
                        stats["by_size_range"]["1MB+"] += 1
                
                except Exception as e:
                    logger.warning(f"Could not analyze {filepath}: {e}")

        # Build grouped output
        if group_by == "extension":
            sorted_items = sorted(
                stats["by_extension"].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            grouped_data = [
                {
                    "group": ext,
                    "count": data["count"],
                    "size_mb": round(data["size"] / (1024 * 1024), 2)
                }
                for ext, data in sorted_items
            ]
        elif group_by == "directory":
            sorted_items = sorted(
                stats["by_directory"].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            grouped_data = [
                {
                    "group": dir_name,
                    "count": data["count"],
                    "size_mb": round(data["size"] / (1024 * 1024), 2)
                }
                for dir_name, data in sorted_items[:20]  # top 20 dirs by count
            ]
        else:  # size
            grouped_data = [
                {"range": k, "count": v}
                for k, v in stats["by_size_range"].items()
            ]
        
        return {
            "status": "completed",
            "total_files": stats["total_files"],
            "total_size_mb": round(stats["total_size"] / (1024 * 1024), 2),
            "group_by": group_by,
            "grouped_data": grouped_data,
            "note": f"Grouped by {group_by}"
        }

    # ==================== 6. Workspace health ====================

    def get_workspace_health(self, directory: str = "") -> Dict[str, Any]:
        """
        Return a coarse health snapshot for the workspace (for the agent).

        Includes:
        - ``entropy``: disorder score in [0, 1]
        - file counts
        - short improvement suggestions
        """
        target_dir = os.path.join(self.sandbox_dir, directory) if directory else self.sandbox_dir
        
        if not os.path.exists(target_dir):
            return {"error": f"Directory does not exist: {directory}"}

        # Base counts
        total_files = 0
        root_files = 0
        files_in_subdirs = 0
        md_count = 0
        naming_violations = 0
        
        # Files directly under scope root
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            if os.path.isfile(item_path):
                root_files += 1
                total_files += 1
        
        # Files under subdirectories
        for root, _, files in os.walk(target_dir):
            if root != target_dir:
                files_in_subdirs += len(files)
                total_files += len(files)
            
            for filename in files:
                if filename.endswith('.md'):
                    md_count += 1
                    # Naming convention heuristic for markdown
                    has_prefix = any([
                        filename.startswith(p) for p in 
                        ["diary_", "search_", "learn_", "report_", "exp_", "plan_", "README"]
                    ]) or any(c.isdigit() for c in filename[:8])
                    if not has_prefix:
                        naming_violations += 1
        
        if total_files == 0:
            return {
                "status": "empty",
                "entropy": 0.0,
                "message": "Workspace is empty"
            }

        # Entropy heuristic
        root_ratio = min(1.0, root_files / 5.0)
        depth_entropy = root_files / total_files if total_files > 0 else 0
        naming_entropy = min(1.0, naming_violations / md_count) if md_count > 0 else 0
        dup_entropy = min(1.0, max(0, (md_count - 100)) / 100)
        
        entropy = (
            0.35 * root_ratio +
            0.25 * depth_entropy +
            0.25 * naming_entropy +
            0.15 * dup_entropy
        )
        
        # Suggestions (English for operators / LLM consumption)
        suggestions = []
        if root_files > 5:
            suggestions.append(
                f"{root_files} files at workspace root; consider moving them into subfolders"
            )
        if naming_violations > 10:
            suggestions.append(
                f"{naming_violations} markdown files may not follow preferred naming prefixes"
            )
        if md_count > 100:
            suggestions.append(
                f"Many markdown files ({md_count}); consider archiving older notes"
            )

        # Health tier labels
        if entropy < 0.3:
            health_level = "excellent"
            health_emoji = "🟢"
        elif entropy < 0.5:
            health_level = "good"
            health_emoji = "🟡"
        elif entropy < 0.7:
            health_level = "fair"
            health_emoji = "🟠"
        else:
            health_level = "needs_cleanup"
            health_emoji = "🔴"
        
        return {
            "status": "completed",
            "entropy": round(entropy, 3),
            "health_level": health_level,
            "health_emoji": health_emoji,
            "stats": {
                "total_files": total_files,
                "root_files": root_files,
                "files_in_subdirs": files_in_subdirs,
                "md_files": md_count,
                "naming_violations": naming_violations
            },
            "suggestions": suggestions,
            "message": (
                f"{health_emoji} Workspace health: {health_level} "
                f"(entropy {entropy:.2f})"
            ),
        }

    # ==================== OpenAI-style tool definitions ====================

    def get_tool_definitions(self) -> List[Dict]:
        """Return tool specs in OpenAI function-calling shape."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "batch_move_files",
                    "description": (
                        "Move many files into one directory. "
                        "Useful for consolidating scattered outputs."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Relative paths under workspace/sandbox to move"
                                ),
                            },
                            "target_dir": {
                                "type": "string",
                                "description": (
                                    "Destination directory relative to workspace/sandbox"
                                ),
                            },
                            "create_dir": {
                                "type": "boolean",
                                "description": "Create target directory if missing (default true)",
                                "default": True,
                            },
                        },
                        "required": ["file_list", "target_dir"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_duplicate_files",
                    "description": (
                        "Detect duplicate files by content hash. "
                        "Returns duplicate groups (capped in payload)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": (
                                    "Scope relative to sandbox; empty string scans whole sandbox"
                                ),
                                "default": "",
                            },
                            "min_size": {
                                "type": "integer",
                                "description": "Minimum file size in bytes (default 100)",
                                "default": 100,
                            },
                            "extensions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Optional extension whitelist (e.g. [\".py\", \".md\"]); "
                                    "omit to include all types"
                                ),
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "archive_by_date",
                    "description": (
                        "Archive files into dated folders using modification time. "
                        "Good for diaries, logs, and time-ordered notes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_pattern": {
                                "type": "string",
                                "description": (
                                    "Source directory or file path (e.g. 'autonomous_diaries/')"
                                ),
                            },
                            "archive_base": {
                                "type": "string",
                                "description": "Archive root folder (default 'archives')",
                                "default": "archives",
                            },
                            "date_format": {
                                "type": "string",
                                "description": (
                                    "strftime format (default '%Y-%m', e.g. 2026-01)"
                                ),
                                "default": "%Y-%m",
                            },
                            "dry_run": {
                                "type": "boolean",
                                "description": "If true, only preview moves (default false)",
                                "default": False,
                            },
                        },
                        "required": ["source_pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remove_duplicates",
                    "description": (
                        "Delete redundant copies in a duplicate group, keeping one file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "duplicate_group": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Paths from detect_duplicate_files (relative to sandbox)"
                                ),
                            },
                            "keep_first": {
                                "type": "boolean",
                                "description": (
                                    "If true, keep the first path; if false, keep newest by mtime "
                                    "(default true)"
                                ),
                                "default": True,
                            },
                            "dry_run": {
                                "type": "boolean",
                                "description": "If true, only preview deletions (default false)",
                                "default": False,
                            },
                        },
                        "required": ["duplicate_group"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_workspace",
                    "description": (
                        "Summarize file distribution (extension, directory, or size buckets)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Scope; empty means whole sandbox",
                                "default": "",
                            },
                            "group_by": {
                                "type": "string",
                                "enum": ["extension", "directory", "size"],
                                "description": (
                                    "Grouping mode: extension | directory | size (default extension)"
                                ),
                                "default": "extension",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_workspace_health",
                    "description": (
                        "Health snapshot: entropy (0-1), markdown naming heuristics, "
                        "and short improvement suggestions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "organize_workspace",
                    "description": (
                        "Smart tidy: analyze files and propose a move plan; "
                        "dry_run=true previews without moving."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dry_run": {
                                "type": "boolean",
                                "description": (
                                    "Preview only when true; execute moves when false (default true)"
                                ),
                                "default": True,
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cleanup_workspace",
                    "description": (
                        "Clean workspace: remove stale temp files; optionally archive old files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temp_days": {
                                "type": "integer",
                                "description": "Keep temp files newer than this many days (default 7)",
                                "default": 7,
                            },
                            "do_archive": {
                                "type": "boolean",
                                "description": "Also archive old files when true",
                                "default": False,
                            },
                            "archive_days": {
                                "type": "integer",
                                "description": (
                                    "Archive files not modified for this many days (default 30)"
                                ),
                                "default": 30,
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_file_recommendation",
                    "description": (
                        "Recommend where to store a file from name (and optional content snippet)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "File name",
                            },
                            "content": {
                                "type": "string",
                                "description": "Optional content snippet for finer classification",
                            },
                        },
                        "required": ["filename"],
                    },
                },
            },
        ]
