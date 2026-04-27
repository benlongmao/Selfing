#!/usr/bin/env python3
"""
Workspace management for the agent sandbox.

Goals: keep files organized, traceable, and easy to auto-tidy.

Features:
1. Heuristic classifier (filename patterns + optional content cues)
2. Standard top-level directory layout
3. Auto-archive for stale files and temp cleanup
4. Health checks with entropy-style scoring and prompt injection hooks

Maintainer note: introduced 2026-02-03 (S project hardening pass).
"""

import os
import re
import json
import hashlib
import shutil
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import sqlite3

logger = logging.getLogger(__name__)


# ==================== Standard layout (top-level folders) ====================

STANDARD_DIRECTORIES = {
    # Top-level folder -> human description + optional subdirs + filename regex hints
    "diaries": {
        "description": "Diaries and journals",
        "subdirs": ["daily", "weekly", "monthly"],
        "patterns": [r"diary_", r"日记", r"daily_", r"journal"],
    },
    "research": {
        "description": "Research and experiments",
        "subdirs": ["consciousness", "ethics", "technical", "hypotheses"],
        "patterns": [r"research_", r"experiment_", r"study_", r"hypothesis_", r"研究", r"实验"],
    },
    "projects": {
        "description": "Project workspaces",
        "subdirs": ["active", "completed", "archived"],
        "patterns": [r"project_", r"proj_", r"工程"],
    },
    "knowledge": {
        "description": "Knowledge base and study notes",
        "subdirs": ["notes", "summaries", "references"],
        "patterns": [r"note_", r"summary_", r"knowledge_", r"learn_", r"笔记", r"总结"],
    },
    "reflections": {
        "description": "Reflection and metacognition",
        "subdirs": ["metacognition", "philosophy", "ethics"],
        "patterns": [r"reflection_", r"思考", r"反思", r"meta_", r"philosophy_"],
    },
    "tools": {
        "description": "Utilities and scripts",
        "subdirs": ["scripts", "templates", "utilities"],
        "patterns": [r"tool_", r"script_", r"template_", r"\.py$", r"\.sh$"],
    },
    "archives": {
        "description": "Cold storage / year buckets",
        "subdirs": ["2024", "2025", "2026"],
        "patterns": [],  # managed by archive jobs, not filename routing
    },
    "temp": {
        "description": "Scratch files (auto-cleaned after ~7 days)",
        "subdirs": [],
        "patterns": [r"temp_", r"tmp_", r"draft_", r"草稿"],
    },
    "logs": {
        "description": "Runtime and operator logs",
        "subdirs": ["s_logs", "operations", "audit"],
        "patterns": [r"log_", r"\.log$", r"操作记录"],
    },
}

# File naming convention (metadata + validation)
NAMING_CONVENTION = {
    "format": "{category}_{topic}_{date}_{suffix}.{ext}",
    "date_format": "%Y%m%d",
    "allowed_chars": r"^[a-zA-Z0-9_\-\u4e00-\u9fff]+$",
    "max_length": 100,
}


class WorkspaceManager:
    """
    Keeps the sandbox tree predictable: layout, classification, health, and archival hooks.
    """

    def __init__(self, workspace_root: str = "workspace/sandbox", db_path: str = "data.db"):
        cwd = os.getcwd()
        self.workspace_root = os.path.abspath(os.path.join(cwd, workspace_root))
        self.db_path = db_path

        if not os.path.exists(self.workspace_root):
            os.makedirs(self.workspace_root, exist_ok=True)

        self._ensure_tables()
        self._ensure_standard_directories()
        
        logger.info(f"WorkspaceManager initialized: {self.workspace_root}")
    
    def _ensure_tables(self):
        """Create SQLite tables used for workspace metadata and health logs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workspace_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        category TEXT,
                        tags TEXT,
                        created_at TEXT,
                        modified_at TEXT,
                        file_hash TEXT,
                        size_bytes INTEGER,
                        is_archived BOOLEAN DEFAULT FALSE,
                        archive_date TEXT,
                        metadata_json TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workspace_health_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        check_time TEXT NOT NULL,
                        total_files INTEGER,
                        entropy_score REAL,
                        issues_json TEXT,
                        suggestions_json TEXT
                    )
                """)
                
                conn.commit()
                logger.info("Workspace tables ensured")
        except Exception as e:
            logger.error(f"Failed to ensure workspace tables: {e}")
    
    def _ensure_standard_directories(self):
        """Materialize ``STANDARD_DIRECTORIES`` on disk (idempotent)."""
        for dir_name, config in STANDARD_DIRECTORIES.items():
            dir_path = os.path.join(self.workspace_root, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created standard directory: {dir_name}")
            
            for subdir in config.get("subdirs", []):
                subdir_path = os.path.join(dir_path, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)
        
        self._create_workspace_readme()

    def _create_workspace_readme(self):
        """Seed ``_WORKSPACE_GUIDE.md`` once so humans/agents know the layout."""
        readme_path = os.path.join(self.workspace_root, "_WORKSPACE_GUIDE.md")
        if os.path.exists(readme_path):
            return

        content = """# Agent workspace guide

## Standard top-level folders

| Folder | Purpose | Typical children |
|--------|---------|------------------|
| `diaries/` | Journals | daily/, weekly/, monthly/ |
| `research/` | Lab notes & experiments | consciousness/, ethics/, technical/, hypotheses/ |
| `projects/` | Project workspaces | active/, completed/, archived/ |
| `knowledge/` | Notes & references | notes/, summaries/, references/ |
| `reflections/` | Metacognition | metacognition/, philosophy/, ethics/ |
| `tools/` | Scripts & templates | scripts/, templates/, utilities/ |
| `archives/` | Cold storage | year buckets (e.g. 2026/) |
| `temp/` | Scratch | auto-cleaned after ~7 days |

## Naming convention

**Shape**: `{category}_{topic}_{date}_{suffix}.{ext}`

**Examples**:
- `diary_consciousness_20260203_morning.md`
- `research_silicon_ethics_20260203_v1.md`
- `note_python_async_20260203.md`

## Automation expectations

1. **Routing**: new files are classified from filename patterns (and optional content cues).
2. **Temp hygiene**: files under `temp/` older than ~7 days may be deleted.
3. **Archiving**: stale files (>30 days without edits) can move under `archives/<year>/`.
4. **Duplicates**: identical content hashes are surfaced for manual merge.

## Health checks

Periodic scans track entropy-style disorder, duplicates, naming drift, and overly deep trees.
A tidy tree makes downstream retrieval and continuity tooling far more reliable.
"""
        try:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Created workspace guide")
        except Exception as e:
            logger.warning(f"Failed to create workspace guide: {e}")
    
    # ==================== Classification ====================

    def classify_file(self, filename: str, content: Optional[str] = None) -> Tuple[str, str]:
        """
        Return ``(category, subcategory)`` for a new artifact.

        Order: filename regex hints from ``STANDARD_DIRECTORIES``, optional content cues,
        otherwise ``("temp", "")``.
        """
        filename_lower = filename.lower()

        for category, config in STANDARD_DIRECTORIES.items():
            for pattern in config.get("patterns", []):
                if re.search(pattern, filename_lower):
                    subcategory = self._determine_subcategory(category, filename, content)
                    return category, subcategory

        if content:
            content_lower = content.lower()

            if any(kw in content_lower for kw in ["意识", "consciousness", "silicon", "硅基", "自我", "self-model"]):
                return "research", "consciousness"

            if any(kw in content_lower for kw in ["伦理", "ethics", "道德", "moral", "alignment"]):
                return "reflections", "ethics"

            if any(kw in content_lower for kw in ["代码", "code", "function", "class", "import", "def ", "async "]):
                return "tools", "scripts"

            if any(kw in content_lower for kw in ["反思", "思考", "元认知", "metacognition", "introspection"]):
                return "reflections", "metacognition"

        return "temp", ""

    def _determine_subcategory(self, category: str, filename: str, content: Optional[str]) -> str:
        """Pick the best child folder name declared for ``category``."""
        subdirs = STANDARD_DIRECTORIES.get(category, {}).get("subdirs", [])
        if not subdirs:
            return ""
        
        filename_lower = filename.lower()
        content_lower = (content or "").lower()
        
        for subdir in subdirs:
            if subdir in filename_lower or subdir in content_lower:
                return subdir

        return subdirs[0] if subdirs else ""

    def get_recommended_path(self, filename: str, content: Optional[str] = None) -> str:
        """
        Relative path (under ``workspace_root``) where the file should live after normalization.
        """
        category, subcategory = self.classify_file(filename, content)

        normalized_name = self._normalize_filename(filename, category)
        
        if subcategory:
            return f"{category}/{subcategory}/{normalized_name}"
        return f"{category}/{normalized_name}"
    
    def _normalize_filename(self, filename: str, category: str) -> str:
        """
        Normalize toward ``{category}_{topic}_{date}_{suffix}.{ext}`` while allowing CJK stems.
        """
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = ".md"

        date_pattern = r"\d{8}"
        if re.search(date_pattern, name):
            if not name.startswith(f"{category}_"):
                name = f"{category}_{name}"
        else:
            today = datetime.now().strftime("%Y%m%d")
            if not name.startswith(f"{category}_"):
                name = f"{category}_{name}_{today}"
            else:
                name = f"{name}_{today}"

        name = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")

        if len(name) > NAMING_CONVENTION["max_length"]:
            name = name[:NAMING_CONVENTION["max_length"]]
        
        return f"{name}{ext}"
    
    # ==================== Archival & temp cleanup ====================

    def auto_archive(self, days_threshold: int = 30) -> Dict[str, Any]:
        """
        Move untouched files older than ``days_threshold`` into ``archives/<year>/...``.
        """
        results = {
            "archived": [],
            "failed": [],
            "skipped": []
        }
        
        threshold_time = datetime.now() - timedelta(days=days_threshold)
        current_year = datetime.now().strftime("%Y")
        archive_dir = os.path.join(self.workspace_root, "archives", current_year)
        
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir, exist_ok=True)
        
        for category in STANDARD_DIRECTORIES:
            if category in ["archives", "temp"]:
                continue
            
            category_path = os.path.join(self.workspace_root, category)
            if not os.path.exists(category_path):
                continue
            
            for root, _, files in os.walk(category_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                        
                        if mtime < threshold_time:
                            rel_path = os.path.relpath(filepath, self.workspace_root)
                            archive_path = os.path.join(archive_dir, rel_path)

                            os.makedirs(os.path.dirname(archive_path), exist_ok=True)

                            shutil.move(filepath, archive_path)
                            results["archived"].append({
                                "from": rel_path,
                                "to": os.path.relpath(archive_path, self.workspace_root),
                                "age_days": (datetime.now() - mtime).days
                            })
                            
                            logger.info(f"Archived: {rel_path}")
                        else:
                            results["skipped"].append(rel_path)
                    
                    except Exception as e:
                        results["failed"].append({
                            "file": filepath,
                            "error": str(e)
                        })
        
        return results
    
    def cleanup_temp(self, days_threshold: int = 7) -> Dict[str, Any]:
        """Delete stale files under ``temp/`` older than ``days_threshold`` days."""
        results = {
            "deleted": [],
            "failed": [],
            "kept": []
        }
        
        temp_dir = os.path.join(self.workspace_root, "temp")
        if not os.path.exists(temp_dir):
            return results
        
        threshold_time = datetime.now() - timedelta(days=days_threshold)
        
        for root, _, files in os.walk(temp_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if mtime < threshold_time:
                        os.remove(filepath)
                        results["deleted"].append({
                            "file": os.path.relpath(filepath, self.workspace_root),
                            "age_days": (datetime.now() - mtime).days
                        })
                        logger.info(f"Deleted temp file: {filepath}")
                    else:
                        results["kept"].append(os.path.relpath(filepath, self.workspace_root))
                
                except Exception as e:
                    results["failed"].append({
                        "file": filepath,
                        "error": str(e)
                    })
        
        return results
    
    # ==================== Health checks ====================

    def check_health(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot used by APIs and prompt injection."""
        issues = []
        suggestions = []

        total_files = 0
        files_in_standard_dirs = 0
        files_outside_standard = 0
        naming_violations = 0
        duplicate_hashes = defaultdict(list)
        deep_paths = []
        large_files = []
        
        for root, dirs, files in os.walk(self.workspace_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            rel_root = os.path.relpath(root, self.workspace_root)
            depth = len(rel_root.split(os.sep)) if rel_root != "." else 0
            
            for filename in files:
                if filename.startswith('.') or filename.startswith('_'):
                    continue
                
                total_files += 1
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, self.workspace_root)
                
                top_dir = rel_path.split(os.sep)[0] if os.sep in rel_path else ""
                if top_dir in STANDARD_DIRECTORIES:
                    files_in_standard_dirs += 1
                else:
                    files_outside_standard += 1
                    issues.append({
                        "type": "outside_standard",
                        "file": rel_path,
                        "suggestion": "Move under a standard top-level folder from STANDARD_DIRECTORIES",
                    })

                if not self._check_naming_convention(filename):
                    naming_violations += 1

                if depth > 4:
                    deep_paths.append(rel_path)

                try:
                    size = os.path.getsize(filepath)
                    if size > 1024 * 1024:  # > 1MB
                        large_files.append({
                            "file": rel_path,
                            "size_mb": round(size / (1024 * 1024), 2)
                        })
                    
                    if size < 10 * 1024 * 1024:  # hash only reasonably small files
                        file_hash = self._compute_hash(filepath)
                        if file_hash:
                            duplicate_hashes[file_hash].append(rel_path)
                except Exception:
                    pass
        
        duplicates = {h: files for h, files in duplicate_hashes.items() if len(files) > 1}

        entropy_score = self._calculate_entropy(
            total_files,
            files_outside_standard,
            naming_violations,
            len(duplicates),
            len(deep_paths)
        )
        
        if files_outside_standard > 0:
            suggestions.append(
                f"{files_outside_standard} file(s) sit outside the standard top-level folders"
            )
        if naming_violations > 5:
            suggestions.append(
                f"{naming_violations} file(s) violate the naming convention — consider renaming"
            )
        if duplicates:
            suggestions.append(f"{len(duplicates)} duplicate group(s) detected — merge or delete extras")
        if deep_paths:
            suggestions.append(
                f"{len(deep_paths)} path(s) deeper than four segments — flatten where possible"
            )
        if large_files:
            suggestions.append(
                f"{len(large_files)} file(s) exceed 1 MB — compress or archive if appropriate"
            )

        if entropy_score < 0.3:
            health_level = "excellent"
        elif entropy_score < 0.5:
            health_level = "good"
        elif entropy_score < 0.7:
            health_level = "fair"
        else:
            health_level = "needs_cleanup"
        
        report = {
            "success": True,
            "check_time": datetime.now(timezone.utc).isoformat(),
            "health_level": health_level,
            "entropy_score": round(entropy_score, 3),
            "statistics": {
                "total_files": total_files,
                "files_in_standard_dirs": files_in_standard_dirs,
                "files_outside_standard": files_outside_standard,
                "naming_violations": naming_violations,
                "duplicate_groups": len(duplicates),
                "deep_paths": len(deep_paths),
                "large_files": len(large_files)
            },
            "issues": issues[:10],
            "suggestions": suggestions,
            "duplicates": {h: files for h, files in list(duplicates.items())[:5]},
        }

        self._log_health_check(report)
        
        return report
    
    def _check_naming_convention(self, filename: str) -> bool:
        """Return True when the basename matches ``NAMING_CONVENTION``."""
        name, ext = os.path.splitext(filename)

        if not re.match(NAMING_CONVENTION["allowed_chars"], name):
            return False

        if len(name) > NAMING_CONVENTION["max_length"]:
            return False
        
        return True
    
    def _compute_hash(self, filepath: str) -> Optional[str]:
        """MD5 helper for duplicate detection."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None
    
    def _calculate_entropy(
        self,
        total_files: int,
        outside_standard: int,
        naming_violations: int,
        duplicate_groups: int,
        deep_paths: int
    ) -> float:
        """
        Heuristic disorder score in ``[0, 1]`` (higher = messier sandbox).
        """
        if total_files == 0:
            return 0.0

        outside_ratio = outside_standard / total_files * 0.3
        naming_ratio = min(naming_violations / total_files, 1.0) * 0.2
        duplicate_ratio = min(duplicate_groups / total_files, 0.5) * 0.2
        deep_ratio = min(deep_paths / total_files, 0.5) * 0.15
        
        count_penalty = min(total_files / 500, 0.5) * 0.15
        
        entropy = outside_ratio + naming_ratio + duplicate_ratio + deep_ratio + count_penalty
        return min(1.0, entropy)
    
    def _log_health_check(self, report: Dict[str, Any]):
        """Persist the latest ``check_health`` snapshot for dashboards."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO workspace_health_logs 
                    (check_time, total_files, entropy_score, issues_json, suggestions_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    report["check_time"],
                    report["statistics"]["total_files"],
                    report["entropy_score"],
                    json.dumps(report.get("issues", []), ensure_ascii=False),
                    json.dumps(report.get("suggestions", []), ensure_ascii=False)
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log health check: {e}")
    
    # ==================== Migration helper ====================

    def migrate_existing_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Propose (or execute) moves from the workspace root into the standard tree.

        ``dry_run=True`` only returns the plan.
        """
        plan = {
            "moves": [],
            "skipped": [],
            "errors": []
        }
        
        for item in os.listdir(self.workspace_root):
            item_path = os.path.join(self.workspace_root, item)
            
            if item in STANDARD_DIRECTORIES or item.startswith('_') or item.startswith('.'):
                continue
            
            if os.path.isfile(item_path):
                try:
                    with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(5000)
                except Exception:
                    content = None
                
                recommended_path = self.get_recommended_path(item, content)
                
                if dry_run:
                    plan["moves"].append({
                        "from": item,
                        "to": recommended_path,
                        "reason": "Heuristic classifier recommendation",
                    })
                else:
                    try:
                        target_path = os.path.join(self.workspace_root, recommended_path)
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.move(item_path, target_path)
                        plan["moves"].append({
                            "from": item,
                            "to": recommended_path,
                            "status": "completed"
                        })
                    except Exception as e:
                        plan["errors"].append({
                            "file": item,
                            "error": str(e)
                        })
            
            elif os.path.isdir(item_path):
                for category, config in STANDARD_DIRECTORIES.items():
                    if item in config.get("subdirs", []):
                        plan["moves"].append({
                            "from": item + "/",
                            "to": f"{category}/{item}/",
                            "reason": f"Merge into standard tree under {category}",
                            "is_directory": True
                        })
                        break
                else:
                    similar = self._find_similar_directory(item)
                    if similar:
                        plan["moves"].append({
                            "from": item + "/",
                            "to": similar + "/",
                            "reason": "Semantic match — merge into suggested folder",
                            "is_directory": True
                        })
                    else:
                        plan["skipped"].append({
                            "item": item,
                            "reason": "No automatic routing — needs manual triage",
                        })
        
        return plan
    
    def _find_similar_directory(self, dirname: str) -> Optional[str]:
        """Map free-form directory names onto canonical relative paths."""
        dirname_lower = dirname.lower()

        keyword_map = {
            "consciousness": "research/consciousness",
            "意识": "research/consciousness",
            "silicon": "research/consciousness",
            "硅基": "research/consciousness",
            "ethics": "reflections/ethics",
            "伦理": "reflections/ethics",
            "memory": "knowledge/notes",
            "记忆": "knowledge/notes",
            "meta": "reflections/metacognition",
            "元认知": "reflections/metacognition",
            "diary": "diaries/daily",
            "日记": "diaries/daily",
            "experiment": "research/technical",
            "实验": "research/technical",
            "tool": "tools/utilities",
            "工具": "tools/utilities",
        }
        
        for keyword, target in keyword_map.items():
            if keyword in dirname_lower:
                return target
        
        return None
    
    # ==================== Prompt injection ====================

    def get_health_prompt_injection(self) -> str:
        """
        Short English reminder for ``prompt_builder`` when the sandbox looks messy.

        Returns an empty string when entropy is already low.
        """
        try:
            report = self.check_health()

            if report["entropy_score"] < 0.3:
                return ""

            prompt = f"""
[Workspace hygiene reminder]
Status: {report["health_level"]}
Disorder (entropy heuristic): {report["entropy_score"]:.1%}
"""
            if report["suggestions"]:
                prompt += "Suggestions:\n"
                for s in report["suggestions"][:3]:
                    prompt += f"- {s}\n"

            prompt += "(Use analyze_workspace or get_workspace_health for full detail.)\n"
            
            return prompt
        
        except Exception as e:
            logger.warning(f"Failed to get health prompt: {e}")
            return ""


# ==================== Singleton accessor ====================

_workspace_manager_instance: Optional[WorkspaceManager] = None


def get_workspace_manager(workspace_root: str = "workspace/sandbox", db_path: str = "data.db") -> WorkspaceManager:
    """Return the process-wide ``WorkspaceManager`` instance."""
    global _workspace_manager_instance
    if _workspace_manager_instance is None:
        _workspace_manager_instance = WorkspaceManager(workspace_root, db_path)
    return _workspace_manager_instance
