#!/usr/bin/env python3
"""
[2026-02-05] Workspace path normalizer.

Goals:
1. Single source of truth for where sandbox files should live.
2. Heuristic routing from filenames / hints into standard buckets.
3. Detect legacy or policy-violating folders and remap safely.
4. Backward-compatible migration keys for older tree layouts.

Notes:
- Anchored to ``WORKSPACE_ROOT`` / ``sandbox/`` with escape checks.
- Complements FileTool path mapping; regex/keyword lists stay bilingual for mixed filenames.
"""

import os
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# ==================== Standard sandbox layout ====================

# v1.3 policy-aligned buckets (see workspace policy doc in repo)
STANDARD_DIRECTORIES = {
    "diaries": {
        "path": "sandbox/diaries",
        "description": "Diaries / personal logs",
        "auto_map_patterns": [
            r"^diary_\d{8}_\d{4,6}\.md$",  # diary_20260205_1530.md
            r"^diary_.*\.md$",  # diary_anything.md
        ],
        "auto_map_keywords": ["日记", "diary"],
    },
    "experiments": {
        "path": "sandbox/experiments",
        "description": "Experiment logs",
        "auto_map_patterns": [
            r"^experiment_\d{8}_.*\.md$",  # experiment_20260205_test.md
            r"^(experiment|test|EXP-).*\.md$",
        ],
        "auto_map_keywords": ["experiment", "test", "实验", "测试"],
    },
    "docs": {
        "path": "sandbox/docs",
        "description": "Docs (specs, summaries, guides)",
        "auto_map_patterns": [
            r".*summary.*\.md$",
            r".*文档.*\.md$",
            r".*规范.*\.md$",
        ],
        "auto_map_keywords": ["summary", "文档", "说明", "规范", "doc"],
    },
    "code": {
        "path": "sandbox/code",
        "description": "Source code files",
        "auto_map_patterns": [
            r".*\.(py|js|ts|java|cpp|c|go|rs)$",
        ],
        "auto_map_keywords": [],
    },
    "drafts": {
        "path": "sandbox/drafts",
        "description": "Drafts / scratch / unclassified",
        "auto_map_patterns": [
            r"^draft_.*",
            r"^temp_.*",
            r"^tmp_.*",
        ],
        "auto_map_keywords": ["draft", "temp", "草稿", "临时"],
    },
    "research": {
        "path": "sandbox/research",
        "description": "Research notes (search, study, deep dives)",
        "auto_map_patterns": [
            r"^(search|learn|research)_.*\.md$",
            r".*研究.*\.md$",
            r".*学习.*\.md$",
        ],
        "auto_map_keywords": ["search", "learn", "research", "研究", "学习", "调研"],
    },
    "projects": {
        "path": "sandbox/projects",
        "description": "Long-lived structured projects",
        "auto_map_patterns": [
            r"^project_.*",
        ],
        "auto_map_keywords": ["project", "项目"],
    },
    "archives": {
        "path": "sandbox/archives",
        "description": "Archives / completed or cold storage",
        "auto_map_patterns": [],
        "auto_map_keywords": ["archive", "归档", "旧"],
    },
    "tools": {
        "path": "sandbox/tools",
        "description": "Utility scripts",
        "auto_map_patterns": [
            r".*_tool\.py$",
            r".*工具\.py$",
        ],
        "auto_map_keywords": ["tool", "工具", "脚本"],
    },
    "silicon_consciousness": {
        "path": "sandbox/silicon_consciousness",
        "description": "Consciousness / silicon-self exploration",
        "auto_map_patterns": [
            r".*consciousness.*\.md$",
            r".*意识.*\.md$",
            r".*硅基.*\.md$",
        ],
        "auto_map_keywords": ["consciousness", "silicon", "意识", "硅基"],
    },
    "metacognition": {
        "path": "sandbox/metacognition",
        "description": "Metacognition notes",
        "auto_map_patterns": [
            r".*metacognition.*\.md$",
            r".*元认知.*\.md$",
        ],
        "auto_map_keywords": ["metacognition", "元认知", "自省"],
    },
    "knowledge": {
        "path": "sandbox/knowledge",
        "description": "Curated knowledge base",
        "auto_map_patterns": [],
        "auto_map_keywords": ["knowledge", "知识"],
    },
    "logs": {
        "path": "sandbox/logs",
        "description": "System and operator logs",
        "auto_map_patterns": [
            r".*\.log$",
            r"^log_.*\.md$",
            r".*_log\.md$",
        ],
        "auto_map_keywords": ["log", "日志", "操作记录"],
    },
}

# Legacy top-level folder names → canonical bucket (CN keys kept for old trees)
LEGACY_DIRECTORY_MAPPING = {
    "autonomous_diaries": "diaries",
    "autonomous_searches": "research",
    "autonomous_learning": "research",
    "consciousness_lab": "silicon_consciousness",
    "硅基意识探索": "silicon_consciousness",
    "temp": "drafts",
    "test": "experiments",
    "memory": "archives",  # legacy memory/ → archives
    "plans": "projects",
    "frameworks": "knowledge",
    "screenshots": "archives",
    "s44_workspace": "drafts",  # nonstandard historical name
}

# ==================== Manager ====================

_POLICY_DOC_NAMES = frozenset({
    "工作空间规章制度.md",
    "WORKSPACE_POLICY.md",
    "workspace_policy.md",
})


class WorkspacePathManager:
    """Resolve, normalize, and auto-place paths under the sandbox."""

    def __init__(self, workspace_root: str = None):
        # Default: backend.project_paths.WORKSPACE_ROOT
        if workspace_root is None:
            from backend.project_paths import WORKSPACE_ROOT
            workspace_root = WORKSPACE_ROOT
        self.workspace_root = os.path.abspath(workspace_root)
        self.sandbox_root = os.path.join(self.workspace_root, "sandbox")
        
        self._ensure_standard_directories()

    def _ensure_standard_directories(self):
        """Create standard bucket folders if missing."""
        for dir_key, dir_info in STANDARD_DIRECTORIES.items():
            dir_path = os.path.join(self.workspace_root, dir_info["path"])
            os.makedirs(dir_path, exist_ok=True)
    
    def normalize_path(self, input_path: str, file_type_hint: Optional[str] = None) -> Tuple[str, str, bool]:
        """
        Normalize ``input_path`` under ``sandbox/`` with legacy repair + optional auto-map.

        Returns:
            ``(absolute_path, relative_to_workspace, was_corrected)``.
        """
        if not input_path:
            return self.sandbox_root, "sandbox", False
        
        input_path = input_path.strip()
        original_path = input_path
        was_corrected = False
        
        # Step 1: strip common prefixes
        prefixes_to_remove = [
            "workspace/sandbox/",
            "workspace/",
            "sandbox/",
        ]
        for prefix in prefixes_to_remove:
            if input_path.startswith(prefix):
                input_path = input_path[len(prefix):]
                break
        
        # Step 2: collapse doubled first segment (e.g. sandbox/sandbox/)
        normalized = input_path.replace('\\', '/')
        parts = normalized.split('/')
        if len(parts) >= 2 and parts[0] == parts[1]:
            input_path = '/'.join(parts[1:])
            was_corrected = True
            logger.info(f"[PATH-NORM] Fixed nested path: {original_path} → {input_path}")

        # Step 3: remap known legacy first-segment names
        if '/' in input_path:
            first_dir = input_path.split('/')[0]
            if first_dir in LEGACY_DIRECTORY_MAPPING:
                new_dir = LEGACY_DIRECTORY_MAPPING[first_dir]
                input_path = input_path.replace(first_dir, new_dir, 1)
                was_corrected = True
                logger.warning(
                    f"[PATH-NORM] Legacy directory segment remapped: {first_dir} → {new_dir}\n"
                    f"  before: {original_path}\n"
                    f"  after: {input_path}"
                )

        # Step 4: auto-map bare filenames at sandbox root
        if '/' not in input_path and '\\' not in input_path:
            basename = os.path.basename(input_path)
            mapped_dir = self._auto_map_file(basename, file_type_hint)
            
            if mapped_dir:
                input_path = f"{mapped_dir}/{basename}"
                was_corrected = True
                logger.info(f"[PATH-NORM] Auto-mapped: {original_path} → {input_path}")

        # Step 5: build absolute path
        abs_path = os.path.join(self.sandbox_root, input_path)
        abs_path = os.path.normpath(abs_path)
        
        # Step 6: sandbox jail — fall back to drafts/
        if not abs_path.startswith(self.sandbox_root):
            logger.error(
                f"[PATH-NORM] Path escape blocked: {input_path} "
                f"(resolved: {abs_path}, sandbox: {self.sandbox_root})"
            )
            abs_path = os.path.join(self.sandbox_root, "drafts", os.path.basename(input_path))
            was_corrected = True

        # Relative to workspace root
        rel_path = os.path.relpath(abs_path, self.workspace_root)
        rel_path = rel_path.replace('\\', '/')
        
        return abs_path, rel_path, was_corrected
    
    def _auto_map_file(self, filename: str, file_type_hint: Optional[str] = None) -> Optional[str]:
        """
        Pick a standard bucket key (``diaries``, ``research``, …) or None.

        Caller should default unknown to ``drafts``.
        """
        # Hint wins when it matches a configured bucket
        if file_type_hint and file_type_hint in STANDARD_DIRECTORIES:
            return STANDARD_DIRECTORIES[file_type_hint]["path"].split('/')[-1]
        
        for dir_key, dir_info in STANDARD_DIRECTORIES.items():
            for pattern in dir_info["auto_map_patterns"]:
                if re.match(pattern, filename, re.IGNORECASE):
                    logger.debug(f"[AUTO-MAP] {filename} matched {pattern} → {dir_key}")
                    return dir_info["path"].split('/')[-1]

            filename_lower = filename.lower()
            for keyword in dir_info["auto_map_keywords"]:
                if keyword.lower() in filename_lower:
                    logger.debug(f"[AUTO-MAP] {filename} keyword {keyword!r} → {dir_key}")
                    return dir_info["path"].split('/')[-1]

        return None

    def get_standard_path_for_action(self, action_type: str, filename_prefix: str = "") -> str:
        """
        Build a relative ``sandbox/<bucket>/<file>.md`` for autonomous writers.

        ``action_type`` examples: ``diary``, ``search``, ``learning``.
        """
        # action → bucket
        action_to_dir = {
            "diary": "diaries",
            "write_diary": "diaries",
            "search": "research",
            "web_search": "research",
            "learning": "research",
            "learn": "research",
            "experiment": "experiments",
            "analysis": "docs",
            "organize": "archives",
        }
        
        target_dir = action_to_dir.get(action_type, "drafts")
        
        if not filename_prefix:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_prefix = f"{action_type}_{timestamp}"
        
        if not filename_prefix.endswith('.md'):
            filename_prefix += '.md'
        
        return f"sandbox/{target_dir}/{filename_prefix}"

    def detect_violations(self) -> Dict:
        """
        Scan immediate sandbox children for unknown dirs / loose files.

        Returns a small report dict for hygiene UIs or prompts.
        """
        violations = {
            "legacy_directories": [],
            "root_files": [],
            "nested_issues": [],
            "total_directories": 0,
        }

        try:
            sandbox_entries = list(Path(self.sandbox_root).iterdir())
            directories = [e for e in sandbox_entries if e.is_dir()]
            files = [
                e for e in sandbox_entries
                if e.is_file() and e.name not in _POLICY_DOC_NAMES
            ]
            
            violations["total_directories"] = len(directories)
            
            standard_dir_names = [
                d["path"].split('/')[-1] 
                for d in STANDARD_DIRECTORIES.values()
            ]
            
            for directory in directories:
                dir_name = directory.name
                if dir_name not in standard_dir_names:
                    if dir_name in LEGACY_DIRECTORY_MAPPING:
                        violations["legacy_directories"].append({
                            "path": dir_name,
                            "should_migrate_to": LEGACY_DIRECTORY_MAPPING[dir_name],
                            "type": "known_legacy"
                        })
                    else:
                        violations["legacy_directories"].append({
                            "path": dir_name,
                            "should_migrate_to": "unknown",
                            "type": "unknown_directory"
                        })
            
            for file in files:
                violations["root_files"].append({
                    "filename": file.name,
                    "suggested_dir": self._auto_map_file(file.name) or "drafts"
                })
        
        except Exception as e:
            logger.error(f"Failed to detect violations: {e}")
        
        return violations


# ==================== Global singleton ====================

_global_path_manager = None

def get_path_manager(workspace_root: str = None) -> WorkspacePathManager:
    """Lazy singleton for ``WorkspacePathManager``."""
    global _global_path_manager
    if _global_path_manager is None:
        _global_path_manager = WorkspacePathManager(workspace_root)
    return _global_path_manager


# ==================== Convenience wrappers ====================

def normalize_path(input_path: str, file_type_hint: Optional[str] = None) -> Tuple[str, str, bool]:
    """``get_path_manager().normalize_path(...)``."""
    manager = get_path_manager()
    return manager.normalize_path(input_path, file_type_hint)


def get_standard_path_for_action(action_type: str, filename_prefix: str = "") -> str:
    """``get_path_manager().get_standard_path_for_action(...)``."""
    manager = get_path_manager()
    return manager.get_standard_path_for_action(action_type, filename_prefix)


def detect_workspace_violations() -> Dict:
    """``get_path_manager().detect_violations()``."""
    manager = get_path_manager()
    return manager.detect_violations()


__all__ = [
    "WorkspacePathManager",
    "get_path_manager",
    "normalize_path",
    "get_standard_path_for_action",
    "detect_workspace_violations",
    "STANDARD_DIRECTORIES",
    "LEGACY_DIRECTORY_MAPPING",
]
