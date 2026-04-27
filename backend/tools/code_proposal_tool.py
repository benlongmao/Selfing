"""
Code proposal tool for the Agent (suggestions only, no direct writes).

The Agent may read project code and file structured change proposals for human review.
Created 2026-02-07, v1.0.
"""

import os
import sqlite3
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Repository root (Agent may read; must not write here via this tool)
S_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Proposal artifacts live under the sandbox workspace
PROPOSALS_DIR = os.path.join(S_PROJECT_ROOT, "workspace/sandbox/code_proposals")


class CodeProposalTool:
    """
    Lets the Agent read the repo and submit change proposals for humans to apply.

    Proposals are persisted (DB + Markdown); the Agent must not patch the tree directly here.
    """
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_proposals_dir()
        self._ensure_db_table()
    
    def _ensure_proposals_dir(self):
        """Create the on-disk proposals directory if missing."""
        os.makedirs(PROPOSALS_DIR, exist_ok=True)
    
    def _ensure_db_table(self):
        """Create the SQLite table for proposals if missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_proposals (
                    id TEXT PRIMARY KEY,
                    session_id TEXT DEFAULT 'selfing-session',
                    target_file TEXT NOT NULL,
                    description TEXT NOT NULL,
                    reason TEXT,
                    old_code TEXT,
                    new_code TEXT,
                    expected_effect TEXT,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'normal',
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewer_comment TEXT,
                    applied_at TEXT
                )
            """)
            conn.commit()
    
    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool definitions for registration in the tool router."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "propose_code_change",
                    "description": (
                        "Propose a code change after reading the repo. "
                        "Suggestions are stored as proposals for a developer to review and apply."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_file": {
                                "type": "string",
                                "description": "Path relative to repo root, e.g. 'backend/chat_service.py'",
                            },
                            "description": {
                                "type": "string",
                                "description": "What should change (concise summary)",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this change is needed",
                            },
                            "old_code": {
                                "type": "string",
                                "description": "Existing snippet to replace (optional but helpful)",
                            },
                            "new_code": {
                                "type": "string",
                                "description": "Replacement snippet",
                            },
                            "expected_effect": {
                                "type": "string",
                                "description": "Expected improvement or behavior after the change",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "normal", "high", "critical"],
                                "description": "Proposal priority",
                            },
                        },
                        "required": ["target_file", "description", "reason", "new_code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_my_proposals",
                    "description": "List code change proposals filed by this session.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["all", "pending", "approved", "rejected", "applied"],
                                "description": "Filter by status",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max rows to return",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_proposal_result",
                    "description": "Fetch review outcome for a proposal by id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "proposal_id": {
                                "type": "string",
                                "description": "Proposal id (e.g. CP-...)",
                            },
                        },
                        "required": ["proposal_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_my_code",
                    "description": "Read a source file under the repo root (optionally a line range).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path relative to repo root, e.g. 'backend/chat_service.py'",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Optional 1-based start line",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Optional 1-based end line (inclusive)",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_my_code_files",
                    "description": "List files and subdirectories under a path inside the repo.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory relative to repo root, e.g. 'backend/' or 'backend/tools/'",
                            },
                        },
                    },
                },
            },
        ]

    def route_tool_call(self, func_name: str, args: Dict, session_id: str = "selfing-session") -> Dict:
        """Dispatch a tool-router call to the matching handler."""
        if func_name == "propose_code_change":
            return self.propose_code_change(
                target_file=args.get("target_file"),
                description=args.get("description"),
                reason=args.get("reason"),
                old_code=args.get("old_code", ""),
                new_code=args.get("new_code"),
                expected_effect=args.get("expected_effect", ""),
                priority=args.get("priority", "normal"),
                session_id=session_id
            )
        elif func_name == "list_my_proposals":
            return self.list_my_proposals(
                status=args.get("status", "all"),
                limit=args.get("limit", 20),
                session_id=session_id
            )
        elif func_name == "check_proposal_result":
            return self.check_proposal_result(args.get("proposal_id"))
        elif func_name == "read_my_code":
            return self.read_my_code(
                file_path=args.get("file_path"),
                start_line=args.get("start_line"),
                end_line=args.get("end_line")
            )
        elif func_name == "list_my_code_files":
            return self.list_my_code_files(args.get("directory", "backend/"))
        else:
            return {"error": f"Unknown function: {func_name}"}
    
    def propose_code_change(
        self,
        target_file: str,
        description: str,
        reason: str,
        new_code: str,
        old_code: str = "",
        expected_effect: str = "",
        priority: str = "normal",
        session_id: str = "selfing-session"
    ) -> Dict[str, Any]:
        """Persist a new proposal row plus Markdown for human review."""
        try:
            # Verify target file exists
            full_path = os.path.join(S_PROJECT_ROOT, target_file)
            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Target file does not exist: {target_file}",
                }
            
            # Generate proposal ID
            proposal_id = f"CP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
            created_at = datetime.now(timezone.utc).isoformat()
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO code_proposals 
                    (id, session_id, target_file, description, reason, old_code, new_code, 
                     expected_effect, status, priority, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """, (proposal_id, session_id, target_file, description, reason, 
                      old_code, new_code, expected_effect, priority, created_at))
                conn.commit()
            
            # At the same time, save it as a Markdown file (convenient for developers to view)
            md_content = self._generate_proposal_markdown(
                proposal_id, target_file, description, reason,
                old_code, new_code, expected_effect, priority, created_at
            )
            md_path = os.path.join(PROPOSALS_DIR, f"{proposal_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            logger.info(f"[CODE-PROPOSAL] Proposal created: {proposal_id}")

            return {
                "success": True,
                "proposal_id": proposal_id,
                "message": "Proposal created; pending developer review",
                "file_path": md_path,
                "target_file": target_file,
                "status": "pending",
            }

        except Exception as e:
            logger.error(f"[CODE-PROPOSAL] Failed to create proposal: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_proposal_markdown(
        self,
        proposal_id: str,
        target_file: str,
        description: str,
        reason: str,
        old_code: str,
        new_code: str,
        expected_effect: str,
        priority: str,
        created_at: str
    ) -> str:
        """Render proposal details as Markdown for reviewers."""
        priority_emoji = {
            "low": "🟢",
            "normal": "🟡", 
            "high": "🟠",
            "critical": "🔴"
        }.get(priority, "🟡")
        
        md = f"""# Code change proposal {proposal_id}

## Basic information
- **Proposal id**: {proposal_id}
- **Created at**: {created_at}
- **Target file**: `{target_file}`
- **Priority**: {priority_emoji} {priority}
- **Status**: ⏳ Pending review

---

## Change description
{description}

## Rationale
{reason}

"""

        if old_code:
            md += f"""## Previous code
```python
{old_code}
```

"""

        md += f"""## Proposed code
```python
{new_code}
```

"""

        if expected_effect:
            md += f"""## Expected impact
{expected_effect}

"""

        md += """---

## Review (for maintainers)
### Outcome
- [ ] ✅ Approve
- [ ] ❌ Reject
- [ ] 🔄 Request changes

### Reviewer notes

### Merge status
- [ ] Applied to the codebase

---
*Generated automatically by the Agent.*
"""
        return md
    
    def list_my_proposals(
        self,
        status: str = "all",
        limit: int = 20,
        session_id: str = "selfing-session"
    ) -> Dict[str, Any]:
        """Return recent proposals for this session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if status == "all":
                    cursor = conn.execute("""
                        SELECT * FROM code_proposals 
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM code_proposals 
                        WHERE session_id = ? AND status = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (session_id, status, limit))
                
                rows = cursor.fetchall()
                
                proposals = []
                for row in rows:
                    proposals.append({
                        "id": row["id"],
                        "target_file": row["target_file"],
                        "description": row["description"][:100] + "..." if len(row["description"]) > 100 else row["description"],
                        "status": row["status"],
                        "priority": row["priority"],
                        "created_at": row["created_at"],
                        "reviewer_comment": row["reviewer_comment"]
                    })
                
                # statistics
                stats_cursor = conn.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM code_proposals 
                    WHERE session_id = ?
                    GROUP BY status
                """, (session_id,))
                stats = {row["status"]: row["count"] for row in stats_cursor.fetchall()}
                
                return {
                    "success": True,
                    "proposals": proposals,
                    "total": len(proposals),
                    "statistics": stats
                }
                
        except Exception as e:
            logger.error(f"[CODE-PROPOSAL] Failed to list proposals: {e}")
            return {"success": False, "error": str(e)}
    
    def check_proposal_result(self, proposal_id: str) -> Dict[str, Any]:
        """Fetch a single proposal including review metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM code_proposals WHERE id = ?",
                    (proposal_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return {"success": False, "error": f"Proposal not found: {proposal_id}"}
                
                return {
                    "success": True,
                    "proposal": {
                        "id": row["id"],
                        "target_file": row["target_file"],
                        "description": row["description"],
                        "reason": row["reason"],
                        "status": row["status"],
                        "priority": row["priority"],
                        "created_at": row["created_at"],
                        "reviewed_at": row["reviewed_at"],
                        "reviewer_comment": row["reviewer_comment"],
                        "applied_at": row["applied_at"]
                    }
                }
                
        except Exception as e:
            logger.error(f"[CODE-PROPOSAL] Failed to fetch proposal: {e}")
            return {"success": False, "error": str(e)}
    
    def read_my_code(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """Read a text file from the repository with optional line slicing."""
        try:
            # Security check: Only files within the s project are allowed to be read
            full_path = os.path.abspath(os.path.join(S_PROJECT_ROOT, file_path))
            if not full_path.startswith(S_PROJECT_ROOT):
                return {
                    "success": False,
                    "error": "Security: path must stay inside the repository root",
                }

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                }

            if not os.path.isfile(full_path):
                return {
                    "success": False,
                    "error": f"Not a file: {file_path}",
                }
            
            # read file
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Process line number ranges
            if start_line is not None and end_line is not None:
                # Convert to 0 index
                start_idx = max(0, start_line - 1)
                end_idx = min(total_lines, end_line)
                selected_lines = lines[start_idx:end_idx]
                content = "".join(selected_lines)
                line_info = f"Lines {start_line}-{end_line} of {total_lines}"
            else:
                content = "".join(lines)
                line_info = f"Full file ({total_lines} lines)"
            
            # If the content is too long, truncate it and prompt
            MAX_CHARS = 15000
            truncated = False
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS]
                truncated = True
            
            return {
                "success": True,
                "file_path": file_path,
                "content": content,
                "total_lines": total_lines,
                "line_info": line_info,
                "truncated": truncated,
                "message": "Output truncated; pass start_line/end_line for more" if truncated else None,
            }

        except Exception as e:
            logger.error(f"[CODE-PROPOSAL] Failed to read file: {e}")
            return {"success": False, "error": str(e)}
    
    def list_my_code_files(self, directory: str = "backend/") -> Dict[str, Any]:
        """List immediate children of a directory under the repo root."""
        try:
            full_path = os.path.abspath(os.path.join(S_PROJECT_ROOT, directory))
            if not full_path.startswith(S_PROJECT_ROOT):
                return {
                    "success": False,
                    "error": "Security: path must stay inside the repository root",
                }

            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {directory}",
                }
            
            files = []
            dirs = []
            
            for item in sorted(os.listdir(full_path)):
                item_path = os.path.join(full_path, item)
                rel_path = os.path.join(directory, item)
                
                # Skip hidden files and __pycache__
                if item.startswith(".") or item == "__pycache__":
                    continue
                
                if os.path.isdir(item_path):
                    dirs.append({
                        "name": item + "/",
                        "path": rel_path + "/",
                        "type": "directory"
                    })
                else:
                    size = os.path.getsize(item_path)
                    files.append({
                        "name": item,
                        "path": rel_path,
                        "type": "file",
                        "size": size,
                        "size_human": f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                    })
            
            return {
                "success": True,
                "directory": directory,
                "directories": dirs,
                "files": files,
                "total_dirs": len(dirs),
                "total_files": len(files)
            }
            
        except Exception as e:
            logger.error(f"[CODE-PROPOSAL] Failed to list directory: {e}")
            return {"success": False, "error": str(e)}


# Developer helpers (not exposed to the model)
def approve_proposal(db_path: str, proposal_id: str, comment: str = "") -> bool:
    """Mark a proposal approved (maintainer helper, not exposed to the model)."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                UPDATE code_proposals 
                SET status = 'approved', 
                    reviewed_at = ?,
                    reviewer_comment = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), comment, proposal_id))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"approve_proposal failed: {e}")
        return False


def reject_proposal(db_path: str, proposal_id: str, comment: str = "") -> bool:
    """Mark a proposal rejected (maintainer helper)."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                UPDATE code_proposals 
                SET status = 'rejected', 
                    reviewed_at = ?,
                    reviewer_comment = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), comment, proposal_id))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"reject_proposal failed: {e}")
        return False


def mark_proposal_applied(db_path: str, proposal_id: str) -> bool:
    """Record that a proposal was merged (maintainer helper)."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                UPDATE code_proposals 
                SET status = 'applied', 
                    applied_at = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), proposal_id))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"mark_proposal_applied failed: {e}")
        return False
