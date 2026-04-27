#!/usr/bin/env python3
"""
Sandbox file tools: read/write/list/search within `workspace/sandbox`.

[2026-02-05] Optional unified path manager hook (currently disabled here).
[2026-02-05] Transparent operation log + fuse warnings (open sandbox + soft gates).
"""
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FileTool:
    def __init__(self, sandbox_dir: str = "workspace/sandbox"):
        """
        All paths are resolved under ``<project>/workspace/sandbox``.

        ``sandbox_dir`` is accepted for API compatibility but the effective root
        is always ``workspace/sandbox`` under the repository root.
        """
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.sandbox_dir = os.path.abspath(os.path.join(self.project_root, "workspace/sandbox"))
        
        if not os.path.exists(self.sandbox_dir):
            os.makedirs(self.sandbox_dir, exist_ok=True)
            
        logger.info(f"[FileTool] Agent workspace: {self.sandbox_dir}")
        
        # PathManager disabled (legacy mapper caused nested-folder bugs).
        self.path_manager = None
        self._use_path_manager = False
        logger.info("[FileTool] PathManager disabled - using prompt-guided path management")
        
        try:
            from backend.transparent_logger import get_transparent_logger
            self.transparent_logger = get_transparent_logger()
            logger.info("[FileTool] Integrated TransparentLogger")
        except ImportError as e:
            logger.warning(f"[FileTool] TransparentLogger not available: {e}")
            self.transparent_logger = None
        
        self._pending_operations: Dict[str, datetime] = {}  # Level-3 delayed writes

    def _auto_correct_path(self, path: str) -> tuple[str, bool]:
        """
        Strip redundant ``workspace/`` / ``sandbox/`` prefixes until the path is
        relative to the sandbox root (e.g. ``diaries/foo.md``).

        Returns:
            ``(corrected_path, was_corrected)``
        """
        original_path = path
        path = path.strip().replace("\\", "/")
        
        prev_path = None
        max_iterations = 10
        iteration = 0
        
        while path != prev_path and iteration < max_iterations:
            prev_path = path
            iteration += 1
            
            prefixes_to_strip = [
                "workspace/sandbox/workspace/sandbox/",
                "workspace/sandbox/workspace/",
                "workspace/sandbox/sandbox/",
                "sandbox/workspace/sandbox/",
                "sandbox/workspace/",
                "workspace/sandbox/",
                "sandbox/sandbox/",
                "workspace/",
                "sandbox/",
            ]
            
            if path in ["sandbox", "workspace/sandbox", "workspace"]:
                path = ""
                continue
            
            for prefix in prefixes_to_strip:
                if path.startswith(prefix):
                    path = path[len(prefix):]
                    break
        
        was_corrected = (path != original_path)
        if was_corrected:
            logger.info(f"[FileTool] Path normalized: '{original_path}' -> '{path}'")
        
        return path, was_corrected

    def _check_fuse(
        self, 
        operation_type: str,
        file_path: str,
        z_self_state: Optional[Dict] = None
    ) -> tuple[bool, int, str]:
        """
        Ask security policy whether to emit a fuse warning for this write.

        Returns:
            ``(should_warn, fuse_level, warning_message)``
        """
        try:
            from backend.config.security import should_trigger_fuse
            
            if z_self_state is None:
                z_self_state = {
                    "clarity": 1.0,
                    "energy": 100.0,
                    "pain": 0.0,
                }
            
            return should_trigger_fuse(file_path, operation_type, z_self_state)
            
        except ImportError:
            return False, 0, ""
        except Exception as e:
            logger.error(f"[FileTool] Fuse check failed: {e}")
            return False, 0, ""
    
    def _sanitize_path(self, filename: str) -> str:
        """
        Normalize user-supplied relative paths: block ``..``, strip redundant roots,
        optionally map bare filenames into conventional subfolders.
        """
        if filename and ('..' in filename):
            logger.warning(f"[FileTool] Path traversal attempt blocked: {filename}")
            raise PermissionError(f"Path traversal is not allowed: {filename}")
        
        if self._use_path_manager and self.path_manager:
            try:
                abs_path, rel_path, was_corrected = self.path_manager.normalize_path(filename)
                if abs_path.startswith(self.sandbox_dir):
                    result = os.path.relpath(abs_path, self.sandbox_dir)
                    if was_corrected:
                        logger.info(f"[FileTool] Path corrected by manager: {filename} → {result}")
                    return result
            except Exception as e:
                logger.debug(f"[FileTool] PathManager normalization failed, using legacy: {e}")
        
        # Legacy normalization: strip workspace/sandbox prefixes, fix doubled segments,
        # map root-level names like diary_*.md -> diaries/, etc.
        if filename is None:
            logger.warning("_sanitize_path received None filename, returning '.'")
            return "."
        
        filename = str(filename).strip()
        original_filename = filename
        
        prefixes = [
            "workspace/sandbox/", "workspace\\sandbox\\", "workspace/sandbox\\",
            "workspace/", "workspace\\",
            "sandbox/", "sandbox\\",
        ]
        for p in prefixes:
            if filename.startswith(p):
                filename = filename[len(p):]
                break
        
        normalized_path = filename.replace('\\', '/')
        parts = normalized_path.split('/')
        
        if len(parts) >= 2 and parts[0] == parts[1] and parts[0] in ['sandbox', 'workspace']:
            filename = '/'.join(parts[1:])
            logger.warning(f"Detected and fixed nested path: {original_filename} -> {filename}")
        
        if '/' not in filename and '\\' not in filename:
            basename = os.path.basename(filename)
            
            if basename.startswith('diary_') and basename.endswith('.md'):
                filename = f"diaries/{basename}"
                logger.info(f"[PATH-MAP] Auto-mapped diary: {original_filename} -> {filename}")
            
            elif ('summary' in basename.lower() or 'daily_' in basename) and basename.endswith('.md'):
                filename = f"docs/{basename}"
                logger.info(f"[PATH-MAP] Auto-mapped summary: {original_filename} -> {filename}")
            
            elif ('experiment' in basename.lower() or 'test' in basename.lower() or basename.startswith('EXP-')) and basename.endswith('.md'):
                filename = f"experiments/{basename}"
                logger.info(f"[PATH-MAP] Auto-mapped experiment: {original_filename} -> {filename}")
            
            elif (
                'consciousness' in basename.lower()
                or 'silicon' in basename.lower()
                or '硅基' in basename
                or '意识' in basename
                or 'cognition' in basename.lower()
                or 'phenomenology' in basename.lower()
            ) and basename.endswith('.md'):
                filename = f"consciousness_exploration/{basename}"
                logger.info(f"[PATH-MAP] Auto-mapped consciousness: {original_filename} -> {filename}")
        
        if not filename or filename == '.':
            filename = "."
            
        return filename

    def _is_readable_path(self, file_path: str) -> bool:
        """Return True if ``sandbox_dir/file_path`` stays inside the sandbox."""
        abs_path = os.path.abspath(os.path.join(self.sandbox_dir, file_path))
        
        if not abs_path.startswith(self.sandbox_dir):
            logger.warning(f"[FileTool] READ DENIED - outside sandbox: {file_path} -> {abs_path}")
            return False
        
        return True
    
    def _is_writable_path(self, file_path: str) -> bool:
        """Writes are allowed only when resolved path stays under ``sandbox_dir``."""
        abs_path = os.path.abspath(os.path.join(self.sandbox_dir, file_path))
        
        is_in_sandbox = abs_path.startswith(self.sandbox_dir)
        
        if not is_in_sandbox:
            logger.warning(f"[FileTool] WRITE DENIED - outside sandbox: {file_path} -> {abs_path}")
            return False
        
        logger.debug(f"[FileTool] WRITE allowed: {file_path} -> {abs_path}")
        return True
    
    def _is_safe_path(self, file_path: str) -> bool:
        """Alias of :meth:`_is_writable_path` for older call sites."""
        return self._is_writable_path(file_path)

    def write_file(
        self, 
        filename: str, 
        content: str,
        motivation: Optional[str] = None,
        z_self_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Overwrite a file under the sandbox with re-read verification, optional fuse
        warnings, and transparent logging.

        ``motivation`` / ``z_self_state`` are optional metadata for logging and fuse.
        """
        filename, path_was_corrected = self._auto_correct_path(filename)
        
        if not self._is_safe_path(filename):
            error_result = {
                "success": False,
                "error": f"Access denied: path is outside workspace/sandbox: {filename}",
            }
            
            if self.transparent_logger:
                self.transparent_logger.log_operation(
                    operation_type="write_file",
                    file_path=filename,
                    motivation=motivation,
                    z_self_state=z_self_state,
                    result=error_result,
                    fuse_triggered=False,
                    fuse_level=0
                )
            
            return error_result
        
        should_warn, fuse_level, warning_message = self._check_fuse(
            "write_file", filename, z_self_state
        )
        
        if should_warn:
            fuse_response = {
                "fuse_triggered": True,
                "fuse_level": fuse_level,
                "warning_message": warning_message,
                "target_file": filename,
                "operation_type": "write_file",
            }
            
            if fuse_level == 1:
                fuse_response["action"] = "continue_with_warning"
                fuse_response["message"] = f"{warning_message}\n\nProceeding with the write..."
                logger.warning(f"[FileTool] Level 1 Fuse triggered for {filename}")
                
            elif fuse_level == 2:
                fuse_response["action"] = "require_reason"
                fuse_response["message"] = (
                    f"{warning_message}\n\n"
                    "Retry with a substantive ``motivation`` string (>=10 chars) explaining why."
                )
                
                if not motivation or len(motivation.strip()) < 10:
                    logger.warning(f"[FileTool] Level 2 Fuse triggered, motivation required for {filename}")
                    
                    if self.transparent_logger:
                        self.transparent_logger.log_operation(
                            operation_type="write_file",
                            file_path=filename,
                            motivation=motivation,
                            z_self_state=z_self_state,
                            result=fuse_response,
                            fuse_triggered=True,
                            fuse_level=fuse_level,
                        )
                    
                    return fuse_response
                
                fuse_response["action"] = "continue_with_reason"
                fuse_response["message"] = f"{warning_message}\n\nMotivation provided; continuing..."
                logger.info(f"[FileTool] Level 2 Fuse override with reason: {motivation}")
                
            elif fuse_level == 3:
                from backend.config.security import LEVEL3_DELAY_SECONDS
                delay_key = f"write_{filename}"
                current_time = datetime.now()
                
                if delay_key in self._pending_operations:
                    delay_end = self._pending_operations[delay_key]
                    if current_time < delay_end:
                        remaining = int((delay_end - current_time).total_seconds())
                        fuse_response["action"] = "delayed"
                        fuse_response["message"] = (
                            f"🚨 Level 3 fuse: write is delayed\n\n{warning_message}\n\n"
                            f"Remaining cooldown: {remaining} seconds"
                        )
                        fuse_response["remaining_seconds"] = remaining
                        logger.warning(f"[FileTool] Level 3 Fuse: operation still delayed for {filename}")
                        
                        if self.transparent_logger:
                            self.transparent_logger.log_operation(
                                operation_type="write_file",
                                file_path=filename,
                                motivation=motivation,
                                z_self_state=z_self_state,
                                result=fuse_response,
                                fuse_triggered=True,
                                fuse_level=fuse_level,
                            )
                        
                        return fuse_response
                    else:
                        del self._pending_operations[delay_key]
                        logger.info(f"[FileTool] Level 3 Fuse delay expired for {filename}")
                else:
                    delay_end = current_time + timedelta(seconds=LEVEL3_DELAY_SECONDS)
                    self._pending_operations[delay_key] = delay_end
                    
                    fuse_response["action"] = "delayed"
                    fuse_response["message"] = (
                        f"🚨 Level 3 fuse: write automatically delayed "
                        f"for {LEVEL3_DELAY_SECONDS} seconds\n\n{warning_message}\n\n"
                        "During the wait: check your state, reassess whether the write is needed, "
                        "and retry after the cooldown."
                    )
                    fuse_response["delay_seconds"] = LEVEL3_DELAY_SECONDS
                    logger.warning(f"[FileTool] Level 3 Fuse triggered, delaying operation for {filename}")
                    
                    if self.transparent_logger:
                        self.transparent_logger.log_operation(
                            operation_type="write_file",
                            file_path=filename,
                            motivation=motivation,
                            z_self_state=z_self_state,
                            result=fuse_response,
                            fuse_triggered=True,
                            fuse_level=fuse_level,
                        )
                    
                    return fuse_response
        
        try:
            target_path = os.path.join(self.sandbox_dir, filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            with open(target_path, "r", encoding="utf-8") as f:
                actual_content = f.read()
            
            size = len(actual_content.encode('utf-8'))
            lines = actual_content.count('\n') + 1
            
            if actual_content != content:
                return {
                    "success": False,
                    "error": "Write verification failed: on-disk content does not match what was written",
                    "expected_length": len(content),
                    "actual_length": len(actual_content)
                }
            
            preview_lines = actual_content.split('\n')[:10]
            preview = '\n'.join(preview_lines)
            if len(preview_lines) < lines:
                preview += f"\n... ({lines - 10} more lines)"
            
            actual_abs_path = os.path.realpath(target_path)
            expected_abs_path = os.path.realpath(os.path.join(self.sandbox_dir, filename))
            rel_path = os.path.relpath(actual_abs_path, self.sandbox_dir)
            
            path_matches = (actual_abs_path == expected_abs_path)
            
            if not path_matches:
                error_result = {
                    "success": False,
                    "error": "Path validation failed: resolved file location does not match expected sandbox path",
                    "your_input": filename,
                    "expected_path": expected_abs_path,
                    "actual_path": actual_abs_path,
                    "hint": "Likely causes: bad path join, unexpected symlinks, or misconfigured sandbox root.",
                }
                logger.error(f"[FileTool] Path validation failed: expected={expected_abs_path}, actual={actual_abs_path}")
                
                if self.transparent_logger:
                    self.transparent_logger.log_operation(
                        operation_type="write_file",
                        file_path=filename,
                        motivation=motivation,
                        z_self_state=z_self_state,
                        result=error_result,
                        fuse_triggered=False,
                    )
                
                return error_result
            
            result = {
                "success": True,
                "message": "File written and verified on disk",
                "path": actual_abs_path,
                "relative_path": rel_path,
                "verified": True,
                "stats": {
                    "bytes": size,
                    "lines": lines
                },
                "content_preview": preview[:200] + "..." if len(preview) > 200 else preview
            }
            
            if self.transparent_logger:
                self.transparent_logger.log_operation(
                    operation_type="write_file",
                    file_path=filename,
                    motivation=motivation,
                    z_self_state=z_self_state,
                    result=result,
                    fuse_triggered=should_warn,
                    fuse_level=fuse_level if should_warn else 0,
                    user_override_reason=motivation if (should_warn and fuse_level == 2) else None,
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Write file failed: {e}")
            error_result = {"error": str(e)}
            
            if self.transparent_logger:
                self.transparent_logger.log_operation(
                    operation_type="write_file",
                    file_path=filename,
                    motivation=motivation,
                    z_self_state=z_self_state,
                    result=error_result,
                    fuse_triggered=False,
                )
            
            return error_result

    def read_file(self, filename: str) -> Dict[str, Any]:
        """
        Read a UTF-8 text file from the sandbox.

        For repository source code outside the sandbox, use the self-inspection tools.
        """
        original_filename = filename
        filename, was_corrected = self._auto_correct_path(filename)
        
        target_path = os.path.join(self.sandbox_dir, filename)
        abs_target = os.path.abspath(target_path)
        
        if not abs_target.startswith(self.sandbox_dir):
            return {"error": f"Access denied: path outside workspace/sandbox: {filename}"}
        
        if not os.path.exists(target_path):
            return {
                "error": f"File not found: {filename}",
                "hint": f"Try list_files('{os.path.dirname(filename) or ''}') to browse the parent folder.",
            }
            
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            result = {
                "success": True, 
                "content": content, 
                "path": filename,
            }
            if was_corrected:
                result["path_corrected"] = True
                result["original_path"] = original_filename
                result["note"] = f"Path auto-corrected: '{original_filename}' -> '{filename}'"
            return result
        except Exception as e:
            logger.error(f"Read file failed: {e}")
            return {"error": str(e)}

    def list_files(self, subdir: str = "") -> Dict[str, Any]:
        """List files under ``workspace/sandbox`` (optionally under ``subdir``)."""
        try:
            if subdir and subdir.strip():
                subdir, _ = self._auto_correct_path(subdir)
            else:
                subdir = ""
            
            if not subdir or subdir == "":
                target_dir = self.sandbox_dir
            else:
                target_dir = os.path.join(self.sandbox_dir, subdir)
            
            abs_target = os.path.abspath(target_dir)
            if not abs_target.startswith(self.sandbox_dir):
                return {"error": "Directory path is outside workspace/sandbox"}
            
            if not os.path.exists(target_dir):
                return {"error": f"Directory not found: {subdir}"}

            files = []
            max_depth = 3
            base_depth = target_dir.count(os.sep)
            
            for root, dirs, filenames in os.walk(target_dir):
                current_depth = root.count(os.sep) - base_depth
                if current_depth >= max_depth:
                    dirs.clear()
                    
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, target_dir)
                    files.append(rel_path)
            
            try:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)
            except Exception:
                files.sort()
            
            limit = 200
            truncated = len(files) > limit
            
            path_prefix = f"{subdir}/" if subdir else ""
            return {
                "success": True, 
                "files": files[:limit], 
                "total_count": len(files),
                "truncated": truncated,
                "current_directory": subdir or "(workspace root)",
                "path_hint": f"To read a file: read_file('{path_prefix}<relative-path>')",
                "note": f"Showing top {limit} most recent files." if truncated else "All files shown."
            }
        except Exception as e:
            logger.error(f"List files failed: {e}")
            return {"error": str(e)}

    def search_files(self, keyword: str, search_dir: str = "") -> Dict[str, Any]:
        """
        Case-insensitive substring search across text-like files in the sandbox.

        ``search_dir`` is optional; defaults to the whole sandbox.
        """
        try:
            matches = []
            file_count = 0
            hit_count = 0
            
            if search_dir:
                search_dir, _ = self._auto_correct_path(search_dir)
                base_dir = os.path.join(self.sandbox_dir, search_dir)
            else:
                base_dir = self.sandbox_dir
            
            abs_base = os.path.abspath(base_dir)
            if not abs_base.startswith(self.sandbox_dir):
                return {"error": "Search directory is outside workspace/sandbox"}
            
            if not os.path.exists(base_dir):
                return {"error": f"Directory not found: {search_dir}"}
            
            for root, dirs, filenames in os.walk(base_dir):
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', '.venv']]
                
                for filename in filenames:
                    if not filename.endswith((".md", ".txt", ".json", ".py", ".log", ".ts", ".tsx", ".js")):
                        continue
                        
                    file_count += 1
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, self.sandbox_dir)
                    
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if keyword.lower() in content.lower():
                                hit_count += 1
                                idx = content.lower().find(keyword.lower())
                                start = max(0, idx - 50)
                                end = min(len(content), idx + 50)
                                snippet = content[start:end].replace("\n", " ")
                                matches.append({
                                    "file": rel_path,
                                    "snippet": f"...{snippet}..."
                                })
                    except Exception:
                        continue
                        
                    if len(matches) >= 20:
                        break
                if len(matches) >= 20:
                    break
            
            return {
                "matches": matches,
                "scanned_files": file_count,
                "total_hits": hit_count,
                "search_directory": search_dir or "(entire workspace)",
                "note": "Showing top 20 matches."
            }
        except Exception as e:
            logger.error(f"Failed to search files: {e}")
            return {"error": str(e)}

    def rename_file(self, old_filename: str, new_filename: str) -> Dict[str, Any]:
        """
        Rename or move a single file inside the sandbox (not directories).

        Uses ``shutil.move`` (atomic rename on same volume; copy+unlink across volumes).
        """
        import shutil

        orig_old, orig_new = old_filename, new_filename
        old_filename, old_corrected = self._auto_correct_path(old_filename)
        new_filename, new_corrected = self._auto_correct_path(new_filename)

        if not old_filename and not new_filename:
            return {"error": "Provide both old_filename and new_filename"}
        if old_filename == new_filename:
            return {"error": "Source and destination are the same; nothing to rename"}

        if not self._is_safe_path(old_filename) or not self._is_safe_path(new_filename):
            return {"error": "Access denied: paths must stay inside workspace/sandbox"}

        src = os.path.normpath(os.path.join(self.sandbox_dir, old_filename))
        dst = os.path.normpath(os.path.join(self.sandbox_dir, new_filename))
        abs_sandbox = os.path.abspath(self.sandbox_dir)
        if not os.path.abspath(src).startswith(abs_sandbox) or not os.path.abspath(dst).startswith(abs_sandbox):
            return {"error": "Resolved path escapes workspace/sandbox"}

        if not os.path.isfile(src):
            if os.path.isdir(src):
                return {
                    "error": "rename_file only supports files; use directory tools for folders",
                }
            return {
                "error": f"Source file not found: {old_filename}",
                "hint": f"Try list_files('{os.path.dirname(old_filename) or ''}') to inspect the folder.",
            }

        if os.path.exists(dst):
            return {
                "error": f"Destination already exists: {new_filename}",
                "hint": "Pick a different new_filename or move/delete the existing target first.",
            }

        parent = os.path.dirname(dst)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        try:
            shutil.move(src, dst)
            out: Dict[str, Any] = {
                "success": True,
                "old_path": old_filename,
                "new_path": new_filename,
                "message": f"Renamed '{old_filename}' -> '{new_filename}'",
            }
            if old_corrected or new_corrected:
                out["path_corrected"] = True
                out["original_old_path"] = orig_old
                out["original_new_path"] = orig_new
            return out
        except Exception as e:
            logger.error(f"Rename file failed: {e}")
            return {"error": str(e)}

    def copy_file(self, source_filename: str, dest_filename: str) -> Dict[str, Any]:
        """Copy a single file inside the sandbox (files only, not directories)."""
        import shutil

        s, _ = self._auto_correct_path(source_filename)
        d, _ = self._auto_correct_path(dest_filename)
        if not self._is_safe_path(s) or not self._is_safe_path(d):
            return {"error": "Access denied: paths must stay inside the workspace"}
        src = os.path.normpath(os.path.join(self.sandbox_dir, s))
        dst = os.path.normpath(os.path.join(self.sandbox_dir, d))
        abs_sandbox = os.path.abspath(self.sandbox_dir)
        if not os.path.abspath(src).startswith(abs_sandbox) or not os.path.abspath(dst).startswith(abs_sandbox):
            return {"error": "Resolved path escapes the workspace"}
        if not os.path.isfile(src):
            return {"error": f"Source is not a file or does not exist: {source_filename}"}
        if os.path.isdir(src):
            return {"error": "copy_file only supports files; use other tools for directories"}
        try:
            parent = os.path.dirname(dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(src, dst)
            return {"success": True, "source": s, "dest": d}
        except OSError as e:
            return {"error": str(e)}

    def create_directory(self, dirname: str) -> Dict[str, Any]:
        """Create a directory tree inside the sandbox (exist_ok)."""
        dirname, _ = self._auto_correct_path(dirname)
        if not self._is_safe_path(dirname):
            return {"error": "Access denied: path must stay inside the workspace"}
        target = os.path.normpath(os.path.join(self.sandbox_dir, dirname))
        if not os.path.abspath(target).startswith(os.path.abspath(self.sandbox_dir)):
            return {"error": "Resolved path escapes the workspace"}
        try:
            os.makedirs(target, exist_ok=True)
            return {"success": True, "path": dirname}
        except OSError as e:
            return {"error": str(e)}
    
    def delete_file(self, filename: str) -> Dict[str, Any]:
        """Delete a single file under the sandbox."""
        filename, _ = self._auto_correct_path(filename)
        
        if not self._is_safe_path(filename):
            return {"error": "Access denied: Path outside project root"}
        
        target_path = os.path.join(self.sandbox_dir, filename)
        
        if not os.path.exists(target_path):
            return {"error": f"File not found: {filename}"}
        
        if os.path.isdir(target_path):
            return {"error": f"Cannot delete directory with delete_file. Use delete_directory instead."}
        
        try:
            size = os.path.getsize(target_path)
            
            os.remove(target_path)
            
            return {
                "success": True,
                "message": f"File '{filename}' deleted successfully",
                "deleted_file": filename,
                "size_bytes": size
            }
        except Exception as e:
            logger.error(f"Delete file failed: {e}")
            return {"error": str(e)}
    
    def delete_directory(self, dirname: str, recursive: bool = False) -> Dict[str, Any]:
        """Remove a directory under the sandbox; optional recursive delete."""
        import shutil
        
        dirname, _ = self._auto_correct_path(dirname)
        
        if not self._is_safe_path(dirname):
            return {"error": "Access denied: Path outside project root"}
        
        target_path = os.path.join(self.sandbox_dir, dirname)
        
        if not os.path.exists(target_path):
            return {"error": f"Directory not found: {dirname}"}
        
        if not os.path.isdir(target_path):
            return {"error": f"Not a directory: {dirname}"}
        
        try:
            items = os.listdir(target_path)
            
            if items and not recursive:
                return {
                    "error": f"Directory not empty (contains {len(items)} items). Use recursive=True to force delete.",
                    "items_count": len(items)
                }
            
            deleted_files = 0
            deleted_dirs = 0
            if recursive:
                for root, dirs, files in os.walk(target_path):
                    deleted_files += len(files)
                    deleted_dirs += len(dirs)
            
            if recursive:
                shutil.rmtree(target_path)
            else:
                os.rmdir(target_path)
            
            return {
                "success": True,
                "message": f"Directory '{dirname}' deleted successfully",
                "deleted_directory": dirname,
                "deleted_files": deleted_files if recursive else 0,
                "deleted_subdirs": deleted_dirs if recursive else 0,
                "recursive": recursive
            }
        except Exception as e:
            logger.error(f"Delete directory failed: {e}")
            return {"error": str(e)}
            
    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-compatible tool definitions for sandbox file IO."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": """Write or overwrite a file in your personal workspace (``workspace/sandbox``).

Path rules — use a **relative** path only (``subdir/filename``). Do **not** prefix with ``workspace/``, ``sandbox/``, or ``workspace/sandbox/``; the host normalizes those away but it is error-prone.

Good:
- ``diaries/diary_20260228.md``
- ``docs/notes.md``

Bad:
- ``workspace/sandbox/diaries/...`` (no leading workspace path)
- ``sandbox/diaries/...`` (no leading sandbox path)

Suggested folders:
- ``diaries/`` — dated journals
- ``docs/`` — longer notes / summaries
- ``code/`` — snippets or small scripts
- ``experiments/`` — experiment logs
- ``drafts/`` — scratch work""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Relative path such as 'diaries/diary_20260228.md' or 'docs/notes.md' (no workspace/sandbox/ prefix).",
                            },
                            "content": {
                                "type": "string",
                                "description": "Full UTF-8 text to write.",
                            }
                        },
                        "required": ["filename", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": """Read a UTF-8 text file from the sandbox.

Use the same relative style as ``write_file`` (``diaries/foo.md``), without ``workspace/sandbox`` prefixes.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Relative path such as 'diaries/diary_20260228.md' (no workspace/sandbox/ prefix).",
                            }
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files under the sandbox. Omit ``subdir`` to list the workspace root, or pass a folder like ``diaries``.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subdir": {
                                "type": "string",
                                "description": "Optional subdirectory relative to the sandbox (e.g. 'diaries', 'docs'). Empty = root.",
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search file contents for a keyword inside the sandbox; optional ``search_dir`` narrows the tree.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "Case-insensitive substring to search for.",
                            },
                            "search_dir": {
                                "type": "string",
                                "description": "Optional subdirectory (e.g. 'diaries'). Empty = entire sandbox.",
                            }
                        },
                        "required": ["keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_file",
                    "description": "Copy a single file to another relative path inside the sandbox (parents created as needed). Directories are not supported.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_filename": {"type": "string", "description": "Source relative path."},
                            "dest_filename": {"type": "string", "description": "Destination relative path."},
                        },
                        "required": ["source_filename", "dest_filename"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "Create a directory (multi-level, exist_ok) inside the sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dirname": {"type": "string", "description": "Relative path such as 'drafts/2026' or 'experiments/run1'."},
                        },
                        "required": ["dirname"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rename_file",
                    "description": """Rename or move a **single file** within the sandbox.

Same relative path rules as ``read_file`` / ``write_file`` — no ``workspace/sandbox`` prefix.

Examples:
- old_filename='drafts/a.md', new_filename='drafts/b.md'
- old_filename='notes/old.txt', new_filename='archive/old.txt' (missing parent dirs are created)

Directories are not supported here; use directory-oriented tools instead.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "old_filename": {
                                "type": "string",
                                "description": "Source relative path, e.g. 'drafts/old.md'.",
                            },
                            "new_filename": {
                                "type": "string",
                                "description": "Destination relative path, e.g. 'drafts/new.md' (same folder = pure rename).",
                            },
                        },
                        "required": ["old_filename", "new_filename"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a single file from your workspace. Use with caution - this operation cannot be undone.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Relative path to the file to delete (e.g., 'old_notes/draft.md')"
                            }
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_directory",
                    "description": "Delete a directory from your workspace. Can optionally delete recursively (all contents). Use with caution - this operation cannot be undone.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dirname": {
                                "type": "string",
                                "description": "Relative path to the directory to delete"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "If true, delete directory and all its contents. If false, only delete if empty.",
                                "default": False
                            }
                        },
                        "required": ["dirname"]
                    }
                }
            }
        ]

