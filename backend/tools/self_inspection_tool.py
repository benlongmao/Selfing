#!/usr/bin/env python3
"""
Self-inspection helpers — read and analyze this repository’s own source from a tight allowlist.

v1.0 [2026-01-16]: read/list/search under ``backend/``, ``config/``, etc.
v2.0 [2026-02-07]: AST summaries, pattern lint, and function drill-down for self-directed refactors.
"""
import os
import ast
import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Repository root (…/s-main)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALLOWED_DIRS = [
    "backend",
    "config",
    "scripts",
    "docs",
]

MAX_FILE_SIZE = 500000  # 500KB
MAX_LINES_DEFAULT = 1000
MAX_LINES_PER_CALL_CAP = 5000


def _coerce_positive_int(value: Any, default: int, *, minimum: int = 1, maximum: Optional[int] = None) -> int:
    try:
        if value is None:
            n = default
        else:
            n = int(value)
    except (TypeError, ValueError):
        n = default
    n = max(minimum, n)
    if maximum is not None:
        n = min(n, maximum)
    return n


class SelfInspectionTool:
    """Sandboxed filesystem + AST introspection for the running S codebase."""

    def __init__(self):
        self.base_dir = BASE_DIR
        self.allowed_dirs = ALLOWED_DIRS

    def _is_path_allowed(self, file_path: str) -> tuple[bool, str]:
        """
        Returns:
            ``(allowed, error_message)`` — ``error_message`` empty when allowed.
        """
        abs_path = os.path.abspath(os.path.join(self.base_dir, file_path))

        if not abs_path.startswith(self.base_dir):
            return False, "Path must stay inside the S project root"

        rel_path = os.path.relpath(abs_path, self.base_dir)
        first_dir = rel_path.split(os.sep)[0]

        if first_dir not in self.allowed_dirs:
            allowed_str = ", ".join(self.allowed_dirs)
            return False, f"Only these top-level dirs are readable: {allowed_str}"

        return True, ""

    def read_self_code(
        self,
        file_path: str,
        max_lines: int = MAX_LINES_DEFAULT,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read a UTF-8 text file under the allowlist, with optional 1-based line windowing."""
        try:
            is_allowed, error_msg = self._is_path_allowed(file_path)
            if not is_allowed:
                return {
                    "error": f"Access denied: {error_msg}",
                    "file_path": file_path,
                    "allowed_dirs": self.allowed_dirs
                }

            abs_path = os.path.abspath(os.path.join(self.base_dir, file_path))

            if not os.path.exists(abs_path):
                return {
                    "error": f"File not found: {file_path}",
                    "suggestion": "Call list_self_files to see what exists"
                }

            if not os.path.isfile(abs_path):
                return {
                    "error": f"Path is a directory, not a file: {file_path}",
                    "suggestion": "Call list_self_files to enumerate children"
                }

            file_size = os.path.getsize(abs_path)
            if file_size > MAX_FILE_SIZE:
                return {
                    "error": f"File too large: {file_size} bytes (limit {MAX_FILE_SIZE})",
                    "suggestion": "Call search_self_code to narrow in on a symbol"
                }

            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            total_lines = len(lines)
            sl = _coerce_positive_int(start_line, 1, minimum=1)
            ml = _coerce_positive_int(
                max_lines, MAX_LINES_DEFAULT, minimum=1, maximum=MAX_LINES_PER_CALL_CAP
            )

            if sl > total_lines:
                return {
                    "file_path": file_path,
                    "total_lines": total_lines,
                    "start_line": sl,
                    "end_line": total_lines,
                    "displayed_lines": 0,
                    "truncated": False,
                    "has_more_before": False,
                    "has_more_after": False,
                    "content": "",
                    "file_size": file_size,
                    "note": f"start_line ({sl}) is past EOF (total_lines={total_lines})",
                }

            if end_line is None:
                el = min(sl + ml - 1, total_lines)
            else:
                el = _coerce_positive_int(end_line, sl, minimum=sl, maximum=total_lines)
                if el - sl + 1 > MAX_LINES_PER_CALL_CAP:
                    el = min(sl + MAX_LINES_PER_CALL_CAP - 1, total_lines)

            chunk = lines[sl - 1 : el]
            content = "".join(chunk)
            displayed = len(chunk)
            has_more_before = sl > 1
            has_more_after = el < total_lines
            truncated = has_more_before or has_more_after

            hints: List[str] = []
            if has_more_before:
                hints.append(
                    f"... lines 1–{sl - 1} omitted ({total_lines} total); lower start_line to include them."
                )
            if has_more_after:
                hints.append(
                    f"... lines {el + 1}–{total_lines} omitted; next call use start_line={el + 1} with max_lines or end_line."
                )
            if hints:
                content = content.rstrip("\n") + "\n\n" + " ".join(hints)

            return {
                "file_path": file_path,
                "total_lines": total_lines,
                "start_line": sl,
                "end_line": el,
                "displayed_lines": displayed,
                "truncated": truncated,
                "has_more_before": has_more_before,
                "has_more_after": has_more_after,
                "content": content,
                "file_size": file_size,
            }

        except Exception as e:
            logger.error(f"read_self_code failed for {file_path}: {e}")
            return {
                "error": f"Error while reading file: {str(e)}",
                "file_path": file_path
            }

    def list_self_files(self, directory: str = "backend/") -> Dict[str, Any]:
        """List files and immediate subdirectories for one allowlisted folder."""
        try:
            is_allowed, error_msg = self._is_path_allowed(directory)
            if not is_allowed:
                return {
                    "error": f"Access denied: {error_msg}",
                    "directory": directory,
                    "allowed_dirs": self.allowed_dirs
                }

            abs_path = os.path.abspath(os.path.join(self.base_dir, directory))

            if not os.path.exists(abs_path):
                return {
                    "error": f"Directory not found: {directory}",
                    "allowed_dirs": self.allowed_dirs
                }

            if not os.path.isdir(abs_path):
                return {
                    "error": f"Path is not a directory: {directory}",
                    "suggestion": "Call read_self_code when pointing at a file"
                }

            files = []
            directories = []

            for item in sorted(os.listdir(abs_path)):
                item_path = os.path.join(abs_path, item)
                rel_path = os.path.join(directory, item)

                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    files.append({
                        "name": item,
                        "path": rel_path,
                        "size": size,
                        "size_kb": round(size / 1024, 2)
                    })
                elif os.path.isdir(item_path):
                    try:
                        file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                    except Exception:
                        file_count = 0

                    directories.append({
                        "name": item,
                        "path": rel_path,
                        "file_count": file_count
                    })

            return {
                "directory": directory,
                "total_files": len(files),
                "total_directories": len(directories),
                "files": files,
                "directories": directories
            }

        except Exception as e:
            logger.error(f"list_self_files failed for {directory}: {e}")
            return {
                "error": f"Error while listing directory: {str(e)}",
                "directory": directory
            }

    def search_self_code(self, keyword: str, directory: str = "backend/", max_results: int = 50) -> Dict[str, Any]:
        """Case-insensitive substring search across ``.py`` / ``.yaml`` under ``directory``."""
        try:
            is_allowed, error_msg = self._is_path_allowed(directory)
            if not is_allowed:
                return {
                    "error": f"Access denied: {error_msg}",
                    "directory": directory,
                    "allowed_dirs": self.allowed_dirs
                }

            abs_path = os.path.abspath(os.path.join(self.base_dir, directory))

            if not os.path.exists(abs_path):
                return {
                    "error": f"Directory not found: {directory}"
                }

            results = []
            total_matches = 0

            for root, dirs, files in os.walk(abs_path):
                dirs[:] = [d for d in dirs if not d.startswith('__') and d != '.git']

                for file in files:
                    if not (file.endswith('.py') or file.endswith('.yaml') or file.endswith('.yml')):
                        continue

                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.base_dir)

                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            lines = f.readlines()

                        for line_num, line in enumerate(lines, 1):
                            if keyword.lower() in line.lower():
                                total_matches += 1

                                if len(results) < max_results:
                                    context_before = lines[line_num-2] if line_num > 1 else ""
                                    context_after = lines[line_num] if line_num < len(lines) else ""

                                    results.append({
                                        "file": rel_path,
                                        "line": line_num,
                                        "content": line.strip(),
                                        "context_before": context_before.strip(),
                                        "context_after": context_after.strip()
                                    })

                    except Exception:
                        continue

            return {
                "keyword": keyword,
                "directory": directory,
                "total_matches": total_matches,
                "displayed_results": len(results),
                "truncated": total_matches > max_results,
                "results": results
            }

        except Exception as e:
            logger.error(f"search_self_code failed for {keyword}: {e}")
            return {
                "error": f"Error while searching: {str(e)}",
                "keyword": keyword
            }

    def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
        """Parse a Python module and return classes, top-level functions, and imports."""
        try:
            is_allowed, error_msg = self._is_path_allowed(file_path)
            if not is_allowed:
                return {"error": f"Access denied: {error_msg}"}

            abs_path = os.path.abspath(os.path.join(self.base_dir, file_path))

            if not os.path.exists(abs_path):
                return {"error": f"File not found: {file_path}"}

            if not file_path.endswith('.py'):
                return {"error": "Only .py files can be analyzed"}

            with open(abs_path, 'r', encoding='utf-8') as f:
                source = f.read()

            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                return {
                    "error": f"Syntax error: {e}",
                    "file_path": file_path
                }

            classes = []
            functions = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    class_vars = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                "name": item.name,
                                "line": item.lineno,
                                "args": [arg.arg for arg in item.args.args],
                                "docstring": ast.get_docstring(item) or ""
                            })
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    class_vars.append(target.id)

                    base_classes = []
                    for base in node.bases:
                        try:
                            if isinstance(base, ast.Name):
                                base_classes.append(base.id)
                            elif isinstance(base, ast.Attribute):
                                base_classes.append(f"{base.value.id if isinstance(base.value, ast.Name) else '?'}.{base.attr}")
                            else:
                                base_classes.append("?")
                        except Exception:
                            base_classes.append("?")

                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node) or "",
                        "methods": methods,
                        "class_variables": class_vars,
                        "base_classes": base_classes
                    })

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    try:
                        functions.append({
                            "name": node.name,
                            "line": node.lineno,
                            "args": [arg.arg for arg in node.args.args],
                            "docstring": ast.get_docstring(node) or ""
                        })
                    except Exception:
                        pass

                elif isinstance(node, ast.Import):
                    try:
                        for alias in node.names:
                            imports.append({
                                "module": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno
                            })
                    except Exception:
                        pass

                elif isinstance(node, ast.ImportFrom):
                    try:
                        for alias in node.names:
                            imports.append({
                                "module": f"{node.module}.{alias.name}" if node.module else alias.name,
                                "alias": alias.asname,
                                "line": node.lineno,
                                "from": node.module
                            })
                    except Exception:
                        pass

            total_lines = len(source.split('\n'))
            code_lines = len([l for l in source.split('\n') if l.strip() and not l.strip().startswith('#')])

            return {
                "success": True,
                "file_path": file_path,
                "statistics": {
                    "total_lines": total_lines,
                    "code_lines": code_lines,
                    "classes_count": len(classes),
                    "functions_count": len(functions),
                    "imports_count": len(imports)
                },
                "classes": classes,
                "functions": functions,
                "imports": imports[:20]
            }

        except Exception as e:
            logger.error(f"analyze_code_structure failed for {file_path}: {e}")
            return {"error": f"Structure analysis failed: {str(e)}"}

    def find_code_patterns(self, file_path: str) -> Dict[str, Any]:
        """Lightweight static hints: long functions, TODOs, magic numbers, broad excepts, long lines."""
        try:
            is_allowed, error_msg = self._is_path_allowed(file_path)
            if not is_allowed:
                return {"error": f"Access denied: {error_msg}"}

            abs_path = os.path.abspath(os.path.join(self.base_dir, file_path))

            if not os.path.exists(abs_path):
                return {"error": f"File not found: {file_path}"}

            with open(abs_path, 'r', encoding='utf-8') as f:
                source = f.read()
                lines = source.split('\n')

            patterns_found = []

            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        if func_lines > 50:
                            patterns_found.append({
                                "type": "long_function",
                                "severity": "info",
                                "line": node.lineno,
                                "message": f"Function '{node.name}' spans {func_lines} lines — consider splitting",
                                "suggestion": "Break large functions into smaller units for readability"
                            })
            except Exception:
                pass

            for i, line in enumerate(lines, 1):
                if 'TODO' in line or 'FIXME' in line or 'XXX' in line:
                    patterns_found.append({
                        "type": "todo_comment",
                        "severity": "info",
                        "line": i,
                        "message": line.strip(),
                        "suggestion": "Track or resolve this TODO when you touch the area"
                    })

            magic_number_pattern = re.compile(r'(?<![a-zA-Z_])(\d{3,})(?![a-zA-Z_\d])')
            for i, line in enumerate(lines, 1):
                if not line.strip().startswith('#'):
                    matches = magic_number_pattern.findall(line)
                    for match in matches:
                        if int(match) > 100 and 'sleep' not in line.lower():
                            patterns_found.append({
                                "type": "magic_number",
                                "severity": "suggestion",
                                "line": i,
                                "message": f"Literal number {match} embedded in code",
                                "suggestion": "Promote repeated literals to named constants"
                            })

            except_count = source.count('except Exception')
            if except_count > 5:
                patterns_found.append({
                    "type": "broad_exception",
                    "severity": "suggestion",
                    "line": 0,
                    "message": f"{except_count} `except Exception` handlers — review whether narrower types suffice",
                    "suggestion": "Prefer targeted exceptions where failure modes are known"
                })

            for i, line in enumerate(lines, 1):
                if len(line) > 120:
                    patterns_found.append({
                        "type": "long_line",
                        "severity": "style",
                        "line": i,
                        "message": f"Line length {len(line)} chars (> 120)",
                        "suggestion": "Wrap or refactor to keep within style limits"
                    })

            return {
                "success": True,
                "file_path": file_path,
                "patterns_found": len(patterns_found),
                "patterns": patterns_found[:30]
            }

        except Exception as e:
            logger.error(f"find_code_patterns failed for {file_path}: {e}")
            return {"error": f"Pattern scan failed: {str(e)}"}

    def get_function_detail(self, file_path: str, function_name: str) -> Dict[str, Any]:
        """Return source slice + args + return annotation + simple call graph for one function."""
        try:
            is_allowed, error_msg = self._is_path_allowed(file_path)
            if not is_allowed:
                return {"error": f"Access denied: {error_msg}"}

            abs_path = os.path.abspath(os.path.join(self.base_dir, file_path))

            if not os.path.exists(abs_path):
                return {"error": f"File not found: {file_path}"}

            with open(abs_path, 'r', encoding='utf-8') as f:
                source = f.read()
                lines = source.split('\n')

            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                return {"error": f"Syntax error: {e}"}

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20

                    func_code = '\n'.join(lines[start_line-1:end_line])

                    args_info = []
                    for arg in node.args.args:
                        arg_info = {"name": arg.arg}
                        if arg.annotation:
                            arg_info["type"] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                        args_info.append(arg_info)

                    return_type = None
                    if node.returns:
                        return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

                    calls = []
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                calls.append(child.func.id)
                            elif isinstance(child.func, ast.Attribute):
                                calls.append(f"*.{child.func.attr}")

                    return {
                        "success": True,
                        "file_path": file_path,
                        "function_name": function_name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "docstring": ast.get_docstring(node) or "",
                        "arguments": args_info,
                        "return_type": return_type,
                        "calls": list(set(calls))[:20],
                        "code": func_code
                    }

            return {
                "success": False,
                "error": f"Function not found: {function_name}",
                "file_path": file_path
            }

        except Exception as e:
            logger.error(f"get_function_detail failed for {file_path}/{function_name}: {e}")
            return {"error": f"Function inspection failed: {str(e)}"}

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """OpenAI-style tool specs consumed by ``tool_router``."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_self_code",
                    "description": (
                        "Read a source file from this repo (allowed roots: backend/, config/, scripts/, docs/). "
                        "Supports 1-based line paging via start_line / end_line / max_lines. "
                        "Example: read_self_code('backend/self_model.py', start_line=1001, max_lines=1000)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path relative to repo root, e.g. 'backend/self_model.py'"
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "First line to include (1-based). Default 1",
                                "default": 1
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Last line to include (1-based). If omitted, reads up to max_lines lines from start_line"
                            },
                            "max_lines": {
                                "type": "integer",
                                "description": "When end_line is omitted, read at most this many lines from start_line (default 1000)",
                                "default": 1000
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_self_files",
                    "description": (
                        "List files and subdirectories under an allowed folder. "
                        "Example: list_self_files('backend/tools/')."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory path such as 'backend/' or 'config/'",
                                "default": "backend/"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_self_code",
                    "description": (
                        "Search for a keyword across .py/.yaml under an allowed directory. "
                        "Example: search_self_code('z_self', 'backend/')."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "Substring to match (case-insensitive)"
                            },
                            "directory": {
                                "type": "string",
                                "description": "Directory to walk (default 'backend/')",
                                "default": "backend/"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Cap on returned hits (default 50)",
                                "default": 50
                            }
                        },
                        "required": ["keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_code_structure",
                    "description": (
                        "Parse a Python module: classes, methods, imports, and coarse stats. "
                        "Useful before proposing refactors."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Python path relative to root, e.g. 'backend/chat_service.py'"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_code_patterns",
                    "description": (
                        "Heuristic lint pass: long functions, TODO/FIXME markers, magic literals, "
                        "frequent broad exceptions, and >120 char lines."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Python file path relative to root"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_function_detail",
                    "description": (
                        "Inspect a single function: signature, return annotation, callees, and full source slice."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Python file path relative to root"
                            },
                            "function_name": {
                                "type": "string",
                                "description": "Function name as defined in that module"
                            }
                        },
                        "required": ["file_path", "function_name"]
                    }
                }
            }
        ]


_self_inspection_tool = None


def get_self_inspection_tool() -> SelfInspectionTool:
    """Singleton accessor for ``SelfInspectionTool``."""
    global _self_inspection_tool
    if _self_inspection_tool is None:
        _self_inspection_tool = SelfInspectionTool()
    return _self_inspection_tool
