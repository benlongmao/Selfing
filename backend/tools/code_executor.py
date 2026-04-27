#!/usr/bin/env python3
"""
Code executor — run Python inside a locked-down workspace sandbox.

Safety goals:
1. Time and output caps
2. Files confined to ``workspace/sandbox``
3. Block obvious network / host-escape patterns
4. Subprocess timeout control
"""
import os
import sys
import subprocess
import tempfile
import json
import logging
import signal
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Real-module alias (``_os`` only exists inside the injected SAFE_PRELUDE string).
_os = os

# Security configuration
MAX_EXECUTION_TIME = 30
MAX_OUTPUT_SIZE = 100000
MAX_MEMORY_MB = 256

# Modules that are prohibited from importing
FORBIDDEN_IMPORTS = [
    "os.system", "subprocess", "shutil.rmtree",
    "socket", "requests", "urllib", "http.client",
    "ftplib", "smtplib", "telnetlib",
    "__import__", "exec", "eval", "compile",
    "open",  # replaced in-prelude with sandbox-safe open
]

# Injected into user code before execution (simplified, keeps sandbox open)
SAFE_PRELUDE = '''
# Pre-import common security modules
import math
import random
import datetime
import json
import re
import collections
import itertools
import functools
import statistics
import decimal
import time
import sys
from typing import List, Dict, Any, Optional, Tuple

# Try numpy (optional)
try:
    import numpy as np
except ImportError:
    np = None

# Pygame (optional)
try:
    import pygame
    pygame.init()
except ImportError:
    pygame = None

# Curses (optional)
try:
    import curses
except ImportError:
    curses = None

import builtins as _builtins
import os as _os
_original_open = _builtins.open

def open(file, mode='r', *args, **kwargs):
    """
    Sandbox-scoped ``open`` replacement.

    [2026-02-28] Option A: reads/writes must stay under ``WORKSPACE_DIR`` (workspace/sandbox).
    """
    abs_path = _os.path.abspath(str(file))
    
    sandbox_root = _os.environ.get('WORKSPACE_DIR', _os.getcwd())
    
    if abs_path.startswith(sandbox_root):
        return _original_open(file, mode, *args, **kwargs)
    raise PermissionError(f"Access denied: {file} is outside the workspace sandbox")
'''

# Executor helpers (not injected; real module code)
_subprocess = __import__('subprocess')
_PYTHON_PATH = _os.environ.get('PYTHON_PATH', sys.executable)
_WORKSPACE_ROOT = _os.environ.get('WORKSPACE_DIR', _os.getcwd())
_TOOLS_DIR = _os.path.join(_WORKSPACE_ROOT, 'tools')

def run_script(script_name, args=None, timeout=30):
    """
    Run a helper ``.py`` that lives inside the workspace sandbox.

    Examples:
        run_script('list_python.py')
        run_script('analyze_python.py', ['../code/my_script.py'])

    Args:
        script_name: Bare script name or relative path under the workspace.
        args: CLI args forwarded to the script.
        timeout: Seconds before the subprocess is killed.

    Returns:
        ``{'success': bool, 'stdout': str, 'stderr': str, 'exit_code': int}``
    """
    args = args or []
    
    # Parse script path
    if _os.path.isabs(script_name):
        script_path = script_name
    elif '/' in script_name:
        script_path = _os.path.join(_WORKSPACE_ROOT, script_name)
    else:
        script_path = _os.path.join(_TOOLS_DIR, script_name)
    
    script_path = _os.path.abspath(script_path)
    
    # Security Check: Must be within work space
    if not script_path.startswith(_WORKSPACE_ROOT):
        return {'success': False, 'error': f'Script must stay inside workspace: {_WORKSPACE_ROOT}'}
    
    if not _os.path.exists(script_path):
        return {'success': False, 'error': f'Script not found: {script_name}'}
    
    if not script_path.endswith('.py'):
        return {'success': False, 'error': 'Only .py scripts are allowed'}
    
    try:
        result = _subprocess.run(
            [_PYTHON_PATH, script_path] + [str(a) for a in args],
            capture_output=True,
            timeout=timeout,
            cwd=_os.path.dirname(script_path)
        )
        
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        
        # Printout (for easy viewing)
        if stdout.strip():
            print(stdout)
        if stderr.strip():
            print(f"[stderr] {stderr}", file=sys.stderr)
        
        return {
            'success': result.returncode == 0,
            'stdout': stdout,
            'stderr': stderr,
            'exit_code': result.returncode
        }
    except _subprocess.TimeoutExpired:
        return {'success': False, 'error': f'Execution timed out after {timeout}s'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def list_scripts():
    """Print helper scripts available under the sandbox tools directory."""
    scripts = []
    if _os.path.exists(_TOOLS_DIR):
        for f in _os.listdir(_TOOLS_DIR):
            if f.endswith('.py'):
                scripts.append(f)
    print(f"Available scripts ({_TOOLS_DIR}):")
    for s in sorted(scripts):
        print(f"  - {s}")
    return scripts


class CodeExecutor:
    """Sandboxed Python runner used by the agent."""
    
    def __init__(self, sandbox_base: str = "workspace/sandbox"):
        """
        Configure the workspace sandbox root.

        [2026-02-28] Option A: all execution IO stays under ``workspace/sandbox``.
        """
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.sandbox_base = os.path.abspath(os.path.join(self.project_root, "workspace/sandbox"))
        os.makedirs(self.sandbox_base, exist_ok=True)
        
        self.execution_history: List[Dict] = []
        
        logger.info(f"[CodeExecutor] Agent code sandbox: {self.sandbox_base}")
    
    def execute_python(
        self,
        code: str,
        timeout: int = MAX_EXECUTION_TIME,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Execute arbitrary Python inside the per-session sandbox.

        Args:
            code: Python source to run.
            timeout: Wall-clock limit in seconds.
            session_id: Subdirectory name under the sandbox for isolation.

        Returns:
            Dict with ``success``, ``stdout``, ``stderr``, ``return_value``,
            ``execution_time``, and optional ``error``.
        """
        execution_id = f"exec-{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        # 1. Security Check
        security_check = self._security_check(code)
        if not security_check["safe"]:
            return {
                "success": False,
                "error": f"Security check failed: {security_check['reason']}",
                "blocked_patterns": security_check.get("blocked_patterns", [])
            }
        
        # 2. Create a session sandbox directory
        session_sandbox = os.path.join(self.sandbox_base, session_id)
        os.makedirs(session_sandbox, exist_ok=True)
        
        # 3. Create a temporary executable file
        script_path = os.path.join(session_sandbox, f"{execution_id}.py")
        result_path = os.path.join(session_sandbox, f"{execution_id}_result.json")
        
        # 4. Wrapping code (adding safe preprocessing and result capture)
        wrapped_code = self._wrap_code(code, result_path)
        
        try:
            # write script
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(wrapped_code)
            
            # [2026-02-28] Option A: cwd and WORKSPACE_DIR both point to sandbox
            env = os.environ.copy()
            env['SANDBOX_DIR'] = session_sandbox
            env['WORKSPACE_DIR'] = self.sandbox_base
            env['PYTHON_PATH'] = sys.executable
            env['PYTHONPATH'] = ''
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.sandbox_base,
                env=env,
                preexec_fn=self._limit_resources if os.name != 'nt' else None
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                stdout = stdout.decode('utf-8', errors='replace')[:MAX_OUTPUT_SIZE]
                stderr = stderr.decode('utf-8', errors='replace')[:MAX_OUTPUT_SIZE]
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # 6. Read the results
                return_value = None
                if os.path.exists(result_path):
                    try:
                        with open(result_path, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                            return_value = result_data.get("return_value")
                    except Exception:
                        pass
                
                # [Validation improvements] Build detailed execution results
                result = {
                    "success": process.returncode == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_value": return_value,
                    "execution_time": execution_time,
                    "execution_id": execution_id
                }
                
                if process.returncode != 0:
                    result["error"] = f"Exit code: {process.returncode}"
                
                # [Verification improvements] Add output summary and verification reminder
                result["output_summary"] = {
                    "stdout_lines": stdout.count('\n') + 1 if stdout.strip() else 0,
                    "stderr_lines": stderr.count('\n') + 1 if stderr.strip() else 0,
                    "has_output": bool(stdout.strip()),
                    "has_errors": bool(stderr.strip())
                }
                
                # [Verification Improvement] Force reminder S to view the actual output
                if result["success"]:
                    result["verification_reminder"] = (
                        "⚠️ Execution finished. The stdout above is the real process output. "
                        "Verify it matches what you intend before summarizing for the user; "
                        "if stdout is empty or surprising, say so honestly."
                    )
                else:
                    result["verification_reminder"] = (
                        "⚠️ Execution failed. Read stderr, explain the root cause to the user, "
                        "and do not imply success."
                    )
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                result = {
                    "success": False,
                    "error": f"Execution timed out after {timeout}s",
                    "stdout": "",
                    "stderr": "",
                    "execution_time": timeout,
                    "verification_reminder": (
                        "⚠️ Timed out—likely an infinite loop or very heavy compute. "
                        "Tell the user the code did not finish."
                    )
                }
            
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            result = {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }
        
        finally:
            # 7. Clean up temporary files
            try:
                if os.path.exists(script_path):
                    os.remove(script_path)
                if os.path.exists(result_path):
                    os.remove(result_path)
            except Exception:
                pass
        
        # Record execution history
        self.execution_history.append({
            "execution_id": execution_id,
            "session_id": session_id,
            "code_preview": code[:200] + "..." if len(code) > 200 else code,
            "success": result.get("success", False),
            "timestamp": start_time.isoformat()
        })
        
        # Limit history length
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return result
    
    def _security_check(self, code: str) -> Dict[str, Any]:
        """Static scan for obviously unsafe constructs."""
        blocked = []
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ("os.system", "host shell via os.system"),
            ("subprocess", "subprocess / host command"),
            ("shutil.rmtree", "recursive delete"),
            ("socket", "raw network socket"),
            ("requests.", "HTTP client"),
            ("urllib", "URL client"),
            ("exec(", "dynamic exec"),
            ("eval(", "dynamic eval"),
            ("compile(", "dynamic compile"),
            ("open('/", "absolute root open"),
            ("open(\"/", "absolute root open"),
        ]
        
        code_lower = code.lower()
        for pattern, reason in dangerous_patterns:
            if pattern.lower() in code_lower:
                # Some modes only warn but do not block
                if pattern in ["import os"]:
                    continue
                blocked.append(f"{pattern}: {reason}")
        
        if blocked:
            return {
                "safe": False,
                "reason": "Dangerous code pattern detected",
                "blocked_patterns": blocked
            }
        
        return {"safe": True}
    
    def _wrap_code(self, code: str, result_path: str) -> str:
        """Inject prelude + JSON result capture around user code."""
        return f'''
{SAFE_PRELUDE}

import sys
import traceback

_result = {{"return_value": None, "error": None}}

try:
{self._indent_code(code, 4)}
except Exception as e:
    _result["error"] = traceback.format_exc()
    print(f"Error: {{e}}", file=sys.stderr)

try:
    with _original_open("{result_path}", "w") as f:
        json.dump(_result, f)
except Exception:
    pass
'''
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent every line of user code before embedding."""
        indent = " " * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line for line in lines)
    
    def _limit_resources(self):
        """Apply RLIMIT caps before exec (Unix only)."""
        try:
            import resource
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (MAX_EXECUTION_TIME, MAX_EXECUTION_TIME + 5))
            # Memory: generous cap so numpy et al. can import
            mem_bytes = MAX_MEMORY_MB * 1024 * 1024 * 4
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except ValueError:
                pass
        except Exception as e:
            logger.warning(f"Failed to set resource limits: {e}")
    
    def list_sandbox_files(self, session_id: str = "default") -> Dict[str, Any]:
        """Enumerate files under a session sandbox folder."""
        session_sandbox = os.path.join(self.sandbox_base, session_id)
        if not os.path.exists(session_sandbox):
            return {"files": [], "message": "Sandbox directory does not exist yet"}
        
        files = []
        for root, dirs, filenames in os.walk(session_sandbox):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, session_sandbox)
                try:
                    size = os.path.getsize(full_path)
                    files.append({
                        "path": rel_path,
                        "size": size,
                        "modified": datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
                    })
                except Exception:
                    continue
        
        return {"files": files, "count": len(files)}
    
    def read_sandbox_file(self, session_id: str, filename: str) -> Dict[str, Any]:
        """
        Read a file produced under the session sandbox.

        [2026-02-28] Option A: paths must remain inside ``workspace/sandbox``.
        """
        session_sandbox = os.path.join(self.sandbox_base, session_id)
        file_path = os.path.join(session_sandbox, filename)
        
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(self.sandbox_base):
            return {"error": "Access denied: path escapes the workspace sandbox"}
        
        if not os.path.exists(file_path):
            return {"error": f"File not found: {filename}"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"success": True, "content": content[:MAX_OUTPUT_SIZE]}
        except Exception as e:
            return {"error": str(e)}
    
    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": (
                        "Execute Python in the locked workspace sandbox (math, json, re, statistics, etc.). "
                        "Networking via urllib/requests/socket is blocked. "
                        "For remote assets (e.g. arXiv PDFs) use fetch_url_to_workspace, then read_pdf—do not download via this tool."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python source to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default 30)",
                                "default": 30
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "code_sandbox_list",
                    "description": "List files stored under the code sandbox for this session",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "code_sandbox_read",
                    "description": (
                        "Read an artifact written by execute_python inside the sandbox. "
                        "For normal workspace files, use read_file instead."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Relative filename produced by execute_python"
                            }
                        },
                        "required": ["filename"]
                    }
                }
            }
        ]
    
    def route_tool_call(self, tool_name: str, args: Dict, session_id: str) -> Dict:
        """Dispatch execute_python / sandbox helper tools."""
        if tool_name == "execute_python":
            return self.execute_python(
                code=args.get("code", ""),
                timeout=args.get("timeout", MAX_EXECUTION_TIME),
                session_id=session_id
            )
        elif tool_name == "code_sandbox_list":
            return self.list_sandbox_files(session_id)
        elif tool_name == "code_sandbox_read":
            return self.read_sandbox_file(session_id, args.get("filename", ""))
        else:
            return {"error": f"Unknown tool: {tool_name}"}

