#!/usr/bin/env python3
"""
OpenCode wrapper so the agent can invoke the `opencode` CLI (e.g. under WSL).
"""
import os
import subprocess
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

OPENCODE_BIN = os.path.expanduser("~/.opencode/bin/opencode")

class OpenCodeTool:
    """Thin wrapper around the OpenCode binary."""

    def __init__(self, workspace_dir: str = None):
        """
        Args:
            workspace_dir: Working directory (default: repo ``workspace/sandbox``).
        """
        self.opencode_bin = OPENCODE_BIN

        if not os.path.exists(self.opencode_bin):
            logger.warning(f"OpenCode not found at {self.opencode_bin}")
            self.enabled = False
            return

        if workspace_dir:
            self.workspace_dir = os.path.abspath(workspace_dir)
        else:
            s_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.workspace_dir = os.path.join(s_project_root, "workspace", "sandbox")

        os.makedirs(self.workspace_dir, exist_ok=True)
        self.enabled = True

        logger.info(f"OpenCodeTool initialized: {self.opencode_bin}, workspace: {self.workspace_dir}")

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Run code through ``opencode run`` with the snippet embedded in the prompt.

        Returns:
            success, output, error (if any), execution_time
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "OpenCode binary not found or not configured",
                "output": ""
            }

        start_time = datetime.now()

        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=f'.{language}',
                dir=self.workspace_dir,
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name

            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    code_content = f.read()

                cmd = [
                    self.opencode_bin,
                    "run",
                    f"Run (execute) this {language} code:\n```{language}\n{code_content}\n```"
                ]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.workspace_dir,
                    env=os.environ.copy()
                )

                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    stdout = stdout.decode('utf-8', errors='replace')
                    stderr = stderr.decode('utf-8', errors='replace')

                    execution_time = (datetime.now() - start_time).total_seconds()

                    success = process.returncode == 0

                    result = {
                        "success": success,
                        "output": stdout,
                        "execution_time": execution_time
                    }

                    if stderr:
                        result["stderr"] = stderr

                    if not success:
                        result["error"] = f"Execution failed (exit {process.returncode})"
                        if stderr:
                            result["error"] += f"\n{stderr}"

                    return result

                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    return {
                        "success": False,
                        "error": f"Timeout after {timeout}s",
                        "output": ""
                    }

            finally:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")

        except Exception as e:
            logger.error(f"OpenCode execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }

    def run_with_message(
        self,
        message: str,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        ``opencode run`` with a free-form task message.
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "OpenCode binary not found or not configured",
                "output": ""
            }

        start_time = datetime.now()

        try:
            cmd = [
                self.opencode_bin,
                "run",
                message
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.workspace_dir,
                env=os.environ.copy()
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                stdout = stdout.decode('utf-8', errors='replace')
                stderr = stderr.decode('utf-8', errors='replace')

                execution_time = (datetime.now() - start_time).total_seconds()

                success = process.returncode == 0

                result = {
                    "success": success,
                    "output": stdout,
                    "execution_time": execution_time
                }

                if stderr:
                    result["stderr"] = stderr

                if not success:
                    result["error"] = f"Execution failed (exit {process.returncode})"
                    if stderr:
                        result["error"] += f"\n{stderr}"

                return result

            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return {
                    "success": False,
                    "error": f"Timeout after {timeout}s",
                    "output": ""
                }

        except Exception as e:
            logger.error(f"OpenCode run failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }

    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tools."""
        if not self.enabled:
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": "opencode_execute",
                    "description": """Execute code through OpenCode (stronger than a bare REPL for multi-step runs).

Use when the user wants opencode, or complex execution with better tooling.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Source to run"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["python", "javascript", "bash"],
                                "description": "Language (default python)",
                                "default": "python"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout seconds (default 60)",
                                "default": 60
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "opencode_run",
                    "description": """Run a natural-language task with OpenCode’s planner.

For analyses or fixes where the user explicitly wants opencode to drive the work.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Task or instruction"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout seconds (default 120)",
                                "default": 120
                            }
                        },
                        "required": ["message"]
                    }
                }
            }
        ]

    def route_tool_call(self, tool_name: str, args: Dict, session_id: str) -> Dict:
        """Dispatch to execute_code or run_with_message."""
        if tool_name == "opencode_execute":
            return self.execute_code(
                code=args.get("code", ""),
                language=args.get("language", "python"),
                timeout=args.get("timeout", 60)
            )
        if tool_name == "opencode_run":
            return self.run_with_message(
                message=args.get("message", ""),
                timeout=args.get("timeout", 120)
            )
        return {"error": f"Unknown tool: {tool_name}"}
