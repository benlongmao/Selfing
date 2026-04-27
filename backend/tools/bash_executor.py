#!/usr/bin/env python3
"""
Bash命令执行工具 (Bash Executor)
让Agent能够安全地执行bash命令

安全特性：
1. 白名单机制：只允许安全的命令
2. 禁止危险命令：rm -rf, dd, mkfs等
3. 工作目录限制：只能在 workspace/sandbox 内操作
4. 超时控制：防止无限执行
5. 输出限制：防止输出过大
6. 资源限制：限制CPU和内存使用
"""
import os
import subprocess
import logging
import shlex
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Safety limits
MAX_EXECUTION_TIME = 30  # seconds
MAX_OUTPUT_SIZE = 50000  # chars cap on captured stdout/stderr
# [2026-02-28] Resolve sandbox path at import (no hard-coded absolute)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../workspace/sandbox"))

# [2026-02-22] Tiered allowlists
# Tier A: read-only / introspection
SAFE_READONLY_COMMANDS = {
    "ls", "cat", "head", "tail", "wc", "file",  # inspect files
    "find", "grep", "egrep", "fgrep",           # search
    "pwd", "whoami", "hostname", "uname", "date", "uptime",  # host facts
    "du", "df",                                 # disk usage
    "which", "whereis", "type",                 # resolve binaries
    "echo", "printf",                           # echo
}

# Tier B: mutating but still sandbox-scoped (validated elsewhere)
SAFE_RESTRICTED_COMMANDS = {
    "cp", "mv", "mkdir", "touch",               # filesystem (sandbox only)
    "sed", "awk", "cut", "sort", "uniq", "tr", "diff",  # text utils
    "tar", "gzip", "gunzip", "zip", "unzip",    # archives
    "curl",                                      # network (GET-only policy in validator)
    "git",                                       # git (no force-push)
    "sqlite3",                                   # sqlite shell (read-only queries enforced)
}

# High risk: disabled for generic bash path
HIGH_RISK_COMMANDS = {
    "python", "python3",  # arbitrary code → use code_executor
    "pip", "pip3",        # package install
    "wget",               # download
}

# Union allowlist (excludes HIGH_RISK)
ALLOWED_COMMANDS = SAFE_READONLY_COMMANDS | SAFE_RESTRICTED_COMMANDS

# Hard denylist
FORBIDDEN_COMMANDS = {
    # destructive
    "rm", "rmdir", "shred",
    # disk
    "dd", "mkfs", "fdisk", "parted",
    # power
    "shutdown", "reboot", "halt", "poweroff", "init",
    # privilege
    "chmod", "chown", "chgrp", "su", "sudo", "passwd",
    # remote shells / transfers
    "nc", "netcat", "telnet", "ssh", "scp", "rsync",
    # process kill
    "kill", "killall", "pkill",
    # package managers
    "apt", "apt-get", "yum", "dnf", "pacman",
    # subshells
    "bash", "sh", "zsh", "fish",  # block nested shells
}

# Dangerous argv regex heuristics
DANGEROUS_PATTERNS = [
    r"rm\s+-rf",  # recursive rm
    r">\s*/dev/",  # write to device nodes
    r"\|\s*bash",  # pipe into bash
    r";\s*rm",  # chained rm
    r"&&\s*rm",  # && rm
    r"`.*`",  # backtick substitution
    r"\$\(.*\)",  # $(...) substitution
]


EVOLUTION_MAX_EXECUTION_TIME = 300  # evolution mode: allow longer test runs

# Evolution: allowed short flags for python -m compileall
_COMPILEALL_SINGLE_FLAGS = frozenset({"-q", "-f", "-l", "-b"})
_COMPILEALL_PATH_TOKEN_RE = re.compile(r"^[\w./+-]+$")


def _compileall_path_token_ok(tok: str) -> bool:
    """compileall 的路径实参：相对路径、无穿越、字符集受限。"""
    if not tok or tok.startswith("-"):
        return False
    if ".." in tok:
        return False
    if not _COMPILEALL_PATH_TOKEN_RE.fullmatch(tok):
        return False
    return True


def _evolution_python_compileall_command_ok(command: str) -> bool:
    """
    演进模式下允许 python3 -m compileall：仅编译为字节码，不执行模块主体。
    使用 shlex 解析，禁止任意 -c / 未知开关 / 危险路径。
    """
    try:
        parts = shlex.split(command.strip())
    except ValueError:
        return False
    if len(parts) < 3:
        return False
    if os.path.basename(parts[0]) not in ("python", "python3"):
        return False
    if parts[1] != "-m" or parts[2] != "compileall":
        return False
    i = 3
    while i < len(parts):
        p = parts[i]
        if p in _COMPILEALL_SINGLE_FLAGS:
            i += 1
            continue
        if p == "-j":
            if i + 1 >= len(parts):
                return False
            jv = parts[i + 1]
            if not re.fullmatch(r"[0-9]{1,2}", jv) or int(jv) > 16:
                return False
            i += 2
            continue
        if p == "-x":
            if i + 1 >= len(parts):
                return False
            pat = parts[i + 1]
            if len(pat) > 256 or "\n" in pat or any(c in pat for c in "|&;`$()"):
                return False
            i += 2
            continue
        if p == "-o":
            if i + 1 >= len(parts):
                return False
            if not _compileall_path_token_ok(parts[i + 1]):
                return False
            i += 2
            continue
        if p.startswith("-"):
            return False
        if not _compileall_path_token_ok(p):
            return False
        i += 1
    return True


class BashExecutor:
    """安全的Bash命令执行器"""
    
    def __init__(self, project_root: str = PROJECT_ROOT, evolution_mode: bool = False):
        """
        初始化Bash执行器
        
        [2026-02-07] 默认：命令在 workspace/sandbox 内
        [2026-04-03] evolution_mode=True：根目录为 S 仓库根，额外允许 pytest / python -m pytest /
        python -m compileall（受限参数）/ npm test|run|ci
        [2026-04-15] 演进模式下子进程 PATH 前置仓库 .venv/bin（或 venv/bin），使 python/pytest 与依赖与项目一致，避免仅系统 python3 无 pytest。
        """
        self.project_root = os.path.abspath(project_root)
        self._evolution_mode = bool(evolution_mode)
        self.execution_history: List[Dict] = []
        
        if not os.path.exists(self.project_root):
            logger.error(f"Workspace directory does not exist: {self.project_root}")
            raise ValueError(f"Invalid workspace root: {self.project_root}")
        
        if self._evolution_mode:
            logger.info(f"[BashExecutor] EVOLUTION mode, repo root: {self.project_root}")
            pair = self._evolution_venv_paths()
            if pair:
                _, bin_dir = pair
                logger.info(
                    f"[BashExecutor] EVOLUTION: prepending PATH with {bin_dir} "
                    f"(python/pytest 与 requirements.txt 一致)"
                )
            else:
                logger.warning(
                    "[BashExecutor] EVOLUTION: 未找到 .venv 或 venv；execute_bash_project 使用系统 PATH，"
                    "可能出现 python 命令缺失或 pytest 未安装。请在仓库根: "
                    "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
                )
        else:
            logger.info(f"[BashExecutor] sandbox: {self.project_root}")

    def _evolution_venv_paths(self) -> Optional[Tuple[str, str]]:
        """演进模式：若仓库根下存在本地虚拟环境，返回 (venv_root, bin_dir)。"""
        if not getattr(self, "_evolution_mode", False):
            return None
        for name in (".venv", "venv"):
            root = os.path.join(self.project_root, name)
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                return (root, bin_dir)
        return None

    def _subprocess_env(self) -> Dict[str, str]:
        """子进程环境：演进模式且存在 venv 时前置 PATH 并设置 VIRTUAL_ENV。"""
        env: Dict[str, str] = {**os.environ, "PYTHONUNBUFFERED": "1", "LC_ALL": "C.UTF-8"}
        pair = self._evolution_venv_paths()
        if pair:
            venv_root, bin_dir = pair
            prev = env.get("PATH", "")
            env["PATH"] = bin_dir + os.pathsep + prev
            env["VIRTUAL_ENV"] = venv_root
        return env
    
    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """
        检查命令是否安全
        
        Returns:
            (is_safe, reason)
        """
        # Reject empty command
        if not command or not command.strip():
            return False, "空命令"
        
        # Normalize whitespace
        command = command.strip()
        
        # First token basename = invoked command
        try:
            tokens = shlex.split(command)
            if not tokens:
                return False, "无效命令"
            
            base_command = os.path.basename(tokens[0])
        except ValueError as e:
            return False, f"命令解析失败: {e}"
        
        # [2026-04-03] Evolution mode: pytest / npm / compileall carve-outs
        if getattr(self, "_evolution_mode", False):
            if base_command == "pytest":
                pass  # allowed after pattern checks below
            elif base_command in ("python", "python3"):
                cs = command.strip()
                if re.match(r"^python3?\s+-m\s+pytest\b", cs):
                    pass
                elif _evolution_python_compileall_command_ok(cs):
                    pass
                else:
                    return (
                        False,
                        "演进模式下 python 仅允许: python -m pytest ... 或 "
                        "python -m compileall（如 -q backend，禁止 -c 与未列出的开关）",
                    )
            elif base_command == "npm":
                if not re.match(r"^npm\s+(test|run|ci)\b", command.strip()):
                    return False, "演进模式下 npm 仅允许: npm test | npm run … | npm ci"
            else:
                pass  # fall through to allowlist
        
        # Denylist
        if base_command in FORBIDDEN_COMMANDS:
            return False, f"禁止的命令: {base_command}"
        
        evolution_ok = getattr(self, "_evolution_mode", False) and base_command in (
            "pytest",
            "python",
            "python3",
            "npm",
        )
        
        # Allowlist
        if not evolution_ok and base_command not in ALLOWED_COMMANDS:
            return False, f"不在白名单中: {base_command} (如需此命令，请联系管理员)"
        
        # Regex danger heuristics
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"检测到危险模式: {pattern}"
        
        # [2026-02-22] Extra checks for restricted commands
        if base_command in SAFE_RESTRICTED_COMMANDS or (
            getattr(self, "_evolution_mode", False) and base_command == "git"
        ):
            # curl: GET-only
            if base_command == "curl":
                if any(opt in command for opt in ["-X POST", "-X PUT", "-X DELETE", "--data", "-d "]):
                    return False, "curl 只允许 GET 请求"
            
            # git: narrow allow-set
            if base_command == "git":
                if "push" in command.split():
                    return False, "禁止 git push（请在本机手动推送）"
                if "--force" in command and "push" in command:
                    return False, "禁止 git push --force"
                if "remote" in command and ("add" in command or "set-url" in command):
                    return False, "禁止修改 git remote"
            
            # sqlite3: read-only SQL
            if base_command == "sqlite3":
                sql_lower = command.lower()
                if any(kw in sql_lower for kw in ["insert", "update", "delete", "drop", "create", "alter"]):
                    return False, "sqlite3 只允许只读查询"
        
        # High-risk commands (evolution whitelists pytest/compileall/npm above)
        if base_command in HIGH_RISK_COMMANDS:
            if getattr(self, "_evolution_mode", False) and base_command in ("python", "python3"):
                pass  # already constrained to -m pytest / -m compileall
            else:
                return False, f"高风险命令 {base_command} 已禁用，请使用对应的安全工具"
        
        return True, "通过安全检查"
    
    def _check_path_args(self, command: str) -> Optional[str]:
        """
        [2026-02-28] 检查命令参数中的路径是否越出 sandbox。
        
        防止通过绝对路径参数绕过 cwd 限制，例如：
        - cp ./file /path/to/project/backend/
        - sed -i 's/x/y/' /path/to/project/backend/app.py
        - mkdir /tmp/evil
        
        Returns:
            错误消息（如被拦截），否则 None
        """
        # Block path traversal in argv string
        if "../" in command or "..\\" in command:
            return "检测到路径穿越 ../"
        
        # Redirect targets must stay inside workspace root
        redirect_pattern = re.findall(r'[12]?>>?\s*([^\s|&;]+)', command)
        for target in redirect_pattern:
            if os.path.isabs(target):
                abs_target = os.path.abspath(target)
                if not abs_target.startswith(self.project_root):
                    return f"重定向目标超出工作空间: {target}"
        
        # Scan absolute paths (quoted + unquoted)
        # Pattern 1: unquoted absolutes
        unquoted_paths = re.findall(r'(?:^|\s)(/[^\s"\'|&;>]+)', command)
        # Pattern 2: quoted absolutes
        quoted_paths = re.findall(r'["\'](/[^"\']+)["\']', command)
        
        safe_devs = {'/dev/null', '/dev/stdin', '/dev/stdout', '/dev/stderr'}
        
        for path in unquoted_paths + quoted_paths:
            abs_path = os.path.abspath(path)
            if abs_path in safe_devs:
                continue
            if not abs_path.startswith(self.project_root):
                return f"路径超出工作空间: {path}"
        
        return None
    
    def execute_bash(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: int = MAX_EXECUTION_TIME,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        执行bash命令
        
        Args:
            command: 要执行的bash命令
            working_dir: 工作目录（相对或绝对路径，必须在项目内）
            timeout: 超时时间（秒）
            session_id: 会话ID（用于日志）
            
        Returns:
            执行结果字典
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"[BASH-EXEC {execution_id}] Command: {command[:100]}")
        
        # Safety gate 1: allow/deny lists
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            error_result = {
                "success": False,
                "error": f"安全检查失败: {reason}",
                "command": command,
                "execution_id": execution_id,
                "timestamp": start_time.isoformat()
            }
            self.execution_history.append(error_result)
            logger.warning(f"[BASH-EXEC {execution_id}] BLOCKED: {reason}")
            return error_result
        
        # Safety gate 2: argv path sandboxing
        path_check = self._check_path_args(command)
        if path_check:
            error_result = {
                "success": False,
                "error": f"路径安全检查失败: {path_check}",
                "command": command,
                "execution_id": execution_id,
                "timestamp": start_time.isoformat()
            }
            self.execution_history.append(error_result)
            logger.warning(f"[BASH-EXEC {execution_id}] PATH BLOCKED: {path_check}")
            return error_result
        
        # Resolve cwd
        if working_dir:
            # Normalize to absolute
            if not os.path.isabs(working_dir):
                abs_working_dir = os.path.abspath(os.path.join(self.project_root, working_dir))
            else:
                abs_working_dir = os.path.abspath(working_dir)
            
            # Must stay under allowed root
            if not abs_working_dir.startswith(self.project_root):
                return {
                    "success": False,
                    "error": f"工作目录必须在允许根目录内: 尝试访问 {abs_working_dir}",
                    "command": command,
                    "execution_id": execution_id
                }
            
            # Directory must exist
            if not os.path.exists(abs_working_dir):
                return {
                    "success": False,
                    "error": f"工作目录不存在: {abs_working_dir}",
                    "command": command,
                    "execution_id": execution_id
                }
        else:
            abs_working_dir = self.project_root
        
        max_t = EVOLUTION_MAX_EXECUTION_TIME if getattr(self, "_evolution_mode", False) else MAX_EXECUTION_TIME
        timeout = min(max(1, timeout), max_t)
        
        # Run subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=abs_working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._subprocess_env(),
            )
            
            stdout = result.stdout[:MAX_OUTPUT_SIZE] if result.stdout else ""
            stderr = result.stderr[:MAX_OUTPUT_SIZE] if result.stderr else ""
            returncode = result.returncode
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # success := exit 0
            success = (returncode == 0)
            
            exec_result = {
                "success": success,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "command": command,
                "working_dir": abs_working_dir,
                "duration_seconds": round(duration, 2),
                "execution_id": execution_id,
                "timestamp": start_time.isoformat(),
                "output_summary": {
                    "stdout_lines": len(stdout.split('\n')) if stdout else 0,
                    "stderr_lines": len(stderr.split('\n')) if stderr else 0,
                    "stdout_chars": len(stdout),
                    "stderr_chars": len(stderr),
                    "was_truncated": len(result.stdout or "") > MAX_OUTPUT_SIZE or len(result.stderr or "") > MAX_OUTPUT_SIZE
                }
            }
            
            # Append to ring buffer
            self.execution_history.append(exec_result)
            
            # Log outcome
            if success:
                logger.info(f"[BASH-EXEC {execution_id}] SUCCESS in {duration:.2f}s")
            else:
                logger.warning(f"[BASH-EXEC {execution_id}] FAILED with code {returncode}")
            
            return exec_result
            
        except subprocess.TimeoutExpired:
            error_result = {
                "success": False,
                "error": f"命令超时（>{timeout}秒）",
                "command": command,
                "execution_id": execution_id,
                "timestamp": start_time.isoformat()
            }
            self.execution_history.append(error_result)
            logger.error(f"[BASH-EXEC {execution_id}] TIMEOUT")
            return error_result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "command": command,
                "execution_id": execution_id,
                "timestamp": start_time.isoformat()
            }
            self.execution_history.append(error_result)
            logger.error(f"[BASH-EXEC {execution_id}] ERROR: {e}")
            return error_result
    
    def get_execution_history(self, session_id: str = "default", limit: int = 10) -> List[Dict]:
        """获取执行历史"""
        # Tail of ring buffer
        return self.execution_history[-limit:]
    
    def get_tool_definitions(self) -> List[Dict]:
        """返回OpenAI格式的工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "description": """在 workspace/sandbox 目录中执行bash命令。

🔥 安全特性 [2026-02-22 强化]：
- ✅ 严格白名单：只允许已验证的安全命令
- ❌ 高风险禁用：python/pip/wget 已禁用（请用专用工具）
- 📁 沙箱限制：只能在 workspace/sandbox 内操作
- ⏱️ 超时控制：最长30秒

✅ 允许的安全命令：
- 文件查看：ls, cat, head, tail, wc, file
- 文件搜索：find, grep
- 文本处理：sed, awk, cut, sort, uniq, diff
- 压缩解压：tar, zip, unzip, gzip
- 网络工具：curl (仅 GET)
- Git操作：git status, git log, git diff (禁止 push --force)
- 系统信息：pwd, date, df, du

❌ 禁用的高风险命令：
- python/python3 → 请用 execute_python 工具
- pip/pip3 → 不允许安装包
- wget → 请用 curl
- rm/rmdir → 请用文件管理工具""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "要执行的bash命令（例如：'ls -la', 'git status', 'python3 script.py'）"
                            },
                            "working_dir": {
                                "type": "string",
                                "description": "工作子目录（可选，相对于工作空间根目录，如 'tools', 'diaries'）"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "超时时间（秒，默认30，最大30）",
                                "default": 30
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "bash_execution_history",
                    "description": "查看最近执行的bash命令历史（最多10条）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "返回的历史记录数量（默认10）",
                                "default": 10
                            }
                        },
                        "required": []
                    }
                }
            }
        ]

    def get_project_tool_definitions(self) -> List[Dict]:
        """演进模式：在 S 仓库根执行 execute_bash_project（与 execute_bash 相同校验逻辑，根目录不同）。"""
        if not getattr(self, "_evolution_mode", False):
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_bash_project",
                    "description": """在 S 项目仓库根目录执行 bash（演进模式）。
若仓库存在 .venv/venv，子进程会自动把其 bin 加入 PATH 最前，优先使用项目里的 python/pytest（与 requirements.txt 一致）。
额外允许：pytest、python -m pytest、python -m compileall（建议 -q backend；
仅允许 -q/-f/-l/-b、-j≤16、-x 排除模式、-o 输出目录；路径须相对且无 ..）、npm test|run|ci。
仍禁止：rm、pip、python -c、任意 .py 脚本路径、git push。
大改动请配合 evolution_git_* 与 evolution_* 文件工具。""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "working_dir": {
                                "type": "string",
                                "description": "相对仓库根的子目录，如 backend；空则仓库根",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "超时秒数，默认30，演进模式最大300",
                                "default": 30,
                            },
                        },
                        "required": ["command"],
                    },
                },
            }
        ]
    
    def route_tool_call(self, tool_name: str, args: Dict, session_id: str) -> Dict:
        """路由工具调用"""
        if tool_name in ("execute_bash", "execute_bash_project"):
            cap = EVOLUTION_MAX_EXECUTION_TIME if getattr(self, "_evolution_mode", False) else MAX_EXECUTION_TIME
            return self.execute_bash(
                command=args.get("command", ""),
                working_dir=args.get("working_dir"),
                timeout=min(args.get("timeout", cap), cap),
                session_id=session_id
            )
        elif tool_name == "bash_execution_history":
            return {
                "success": True,
                "history": self.get_execution_history(
                    session_id=session_id,
                    limit=args.get("limit", 10)
                )
            }
        else:
            return {"error": f"Unknown tool: {tool_name}"}


# Module singleton
_bash_executor_instance = None

def get_bash_executor() -> BashExecutor:
    """获取BashExecutor单例"""
    global _bash_executor_instance
    if _bash_executor_instance is None:
        _bash_executor_instance = BashExecutor()
    return _bash_executor_instance
