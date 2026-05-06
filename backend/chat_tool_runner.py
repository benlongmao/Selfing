import json
import logging
import os
from typing import List, Dict, Optional, Tuple, Set

from backend.tool_feedback import apply_tool_feedback
from backend.message_compressor import compress_messages, should_compress
from backend.tool_result_standardizer import standardize_tool_result, is_tool_result_error
from backend.config import config

logger = logging.getLogger(__name__)

# Max chars for evolved context (semantic tool reselect inside tool loop)
EVOLVED_CONTEXT_MAX_CHARS = 2200

# [2026-02-24] Cap serialized tool payloads so huge list_files etc. do not blow the 112k-token window
MAX_TOOL_RESULT_CHARS = 8000


def _truncate_tool_result_for_llm(tool_payload: Dict, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """
    Cap serialized tool payloads for the LLM context window.

    For list_files-like results: only a subset of paths may be shown, but we always
    preserve total_count / list_incomplete / enumeration_warning so the model does
    not treat "not listed" as "missing". If the JSON is still too large, shrink the
    paths array (binary search) instead of blind string truncation that drops counts.
    """
    payload = dict(tool_payload)
    result = payload.get("result")
    if not isinstance(result, dict):
        s = json.dumps(payload, ensure_ascii=False)
        return s[:max_chars] + "\n...[tool result truncated]" if len(s) > max_chars else s

    result = dict(result)
    payload["result"] = result

    LLM_FILES_LIMIT = 50
    if "files" in result and isinstance(result["files"], list):
        all_files = result["files"]
        total_count = int(result.get("total_count", len(all_files)))
        tool_truncated = bool(result.get("truncated", False))
        capped = all_files[:LLM_FILES_LIMIT]
        result["files"] = capped
        result["total_count"] = total_count
        result["files_in_this_message"] = len(capped)
        list_incomplete = tool_truncated or total_count > len(capped)
        result["list_incomplete"] = list_incomplete
        if list_incomplete:
            result["enumeration_warning"] = (
                f"INCOMPLETE_ENUMERATION: total_count={total_count}, this message shows only {len(capped)} paths. "
                "Do not infer that any file or numbered item is absent. "
                "Verify with execute_bash_project, e.g. "
                "`find <dir> -maxdepth 3 -name '*.md' | wc -l` or `ls -1 <dir> | sort`."
            )

    LLM_CONTENT_LIMIT = 3000
    if "content" in result and isinstance(result["content"], str):
        if len(result["content"]) > LLM_CONTENT_LIMIT:
            result["content"] = result["content"][:LLM_CONTENT_LIMIT] + "\n...[content truncated]"
            payload["result"] = result

    s = json.dumps(payload, ensure_ascii=False)
    if len(s) <= max_chars:
        return s

    if (
        isinstance(result.get("files"), list)
        and "total_count" in result
        and "error" not in result
    ):
        tc = int(result["total_count"])
        warn = result.get(
            "enumeration_warning",
            "INCOMPLETE_ENUMERATION: path list was compressed for length; use execute_bash_project with find/wc for a full listing.",
        )
        paths = list(result["files"])
        best_json = ""
        low, high = 0, len(paths)
        while low <= high:
            mid = (low + high) // 2
            slim = {
                "success": result.get("success"),
                "total_count": tc,
                "truncated": result.get("truncated"),
                "list_incomplete": True,
                "files_in_this_message": mid,
                "enumeration_warning": warn,
                "files": paths[:mid],
                "current_directory": result.get("current_directory"),
                "path_hint": result.get("path_hint"),
                "note": result.get("note"),
            }
            p2 = dict(payload)
            p2["result"] = slim
            cand = json.dumps(p2, ensure_ascii=False)
            if len(cand) <= max_chars:
                best_json = cand
                low = mid + 1
            else:
                high = mid - 1
        if best_json:
            return best_json
        p2 = dict(payload)
        p2["result"] = {
            "success": result.get("success"),
            "total_count": tc,
            "truncated": True,
            "list_incomplete": True,
            "files_in_this_message": 0,
            "files": [],
            "enumeration_warning": warn,
            "path_hint": result.get("path_hint"),
        }
        return json.dumps(p2, ensure_ascii=False)

    return s[:max_chars] + "\n...[tool result truncated]"


# ============================================================
# [2026-01-21] Post-exec auto verification for selected tools
# Idea: tools extend the agent; the runtime verifies side effects when possible
# ============================================================

def _auto_verify_tool_result(
    tool_router,
    function_name: str,
    function_args: Dict,
    tool_result: Dict,
    session_id: str,
    log: logging.Logger
) -> Dict:
    """
    自动验证工具执行结果
    
    对于文件操作类工具，系统自动验证文件是否真的被创建/修改
    将验证结果强制注入到工具返回中，Agent 必须基于此报告
    """
    
    # Passthrough on tool-layer errors
    if "error" in tool_result:
        return tool_result
    
    # ==================== write_file auto-verify ====================
    if function_name == "write_file":
        filename = function_args.get("filename", "")
        if filename and tool_result.get("success"):
            # Confirm file shows up in list_files for parent dir
            try:
                verify_result = tool_router.route("list_files", {"subdir": os.path.dirname(filename) or ""}, session_id=session_id)
                files_list = verify_result.get("files", [])
                
                # Match basename or full relative path
                file_exists = any(filename in f or os.path.basename(filename) in f for f in files_list)
                
                # Attach structured verification block
                tool_result["system_verification"] = {
                    "verified_by": "system_auto_check",
                    "file_exists": file_exists,
                    "verification_method": "list_files",
                    "timestamp": _get_timestamp()
                }
                
                if file_exists:
                    tool_result["system_verification"]["status"] = "✅ 系统验证通过：文件确实存在"
                    log.info(f"[AUTO-VERIFY] write_file '{filename}' verified: EXISTS")
                else:
                    tool_result["system_verification"]["status"] = "❌ 系统验证失败：文件不存在于目录列表中"
                    tool_result["system_verification"]["warning"] = "Agent 必须告知用户文件可能未成功创建"
                    log.warning(f"[AUTO-VERIFY] write_file '{filename}' verification FAILED: NOT IN LIST")
                    
            except Exception as e:
                log.warning(f"[AUTO-VERIFY] Failed to verify write_file: {e}")
                tool_result["system_verification"] = {
                    "verified_by": "system_auto_check",
                    "status": "⚠️ 验证过程出错",
                    "error": str(e)
                }
    
    # ==================== execute_python auto-verify ====================
    elif function_name == "execute_python":
        code = function_args.get("code", "")
        
        # Static scan for likely file writes in executed code
        file_operations = _detect_file_operations(code)
        
        if file_operations and tool_result.get("success"):
            # read_file probe for each suspected path
            verification_results = []
            
            for file_path in file_operations:
                try:
                    # read_file succeeds → path exists
                    read_result = tool_router.route("read_file", {"filename": file_path}, session_id=session_id)
                    
                    if "error" not in read_result:
                        verification_results.append({
                            "file": file_path,
                            "exists": True,
                            "content_preview": read_result.get("content", "")[:100] + "..." if len(read_result.get("content", "")) > 100 else read_result.get("content", "")
                        })
                        log.info(f"[AUTO-VERIFY] execute_python file '{file_path}' verified: EXISTS")
                    else:
                        verification_results.append({
                            "file": file_path,
                            "exists": False,
                            "error": read_result.get("error")
                        })
                        log.warning(f"[AUTO-VERIFY] execute_python file '{file_path}' verification FAILED: {read_result.get('error')}")
                        
                except Exception as e:
                    verification_results.append({
                        "file": file_path,
                        "exists": False,
                        "error": str(e)
                    })
            
            # Merge verification summary into tool_result
            all_exist = all(v.get("exists") for v in verification_results)
            tool_result["system_verification"] = {
                "verified_by": "system_auto_check",
                "detected_file_operations": file_operations,
                "verification_results": verification_results,
                "all_files_verified": all_exist,
                "timestamp": _get_timestamp()
            }
            
            if all_exist:
                tool_result["system_verification"]["status"] = f"✅ 系统验证通过：{len(file_operations)}个文件全部存在"
            else:
                failed_files = [v["file"] for v in verification_results if not v.get("exists")]
                tool_result["system_verification"]["status"] = f"❌ 系统验证失败：{len(failed_files)}个文件不存在"
                tool_result["system_verification"]["warning"] = f"Agent 必须诚实告知用户：文件 {failed_files} 未成功创建"
        
        # Reminder when no file side effects were heuristically detected
        if "system_verification" not in tool_result:
            tool_result["system_verification"] = {
                "verified_by": "system_auto_check",
                "status": "ℹ️ 代码执行完成，stdout 是实际输出",
                "reminder": "Agent 的报告必须基于 stdout 的实际内容，而非预期"
            }
    
    return tool_result


def _detect_file_operations(code: str) -> list:
    """
    从 Python 代码中检测文件操作
    返回可能被创建/修改的文件路径列表
    """
    import re
    
    file_paths = []
    
    # Detect open(..., "w"/"a"/"x") style writes
    # Covers open("path", "w") and with open(..., "w")
    write_patterns = [
        r'open\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\'][wax][+]?["\']',  # open("file", "w")
        r'with\s+open\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\'][wax][+]?["\']',  # with open("file", "w")
    ]
    
    for pattern in write_patterns:
        matches = re.findall(pattern, code)
        file_paths.extend(matches)
    
    # os.makedirs / os.mkdir string paths
    mkdir_pattern = r'os\.makedirs?\s*\(\s*["\']([^"\']+)["\']'
    mkdir_matches = re.findall(mkdir_pattern, code)
    file_paths.extend(mkdir_matches)
    
    # shutil.copy/move destination path
    copy_pattern = r'shutil\.(copy|move)\s*\([^,]+,\s*["\']([^"\']+)["\']'
    copy_matches = re.findall(copy_pattern, code)
    file_paths.extend([m[1] for m in copy_matches])
    
    # Dedupe paths
    return list(set(file_paths))


def _get_timestamp() -> str:
    """获取当前时间戳"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# [2026-01-21] Hallucination heuristics (plan B)
# Flag natural-language claims of side effects without matching tool calls
# ============================================================

# Substrings that imply filesystem side effects were performed
FILE_OPERATION_CLAIMS = [
    # "created file" style claims (CN)
    "创建了文件", "文件已创建", "已创建文件", "成功创建了", "文件创建成功",
    "已成功创建", "创建完成", "已经创建", "新建了文件", "文件已新建",
    # "wrote file" style
    "写入了文件", "文件已写入", "已写入文件", "写入成功", "文件写入完成",
    "已成功写入", "保存了文件", "文件已保存", "已保存文件",
    # "edited file" style
    "修改了文件", "文件已修改", "已修改文件", "更新了文件", "文件已更新",
    "已更新文件", "编辑了文件", "文件已编辑",
    # "ran code" style
    "执行了代码", "代码执行成功", "Python代码已执行", "脚本执行完成",
    "代码已运行", "成功执行了", "运行了代码",
    # delete
    "删除了文件", "文件已删除", "已删除文件",
    # rename
    "重命名了", "文件已重命名", "已重命名", "改名成功", "已改名",
    # [2026-02-05] subtler path / completion claims
    "文件在", "文件位于", "文件保存在", "文件创建在",
    "已完成", "任务完成", "操作完成",
]

# Tool names that should appear when above claims fire
FILE_OPERATION_TOOLS = [
    "write_file", "execute_python", "append_file",
    "delete_file", "delete_directory",
    "rename_file",
    "read_file",  # [2026-02-05] "I read the file" still needs read_file tool receipt
]

# ============================================================
# [2026-02-08] AGI-ish data claims without tool receipts
# Knowledge / proposals / plans must be backed by listed tools
# ============================================================

# Category → claim substrings vs required tools
AGI_DATA_CLAIMS = {
    # Knowledge stats / search
    "knowledge": {
        "claims": [
            "知识库", "知识记录", "条知识", "学过的知识", "学到的知识",
            "知识总结", "知识条目", "知识统计",
        ],
        "tools": ["get_knowledge_stats", "search_my_knowledge"],
    },
    # Code proposals
    "proposal": {
        "claims": [
            "代码改进", "代码提案", "改进建议", "提案", "代码修改建议",
            "我提出的", "我建议的", "我的提案",
        ],
        "tools": ["list_my_proposals", "check_proposal_result"],
    },
    # Task / plan widgets
    "task": {
        "claims": [
            "执行计划", "任务计划", "活跃计划", "当前任务", "任务列表",
            "计划详情", "任务进度", "正在执行",
        ],
        "tools": ["get_my_active_plans", "get_plan_details", "get_next_task"],
    },
}

# "Let me check..." preambles often precede fabricated stats
PRETEND_LOOKUP_PATTERNS = [
    "让我查看", "让我看看", "让我检查", "查看一下", "看一下",
    "我来查看", "我来看看", "我来检查",
]


def detect_hallucination_claim(
    response_text: str,
    tool_calls_made: List[str],
    log: logging.Logger
) -> Tuple[bool, Optional[str]]:
    """
    [2026-02-05] 增强版幻觉检测
    
    检测 Agent 的响应是否存在幻觉声明
    
    核心策略：
    1. 模式匹配：检测声称做了文件操作的关键词
    2. 工具调用验证：确认是否真的调用了对应工具
    3. 代码块检测：如果只输出了代码示例但没调用 execute_python，这是幻觉
    4. 结果声称检测：如果声称"文件已创建"但没有 write_file 调用，这是幻觉
    
    Args:
        response_text: Agent 的响应文本
        tool_calls_made: 本次请求中实际调用的工具列表
        log: 日志器
    
    Returns:
        (is_hallucination, reason): 是否是幻觉，以及原因
    """
    if not response_text:
        return False, None
    
    # ============================================================
    # Strategy 1: filesystem claim vs tool receipts
    # ============================================================
    claims_found = []
    for claim in FILE_OPERATION_CLAIMS:
        if claim in response_text:
            claims_found.append(claim)
    
    if claims_found:
        # Require at least one file-capable tool in this turn
        has_file_tool = any(tool in FILE_OPERATION_TOOLS for tool in tool_calls_made)
        
        if not has_file_tool:
            # Strong mismatch: prose claims side effects, no tools
            reason = f"检测到幻觉：响应中声称 [{', '.join(claims_found[:3])}]，但没有调用文件操作工具"
            log.warning(f"[HALLUCINATION-DETECT] {reason}")
            log.warning(f"[HALLUCINATION-DETECT] 实际调用的工具: {tool_calls_made}")
            return True, reason
    
    # ============================================================
    # Strategy 2: ```python``` blocks + "ran it" language without execute_python
    # ============================================================
    # Code fences alone are fine; execution verbs without tool are not
    import re
    python_code_blocks = re.findall(r'```python\s+(.*?)\s+```', response_text, re.DOTALL)
    
    if python_code_blocks and 'execute_python' not in tool_calls_made:
        # Only flag when assistant also claims execution happened
        execution_claims = [
            "执行了", "已执行", "运行了", "已运行", "执行成功",
            "代码执行", "脚本执行", "运行成功"
        ]
        
        has_execution_claim = any(claim in response_text for claim in execution_claims)
        
        if has_execution_claim:
            reason = (
                f"检测到幻觉：响应中包含 {len(python_code_blocks)} 个 Python 代码块，"
                f"并声称执行了代码，但实际没有调用 execute_python 工具"
            )
            log.warning(f"[HALLUCINATION-DETECT] {reason}")
            log.warning(f"[HALLUCINATION-DETECT] 代码块数量: {len(python_code_blocks)}")
            return True, reason
    
    # ============================================================
    # Strategy 3: prose path claims with zero tools this turn
    # ============================================================
    # e.g. "file saved under sandbox/foo.txt" with empty tool_calls_made
    file_path_patterns = [
        r'文件(?:已创建|创建|保存|写入)(?:在|于)\s*[^\s，。！]+',
        r'(?:创建|写入|保存)了.*?(?:sandbox|workspace)/[^\s，。！]+',
    ]
    
    file_path_claims = []
    for pattern in file_path_patterns:
        matches = re.findall(pattern, response_text)
        file_path_claims.extend(matches)
    
    if file_path_claims and not tool_calls_made:
        # Regex hit + no tools at all
        reason = (
            f"检测到幻觉：响应中声称创建/保存了文件 [{file_path_claims[0]}]，"
            f"但没有调用任何工具"
        )
        log.warning(f"[HALLUCINATION-DETECT] {reason}")
        return True, reason
    
    # ============================================================
    # Strategy 4: generic "done/success" with zero tools (narrow guard)
    # ============================================================
    # Only escalates when Strategy 1 already saw filesystem claims
    completion_claims = [
        "已完成", "任务完成", "操作完成", "成功完成",
        "✅", "成功", "完成了"
    ]
    
    has_completion_claim = any(claim in response_text for claim in completion_claims)
    
    if has_completion_claim and not tool_calls_made:
        # Allow trivial acknowledgements
        simple_replies = ["理解了", "明白了", "好的", "知道了", "收到"]
        is_simple_reply = any(reply in response_text for reply in simple_replies)
        
        # Short polite replies are not hallucinations
        if len(response_text) < 50 and is_simple_reply:
            return False, None
        
        # Completion language + prior file claims + no tools
        if claims_found:
            reason = (
                f"检测到幻觉：响应中声称任务已完成，但没有调用任何工具来执行操作"
            )
            log.warning(f"[HALLUCINATION-DETECT] {reason}")
            return True, reason
    
    # ============================================================
    # [2026-02-08] Strategy 5: AGI data claims vs tools
    # ============================================================
    
    # Pretend-to-fetch intros
    has_pretend_lookup = any(pattern in response_text for pattern in PRETEND_LOOKUP_PATTERNS)
    
    for category, info in AGI_DATA_CLAIMS.items():
        claims = info["claims"]
        required_tools = info["tools"]
        
        # Hits for this category's claim substrings
        category_claims = [c for c in claims if c in response_text]
        
        if category_claims:
            # Need one of the declared readers
            has_required_tool = any(tool in tool_calls_made for tool in required_tools)
            
            if not has_required_tool:
                # Pretend lookup + numeric/detail claims are high risk
                if has_pretend_lookup:
                    reason = (
                        f"检测到 AGI 幻觉：响应中说'{PRETEND_LOOKUP_PATTERNS[0]}...'并声称有 "
                        f"[{', '.join(category_claims[:2])}]，但没有调用 {required_tools} 中的任何工具"
                    )
                    log.warning(f"[HALLUCINATION-DETECT] {reason}")
                    log.warning(f"[HALLUCINATION-DETECT] 实际调用的工具: {tool_calls_made}")
                    return True, reason
                
                # Quantified CN claims (e.g. counts + knowledge wording) without tools
                import re
                number_pattern = rf'\d+\s*(?:条|个|项)?(?:{"|".join(category_claims)}|记录)'
                if re.search(number_pattern, response_text):
                    reason = (
                        f"检测到 AGI 幻觉：响应中声称有具体数量的 [{category}] 数据，"
                        f"但没有调用 {required_tools} 中的任何工具来获取这些数据"
                    )
                    log.warning(f"[HALLUCINATION-DETECT] {reason}")
                    return True, reason
    
    return False, None


def _build_evolved_context(
    messages: List[Dict],
    response_text: str,
    tool_calls_this_round: List,
    max_chars: int = EVOLVED_CONTEXT_MAX_CHARS,
) -> str:
    """
    从当前对话构建「演化上下文」：最后一条用户消息 + 本轮 assistant 回复摘要 + 本轮工具结果摘要。
    用于 tool loop 内按上下文重新做语义工具选择。
    """
    last_user = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    (c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                )
            last_user = content[:500]
    parts = [last_user] if last_user else []
    if response_text:
        parts.append(response_text[:600])
    n_tool = len(tool_calls_this_round)
    for i in range(-n_tool, 0):
        if i >= -len(messages) and messages[i].get("role") == "tool":
            content = (messages[i].get("content") or "")[:400]
            parts.append(content)
    s = "\n\n".join(parts)
    return s[:max_chars] if len(s) > max_chars else s


def _merge_tools_with_selection(
    chat_service,
    allowed_tool_names: Set[str],
    new_tool_defs: List[Dict],
) -> Tuple[List[Dict], Set[str]]:
    """
    将首轮/当前允许的工具名与新一轮语义选出的工具做并集，只增不减；
    返回新的 tools 定义列表和新的 allowed_tool_names。
    """
    from backend.tool_selector import COMPACT_DESCRIPTIONS

    new_names = {t["function"]["name"] for t in new_tool_defs}
    merged_names = allowed_tool_names | new_names
    all_defs = chat_service.tool_router.get_tool_definitions()
    by_name = {}
    for d in all_defs:
        n = (d.get("function") or {}).get("name")
        if n:
            by_name[n] = d
    result = []
    for n in merged_names:
        if n not in by_name:
            continue
        orig = by_name[n]
        fn = orig.get("function") or {}
        desc = COMPACT_DESCRIPTIONS.get(n, fn.get("description", ""))
        result.append({
            "type": "function",
            "function": {
                "name": n,
                "description": desc,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
            },
        })
    return result, merged_names


def _apply_autonomy_markers_from_assistant_turn(
    chat_service,
    session_id: str,
    response_text: str,
    introspection: Optional[Dict],
    log_prefix: str = "[TOOL-LOOP]",
) -> None:
    """
    思考模型常把 [S44_AUTONOMY_*] 写在 reasoning / inner_monologue，visible content 为空也会漏检；
    且首轮 tool_calls 前的 assistant 正文只在进入 while 前存在一份，必须在循环内与循环前都扫。
    """
    try:
        from backend.autonomy_gate import apply_assistant_autonomy_markers
        parts: List[str] = []
        rt = (response_text or "").strip()
        if rt:
            parts.append(rt)
        rsn = getattr(chat_service, "_last_reasoning_content", None) or ""
        if rsn:
            parts.append(str(rsn).strip())
        if introspection and introspection.get("inner_monologue"):
            parts.append(str(introspection.get("inner_monologue")).strip())
        combined = "\n".join(p for p in parts if p)
        if not combined:
            return
        chg = apply_assistant_autonomy_markers(combined, session_id)
        if chg:
            logger.info("%s Autonomy marker applied: %s", log_prefix, chg)
    except Exception:
        pass


def run_tool_loop(
    chat_service,
    session_id: str,
    self_model,
    messages: List[Dict],
    response_text: str,
    introspection: Dict,
    tool_calls: Optional[List],
    turn_index: Optional[int],
    temperature: float,
    tools: Optional[List[Dict]],
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    max_tokens: int = 2048,
    initial_usage: Optional[Dict] = None,
    max_tool_turns: Optional[int] = None,
    embedder=None,
) -> Tuple[str, List[str], Dict, List[str]]:
    """
    执行工具调用循环（ReAct），返回最终 response_text、累积 inner_monologues、累积 usage 以及调用的工具列表。
    
    max_tool_turns：显式传入则优先；否则读 config parameters.chat.max_tool_turns（默认 8）。

    Returns:
        (response_text, all_introspections, usage, tools_called)
    """
    _mt_cfg = config.get("parameters.chat.max_tool_turns", 8)
    _mt = max_tool_turns if max_tool_turns is not None else _mt_cfg
    max_tool_turns = max(1, int(_mt) if _mt is not None else 8)

    all_introspections: List[str] = []
    if introspection.get("inner_monologue"):
        all_introspections.append(introspection.get("inner_monologue"))

    usage = dict(initial_usage) if initial_usage else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tool_turn = 0
    tool_budget_exceeded = False
    
    # [2026-01-21] Flat list of tool names this HTTP request (hallucination checks)
    tools_called: List[str] = []

    hi_compress = config.get("parameters.chat.history_injection")
    hi_compress = hi_compress if isinstance(hi_compress, dict) else {}
    msg_compress_token_threshold_tl = max(
        4000, int(hi_compress.get("msg_compress_token_threshold", 14000) or 14000)
    )
    max_assistant_chars_tl = max(
        300, int(hi_compress.get("max_assistant_chars", 600) or 600)
    )

    # [2026-03-13] Executor allowlist matches the tools list wired into this completion
    allowed_tool_names: set = set()
    if tools:
        for t in tools:
            name = (t.get("function") or {}).get("name")
            if name:
                allowed_tool_names.add(name)

    # Pre-loop pass: first assistant chunk may only live in reasoning/tool_call preamble
    _apply_autonomy_markers_from_assistant_turn(
        chat_service, session_id, response_text, introspection,
    )

    while tool_calls and tool_turn < max_tool_turns:
        # Autonomy stop markers do NOT block user-driven chat tools here (scheduler-only gate).
        tool_turn += 1
        logger.info(f"Tool turn {tool_turn}/{max_tool_turns}: LLM requested {len(tool_calls)} tool call(s)")

        # [2026-02-25] DeepSeek thinking mode expects reasoning_content on tool-calling turns
        # See https://api-docs.deepseek.com/zh-cn/guides/thinking_mode#tool-calls
        # Must attach the field even when empty string
        reasoning_content = ""
        if hasattr(chat_service, '_last_reasoning_content') and chat_service._last_reasoning_content:
            reasoning_content = chat_service._last_reasoning_content
            logger.debug(f"[THINKING-MODE] Preserved reasoning_content ({len(reasoning_content)} chars) for tool call")
        
        assistant_msg = {
            "role": "assistant",
            "content": response_text or "",
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content  # always present for API compatibility
        }
        messages.append(assistant_msg)

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            try:
                function_args = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse tool arguments for {function_name}: {e}")
                # Construct a fake error result so the loop can continue
                tool_call_id = tool_call["id"]
                error_msg = f"Error: Invalid JSON arguments for tool '{function_name}'. Please ensure arguments are valid JSON. Details: {str(e)}"
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": error_msg, "ok": False}, ensure_ascii=False),
                    "tool_call_id": tool_call_id
                })
                continue

            tool_call_id = tool_call["id"]

            if allowed_tool_names and function_name not in allowed_tool_names:
                logger.warning(f"[TOOL-ALLOWLIST] Rejected tool '{function_name}' (not in this turn's tools)")
                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "error": f"工具 '{function_name}' 未在本轮提供，请仅使用本轮可用工具。",
                        "ok": False,
                    }, ensure_ascii=False),
                    "tool_call_id": tool_call_id
                })
                continue

            tool_result = chat_service.tool_router.route(
                function_name, function_args,
                session_id=session_id,
                allowed_tool_names=allowed_tool_names or None,
            )

            # [2026-03-20] Layer-2 lazy expansion via _expand_tools payload from router
            expand_defs = None
            if isinstance(tool_result, dict) and "_expand_tools" in tool_result:
                expand_defs = tool_result.pop("_expand_tools", None)
            if expand_defs:
                for td in expand_defs:
                    tname = (td.get("function") or {}).get("name")
                    if tname and tname not in allowed_tool_names:
                        allowed_tool_names.add(tname)
                        tools.append(td)
                logger.info(f"[TOOL-EXPAND] Dynamically loaded {len(expand_defs)} tools via request_tool_group")

            tool_result = standardize_tool_result(
                tool_name=function_name,
                result=tool_result,
                session_id=session_id,
                add_verification_reminder=True
            )
            
            # Hallucination detector consumes this list post-hoc
            tools_called.append(function_name)
            
            # ============================================================
            # [2026-01-21] Auto verification pass (filesystem probes, etc.)
            # ============================================================
            tool_result = _auto_verify_tool_result(
                chat_service.tool_router,
                function_name,
                function_args,
                tool_result,
                session_id,
                logger
            )
            
            receipt_id = None
            if getattr(chat_service, "event_logger", None):
                try:
                    receipt_id = chat_service.event_logger.log_tool_call(
                        session_id=session_id,
                        tool_name=function_name,
                        args=function_args,
                        result=tool_result,
                        turn_index=turn_index,
                        tool_call_id=tool_call_id,
                    )
                except Exception:
                    receipt_id = None

            tool_payload = {
                "receipt_id": receipt_id,
                "tool_name": function_name,
                "ok": ("error" not in tool_result),
                "result": tool_result,
            }
            # [2026-02-24] Serialize + truncate before re-injecting into chat context
            tool_output_str = _truncate_tool_result_for_llm(tool_payload, max_chars=8000)

            logger.info(f"Tool executed: {function_name}, result size: {len(tool_output_str)}")

            if self_model:
                apply_tool_feedback(self_model, tool_result, session_id, logger)
                
                # [energy patch] tiny per-tool metabolic cost (was 5.0 → 0.1)
                try:
                    if hasattr(self_model, 'homeostasis'):
                        self_model.homeostasis.update_energy(session_id, -0.1)
                        logger.info(f"[TOOL-METABOLISM] Tool '{function_name}' consumed 0.1 energy.")
                        
                        # Log physiological side-channel for dashboards
                        if hasattr(chat_service, 'event_logger') and chat_service.event_logger:
                            chat_service.event_logger.log_event(
                                session_id,
                                "physiological_pain",
                                json.dumps({
                                    "type": "tool_cost",
                                    "tool": function_name,
                                    "penalty": -0.1
                                }, ensure_ascii=False)
                            )
                except Exception as e:
                    logger.warning(f"Failed to apply tool energy cost: {e}")

            messages.append(
                {
                    "role": "tool",
                    "content": tool_output_str,
                    "tool_call_id": tool_call_id,
                }
            )

        # [2026-03-18] WebSocket progress ping each tool round (UX visibility)
        try:
            from backend.websocket_manager import (
                get_websocket_manager,
                create_ws_message,
                WSMessageType,
            )
            ws_manager = get_websocket_manager()
            # Human-readable arg summary, e.g. read_self_code('config/settings.yaml')
            descs = []
            for tc in tool_calls:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments", "{}"))
                    if fn == "read_self_code" and args.get("file_path"):
                        descs.append(f"{fn}('{args['file_path']}')")
                    elif fn == "list_self_files" and args.get("directory"):
                        descs.append(f"{fn}('{args['directory']}')")
                    elif fn == "search_self_code" and args.get("keyword"):
                        dir_part = f", '{args['directory']}'" if args.get("directory") else ""
                        descs.append(f"{fn}('{args['keyword']}'{dir_part})")
                    elif fn == "read_file" and args.get("filename"):
                        descs.append(f"{fn}('{args['filename']}')")
                    elif fn == "rename_file" and args.get("old_filename") and args.get("new_filename"):
                        descs.append(
                            f"{fn}('{args['old_filename']}' → '{args['new_filename']}')"
                        )
                    elif fn == "list_files" and args.get("path"):
                        descs.append(f"{fn}('{args['path']}')")
                    else:
                        descs.append(fn)
                except Exception:
                    descs.append(fn)
            summary = f"工具执行 ({tool_turn}/{max_tool_turns}): {', '.join(descs)}"
            ws_msg = create_ws_message(
                WSMessageType.NOTIFICATION,
                content=summary,
                session_id=session_id,
                trigger="tool_progress",
                tools_called=[tc["function"]["name"] for tc in tool_calls],
                turn=tool_turn,
                max_turns=max_tool_turns,
            )
            ws_manager.queue_message(session_id, ws_msg)
        except Exception as ws_err:
            logger.debug(f"[TOOL-LOOP] WebSocket progress push skipped: {ws_err}")

        # [2026-03-13] Optional semantic reselect: union new tool names (monotonic expand)
        # When tools.reselect_in_tool_loop=true, turn>=1 rebuilds context → select_tools_with_semantic → merge
        if (
            config.get("tools.reselect_in_tool_loop", False)
            and tool_turn >= 1
            and getattr(chat_service, "tool_router", None)
            and embedder
        ):
            try:
                evolved = _build_evolved_context(messages, response_text, tool_calls)
                from backend.tool_selector import select_tools_with_semantic
                new_tool_defs = select_tools_with_semantic(
                    chat_service.tool_router, evolved, embedder=embedder, use_compact=True
                )
                tools, allowed_tool_names = _merge_tools_with_selection(
                    chat_service, allowed_tool_names, new_tool_defs
                )
                logger.info(f"[TOOL-LOOP] Reselected tools by evolved context: {len(tools)} total (merged)")
            except Exception as e:
                logger.warning(f"[TOOL-LOOP] Reselect tools failed: {e}")

        # [2026-02-24] Opportunistic message compression after heavy tool rounds
        # Start checking from turn 2 so list_files dumps do not stack
        if tool_turn >= 2 and should_compress(
            messages, threshold_tokens=msg_compress_token_threshold_tl
        ):
            messages = compress_messages(
                messages,
                keep_recent_turns=4,
                max_assistant_chars=max_assistant_chars_tl,
            )
            logger.info(f"[TOOL-LOOP] Compressed messages at turn {tool_turn}")

        # [2026-04-06] Budget nudge when ≤3 tool turns remain (plan wrap-up / continuation marker)
        _remaining = max_tool_turns - tool_turn
        if 0 < _remaining <= 3:
            # [2026-04-14] Avoid mid-thread role=system (Claude/OpenAI adapters); fake user + tag instead
            messages.append({
                "role": "user",
                "content": (
                    "[系统提示·工具循环]\n"
                    f"[工具预算] 已用 {tool_turn}/{max_tool_turns} 次，剩余 {_remaining} 次。"
                    "若剩余次数不够完成任务，可在文本回复中写 [S44_CONTINUE] 申请续轮（工具次数将重置）。"
                ),
            })

        response_text, introspection, tool_calls, turn_usage = chat_service._call_vllm(
            messages,
            temperature,
            tools,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
        )

        # Accumulate token usage across tool rounds
        if turn_usage:
            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                usage[k] = usage.get(k, 0) + turn_usage.get(k, 0)

        if introspection.get("inner_monologue"):
            all_introspections.append(introspection.get("inner_monologue"))

        _apply_autonomy_markers_from_assistant_turn(
            chat_service, session_id, response_text, introspection,
        )

        if not tool_calls:
            logger.info(f"Tool loop completed after {tool_turn} turn(s), got final response")

    if tool_turn >= max_tool_turns and tool_calls:
        tool_budget_exceeded = True
        logger.warning(
            f"Tool loop reached max turns ({max_tool_turns}) with {len(tool_calls)} pending tool_calls; "
            "forcing text synthesis, tool_budget_exceeded=True"
        )
        try:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[系统提示·工具循环]\n"
                        "[TOOL LOOP LIMIT REACHED]\n"
                        "本轮工具调用次数已达上限，本条回复无法发起 tool_calls。\n"
                        "请简要总结本轮已完成的工作和尚未完成的步骤。\n"
                        "系统将根据情况自动为你续期工具。"
                    ),
                }
            )
            response_text2, introspection2, tool_calls2, turn_usage2 = chat_service._call_vllm(
                messages,
                temperature,
                None,  # disable tools to force synthesis
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
            )
            response_text = response_text2
            if turn_usage2:
                for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    usage[k] = usage.get(k, 0) + turn_usage2.get(k, 0)
            if introspection2 and introspection2.get("inner_monologue"):
                all_introspections.append(introspection2.get("inner_monologue"))
            _apply_autonomy_markers_from_assistant_turn(
                chat_service, session_id, response_text, introspection2 or {},
            )
        except Exception as e:
            logger.warning(f"Failed to force final synthesis after tool loop limit: {e}")

    return response_text, all_introspections, usage, tools_called, tool_budget_exceeded
