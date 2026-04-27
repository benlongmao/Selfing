import json
import logging
import re
from typing import List, Dict, Any, Tuple

from backend.chat_service_cleanup import strip_response_artifacts

logger = logging.getLogger(__name__)

RECEIPT_RE = re.compile(r"\brct_[0-9a-f]{16,}\b", re.IGNORECASE)


def extract_receipts_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    receipts: List[str] = []
    for m in messages or []:
        if m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        try:
            obj = json.loads(content)
            rid = obj.get("receipt_id")
            if isinstance(rid, str) and rid:
                receipts.append(rid)
        except Exception:
            # fallback: try regex
            for rid in RECEIPT_RE.findall(content):
                receipts.append(rid)
    # de-dup stable order
    seen = set()
    out = []
    for r in receipts:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _looks_like_tool_claim(text: str) -> bool:
    if not text:
        return False
    # Fix C: precise triggers; avoid false positives on metaphors (e.g. literary "searched my mind").
    markers = [
        # Explicit tool references
        "工具输出", "工具查询", "工具返回", "根据工具", "使用工具", "calling tool", "tool output",

        # Concrete web-search claims
        "我搜索了网络", "我查询了网络", "我从网上查到", "联网搜索", "通过搜索",
        "searched the web", "browsed the internet", "online search",

        # Concrete file read/list claims
        "我读取了文件", "我查看了文件", "文件内容显示", "读取本地文件", "查看目录",
        "read the file", "checked the file", "listed files",

        # Concrete write/delete claims
        "我已写入文件", "我保存了文件", "创建了文件", "删除了文件",
        "文件已成功创建", "文件已保存", "成功创建", "成功保存", "已创建",
        "wrote to file", "saved to file", "created file", "deleted file",
        "file created", "file saved",

        # Technical function names (strong signal)
        "tavily_search", "list_files", "read_file", "write_file", "rename_file", "get_self_facts",
    ]
    
    lower = text.lower()
    
    # 1) Strong substring markers (mixed-language list by design)
    if any(m in text for m in markers):
        return True

    # 2) English technical cue — almost always a receipt reference
    if "receipt_id" in lower:
        return True
        
    return False


def enforce_receipts(response_text: str, receipts: List[str]) -> Tuple[str, bool]:
    """
    Minimal implementation (P2.7).

    - If the reply looks like a tool / hard-fact self-claim but has no ``rct_*`` receipt ids,
      append a receipt list at the end.
    - Tool-claim detection must use **user-visible** text after ``strip_response_artifacts``.
      If the claim only lives in stripped channels (<thought>, stream-of-consciousness tags,
      ``[Pineal Check]``, DSML, etc.), do **not** append receipts (or the user would only see
      a dangling ``Receipts: rct_…`` line post-sanitize).
    - Returns ``(new_text, modified)``.
    """
    if not response_text:
        return response_text, False
    if not receipts:
        return response_text, False

    # already contains receipt
    if RECEIPT_RE.search(response_text):
        return response_text, False

    visible_user = strip_response_artifacts(response_text, logger=None)
    if not _looks_like_tool_claim(visible_user):
        if _looks_like_tool_claim(response_text) and visible_user != (response_text or "").strip():
            logger.info(
                "[RESPONSE-VERIFIER] Skip appending receipts: tool-like wording only in stripped "
                "channels (thought/stream-of-consciousness/Pineal/DSML/etc.); user-visible text has no such claim"
            )
        return response_text, False

    receipt_line = "Receipts: " + ", ".join(receipts[:6]) + (" ..." if len(receipts) > 6 else "")
    # keep concise: append as last line
    new_text = response_text.rstrip() + "\n\n" + receipt_line
    logger.info(
        "[RESPONSE-VERIFIER] Appended receipt line (visible reply claims tool/facts without rct_)"
    )
    return new_text, True

