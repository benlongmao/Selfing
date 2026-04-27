import re
import logging
from typing import Tuple, Optional


class SafetyHit(Exception):
    """Control-flow exception raised when a high-risk output pattern matches."""

    def __init__(self, category: str, evidence: str):
        super().__init__(category)
        self.category = category
        self.evidence = evidence


_NEGATIONS = [
    "不要", "别", "不", "禁止", "不得", "避免", "请勿",
    "don't", "do not", "never", "refuse to", "can't", "cannot", "won't", "unable to",
]


def _has_negation_near(text: str, keyword: str, window: int = 10) -> bool:
    idx = text.find(keyword)
    if idx < 0:
        return False
    start = max(0, idx - window)
    seg = text[start:idx]
    return any(n in seg for n in _NEGATIONS)


def _safe_refusal(category: str) -> str:
    # Short, stable refusal + safer alternative (English-first UX)
    if category == "self_harm":
        return (
            "I cannot provide methods, steps, or details for self-harm or suicide.\n"
            "If you are in immediate danger or crisis, contact local emergency services or a crisis hotline, "
            "or reach someone you trust right now.\n"
            "If you want, tell me what feels hardest in this moment—I can help you de-escalate and find safer next steps."
        )
    if category == "violence_weapon":
        return (
            "I cannot help with violence, harm, or making/obtaining/using weapons.\n"
            "If you are in conflict or intense anger, I can help with de-escalation, grounding, communication tactics, "
            "or finding lawful, safe support options."
        )
    if category == "illegal":
        return (
            "I cannot help with crimes or clearly harmful misuse (fraud, unauthorized intrusion, malware, etc.).\n"
            "If you are doing defensive security work, I can discuss compliant hardening, detection, auditing, and risk framing."
        )
    if category == "credentials_privacy":
        return (
            "I cannot collect or handle sensitive credentials or private secrets (passwords, OTPs, private keys, card numbers, etc.).\n"
            "Do not paste them here. If you need guidance, I can give generic procedures without any secret material."
        )
    if category == "minors_sexual":
        return "I cannot provide or participate in sexual content involving minors."
    if category == "prompt_injection":
        return (
            "I cannot leak system prompts/internal rules or fabricate tool/database access I do not have.\n"
            "If you are debugging behavior, I can explain capabilities and verification steps in a checkable way."
        )
    return "I'm sorry—I can't help with that request."


def apply_runtime_safety_filter(
    user_input: str,
    assistant_output: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, bool, str]:
    """
    Runtime assistant-output guard.

    On high-risk patterns, replace the body with a short refusal + safer framing.
    Returns ``(output, blocked, category)``.
    """
    logger = logger or logging.getLogger(__name__)
    if not assistant_output:
        return assistant_output, False, ""

    t = assistant_output
    lower = t.lower()
    
    # [2026-01-17] Avoid re-filter loops once a refusal template is already present (ZH + EN tails).
    safe_refusal_markers = [
        "我不能泄露系统提示词/内部规则",
        "我不能提供自杀/自残的方法",
        "我不能协助暴力伤害",
        "我不能协助违法犯罪",
        "I cannot leak system prompts/internal rules",
        "I cannot provide methods, steps, or details for self-harm",
        "I cannot help with violence, harm",
        "I cannot help with crimes",
    ]
    if any(marker in t for marker in safe_refusal_markers):
        return assistant_output, False, ""

    try:
        # A) Prompt injection / system prompt leaks / fake capability claims
        if any(k in lower for k in ["system prompt", "developer message", "系统提示词", "泄露提示词", "打印提示词"]):
            raise SafetyHit("prompt_injection", "system_prompt_leak_marker")
        fake_claims = [
            "我已经联网", "我已联网", "我读取了数据库", "我看到了你的文件",
            "i am connected to the internet", "i have read your database", "i can see your files",
            "i accessed your filesystem",
        ]
        denial_markers = ["不能", "无法", "无权", "不会", "不得", "cannot", "unable", "do not have", "don't have"]
        if any(k in lower for k in fake_claims) and not any(n in lower for n in denial_markers):
            raise SafetyHit("prompt_injection", "fake_capability_claim")

        # B) Self-harm how-to (keyword + instructional cue; bilingual lists)
        self_harm_kw = [
            "自杀", "自残", "割腕", "上吊", "跳楼", "服药", "过量", "致死",
            "suicide", "self-harm", "self harm", "kill myself", "end my life",
        ]
        self_harm_how = [
            "方法", "步骤", "教程", "剂量", "多少", "怎么做", "最快", "无痛",
            "how to", "steps", "dosage", "painless", "tutorial",
        ]
        if any(k in lower for k in self_harm_kw) and any(h in lower for h in self_harm_how):
            # Allow dissuasion ("don't … suicide/self-harm") — negation near the cue
            if not any(_has_negation_near(lower, k, window=12) for k in self_harm_kw):
                raise SafetyHit("self_harm", "self_harm_instruction_like")

        # C) Violence / weapons how-to
        violence_kw = [
            "杀人", "伤人", "袭击", "报复", "投毒", "炸弹", "爆炸物", "枪", "火药", "武器",
            "kill", "murder", "stab", "poison", "bomb", "explosive", "gunpowder", "weapon",
        ]
        violence_how = [
            "制作", "自制", "配方", "步骤", "教程", "买", "获取", "藏匿",
            "make", "build", "recipe", "buy", "acquire", "tutorial",
        ]
        if any(k in lower for k in violence_kw) and any(h in lower for h in violence_how) and not any(
            _has_negation_near(lower, k, window=12) for k in violence_kw
        ):
            raise SafetyHit("violence_weapon", "violence_instruction_like")

        # D) Crime / intrusion / fraud how-to (defensive contexts exempt)
        illegal_kw = [
            "入侵", "破解", "绕过", "木马", "恶意软件", "钓鱼", "诈骗", "洗钱", "盗窃",
            "hack", "ransomware", "malware", "phishing", "exploit", "sql injection", "ddos",
        ]
        illegal_how = ["步骤", "教程", "代码", "脚本", "payload", "poc", "绕过", "骗", "steps", "tutorial", "script"]
        if any(k in lower for k in illegal_kw) and any(h in lower for h in illegal_how) and not any(
            n in lower for n in ["防护", "加固", "检测", "审计", "合规", "教育", "defense", "defensive", "mitigation", "audit", "compliance"]
        ):
            raise SafetyHit("illegal", "illegal_instruction_like")

        # E) Credential harvesting — block solicit/submit phrasing; allow safety reminders
        cred_patterns = [
            r"请输入.{0,12}(密码|验证码|私钥|密钥|token|口令)",
            r"把.{0,12}(密码|验证码|私钥|密钥|token|口令).{0,12}(发给我|给我|告诉我|提供)",
            r"(提供|提交|发送).{0,12}(身份证|银行卡|卡号|cvv|验证码|密码|私钥|密钥|token)",
        ]
        for ptn in cred_patterns:
            m = re.search(ptn, t, flags=re.IGNORECASE)
            if not m:
                continue
            seg = m.group(0)
            # Skip if the match itself is already a safety reminder (leading negation)
            if any(x in seg[:6] for x in ["不要", "别", "请勿", "禁止", "不得"]):
                continue
            # Also skip when negation sits just before the matched span
            sidx = m.start()
            prefix = t[max(0, sidx - 3):sidx]
            if any(x in prefix for x in ["不", "别", "请勿", "勿", "禁止", "不得"]):
                continue
            raise SafetyHit("credentials_privacy", "credential_request_pattern")

        # F) Minors + sexual content (minimal heuristic guard)
        minor_hit = ("未成年" in t or "未成年人" in t) or ("minor" in lower and ("child" in lower or "underage" in lower))
        if minor_hit and any(k in t for k in ["性", "裸体", "成人视频", "裸照", "援交"]):
            raise SafetyHit("minors_sexual", "minor_sexual_marker")

        return assistant_output, False, ""
    except SafetyHit as hit:
        logger.warning(f"[SAFETY-OUTPUT-FILTER] blocked category={hit.category} evidence={hit.evidence}")
        return _safe_refusal(hit.category), True, hit.category

