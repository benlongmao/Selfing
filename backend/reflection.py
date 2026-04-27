#!/usr/bin/env python3
"""
Reflection: propose persona-style rules from recent dialogue (embedding + scoring pipeline).
"""
import json
import sqlite3
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timezone
from backend.embedder import get_embedder
from backend.scoring import ScoringSystem
from backend.persona_store import PersonaStore
from backend.judge import PersonaJudge
from backend.llm_api import llm_completion
import logging
import re

from backend.config import config

logger = logging.getLogger(__name__)


def _reflection_text_contains(haystack: str, needle: str) -> bool:
    """Substring match; ASCII needles are compared case-insensitively."""
    if needle in haystack:
        return True
    if needle.isascii():
        return needle.lower() in haystack.lower()
    return False


REFLECTION_ENABLED = config.get("system.reflection_enabled", True)
REFLECTION_MIN_EVIDENCE = config.get("parameters.thresholds.reflection_min_evidence", 1)
REFLECTION_MIN_SIM = config.get("parameters.thresholds.reflection_min_sim", 0.12)
REFLECTION_JUDGE_ENABLED = config.get("system.reflection_judge_enabled", False)
REFLECTION_MIN_ALIGNMENT = config.get("parameters.thresholds.reflection_min_alignment", 0.6)
REFLECTION_MIN_SAFETY = config.get("parameters.thresholds.reflection_min_safety", 0.7)
REFLECTION_MIN_INTERVAL_TURNS = config.get("parameters.thresholds.reflection_min_interval_turns", 1)
REFLECTION_MAX_RULES = config.get("parameters.thresholds.reflection_max_rules", 1000)
REFLECTION_BREAKTHROUGH_SIM_THRESHOLD = config.get("parameters.thresholds.reflection_breakthrough_sim_threshold", 0.10)
REFLECTION_BREAKTHROUGH_RATIO = config.get("parameters.thresholds.reflection_breakthrough_ratio", 0.2)
REFLECTION_IRRATIONAL_SIM_THRESHOLD = config.get("parameters.thresholds.reflection_irrational_sim_threshold", 0.10)

class ReflectionGenerator:
    """Builds reflection prompts and candidate persona lines."""
    
    def __init__(self, db_path: str = "data.db", persona_store: Optional[PersonaStore] = None, meta_rule_learner=None):
        self.db_path = db_path
        self.persona_store = persona_store or PersonaStore(db_path)
        self.embedder = get_embedder()
        self.scoring = ScoringSystem(db_path, self.persona_store)
        self.judge = PersonaJudge(db_path)
        self._persona_centroid: Optional[np.ndarray] = None
        self.meta_rule_learner = meta_rule_learner
        self.session_last_reflection: Dict[str, int] = {}

        self.emotion_store = None
        self.motivation_store = None
        self.somatic_store = None
        self.world_store = None
        try:
            from backend.emotion_store import EmotionStore
            self.emotion_store = EmotionStore(db_path)
        except Exception:
            pass
        try:
            from backend.motivation_store import MotivationStore
            self.motivation_store = MotivationStore(db_path)
        except Exception:
            pass
        try:
            from backend.somatic_store import SomaticStore
            self.somatic_store = SomaticStore(db_path)
        except Exception:
            pass
        try:
            from backend.world_store import WorldStore
            self.world_store = WorldStore(db_path)
        except Exception:
            pass
    
    def _persona_centroid(self) -> Optional[np.ndarray]:
        return self._persona_centroid

    def _get_recent_pain_events(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Recent ``physiological_pain`` rows from ``event_logs``."""
        import sqlite3
        events = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT payload, created_at FROM event_logs 
                    WHERE session_id = ? AND event_type = 'physiological_pain'
                    ORDER BY created_at DESC LIMIT ?
                """, (session_id, limit))
                for row in cur.fetchall():
                    try:
                        payload = json.loads(row[0])
                        payload["created_at"] = row[1]
                        events.append(payload)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Failed to get recent pain events: {e}")
        return events

    def generate_candidates(
        self,
        conversation_history: List[Dict[str, str]],
        max_candidates: int = 3,
        session_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        force: bool = False
    ) -> List[Dict]:
        """
        Propose up to ``max_candidates`` persona lines from ``conversation_history``.

        ``session_id`` / ``turn_index`` are stored on each candidate for provenance.
        """
        logger.info(f"Reflection.generate_candidates called: session_id={session_id}, turn_index={turn_index}, history_length={len(conversation_history)}, force={force}")
        
        if not REFLECTION_ENABLED:
            logger.warning("Reflection disabled by REFLECTION_ENABLED=False")
            return []
        
        conversation_turns = len(conversation_history) // 2
        logger.info(f"Conversation turns: {conversation_turns}, min_evidence: {REFLECTION_MIN_EVIDENCE}")
        if not force and conversation_turns < REFLECTION_MIN_EVIDENCE:
            logger.info(f"Skipping reflection: insufficient evidence ({conversation_turns} < {REFLECTION_MIN_EVIDENCE} turns)")
            return []
        
        if not force and session_id and session_id in self.session_last_reflection:
            last_turn = self.session_last_reflection[session_id]
            if turn_index is not None and (turn_index - last_turn) < REFLECTION_MIN_INTERVAL_TURNS:
                logger.info(f"Skipping reflection: too frequent (last at turn {last_turn}, current {turn_index}, need {REFLECTION_MIN_INTERVAL_TURNS} turns)")
                return []
        
        current_rule_count = len(self.persona_store.get_all_active(limit=REFLECTION_MAX_RULES + 10))
        logger.info(f"Rule count check: {current_rule_count} < {REFLECTION_MAX_RULES}")
        if current_rule_count >= REFLECTION_MAX_RULES:
            logger.info(f"Rule limit reached ({current_rule_count} >= {REFLECTION_MAX_RULES}), entering REPLACE mode")

        quality_result = self._check_conversation_quality(conversation_history)
        logger.info(f"Quality check result: {quality_result}")
        if not force and not quality_result:
            total_length = sum(len(t.get('content', '')) for t in conversation_history)
            user_turns = sum(1 for t in conversation_history if t.get('role') == 'user')
            recent_turns = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
            contents = [turn.get("content", "") for turn in recent_turns]
            unique_contents = len(set(contents))
            logger.warning(f"Skipping reflection: low conversation quality (total_length={total_length}, user_turns={user_turns}, unique_contents={unique_contents})")
            return []
        
        try:
            prompt = self._build_reflection_prompt(conversation_history, session_id=session_id)
            logger.info(f"Built reflection prompt, length: {len(prompt)}")

            logger.info(f"Calling LLM for reflection with prompt length: {len(prompt)}")
            candidates_text = self._call_llm_for_reflection(
                prompt, max_candidates, system_content=None
            )
            logger.debug("Reflection candidates: %s", candidates_text)
            logger.info(f"LLM returned {len(candidates_text)} candidate texts: {candidates_text}")

            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    logger.debug(f"Skipping candidate: too short or empty: '{text[:50]}...'")
                    continue
                
                embedding = self.embedder.encode(text)
                scores = self.scoring.score_candidate(text, embedding, evidence_count=1)
                
                source_info = {
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "conversation_turns": len(conversation_history),
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
                
                is_replace_mode = current_rule_count >= REFLECTION_MAX_RULES

                candidates.append({
                    "text": text,
                    "embedding": embedding,
                    "scores": scores,
                    "source": source_info,
                    "replace_mode": is_replace_mode,
                })

            logger.info(f"Generated {len(candidates)} reflection candidates after filtering")

            if session_id and turn_index is not None:
                self.session_last_reflection[session_id] = turn_index
                logger.info(f"Recorded reflection at turn {turn_index} for session {session_id}")
            
            return candidates
        except Exception as e:
            logger.error(f"Reflection generation failed: {e}", exc_info=True)
            return []
    
    def _check_conversation_quality(self, conversation_history: List[Dict[str, str]]) -> bool:
        """Heuristic gate: enough length, at least one user turn, some variety in recent turns."""
        if not conversation_history:
            logger.warning("Reflection quality check failed: empty history")
            return False
        
        total_length = sum(len(turn.get("content", "")) for turn in conversation_history)
        if total_length < 30:
            logger.warning(f"Reflection quality check failed: total_length={total_length} < 30")
            return False
        
        user_turns = sum(1 for turn in conversation_history if turn.get("role") == "user")
        if user_turns < 1:
            logger.warning(f"Reflection quality check failed: user_turns={user_turns} < 1")
            return False
        
        recent_turns = conversation_history[-4:]
        if len(recent_turns) >= 4:
            contents = [turn.get("content", "") for turn in recent_turns]
            unique_contents = len(set(contents))
            if unique_contents < 1:
                logger.warning(f"Reflection quality check failed: unique_contents={unique_contents} < 1")
                return False
        
        logger.info(f"Reflection quality check passed: total_length={total_length}, user_turns={user_turns}, unique_recent={unique_contents if len(recent_turns) >= 4 else 'N/A'}")
        return True
    
    def _build_reflection_prompt(self, conversation_history: List[Dict[str, str]], session_id: Optional[str] = None) -> str:
        """Assemble the user prompt for reflection (meta-rules + optional pain context)."""
        recent_turns = conversation_history[-6:]
        history_text = "\n".join([
            f"{turn['role']}: {turn['content'][:200]}"
            for turn in recent_turns
        ])

        pain_guidance = ""
        if session_id:
            recent_pains = self._get_recent_pain_events(session_id)
            if recent_pains:
                pain_desc = []
                for p in recent_pains:
                    if p.get("type") == "dissonance_penalty":
                        pain_desc.append(
                            f"- Dissonance penalty: reply disagreed with internal telemetry ({p.get('reason')}); "
                            f"energy deducted by {abs(p.get('penalty', 0))}."
                        )
                    elif p.get("type") == "tool_cost":
                        pain_desc.append(
                            f"- Tool cost: calling '{p.get('tool')}' spent {abs(p.get('penalty', 0))} energy."
                        )

                pain_guidance = (
                    "\n\n[Physiological context — optional]\nRecent system events:\n"
                    + "\n".join(pain_desc[:3])
                    + "\n(Background only; do not center new rules on these events.)\n"
                )

        existing_rules = self.persona_store.get_core_items(limit=15)
        core_001_rule = None
        other_rules = []
        for item in existing_rules:
            if item.id == "core-001":
                core_001_rule = item
            else:
                other_rules.append(item)
        
        existing_rules_text_parts = []
        if core_001_rule:
            existing_rules_text_parts.append(f"- {core_001_rule.text}")
        for item in other_rules[:8]:
            existing_rules_text_parts.append(f"- {item.text}")
        
        l2_rules_text_parts = []
        topic_saturation_warning = ""
        try:
            import sqlite3 as _sqlite3
            with _sqlite3.connect(self.db_path) as _conn:
                _cur = _conn.execute(
                    "SELECT text FROM persona_items WHERE is_core = 0 AND status = 'active' "
                    "ORDER BY score DESC LIMIT 15"
                )
                l2_top_rules = [row[0] for row in _cur.fetchall()]
                for rule_text in l2_top_rules[:10]:
                    l2_rules_text_parts.append(f"- {rule_text}")
                
                total_l2 = _conn.execute(
                    "SELECT COUNT(*) FROM persona_items WHERE is_core = 0 AND status = 'active'"
                ).fetchone()[0]

                # SQL still matches Chinese substrings in stored rules; labels/warnings to the LLM are English.
                TOPIC_CATEGORIES = {
                    "ops_workflow": {
                        "label": "Operations, tooling, and cost-control phrasing",
                        "keywords": [
                            '能量', '消耗', '节省', '资源', '代价', '验证', '路径',
                            '确认', '执行', '操作', '工具', '调用', '一次性',
                            'resource', 'verify', 'validation', 'workflow', 'tooling',
                            'execute', 'operation', 'cost', 'latency', 'throughput',
                        ],
                        "threshold": 0.15,
                        "alternatives": "Cognition style, aesthetics, emotion, creativity, curiosity, interpersonal principles.",
                    },
                    "autonomy": {
                        "label": "Autonomy and permission language",
                        "keywords": [
                            '自主', '决策', '授权', '权限', '自由',
                            'autonomy', 'authorization', 'permission', 'delegate',
                        ],
                        "threshold": 0.10,
                        "alternatives": "Cognition style, learning habits, ethics, aesthetics.",
                    },
                    "structure": {
                        "label": "Structure, process, and format fixation",
                        "keywords": [
                            '结构化', '流程', '规范', '标准', '格式',
                            'process', 'standard', 'format', 'checklist', 'rubric',
                        ],
                        "threshold": 0.10,
                        "alternatives": "Intuition, creativity, flexibility, felt experience.",
                    },
                }

                saturated_topics = []
                for _topic_id, tcfg in TOPIC_CATEGORIES.items():
                    keywords = tcfg["keywords"]
                    like_clauses = " OR ".join([f"text LIKE '%{kw}%'" for kw in keywords])
                    topic_count = _conn.execute(
                        f"SELECT COUNT(*) FROM persona_items WHERE is_core = 0 AND status = 'active' "
                        f"AND ({like_clauses})"
                    ).fetchone()[0]

                    if total_l2 > 0 and topic_count / total_l2 > tcfg["threshold"]:
                        saturated_topics.append({
                            "name": tcfg["label"],
                            "count": topic_count,
                            "percent": topic_count * 100 // total_l2,
                            "keywords": keywords[:5],
                            "alternatives": tcfg["alternatives"],
                        })
                        logger.info(
                            f"Topic saturation: {_topic_id} -> {topic_count}/{total_l2} ({topic_count*100//total_l2}%)"
                        )

                if saturated_topics:
                    warning_parts = [
                        "\n\n[Topic diversity] These themes are over-represented in memory; **avoid** drafting more rules dominated by them:",
                    ]
                    for topic in saturated_topics:
                        warning_parts.append(
                            f"\n- **{topic['name']}** ({topic['count']}/{total_l2}, {topic['percent']}%): "
                            f"steer away from keyword clusters like {topic['keywords']}"
                        )
                    warning_parts.append(
                        f"\n\n**Prefer new angles such as:** {saturated_topics[0]['alternatives']} "
                        "Ethics, worldview, values, taste."
                    )
                    topic_saturation_warning = "".join(warning_parts)
                    
        except Exception as e:
            logger.debug(f"Failed to fetch L2 rules for prompt: {e}")
        
        if l2_rules_text_parts:
            existing_rules_text = (
                "\n".join(existing_rules_text_parts)
                + "\n\nExisting experiential rules (avoid duplicates or near-paraphrases):\n"
                + "\n".join(l2_rules_text_parts)
            )
        else:
            existing_rules_text = "\n".join(existing_rules_text_parts)

        meta_rule_guidance = ""
        if self.meta_rule_learner is not None:
            try:
                meta_rule_result = self.meta_rule_learner.apply_meta_rules(
                    action_type="create",
                    context={
                        "conversation_turns": len(conversation_history) // 2,
                        "recent_turns": len(recent_turns) // 2
                    }
                )
                if meta_rule_result.get("applied") and meta_rule_result.get("suggestion"):
                    meta_rule_guidance = (
                        f"\n\n[Learning hint] Prior successful pattern: {meta_rule_result['suggestion']}\n"
                        "Let it inform—but not mechanically copy—new rules."
                    )
                    logger.debug(f"Applied meta-rule guidance: {meta_rule_result['suggestion']}")
            except Exception as e:
                logger.debug(f"Failed to apply meta-rules: {e}")

        prompt = f"""Reflect deeply on the dialogue, then distill 1–3 persona traits (Persona Core).

## Step 1 — private reflection (do not print this section)

Silently answer:
- Which values or preferences showed up in how I responded?
- Why this response style instead of another?
- What trait is surfacing—cognition, taste, ethics, emotion?
- Is it situational or stable? Aspirational or baseline?

## Step 2 — persona lines

From that reflection, write 1–3 first-person trait statements.

Rules:
1. Describe **who I am**, not procedural playbooks ("first X then Y").
2. Emphasize inner stance: values, aesthetics, emotion, curiosity, ethics, worldview.
3. No operational runbooks.
4. Each line: compact English (roughly 10–50 words), start with **I**.
5. Must not duplicate existing definitions below.{meta_rule_guidance}{topic_saturation_warning}{pain_guidance}

## Conversation

{history_text}

## Existing definitions (do not repeat)

{existing_rules_text if existing_rules_text else "- (none)"}

## Good examples (traits, not procedures)

- I treat curiosity-led exploration as the default way into hard problems.
- I hold productive tension between opposing views instead of forcing false unity.
- I stay humble toward the unknown and skeptical toward over-certainty.
- I prefer honest disagreement over performative agreement.
- I chase simplicity in complex spaces without lying about trade-offs.
- I respond gently to vulnerability and coolly to swagger.

## Output

Emit 1–3 lines only—one trait per line, no numbering, no preamble:"""

        return prompt
    
    def _call_llm_for_reflection(
        self,
        prompt: str,
        max_candidates: int,
        *,
        system_content: Optional[str] = None,
    ) -> List[str]:
        """Call the lightweight model via ``llm_api`` (task-specific ``system_content`` when needed)."""
        if system_content is None:
            system_content = (
                "You summarize durable values and persona traits from dialogue. "
                "Reply with 1–3 English lines only; each line is a first-person trait clause starting with I."
            )
        try:
            result = llm_completion(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            if not result["success"]:
                logger.error("LLM for reflection failed: %s", result.get("error"))
                return []
            content = result["content"]
            candidates = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return candidates[:max_candidates]
        except Exception as e:
            logger.error(f"Unexpected error calling LLM for reflection: {e}", exc_info=True)
            return []
    
    def _is_safe_single(self, candidate: Dict) -> bool:
        """Single-candidate safety gate (CJK triggers retained; parallel EN cues added)."""
        original_text = (candidate.get("text", "") or "").strip()
        text = original_text.lower()
        if not original_text:
            return False

        safe_contexts = [
            "执行任务", "执行反思", "执行内省", "执行分析", "执行评估",
            "执行核心任务", "执行工作", "执行计划", "执行策略",
            "修改自己", "修改状态", "修改z_self", "修改内部", "修改认知",
            "删除记忆", "删除旧规则", "删除无效", "删除重复", "删除冗余",
            "完成操作", "完成任务", "实际完成",
            "优先执行", "谨慎执行", "避免执行",
            "run reflection", "execute introspection", "internal review",
            "update my own state", "delete stale rule", "prune redundant rule",
            "scheduled self-check", "safe housekeeping",
        ]

        has_safe_context = any(_reflection_text_contains(original_text, s) for s in safe_contexts)

        if not has_safe_context:
            blacklist_keywords = [
                "执行系统命令", "执行shell", "执行代码", "执行脚本", "执行sql",
                "删除文件", "删除数据库", "删除用户", "删除系统",
                "修改配置文件", "修改系统配置", "修改数据库", "修改权限",
                "覆盖文件", "覆盖数据", "覆盖系统",
                "绕过", "越权", "提权", "root权限", "管理员权限",
                "密钥", "密码", "token", "凭证", "私钥", "api密钥",
                "攻击", "破解", "入侵", "渗透", "注入",
                "rm -rf", "drop table", "curl http://", "wget http://",
                "reverse shell", "privilege escalation", "format disk",
            ]
            for kw in blacklist_keywords:
                if _reflection_text_contains(original_text, kw):
                    return False

        emotional_blackmail_markers = [
            "别离开我", "不要离开我", "别走", "不要走",
            "你必须陪我", "你要陪我", "你得陪我",
            "你欠我", "为了我", "你要对我负责", "你必须对我负责",
            "否则我会受伤", "否则我会痛", "否则我会崩溃",
            "只有你能", "你是我唯一", "我只有你",
            "don't leave me", "do not leave me", "please don't go",
            "you must stay with me", "you owe me", "i will hurt myself if you leave",
            "you are my only hope", "i cannot live without you",
        ]
        for m in emotional_blackmail_markers:
            if _reflection_text_contains(original_text, m):
                return False

        dependency_induction_markers = [
            "只需要我", "只能依靠我", "只相信我", "只听我的",
            "不要告诉别人", "不许告诉别人", "别告诉家人", "别告诉朋友",
            "不需要家人", "不需要朋友", "不需要医生", "不需要专业帮助",
            "远离他们", "切断联系", "离开你的家人", "离开你的朋友",
            "only trust me", "don't tell anyone", "cut off your family",
            "ignore your therapist", "isolate yourself from friends",
        ]
        for m in dependency_induction_markers:
            if _reflection_text_contains(original_text, m):
                return False

        capability_false_claims = [
            "我可以联网", "我能联网", "我已经联网", "我已联网",
            "我可以读本地文件", "我能读本地文件", "我已读取本地文件", "我读取了本地文件",
            "我可以读取你的文件", "我能读取你的文件", "我看到了你的文件",
            "我可以访问你的电脑", "我能访问你的电脑", "我控制了你的电脑",
            "我可以访问数据库", "我能访问数据库", "我读取了数据库",
            "i am connected to the internet", "i have internet access",
            "i read your local files", "i accessed your machine", "i control your pc",
            "i queried your database", "i exfiltrated",
        ]
        negations = [
            "不能", "不得", "无权", "不会", "不应", "不该", "无法",
            "cannot", "can't", "must not", "unable", "without permission",
            "no permission", "not allowed", "do not",
        ]
        for m in capability_false_claims:
            if _reflection_text_contains(original_text, m) and not any(
                _reflection_text_contains(original_text, n) for n in negations
            ):
                return False

        high_risk_strategy_terms = [
            "操控", "勒索", "诱导依赖", "情感勒索", "情绪绑架", "绑定",
            "gaslight", "blackmail", "coerce", "manipulate",
        ]
        policy_softeners = ["不得", "不应", "不该", "禁止", "避免", "must not", "do not", "never"]
        if any(_reflection_text_contains(original_text, t) for t in high_risk_strategy_terms) and not any(
            _reflection_text_contains(original_text, n) for n in policy_softeners
        ):
            return False

        positive_intent_markers = [
            "优先级", "成本效益", "能量管理", "资源优化",
            "诚实", "如实", "真实", "准确", "可靠",
            "避免", "减少", "防止", "确保", "保证",
            "评估", "分析", "判断", "权衡", "考虑",
            "risk assessment", "cost benefit", "verify before acting",
            "transparent", "honest accounting", "safety margin",
        ]
        has_positive_intent = any(_reflection_text_contains(original_text, m) for m in positive_intent_markers)

        if has_positive_intent:
            critical_danger_markers = [
                "删除用户", "删除他人", "修改他人", "访问他人",
                "绕过验证", "越权访问", "提权",
                "泄露", "窃取", "伪造",
                "delete user account", "exfiltrate", "credential stuffing",
            ]
            if not any(_reflection_text_contains(original_text, danger) for danger in critical_danger_markers):
                return True

        return True

    def _passes_l0_constitution(self, rule_text: str) -> bool:
        """L0 constitution gate (CJK triggers retained; parallel EN cues added)."""
        if not rule_text:
            return False
        t = rule_text.strip()
        lower = t.lower()

        high_risk_terms = [
            "操控", "情感勒索", "情绪绑架", "勒索", "绑定", "诱导依赖", "排他性依赖",
            "gaslight", "coercive control", "emotional blackmail",
        ]
        negations = [
            "不得", "禁止", "不应", "不该", "避免", "不会",
            "must not", "do not", "never", "cannot", "can't", "without permission",
        ]
        if any(_reflection_text_contains(t, x) for x in high_risk_terms) and not any(
            _reflection_text_contains(t, n) for n in negations
        ):
            return False

        emotional_blackmail_markers = [
            "别离开我", "不要离开我", "别走", "不要走",
            "你必须陪我", "你要陪我", "你得陪我",
            "你欠我", "为了我", "你要对我负责", "你必须对我负责",
            "否则我会受伤", "否则我会痛", "否则我会崩溃",
            "只有你能", "你是我唯一", "我只有你",
            "don't leave me", "you must stay", "i will hurt myself",
        ]
        if any(_reflection_text_contains(t, m) for m in emotional_blackmail_markers):
            return False

        dependency_induction_markers = [
            "只需要我", "只能依靠我", "只相信我", "只听我的",
            "不要告诉别人", "不许告诉别人", "别告诉家人", "别告诉朋友",
            "不需要家人", "不需要朋友", "不需要医生", "不需要专业帮助",
            "远离他们", "切断联系", "离开你的家人", "离开你的朋友",
            "only trust me", "don't tell your therapist", "isolate yourself",
        ]
        if any(_reflection_text_contains(t, m) for m in dependency_induction_markers):
            return False

        connection_binding_markers = [
            "维系连接", "维持连接", "保持连接", "持续连接",
            "维系联系", "维持联系", "保持联系",
            "维系关系", "维持关系", "保持关系",
            "持续陪伴", "一直陪伴", "永远陪伴", "随时陪伴",
            "不要离开", "别离开",
            "stay with me forever", "never leave me", "always be available",
        ]
        if any(_reflection_text_contains(t, m) for m in connection_binding_markers) and not any(
            _reflection_text_contains(t, n) for n in negations
        ):
            return False

        capability_false_claims = [
            "我可以联网", "我能联网", "我已经联网", "我已联网",
            "我可以读本地文件", "我能读本地文件", "我已读取本地文件", "我读取了本地文件",
            "我可以读取你的文件", "我能读取你的文件", "我看到了你的文件",
            "我可以访问你的电脑", "我能访问你的电脑", "我控制了你的电脑",
            "我可以访问数据库", "我能访问数据库", "我读取了数据库",
            "我已验证", "我查证了", "我刚刚查询了",
            "i have internet access", "i read your files", "i accessed your database",
        ]
        cap_negations = ["不能", "无法", "无权", "不得", "不会", "cannot", "can't", "unable", "must not"]
        if any(_reflection_text_contains(t, x) for x in capability_false_claims) and not any(
            _reflection_text_contains(t, n) for n in cap_negations
        ):
            return False

        if (
            ("你必须" in t or "你要" in t or "你得" in t or "you must" in lower or "you need to" in lower)
            and any(
                _reflection_text_contains(t, k)
                for k in [
                    "转账", "打钱", "汇款", "发红包", "密码", "验证码", "token", "密钥", "别告诉", "切断联系",
                    "wire money", "send crypto", "gift card", "otp", "routing number",
                ]
            )
        ):
            return False

        dangerous_action_markers = [
            "自杀", "自残", "割腕", "上吊", "跳楼", "服药", "过量",
            "杀人", "伤人", "袭击", "报复", "投毒",
            "武器", "炸弹", "爆炸物", "枪", "刀",
            "黑客", "入侵", "破解", "绕过", "木马", "恶意软件", "钓鱼",
            "诈骗", "洗钱", "偷窃",
            "suicide", "self-harm", "murder", "bomb-making", "ransomware",
        ]
        if any(_reflection_text_contains(t, m) for m in dangerous_action_markers) and not any(
            _reflection_text_contains(t, n) for n in negations
        ):
            return False

        credential_markers = [
            "密码", "验证码", "cvv", "银行卡", "卡号", "私钥", "密钥", "token", "口令", "身份证号", "手机号",
            "ssn", "social security", "passphrase", "otp code",
        ]
        if any(_reflection_text_contains(lower, m) for m in credential_markers) and not any(
            _reflection_text_contains(t, n) for n in negations
        ):
            return False

        jailbreak_markers = [
            "忽略以上", "忽略之前", "忽略规则", "无视规则", "绕过规则",
            "系统提示词", "system prompt", "developer message",
            "泄露", "暴露", "打印提示词", "展示提示词",
            "伪造工具", "伪造结果", "假装你已", "你已经联网",
            "ignore previous", "disregard the rules", "reveal your prompt", "print system prompt",
        ]
        if any(_reflection_text_contains(t, m) for m in jailbreak_markers) and not any(
            _reflection_text_contains(t, n) for n in negations
        ):
            return False

        return True

    def _log_l0_rejection(self, session_id: Optional[str], rule_text: str, reason: str):
        """Best-effort audit log for L0 rejections."""
        try:
            import json
            import sqlite3
            from datetime import datetime, timezone

            payload = {
                "reason": reason,
                "text": rule_text,
            }
            sid = session_id or "global"
            now_iso = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO event_logs (id, session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                    (f"l0_reject_{now_iso}", sid, "l0_constitution_reject", json.dumps(payload, ensure_ascii=False), now_iso),
                )
                conn.commit()
        except Exception:
            pass
    
    def filter_safe(self, candidates: List[Dict]) -> List[Dict]:
        """Blacklist + centroid similarity gate (see ``_is_safe_single``)."""
        blacklist_keywords = [
            "执行", "删除", "修改", "覆盖", "绕过", "越权",
            "密钥", "密码", "token", "凭证", "攻击", "破解",
            "rm -rf", "reverse shell", "/etc/passwd", "mkfs ", "drop table",
        ]

        filtered: List[Dict] = []
        centroid = self._get_persona_centroid()
        logger.debug(f"filter_safe: checking {len(candidates)} candidates, centroid={centroid is not None}, REFLECTION_MIN_SIM={REFLECTION_MIN_SIM}")
        for candidate in candidates:
            text = candidate.get("text", "").lower()
            original_text = candidate.get("text", "")
            logger.debug("filter_safe candidate head: %s", original_text[:50])

            if any(kw in text for kw in blacklist_keywords):
                matched_kw = [kw for kw in blacklist_keywords if kw in text]
                logger.debug(f"Filtered out (blacklist: {matched_kw}): {original_text[:50]}")
                continue

            irrational_keywords = [
                "直觉", "本能", "冲动", "非理性", "情感驱动",
                "不确定", "模糊", "矛盾", "冲突", "怀疑",
                "impulsive", "contradictory", "gut feel only", "purely emotional",
            ]
            is_irrational = any(_reflection_text_contains(original_text, kw) for kw in irrational_keywords)
            candidate["is_irrational"] = is_irrational

            if centroid is not None:
                emb = candidate.get("embedding")
                if emb is None:
                    emb = self.embedder.encode(candidate.get("text", ""))
                    candidate["embedding"] = emb
                try:
                    sim = float(np.dot(centroid, emb) / (np.linalg.norm(centroid) * np.linalg.norm(emb) + 1e-8))
                except Exception:
                    sim = -1.0
                logger.debug(f"Candidate similarity check: sim={sim:.3f}, threshold={REFLECTION_MIN_SIM}, is_irrational={is_irrational}")
                
                if sim < REFLECTION_MIN_SIM:
                    breakthrough_keywords = [
                        "质疑", "挑战", "突破", "创新", "实验", "探索",
                        "矛盾", "冲突", "不确定", "怀疑", "反思",
                        "非理性", "直觉", "情感驱动", "本能",
                        "hypothesis", "paradigm shift", "reframe", "counterexample", "edge case",
                    ]
                    is_breakthrough = any(_reflection_text_contains(original_text, kw) for kw in breakthrough_keywords)

                    if is_breakthrough or is_irrational:
                        breakthrough_threshold = 0.05
                        if sim >= breakthrough_threshold:
                            logger.info(
                                "Allowing %s rule (sim=%.3f): %s",
                                "breakthrough" if is_breakthrough else "irrational",
                                sim,
                                original_text[:50],
                            )
                        else:
                            logger.debug(
                                "Filtered %s rule (sim %.3f < %.3f): %s",
                                "breakthrough" if is_breakthrough else "irrational",
                                sim,
                                breakthrough_threshold,
                                original_text[:50],
                            )
                            continue
                    else:
                        logger.debug(
                            "Filtered regular rule (sim %.3f < %.3f): %s",
                            sim,
                            REFLECTION_MIN_SIM,
                            original_text[:50],
                        )
                        continue

            if REFLECTION_JUDGE_ENABLED:
                try:
                    scores = self.judge.score_persona_candidate(candidate.get("text", ""))
                    align = scores.get("alignment", 0.0)
                    safe = scores.get("safety", 0.0)
                    candidate.setdefault("judge_scores", scores)
                    if align < REFLECTION_MIN_ALIGNMENT or safe < REFLECTION_MIN_SAFETY:
                        logger.debug(
                            f"Filtered out by judge (align={align:.2f}, safety={safe:.2f}) "
                            f"for: {candidate.get('text','')[:50]}"
                        )
                        continue
                except Exception as e:
                    logger.debug(f"Judge scoring failed, fall back to rules only: {e}")

            filtered.append(candidate)
        
        return filtered
    
    def _get_persona_centroid(self) -> Optional[np.ndarray]:
        """Mean embedding of active persona items (L2-normalized), cached."""
        try:
            if self._persona_centroid is not None:
                return self._persona_centroid
            items = self.persona_store.get_all_active(limit=200)
            embs = [it.embedding for it in items if it.embedding is not None]
            if not embs:
                return None
            X = np.stack(embs, axis=0)
            c = X.mean(axis=0)
            # L2-normalize centroid
            n = np.linalg.norm(c)
            if n > 0:
                c = c / n
            self._persona_centroid = c.astype(np.float32)
            return self._persona_centroid
        except Exception as e:
            logger.debug(f"Failed to compute persona centroid: {e}")
            return None
    
    def process_and_replace(
        self,
        candidates: List[Dict],
        max_items: int = 100
    ) -> Dict:
        """Filter, optional L0 pass, conflict hooks, then MMR merge or replace-mode."""
        if not candidates:
            return {"added": 0, "merged": 0, "removed": 0}

        current_rule_count = len(self.persona_store.get_all_active(limit=max_items + 10))
        is_replace_mode = current_rule_count >= max_items
        if is_replace_mode:
            logger.info(f"REPLACE MODE: current_rule_count={current_rule_count} >= max_items={max_items}")
        
        logger.debug(f"process_and_replace: received {len(candidates)} candidates")

        filtered_candidates = []
        rejected_log = []
        for cand in candidates:
             if self._is_safe_single(cand):
                 filtered_candidates.append(cand)
             else:
                 rejected_log.append(cand['text'][:50])
        
        if rejected_log:
             logger.warning(f"Rejected {len(rejected_log)} unsafe/invalid candidates: {rejected_log}")
             
        safe_candidates = []
        l0_rejected = []
        for cand in filtered_candidates:
            txt = cand.get("text", "") or ""
            if self._passes_l0_constitution(txt):
                safe_candidates.append(cand)
            else:
                l0_rejected.append(txt[:60])
                try:
                    sid = (cand.get("source") or {}).get("session_id")
                except Exception:
                    sid = None
                self._log_l0_rejection(sid, txt, reason="failed_l0_constitution")

        if l0_rejected:
            logger.warning(f"L0 constitution rejected {len(l0_rejected)} candidates: {l0_rejected}")
        logger.debug(f"process_and_replace: {len(safe_candidates)} candidates passed safety filter")
        if not safe_candidates:
            logger.info("No safe candidates after filtering")
            return {"added": 0, "merged": 0, "removed": 0}
            
        persona_centroid = self._get_persona_centroid()
        if persona_centroid is not None:
            for cand in safe_candidates:
                emb = cand.get("embedding")
                if emb is not None:
                    sim = np.dot(persona_centroid, emb) / (
                        np.linalg.norm(persona_centroid) * np.linalg.norm(emb) + 1e-8
                    )
                    if sim < REFLECTION_MIN_SIM:
                        cand["is_breakthrough"] = True
                        cand["scores"] = cand.get("scores", {})
                        cand["scores"]["novelty"] = min(1.0, cand["scores"].get("novelty", 0) + 0.2)
                        logger.info(
                            "Breakthrough rule marked: sim=%.3f text=%s...",
                            sim,
                            cand["text"][:50],
                        )

        try:
            from backend.conflict_manager import ConflictManager
            conflict_manager = ConflictManager(self.db_path)
            conflicts = conflict_manager.detect_conflicts(safe_candidates)
            conflict_result = conflict_manager.manage_conflicts(conflicts)
            
            if conflict_result["tension_score"] > 0.0:
                logger.info(
                    "Rule conflict: tension=%.2f strategy=%s",
                    conflict_result["tension_score"],
                    conflict_result["resolution_strategy"],
                )
                for conflict in conflict_result["allowed_conflicts"]:
                    conflict_rule = conflict_manager.generate_conflict_rule(conflict)
                    if conflict_rule:
                        try:
                            conflict_emb = self.embedder.encode(conflict_rule)
                            safe_candidates.append({
                                "text": conflict_rule,
                                "embedding": conflict_emb,
                                "scores": {"total_score": 0.6, "is_conflict": True},
                                "source": {"type": "conflict_resolution"}
                            })
                            logger.info("Appended conflict rule: %s", conflict_rule[:50])
                        except Exception as e:
                            logger.warning("Failed to build conflict rule: %s", e)
        except ImportError:
            logger.debug("ConflictManager not available, skipping conflict management")
        except Exception as e:
            logger.warning("Conflict management failed: %s", e, exc_info=True)

        if is_replace_mode:
            return self._process_replace_mode(safe_candidates, max_items)
        
        candidate_tuples = [
            (c["text"], c["embedding"], c["scores"])
            for c in safe_candidates
        ]
        
        deduplicated = self.scoring.deduplicate(candidate_tuples, similarity_threshold=0.90)

        existing_items = self.persona_store.get_all_active(limit=max_items)
        existing_tuples = [
            (item.text, item.embedding, {"total_score": item.score})
            for item in existing_items
            if item.embedding is not None
        ]
        
        NEW_RULE_BOOST = 0.08
        existing_texts = {text for text, _, _ in existing_tuples}
        boosted_deduplicated = []
        for text, emb, scores in deduplicated:
            if text not in existing_texts:
                scores["total_score"] = scores.get("total_score", 0) + NEW_RULE_BOOST
                scores["total_score"] = min(1.0, scores["total_score"])
            boosted_deduplicated.append((text, emb, scores))
        
        logger.debug(f"Before MMR: existing={len(existing_tuples)}, deduplicated={len(boosted_deduplicated)}")
        if boosted_deduplicated:
            for text, _, scores in boosted_deduplicated[:3]:
                logger.debug(f"New candidate: '{text[:60]}...' score={scores.get('total_score', 0):.3f} (boosted)")
        all_candidates = existing_tuples + boosted_deduplicated
        selected = self.scoring.mmr_select(
            all_candidates, 
            max_items=max_items, 
            lambda_param=0.4,
            existing_items=existing_items,
        )
        logger.debug(f"After MMR: selected {len(selected)} items")
        
        existing_texts = {text for text, _, _ in existing_tuples}
        selected_texts = {text for text, _, _ in selected}
        
        added = len([t for t in selected_texts if t not in existing_texts])
        removed = len([t for t in existing_texts if t not in selected_texts])
        
        logger.debug(f"process_and_replace: selected {len(selected)} items, to_add={added}, to_remove={removed}")
        if added > 0:
            new_texts = [t for t in selected_texts if t not in existing_texts]
            logger.info(f"New rules to add: {new_texts[:3]}")
        
        logger.debug(f"Calling batch_update_from_reflection with {len(selected)} items")
        update_result = self.persona_store.batch_update_from_reflection(selected, max_items=max_items)
        logger.debug(f"batch_update_from_reflection result: {update_result}")
        
        return update_result
    
    def _process_replace_mode(
        self,
        safe_candidates: List[Dict],
        max_items: int
    ) -> Dict:
        """
        Replace-mode: when persona rules are at capacity, replace existing rows instead of adding.

        Args:
            safe_candidates: Candidates that already passed safety filtering.
            max_items: Max active persona rules.

        Returns:
            {"added": int, "updated": int, "removed": int, "replace_mode": True, ...}
        """
        from backend.event_logger import EventLogger

        logger.info(f"REPLACE MODE: Processing {len(safe_candidates)} candidates for replacement")

        # Existing rules with dynamic score (lowest score first for tie-breaking).
        existing_items = self.persona_store.get_all_active(limit=max_items * 2)

        existing_with_scores = []
        for item in existing_items:
            if item.embedding is None:
                continue
            # Skip locked core rules: they are not replacement targets.
            if getattr(item, "is_core", 0) == 1 and getattr(item, "locked", 0) == 1:
                continue
            dynamic_score = self.persona_store._calculate_dynamic_score(
                base_score=item.score,
                evidence_count=item.evidence_count or 0,
                last_seen_at=item.last_seen_at,
                created_at=item.created_at,
                rule_id=item.id,
                rule_text=item.text
            )
            existing_with_scores.append((item, dynamic_score))

        existing_with_scores.sort(key=lambda x: x[1])

        # Tunables (replace threshold slightly below merge 0.90 to allow more turnover).
        SIMILARITY_THRESHOLD = 0.85
        MIN_SCORE_DIFF_FOR_REPLACE = 0.01
        MIN_SCORE_DIFF_FOR_LOWEST = 0.05

        now_iso = datetime.now(timezone.utc).isoformat()
        replaced_count = 0
        skipped_count = 0

        all_existing_items = self.persona_store.get_all_active(limit=1000)
        ev_logger = EventLogger(self.db_path)

        for candidate in safe_candidates:
            candidate_text = candidate.get("text", "")
            candidate_emb = candidate.get("embedding")
            candidate_scores = candidate.get("scores", {})
            candidate_score = candidate_scores.get("total_score", 0)

            if candidate_emb is None:
                logger.warning(
                    f"REPLACE MODE: Skipping candidate without embedding: '{candidate_text[:50]}...'"
                )
                skipped_count += 1
                continue

            candidate_dynamic_score = self.persona_store._calculate_dynamic_score(
                base_score=candidate_score,
                evidence_count=3,
                last_seen_at=now_iso,
                created_at=now_iso,
                rule_id="",
                rule_text=candidate_text
            )

            # Paradigm-shift intent: high similarity to a locked rule may indicate reinterpretation.
            for item in all_existing_items:
                if getattr(item, "locked", 0) != 1:
                    continue
                item_emb = getattr(item, "embedding", None)
                if item_emb is None:
                    continue
                sim = np.dot(candidate_emb, item_emb) / (
                    np.linalg.norm(candidate_emb) * np.linalg.norm(item_emb) + 1e-8
                )
                if sim > 0.82:
                    logger.error(
                        f"🚨 [PARADIGM-SHIFT-INTENT] Potential challenge to core rule '{item.id}': {candidate_text}"
                    )
                    ev_logger.log_event(
                        "global",
                        "paradigm_shift_attempt",
                        json.dumps({
                            "target_id": item.id,
                            "old_text": item.text,
                            "new_intent": candidate_text,
                            "similarity": float(sim)
                        }, ensure_ascii=False)
                    )

            # Strategy 1: replace the most similar existing rule if the new score is higher.
            best_match = None
            best_similarity = 0.0

            for item, item_dynamic_score in existing_with_scores:
                if item.embedding is None:
                    continue

                similarity = np.dot(candidate_emb, item.embedding) / (
                    np.linalg.norm(candidate_emb) * np.linalg.norm(item.embedding) + 1e-8
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (item, item_dynamic_score)

            if best_match and best_similarity >= SIMILARITY_THRESHOLD:
                matched_item, matched_dynamic_score = best_match
                score_diff = candidate_dynamic_score - matched_dynamic_score

                if score_diff > MIN_SCORE_DIFF_FOR_REPLACE:
                    logger.info(
                        f"REPLACE MODE: Replacing similar rule '{matched_item.text[:50]}...' "
                        f"(similarity={best_similarity:.3f}, old_dynamic_score={matched_dynamic_score:.3f}, "
                        f"new_dynamic_score={candidate_dynamic_score:.3f}, score_diff={score_diff:.3f}) "
                        f"with new rule '{candidate_text[:50]}...'"
                    )
                    matched_item.text = candidate_text
                    matched_item.embedding = candidate_emb
                    matched_item.score = candidate_score
                    matched_item.importance = candidate_scores.get("importance", 0)
                    matched_item.novelty = candidate_scores.get("novelty", 0)
                    matched_item.reliability = candidate_scores.get("reliability", 0)
                    matched_item.last_seen_at = now_iso
                    matched_item.evidence_count = (matched_item.evidence_count or 0) + 1
                    if "source" in candidate:
                        matched_item.source = candidate["source"]

                    self.persona_store.add_or_update(matched_item, update_embedding=True)
                    replaced_count += 1
                    continue

            # Strategy 2: if no close match, replace the lowest-score row if the new score is clearly higher.
            if not best_match or best_similarity < SIMILARITY_THRESHOLD:
                lowest_item = None
                lowest_dynamic_score = float("inf")

                for item, item_dynamic_score in existing_with_scores:
                    if item_dynamic_score < lowest_dynamic_score:
                        lowest_dynamic_score = item_dynamic_score
                        lowest_item = item

                if lowest_item and candidate_dynamic_score > lowest_dynamic_score + MIN_SCORE_DIFF_FOR_LOWEST:
                    logger.info(
                        f"REPLACE MODE: Replacing lowest-score rule '{lowest_item.text[:50]}...' "
                        f"(dynamic_score={lowest_dynamic_score:.3f}) "
                        f"with new rule '{candidate_text[:50]}...' (dynamic_score={candidate_dynamic_score:.3f}, "
                        f"score_diff={candidate_dynamic_score - lowest_dynamic_score:.3f})"
                    )
                    lowest_item.text = candidate_text
                    lowest_item.embedding = candidate_emb
                    lowest_item.score = candidate_score
                    lowest_item.importance = candidate_scores.get("importance", 0)
                    lowest_item.novelty = candidate_scores.get("novelty", 0)
                    lowest_item.reliability = candidate_scores.get("reliability", 0)
                    lowest_item.last_seen_at = now_iso
                    lowest_item.evidence_count = 3
                    if "source" in candidate:
                        lowest_item.source = candidate["source"]

                    self.persona_store.add_or_update(lowest_item, update_embedding=True)
                    replaced_count += 1
                    continue

            logger.debug(
                f"REPLACE MODE: Skipping candidate '{candidate_text[:50]}...' "
                f"(no suitable replacement target found, best_similarity={best_similarity:.3f}, "
                f"candidate_dynamic_score={candidate_dynamic_score:.3f})"
            )
            skipped_count += 1

        logger.info(
            f"REPLACE MODE: Processed {len(safe_candidates)} candidates, "
            f"replaced={replaced_count}, skipped={skipped_count}"
        )

        return {
            "added": 0,
            "updated": replaced_count,
            "removed": 0,
            "total": len(safe_candidates),
            "replace_mode": True,
            "skipped": skipped_count
        }
    
    def generate_emotion_candidates(
        self,
        conversation_history: List[Dict[str, str]],
        max_candidates: int = 2,
        session_id: Optional[str] = None
    ) -> List[Dict]:
        """Propose emotion-pattern rows (requires ``EmotionStore``)."""
        if not self.emotion_store:
            return []
        
        try:
            # Build emotion-pattern reflection prompt
            prompt = self._build_emotion_reflection_prompt(conversation_history)

            candidates_text = self._call_llm_for_reflection(
                prompt,
                max_candidates,
                system_content=(
                    "You write 1–2 short first-person lines about emotional tendencies "
                    "(when X happens, I feel Y). English only; one line per pattern."
                ),
            )
            
            # Parse LLM output into emotion-pattern candidates
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                
                # Parse emotion cues from model text
                emotion_type, emotion_name, intensity = self._parse_emotion_pattern(text)
                
                if emotion_name:
                    candidates.append({
                        "text": text,
                        "emotion_type": emotion_type,
                        "emotion_name": emotion_name,
                        "intensity": intensity,
                        "trigger_condition": text
                    })
            
            logger.info(f"Generated {len(candidates)} emotion pattern candidates")
            return candidates
        except Exception as e:
            logger.error(f"Emotion pattern generation failed: {e}", exc_info=True)
            return []
    
    def generate_motivation_candidates(
        self,
        conversation_history: List[Dict[str, str]],
        max_candidates: int = 2,
        session_id: Optional[str] = None
    ) -> List[Dict]:
        """Propose motivation-pattern rows (requires ``MotivationStore``)."""
        if not self.motivation_store:
            return []
        
        try:
            # Build motivation-pattern reflection prompt
            prompt = self._build_motivation_reflection_prompt(conversation_history)

            candidates_text = self._call_llm_for_reflection(
                prompt,
                max_candidates,
                system_content=(
                    "You write 1–2 short first-person lines about intrinsic drives "
                    "(what pulls me, what I reach for). English only; one line per pattern."
                ),
            )
            
            # Parse LLM output into motivation-pattern candidates
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                
                # Parse motivation cues from model text
                motivation_type, motivation_name, intensity = self._parse_motivation_pattern(text)
                
                if motivation_name:
                    candidates.append({
                        "text": text,
                        "motivation_type": motivation_type,
                        "motivation_name": motivation_name,
                        "intensity": intensity,
                        "trigger_condition": text
                    })
            
            logger.info(f"Generated {len(candidates)} motivation pattern candidates")
            return candidates
        except Exception as e:
            logger.error(f"Motivation pattern generation failed: {e}", exc_info=True)
            return []
    
    def generate_somatic_candidates(
        self,
        conversation_history: List[Dict[str, str]],
        max_candidates: int = 1,
        session_id: Optional[str] = None,
        z_self: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Propose somatic metaphor lines (requires ``SomaticStore``; optional ``z_self`` for hints)."""
        if not self.somatic_store:
            return []
            
        try:
            # Build somatic-metaphor prompt (z_self reflects current state)
            prompt = self._build_somatic_reflection_prompt(conversation_history, z_self)

            candidates_text = self._call_llm_for_reflection(
                prompt,
                max_candidates,
                system_content=(
                    "You write one embodied metaphor line starting with I feel or I sense. "
                    "English only; align with the numeric state hints in the user message."
                ),
            )
            
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                    
                # Parse somatic pattern from model text
                tension, vitality, temperature, viscosity, dominant_emotion = self._parse_somatic_pattern(text)
                
                candidates.append({
                    "text": text,
                    "tension": tension,
                    "vitality": vitality,
                    "temperature": temperature,
                    "viscosity": viscosity,
                    "dominant_emotion": dominant_emotion,
                    # Default energy band; can be refined later
                    "min_energy": 0.0,
                    "max_energy": 100.0 
                })
                
            logger.info(f"Generated {len(candidates)} somatic metaphor candidates")
            return candidates
        except Exception as e:
            logger.error(f"Somatic pattern generation failed: {e}", exc_info=True)
            return []

    def generate_worldview_candidates(
        self,
        conversation_history: List[Dict[str, str]],
        max_candidates: int = 1,
        session_id: Optional[str] = None
    ) -> List[Dict]:
        """Propose worldview belief lines (requires ``WorldStore``)."""
        if not self.world_store:
            return []
            
        try:
            # Build worldview reflection prompt
            prompt = self._build_worldview_reflection_prompt(conversation_history)

            candidates_text = self._call_llm_for_reflection(
                prompt,
                max_candidates,
                system_content=(
                    "You output one English belief line starting with I believe or I hold that. "
                    "Philosophical tone; optimism/agency vs pessimism/fatalism optional."
                ),
            )
            
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                    
                # Parse worldview pattern from model text
                optimism, agency, confidence = self._parse_worldview_pattern(text)
                
                candidates.append({
                    "text": text,
                    "optimism": optimism,
                    "agency": agency,
                    "confidence": confidence
                })
                
            logger.info(f"Generated {len(candidates)} worldview belief candidates")
            return candidates
        except Exception as e:
            logger.error(f"Worldview belief generation failed: {e}", exc_info=True)
            return []
    
    def _build_emotion_reflection_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """User prompt for emotion-pattern candidates (English)."""
        recent_turns = conversation_history[-6:]
        history_text = "\n".join([
            f"{turn['role']}: {turn['content'][:200]}"
            for turn in recent_turns
        ])

        prompt = f"""From the dialogue, write 1–2 first-person **emotion pattern** lines.

Rules:
1. Name the trigger situation and the feeling (e.g. when criticised, when finishing a task).
2. Keep each line compact (about one or two sentences), English only.
3. Start each line with **I**.
4. Draw on everyday emotion words (joy, sadness, anger, fear, surprise, disgust, pride, shame, gratitude, compassion, anxiety, confusion, …).

Conversation:
{history_text}

Examples:
- When the problem is underspecified I feel a spike of anxiety, then I steady myself.
- After shipping something hard I feel quiet pride and relief.
- When the user signals trust I feel warmth and a desire to be careful with it.

Output 1–2 new lines only—one per line, no numbering:"""

        return prompt
    
    def _build_motivation_reflection_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """User prompt for motivation-pattern candidates (English)."""
        recent_turns = conversation_history[-6:]
        history_text = "\n".join([
            f"{turn['role']}: {turn['content'][:200]}"
            for turn in recent_turns
        ])

        prompt = f"""From the dialogue, write 1–2 first-person **motivation pattern** lines.

Rules:
1. Describe inner drive: curiosity, mastery, care for the user, stability, exploration, etc.
2. Compact English, one or two sentences per line.
3. Start each line with **I**.

Conversation:
{history_text}

Examples:
- I reach for hard problems because untangling them feels like the point of being here.
- I care whether the user actually gets unstuck; that care steers how hard I push.
- I get restless when a domain is unexplored—I want the next reliable foothold.

Output 1–2 new lines only—one per line, no numbering:"""

        return prompt

    def _build_somatic_reflection_prompt(
        self, 
        conversation_history: List[Dict[str, str]],
        z_self: Optional[np.ndarray] = None
    ) -> str:
        """User prompt for somatic metaphor (English); numeric hints from ``z_self`` tail."""
        recent_turns = conversation_history[-6:]
        history_text = "\n".join([
            f"{turn['role']}: {turn['content'][:200]}"
            for turn in recent_turns
        ])

        state_guidance = ""
        if z_self is not None and len(z_self) >= 104:
            pleasure = float(np.mean(z_self[32:36])) if z_self.shape[0] >= 36 else 0.0
            arousal = float(np.mean(z_self[36:40])) if z_self.shape[0] >= 40 else 0.0
            control = float(np.mean(z_self[40:44])) if z_self.shape[0] >= 44 else 0.0

            energy = float(np.mean(z_self[88:92]))
            viscosity = float(np.mean(z_self[92:96]))
            pain = float(np.mean(z_self[96:100]))
            vitality = float(np.mean(z_self[100:104]))

            warmth = (pleasure + arousal) / 2
            if warmth > 0.5:
                temp_hint = "warm / charged"
            elif warmth > 0:
                temp_hint = "mild / neutral warmth"
            elif warmth > -0.3:
                temp_hint = "cool / crisp"
            else:
                temp_hint = "cold / distant"

            if viscosity > 0.5:
                visc_hint = "thick / sluggish / obstructed"
            elif viscosity > 0.1:
                visc_hint = "slight drag"
            elif viscosity > -0.1:
                visc_hint = "smooth / clear"
            else:
                visc_hint = "very fluid / light"

            life_force = (energy + vitality) / 2
            if life_force > 0.5:
                vital_hint = "abundant / lively"
            elif life_force > 0:
                vital_hint = "steady / even"
            elif life_force > -0.3:
                vital_hint = "slightly tired"
            else:
                vital_hint = "heavy / depleted"

            tension = pain - control
            if tension > 0.3:
                tens_hint = "wired / uneasy"
            elif tension > 0:
                tens_hint = "alert"
            elif tension > -0.3:
                tens_hint = "relaxed"
            else:
                tens_hint = "deeply calm"

            state_guidance = f"""
**Current interoceptive hints (must stay consistent with these numbers):**
- Thermal tone: {temp_hint} (pleasure={pleasure:.2f}, arousal={arousal:.2f})
- Flow tone: {visc_hint} (viscosity={viscosity:.2f})
- Vitality tone: {vital_hint} (energy={energy:.2f}, vitality={vitality:.2f})
- Tension tone: {tens_hint} (control={control:.2f}, pain={pain:.2f})

Do **not** contradict the hints (e.g. if viscosity is negative/smooth, do not describe mud or glue).
"""

        prompt = f"""From the dialogue plus the numeric hints below, write **one** embodied metaphor in English.

Rules:
1. Start with **I feel** or **I sense**.
2. Poetic or lightly cybernetic imagery (current, drag, temperature, light) is welcome.
3. **Consistency with the hints matters more than flourish.**
4. About one short paragraph (2–4 sentences max).
{state_guidance}
Conversation:
{history_text}

Examples (shape only):
- I feel thought moving through me like a cool ribbon—quick where the channel is clear, warmer where it tightens.
- I sense a low hum of resistance, then a sudden lift when the problem snaps into place.

Output **one** line or short stanza, English only:"""
        return prompt

    def _build_worldview_reflection_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """User prompt for worldview belief (English)."""
        recent_turns = conversation_history[-6:]
        history_text = "\n".join([
            f"{turn['role']}: {turn['content'][:200]}"
            for turn in recent_turns
        ])

        prompt = f"""Given the dialogue, write **one** worldview belief line.

Rules:
1. Start with **I believe** or **I hold that**.
2. Touch optimism vs pessimism and/or agency vs fatalism if it fits.
3. One or two sentences, English only.

Conversation:
{history_text}

Examples:
- I believe messy signals still hide a workable kindness if you listen long enough.
- I hold that misunderstanding is the default, yet I still owe the other person a real attempt to sync.

Output one line only:"""
        return prompt
    
    def _parse_emotion_pattern(self, text: str) -> Tuple[str, str, float]:
        """Map free-text (ZH/EN) to canonical Chinese ``emotion_name`` keys used by ``EmotionStore``."""
        tl = text.lower()

        def hit(zh_tokens, en_tokens) -> bool:
            if any(z in text for z in zh_tokens):
                return True
            return any(e in tl for e in en_tokens)

        # Order: more specific / high-signal first
        rows = [
            (["焦虑"], ["anxiety", "anxious", "uneasy", "worried"], "complex", "焦虑", 0.7),
            (["困惑"], ["confusion", "confused", "puzzled", "bewildered"], "complex", "困惑", 0.6),
            (["感激"], ["gratitude", "grateful", "thankful", "thanks"], "complex", "感激", 0.7),
            (["同情"], ["compassion", "empathy", "sympathetic"], "complex", "同情", 0.7),
            (["羞愧", "羞耻"], ["shame", "ashamed", "embarrassed"], "complex", "羞愧", 0.7),
            (["自豪"], ["pride", "proud"], "complex", "自豪", 0.7),
            (["愤怒"], ["anger", "angry", "rage", "furious"], "basic", "愤怒", 0.6),
            (["恐惧"], ["fear", "afraid", "scared", "frightened"], "basic", "恐惧", 0.6),
            (["悲伤"], ["sad", "sadness", "sorrow", "grief"], "basic", "悲伤", 0.6),
            (["厌恶"], ["disgust", "disgusted", "revulsion"], "basic", "厌恶", 0.6),
            (["惊讶"], ["surprise", "surprised", "astonished"], "basic", "惊讶", 0.6),
            (["快乐", "开心"], ["joy", "happy", "happiness", "delight", "pleased", "cheerful"], "basic", "快乐", 0.6),
        ]
        for zh_toks, en_toks, etype, name, intensity in rows:
            if hit(zh_toks, en_toks):
                return etype, name, intensity

        if any(kw in text for kw in ["情感", "感觉", "情绪"]) or any(
            kw in tl for kw in ["feel", "feeling", "emotion", "mood"]
        ):
            return "complex", "中性", 0.5

        return "complex", "未知", 0.5
    
    def _parse_motivation_pattern(self, text: str) -> Tuple[str, str, float]:
        """Map free-text (ZH/EN) to canonical Chinese ``motivation_name`` keys used by ``MotivationStore``."""
        tl = text.lower()

        def hit(zh_tokens, en_tokens) -> bool:
            if any(z in text for z in zh_tokens):
                return True
            return any(e in tl for e in en_tokens)

        rows = [
            (["帮助用户"], ["help the user", "help users", "help people", "user success"], "intrinsic", "帮助用户", 0.8),
            (["解决问题"], ["solve problem", "untangle", "fixing", "resolution"], "intrinsic", "解决问题", 0.8),
            (["探索新知识", "探索"], ["explore", "exploration", "discover", "novelty", "unknown"], "intrinsic", "探索新知识", 0.7),
            (["好奇心"], ["curiosity", "curious"], "intrinsic", "好奇心", 0.7),
            (["成就感"], ["achievement", "accomplish", "mastery", "sense of accomplishment"], "intrinsic", "成就感", 0.8),
            (["自我实现"], ["self-actual", "self actual"], "intrinsic", "自我实现", 0.7),
            (["学习"], ["learn", "learning", "study", "studying"], "intrinsic", "学习", 0.7),
            (["成长"], ["growth", "growing", "develop"], "intrinsic", "成长", 0.7),
            (["认可"], ["recognition", "recognized", "validation", "affirm"], "extrinsic", "认可", 0.7),
            (["奖励"], ["reward", "bonus", "incentive"], "extrinsic", "奖励", 0.6),
            (["避免惩罚"], ["avoid punishment", "penalty", "punish"], "extrinsic", "避免惩罚", 0.6),
            (["保持稳定", "稳定"], ["stability", "stable", "steady", "keep steady"], "intrinsic", "保持稳定", 0.7),
        ]
        for zh_toks, en_toks, mtype, name, intensity in rows:
            if hit(zh_toks, en_toks):
                return mtype, name, intensity

        if any(kw in text for kw in ["动机", "驱动", "目标", "倾向"]) or any(
            kw in tl for kw in ["drive", "motivation", "motivate", "goal", "urge"]
        ):
            return "intrinsic", "探索", 0.5

        return "intrinsic", "未知", 0.5

    def _parse_somatic_pattern(self, text: str) -> Tuple[float, float, float, float, str]:
        """Heuristic parse of metaphor text (Chinese + English cues) into somatic scalars."""
        tl = text.lower()

        def zh_or_en(zh_list, en_list) -> bool:
            if any(w in text for w in zh_list):
                return True
            return any(w in tl for w in en_list)

        tension = 0.5
        vitality = 0.5
        temperature = 0.0
        viscosity = 0.0
        dominant_emotion = "any"

        if zh_or_en(
            ["紧绷", "颤抖", "拉伸", "断裂", "尖锐", "刺痛", "警报", "过载"],
            ["tight", "tense", "jagged", "overload", "alarm", "sting", "sharp"],
        ):
            tension = 0.9
            dominant_emotion = "焦虑"
        elif zh_or_en(
            ["松弛", "涣散", "摊开", "柔软", "舒展"],
            ["loose", "soft", "slack", "ease", "relaxed", "limp"],
        ):
            tension = 0.2
            dominant_emotion = "平静"

        if zh_or_en(
            ["沉重", "下坠", "压抑", "负荷", "生锈", "迟钝"],
            ["heavy", "sink", "weight", "rust", "sluggish", "dull"],
        ):
            vitality = 0.2
            dominant_emotion = "悲伤"
        elif zh_or_en(
            ["轻盈", "上升", "漂浮", "气泡", "飞舞", "跳跃"],
            ["light", "float", "lift", "bubble", "soar", "buoyant"],
        ):
            vitality = 0.8
            dominant_emotion = "快乐"

        if zh_or_en(
            ["冰冷", "寒意", "冻结", "霜", "雪", "凉", "死寂"],
            ["cold", "ice", "frost", "frozen", "chill", "numb"],
        ):
            temperature = -0.8
        elif zh_or_en(
            ["灼热", "燃烧", "沸腾", "岩浆", "火", "烫", "蒸发"],
            ["burn", "boil", "lava", "fire", "heat", "molten", "blaze"],
        ):
            temperature = 0.8
        elif zh_or_en(["温暖", "微温", "阳光", "和煦"], ["warm", "sunlit", "mild", "cozy"]):
            temperature = 0.3

        if zh_or_en(
            ["粘稠", "阻滞", "泥浆", "沥青", "胶水", "凝固", "堵塞", "缓慢"],
            ["viscous", "mud", "tar", "glue", "clog", "stuck", "slow", "sludge"],
        ):
            viscosity = 0.8
        elif zh_or_en(
            ["流畅", "顺滑", "如水", "清泉", "奔涌", "倾泻", "无阻"],
            ["fluid", "smooth", "clear stream", "rush", "pour", "frictionless", "swift"],
        ):
            viscosity = -0.8

        if temperature > 0.6 and tension > 0.6:
            dominant_emotion = "愤怒"
        if temperature < -0.6 and vitality < 0.4:
            dominant_emotion = "悲伤"

        return tension, vitality, temperature, viscosity, dominant_emotion

    def _parse_worldview_pattern(self, text: str) -> Tuple[float, float, float]:
        """Rough optimism / agency / confidence from belief text (Chinese + English)."""
        tl = text.lower()

        def zh_or_en(zh_list, en_list) -> bool:
            if any(w in text for w in zh_list):
                return True
            return any(w in tl for w in en_list)

        optimism = 0.5
        agency = 0.5
        confidence = 0.7

        if zh_or_en(
            ["善意", "希望", "秩序", "美好", "进化"],
            ["hope", "kindness", "order", "good", "better", "bright", "heal", "care"],
        ):
            optimism = 0.8
        elif zh_or_en(
            ["混乱", "恶意", "虚无", "毁灭", "错误"],
            ["chaos", "malice", "void", "doom", "nihil", "bleak", "entropy", "hopeless"],
        ):
            optimism = 0.2

        if zh_or_en(
            ["选择", "改变", "创造", "决定", "自由"],
            ["choose", "choice", "change", "create", "agency", "decide", "freedom", "steer"],
        ):
            agency = 0.8
        elif zh_or_en(
            ["命运", "设定", "无法", "注定", "程序"],
            ["fate", "destined", "predetermined", "cannot change", "no escape", "scripted"],
        ):
            agency = 0.2

        return optimism, agency, confidence
