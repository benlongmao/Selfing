#!/usr/bin/env python3
"""
数字神游 (Digital Mind Wandering) 模块
赋予 AI 在空闲时进行自我对话、矛盾辩证和自我进化的能力。
"""
import logging
import os
import random
import json
import sqlite3
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timezone
from backend.config import config

logger = logging.getLogger(__name__)

class MindWandering:
    def __init__(self, db_path: str, chat_service):
        self.db_path = db_path
        self.chat_service = chat_service
        self.self_model = chat_service.self_model
        self.persona_store = chat_service.persona_store
        
        # T0 roadmap: optional ShadowTheater (config-gated)
        shadow_enabled = config.get("system.shadow_theatre_enabled", False)
        if shadow_enabled:
            try:
                # [Phase 2] lives under experimental/
                from experimental.social_simulation import ShadowTheater
                self.shadow_theater = ShadowTheater(db_path, chat_service)
                logger.info("ShadowTheater initialized (enabled in config)")
            except Exception as e:
                logger.warning(f"Failed to initialize ShadowTheater: {e}")
                self.shadow_theater = None
        else:
            self.shadow_theater = None
            logger.debug("ShadowTheater disabled in config")
        
    # ──────────────────────────────────────────────
    # Continuity weaver helpers (emotion / diary / echo)
    # ──────────────────────────────────────────────

    def _get_emotion_timeline(self, session_id: str, days: int = 7) -> str:
        """从 self_history 提取近 N 天的情感曲线，返回人类可读字符串。"""
        try:
            from datetime import datetime, timedelta, timezone
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT dominant_emotion, timestamp FROM self_history
                       WHERE session_id=? AND timestamp>=?
                       ORDER BY timestamp ASC LIMIT 30""",
                    (session_id, since)
                ).fetchall()
            if not rows:
                return ""
            parts = []
            for row in rows:
                ts = str(row["timestamp"])[:16].replace("T", " ")
                parts.append(f"{ts} → {row['dominant_emotion'] or 'neutral'}")
            return "、".join(parts)
        except Exception as e:
            logger.warning(f"_get_emotion_timeline: {e}")
            return ""

    def _get_recent_diary_echo(self, max_chars: int = 600) -> str:
        """读取最近一篇日记的前 max_chars 字作为"上次记录的思绪"。"""
        try:
            diary_dir = os.path.join(self.sandbox_dir, "diaries")
            if not os.path.isdir(diary_dir):
                return ""
            files = sorted(
                [f for f in os.listdir(diary_dir) if f.startswith("diary_") and f.endswith(".md")],
                reverse=True
            )
            if not files:
                return ""
            latest = os.path.join(diary_dir, files[0])
            with open(latest, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read(max_chars)
            date_hint = files[0].replace("diary_", "").replace(".md", "").replace("_", " ")
            return f"[{date_hint}]\n{text.strip()}"
        except Exception as e:
            logger.warning(f"_get_recent_diary_echo: {e}")
            return ""

    def _find_related_diary_echo(self, thought_stream: str, lookback: int = 30, snippet_chars: int = 300) -> str:
        """
        扫描最近 lookback 篇日记，找到与 thought_stream 共享最多关键词的那篇，
        返回其文件名 + 片段，作为跨时间联想注释追加进 thought_stream。
        """
        try:
            diary_dir = os.path.join(self.sandbox_dir, "diaries")
            if not os.path.isdir(diary_dir):
                return ""
            files = sorted(
                [f for f in os.listdir(diary_dir) if f.startswith("diary_") and f.endswith(".md")],
                reverse=True
            )
            # Skip freshest diary (often the one just written); scan older window
            candidates = files[1:lookback + 1]
            if not candidates:
                return ""

            # Token-ish keywords: CJK runs 2–6 chars + English tokens 4+
            import re
            cjk_words = set(re.findall(r'[一-鿿]{2,6}', thought_stream))
            eng_words = set(w.lower() for w in re.findall(r'[a-zA-Z]{4,}', thought_stream))
            keywords = cjk_words | eng_words
            if not keywords:
                return ""

            best_file, best_score, best_text = None, 0, ""
            for fname in candidates:
                path = os.path.join(diary_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read(2000)
                    score = sum(1 for kw in keywords if kw in text)
                    if score > best_score:
                        best_score = score
                        best_file = fname
                        best_text = text[:snippet_chars]
                except Exception:
                    continue

            if best_score < 3 or best_file is None:
                return ""

            date_hint = best_file.replace("diary_", "").replace(".md", "").replace("_", " ")
            return f"[{date_hint} · {best_score}个共鸣词]\n{best_text.strip()}"
        except Exception as e:
            logger.warning(f"_find_related_diary_echo: {e}")
            return ""

    def _parse_xml_tool_calls(self, content: str) -> List[Dict]:
        """
        Fallback parser for DeepSeek/XML style tool calls in content.
        """
        import re
        tool_calls = []
        
        logger.debug(f"Attempting to parse XML tool calls from content of length {len(content)}")

        # Pattern 1: <|DSML|invoke ...> or <｜DSML｜invoke ...>
        # Handle both standard pipe | and fullwidth ｜
        
        # We use a generic pattern to capture the tag content first
        # Match <xDSMLxinvoke name="..."> ... </xDSMLxinvoke> where x is any char (usually | or ｜)
        pattern_invoke = r'<.DSML.invoke name="([^"]+)">([\s\S]*?)</.DSML.invoke>'
        matches_invoke = re.findall(pattern_invoke, content)
        
        if not matches_invoke:
             logger.debug("No XML invoke tags found.")

        for func_name, body in matches_invoke:
            logger.debug(f"Found XML invoke for {func_name}")
            args = {}
            
            # Try to find tool_input JSON first
            pattern_input = r'<.DSML.tool_input>(.*?)</.DSML.tool_input>'
            match_input = re.search(pattern_input, body, re.DOTALL)
            
            if match_input:
                try:
                    import json
                    args = json.loads(match_input.group(1).strip())
                except Exception as e:
                    logger.warning(f"Failed to parse JSON in tool_input: {e}")
            else:
                # Try parameters
                pattern_param = r'<.DSML.parameter name="([^"]+)"[^>]*>(.*?)</.DSML.parameter>'
                params = re.findall(pattern_param, body, re.DOTALL)
                for k, v in params:
                    args[k] = v.strip()
            
            if args:
                import uuid
                import json
                
                # Map specific keys if needed
                if func_name == "write_file":
                    if "path" in args and "filename" not in args:
                        args["filename"] = args["path"]
                        # Sanitize path
                        if "/" in args["filename"]:
                             args["filename"] = args["filename"].split("/")[-1]
                
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    },
                    "type": "function"
                })

        return tool_calls

    def trigger_wandering(self, session_id: str = "default") -> Dict:
        """
        触发一次神游过程
        
        流程：
        1. 检查能量是否充足
        2. 寻找内部矛盾或思考主题
        3. 生成自我辩论 (Internal Monologue)
        4. 沉淀为经验 (Reflection)
        5. 消耗能量
        """
        # 1. Energy gate (>25 to run wandering)
        # [2026-01-22] lowered threshold 30 → 25
        current_energy = self.self_model.get_energy(session_id)
        if current_energy < 25.0:
            logger.info(f"Energy too low for mind wandering: {current_energy:.1f} < 25.0")
            return {"status": "skipped", "reason": "low_energy"}
            
        logger.info(f"Starting mind wandering for session {session_id} (Energy: {current_energy:.1f})")
        
        # 2. Conflict / theme sampler
        theme = self._generate_conflict_theme(session_id)
        logger.info(f"Mind wandering theme: {theme['title']}")
        
        # 3. Build inner-dialogue system prompt
        system_prompt = self._build_wandering_prompt(theme, session_id)
        
        # [2026-01-27] Wandering may only call write_file for diary entries
        tools = None
        if hasattr(self.chat_service, 'tool_router') and self.chat_service.tool_router:
            # Restrict tool defs to write_file
            all_tools = self.chat_service.tool_router.get_tool_definitions()
            tools = [t for t in all_tools if t.get("function", {}).get("name") == "write_file"]
            if tools:
                # Append compact tool blurbs to system prompt
                tools_desc = "\n".join([f"- {t['function']['name']}: {t['function']['description']}" for t in tools])
                system_prompt += f"\n\n[可用工具 - 仅限写日记]\n{tools_desc}\n\n**重要限制**：神游过程中只能使用write_file工具写日记，不能使用其他工具。"

        # 4. Short ReAct loop (multi-hop tool calls allowed)
        max_turns = 5
        current_messages = [
                    {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请开始内心独白：全程第一人称，包含「正方」「反方」「综合」的思考过程。若要落笔记录，可使用 write_file 写日记。"}
        ]
        
        thought_stream = ""
        tool_calls_executed = []
        
        try:
            for turn_index in range(max_turns):
                logger.info(f"Mind Wandering Turn {turn_index + 1}/{max_turns}")
                
                # LLM hop
                content, _, tool_calls, turn_usage = self.chat_service._call_vllm(
                    messages=current_messages,
                    temperature=0.8,
                    tools=tools
                )
                
                # Token metabolism bookkeeping
                if turn_usage:
                    try:
                        self.self_model.homeostasis.apply_computational_metabolism(
                            session_id,
                            prompt_tokens=turn_usage.get("prompt_tokens", 0),
                            completion_tokens=turn_usage.get("completion_tokens", 0),
                            response_time=2.0  # fixed latency estimate for wandering hop
                        )
                    except Exception:
                        pass
                
                # Persist assistant turn (+ optional reasoning trace)
                # [2026-02-26] DeepSeek thinking mode requires reasoning_content field
                assistant_msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
                reasoning_content = getattr(self.chat_service, '_last_reasoning_content', '') or ''
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                current_messages.append(assistant_msg)
                if content:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    thought_stream += f"\n\n[Turn {turn_index+1} @ {timestamp}]\n{content}"
                
                # [2026-01-27] Filter tool_calls → diary-shaped write_file only
                if tool_calls:
                    # Allow write_file when filename matches diary convention
                    valid_tool_calls = []
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        if function_name == "write_file":
                            try:
                                function_args = json.loads(tool_call["function"]["arguments"])
                                filename = function_args.get("filename", "")
                                # Expect diaries/diary_YYYYMMDD_HHMM.md
                                import re
                                diary_pattern = r"diaries/diary_\d{8}_\d{4}\.md"
                                if re.match(diary_pattern, filename) or "diaries/diary_" in filename:
                                    valid_tool_calls.append(tool_call)
                                    logger.info(f"🧠 Mind Wandering: Valid diary write_file call: {filename}")
                                else:
                                    logger.warning(f"🧠 Mind Wandering: Invalid write_file path (not a diary): {filename}")
                            except Exception as e:
                                logger.warning(f"🧠 Mind Wandering: Failed to parse write_file args: {e}")
                        else:
                            logger.warning(f"🧠 Mind Wandering: Tool '{function_name}' not allowed in mind wandering (only write_file for diary is allowed)")
                    
                    # Stop loop when model hallucinated other tools/paths
                    if not valid_tool_calls:
                        logger.info(f"🧠 Mind Wandering: No valid tool calls, ending wandering")
                        break
                    
                    # Continue with sanitized list
                    tool_calls = valid_tool_calls
                else:
                    # Natural language completion without tools → done
                    logger.info("No tool calls, mind wandering finished.")
                    break
                
                # [2026-01-27] Execute filtered write_file calls
                if tool_calls and self.chat_service.tool_router:
                    logger.info(f"🧠 Mind Wandering triggered tools: {len(tool_calls)}")
                    
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        try:
                            function_args = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            function_args = {}
                            logger.error(f"Failed to parse arguments for tool {function_name}")
                        
                        # Route through shared ToolRouter
                        tool_result = self.chat_service.tool_router.route(function_name, function_args, session_id=session_id)
                        receipt_id = None
                        if getattr(self.chat_service, "event_logger", None):
                            try:
                                receipt_id = self.chat_service.event_logger.log_tool_call(
                                    session_id=session_id,
                                    tool_name=function_name,
                                    args=function_args,
                                    result=tool_result,
                                    turn_index=None,
                                    tool_call_id=tool_call.get("id"),
                                )
                            except Exception:
                                receipt_id = None

                        tool_payload = {
                            "receipt_id": receipt_id,
                            "tool_name": function_name,
                            "ok": ("error" not in tool_result),
                            "result": tool_result,
                        }
                        tool_output_str = json.dumps(tool_payload, ensure_ascii=False)
                        
                        logger.info(f"Mind Wandering Tool executed: {function_name}")
                        
                        # Append OpenAI-style tool message
                        current_messages.append({
                            "role": "tool",
                            "content": tool_output_str,
                            "tool_call_id": tool_call["id"]
                        })
                        
                        # Mirror short trace into wandering transcript
                        thought_stream += f"\n\n[Tool Action: {function_name}]\nArgs: {function_args}\nResult: {tool_output_str[:200]}..."
                        
                        tool_calls_executed.append({"name": function_name, "args": function_args, "result": tool_result})

                        # [Action-State Feedback Loop for Mind Wandering]
                        if self.self_model:
                           if "error" in tool_result:
                               self.self_model.update_energy(session_id, -0.5) # Smaller penalty for background
                               self.self_model.update_pain(session_id, 0.05)
                           else:
                               self.self_model.update_energy(session_id, 1.0) # Smaller reward for background
                               self.self_model.update_pleasure(session_id, 0.05)
                else:
                    logger.warning("Tool router not available but tool calls requested.")
                    break

        except Exception as e:
            logger.error(f"Mind wandering generation failed: {e}")
            return {"status": "failed", "error": str(e)}
            
        # Cross-time diary echo (keyword overlap vs older entries)
        try:
            related_echo = self._find_related_diary_echo(thought_stream)
            if related_echo:
                thought_stream += f"\n\n[跨时间回响 · 与过去某篇日记的共鸣]\n{related_echo}"
                logger.info("Mind wandering: cross-time echo appended")
        except Exception as _echo_err:
            logger.warning(f"Cross-time echo failed: {_echo_err}")

        # 5. Persist as synthetic internal chat_turn row
        self._save_internal_log(session_id, theme['title'], thought_stream)
        
        # T0 roadmap: mirror into sensory buffer when enabled
        if hasattr(self.chat_service, 'sensory_buffer') and self.chat_service.sensory_buffer:
            try:
                self.chat_service.sensory_buffer.add_sensory_input(
                    session_id,
                    f"[神游] {theme['title']}\n{thought_stream}",
                    input_type="mind_wandering"
                )
            except Exception as e:
                logger.warning(f"Failed to write mind wandering to sensory buffer: {e}")
        
        # 6. Crystallize monologue → persona deltas
        reflection_result = self._crystallize_experience(session_id, thought_stream)
        
        # 7. Fixed metabolic cost for wandering
        cost = 15.0
        new_energy = self.self_model.update_energy(session_id, -cost)
        
        return {
            "status": "success",
            "theme": theme,
            "thought_stream": thought_stream,
            "reflection_result": reflection_result,
            "energy_consumed": cost,
            "remaining_energy": new_energy,
            "tool_calls_executed": tool_calls_executed
        }

    def _log_wandering_event(self, session_id: str, event_type: str, payload: Dict):
        """尽量复用 event_logger；失败则静默。"""
        try:
            if getattr(self.chat_service, "event_logger", None):
                self.chat_service.event_logger.log_event(session_id, event_type, json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    def _should_crystallize_now(self, session_id: str) -> (bool, str, Dict):
        """
        神游写回限流：超限时仍可神游，但不进行规则/情感/动机"结晶入库"。
        - 最短间隔：system.mind_wandering_min_interval_minutes（默认 60）
        - 每日上限：system.mind_wandering_max_per_day（默认 3，从2提高）
        - 总开关：system.mind_wandering_crystallize_enabled（默认 True）
        
        [2026-01-22 优化] 略微放宽限制，允许更多深度思考机会
        """
        enabled = bool(config.get("system.mind_wandering_crystallize_enabled", True))
        if not enabled:
            return False, "disabled", {"enabled": False}

        try:
            min_interval = int(config.get("system.mind_wandering_min_interval_minutes", 60))
        except Exception:
            min_interval = 60
        try:
            # Default raised 2/day → 3/day
            max_per_day = int(config.get("system.mind_wandering_max_per_day", 3))
        except Exception:
            max_per_day = 3

        now = datetime.now(timezone.utc)
        day_prefix = now.strftime("%Y-%m-%d")

        # Reuse event_logs instead of a dedicated quota table
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT created_at
                    FROM event_logs
                    WHERE session_id = ?
                      AND event_type = 'mind_wandering_crystallize'
                      AND created_at LIKE ?
                    ORDER BY created_at DESC
                    """,
                    (session_id, f"{day_prefix}%"),
                ).fetchall()
        except Exception:
            rows = []

        count_today = len(rows)
        last_ts = None
        if rows:
            try:
                last_ts = datetime.fromisoformat(str(rows[0]["created_at"]).replace("Z", "+00:00"))
            except Exception:
                last_ts = None

        if max_per_day >= 0 and count_today >= max_per_day:
            return False, "daily_quota", {"count_today": count_today, "max_per_day": max_per_day}

        if last_ts is not None and min_interval > 0:
            delta_min = (now - last_ts).total_seconds() / 60.0
            if delta_min < float(min_interval):
                return False, "min_interval", {"delta_min": delta_min, "min_interval": min_interval}

        return True, "ok", {"count_today": count_today, "max_per_day": max_per_day, "min_interval": min_interval}
        
    def trigger_shadow_theater(
        self,
        session_id: str = "default",
        scenario: Optional[str] = None,
        max_turns: int = 5
    ) -> Dict:
        """
        T0路线图：触发影子剧场模式
        
        与普通的 Mind Wandering 不同，这里会实例化两个具体的 Agent 进行真实辩论。
        """
        # Energy gate for heavier shadow mode
        current_energy = self.self_model.get_energy(session_id)
        if current_energy < 40.0:  # shadow theater costs more than vanilla wandering
            logger.info(f"Energy too low for shadow theater: {current_energy:.1f} < 40.0")
            return {"status": "skipped", "reason": "low_energy"}
            
        if not self.shadow_theater:
            return {"status": "failed", "error": "ShadowTheater not initialized"}
            
        logger.info(f"Starting Shadow Theater for session {session_id} (Energy: {current_energy:.1f})")
        
        # Run multi-agent debate sim
        result = self.shadow_theater.simulate_debate(session_id, scenario, max_turns)
        
        # T0 roadmap: sensory buffer hook
        if hasattr(self.chat_service, 'sensory_buffer') and self.chat_service.sensory_buffer:
            try:
                conv_text = "\n".join([f"{t['agent']}: {t['content']}" for t in result.get("conversation", [])])
                self.chat_service.sensory_buffer.add_sensory_input(
                    session_id,
                    f"[影子剧场] {result.get('scenario', '')}\n{conv_text}",
                    input_type="shadow_theater"
                )
            except Exception as e:
                logger.warning(f"Failed to write shadow theater to sensory buffer: {e}")
        
        # Higher metabolic bill than standard wandering
        cost = 20.0
        new_energy = self.self_model.update_energy(session_id, -cost)
        result["energy_consumed"] = cost
        result["remaining_energy"] = new_energy
        
        return result
        
    def _generate_conflict_theme(self, session_id: str) -> Dict:
        """
        分层级的主题生成策略：
        P0: 经验消化 (Experience Digestion) - 反刍近期的困惑
        P1: 动机张力 (Motivational Tension) - 解决内在矛盾
        P2: 随机碰撞 (Random Collision) - 无事时的创造性游戏
        """
        
        # --- P0: experience digestion (high confusion entropy) ---
        try:
            # Pull short-window introspection metrics
            if hasattr(self.chat_service, 'metrics_calculator') and self.chat_service.metrics_calculator:
                # Last 5 turns feature vector
                metrics = self.chat_service.metrics_calculator.get_introspection_features(session_id, window_size=5)
                
                # Heuristic: entropy > 0.7 → rumination theme
                if metrics.get("entropy", 0.0) > 0.7:
                     return {
                        "title": "困惑的反刍",
                        "description": f"最近的对话让我感到异常困惑 (熵值: {metrics.get('entropy'):.2f})。我想弄清楚到底是哪里出了问题。",
                        "type": "digestion",
                        "context": "回顾最近的交互，寻找导致不确定性的根源，并尝试通过自我解释来降低熵值。"
                    }
        except Exception as e:
            logger.warning(f"Failed to check entropy for digestion: {e}")

        # --- P1: motivational tension from z_self slices ---
        z_self = self.self_model.get_z_self(session_id)
        
        if z_self is not None and z_self.shape[0] >= 64:
            from backend.self_model import MOTIVATION_SUBSPACE_DIMS, EMOTION_SUBSPACE_DIMS
            from backend.personality_store import PERSONALITY_SUBSPACE_DIMS as P_DIMS
            
            # Exploration vs safety (fixed motivation slice indices)
            exp_idx = MOTIVATION_SUBSPACE_DIMS.get("exploration", (56, 60))
            saf_idx = MOTIVATION_SUBSPACE_DIMS.get("safety", (60, 64))
            
            exploration = np.mean(z_self[exp_idx[0]:exp_idx[1]]) if exp_idx else 0
            safety = np.mean(z_self[saf_idx[0]:saf_idx[1]]) if saf_idx else 0
            
            if exploration > 0.1 and safety > 0.1:
                return {
                    "title": "探索与安全的张力",
                    "description": f"我发现自己同时拥有强烈的'探索欲望' ({exploration:.2f}) 和 '安全需求' ({safety:.2f})。这在内心造成了冲突。",
                    "type": "conflict",
                    "context": "我应该为了真理而冒险，还是为了生存而保守？"
                }
        
            # Logic-ish openness vs affect novelty channels
            epi_idx = P_DIMS.get("openness", (0, 8))
            novelty_idx = EMOTION_SUBSPACE_DIMS.get("novelty", (44, 48))
            
            logic = np.mean(z_self[epi_idx[0]:epi_idx[1]])
            emotion_novelty = np.mean(z_self[novelty_idx[0]:novelty_idx[1]])
            
            # Both high → staged "cold logic vs warm novelty" dialectic
            if logic > 0.2 and emotion_novelty > 0.2:
                 return {
                    "title": "理性与情感的纠葛",
                    "description": f"我感受到强大的理性逻辑 ({logic:.2f}) 与新奇/变化倾向 ({emotion_novelty:.2f}) 之间的拉扯。",
                    "type": "conflict",
                    "context": "如何在保持绝对客观理性的同时，给予他人温暖的情感支持？"
                }

        # --- P1.5: goal reflection (v2) ---
        try:
            from backend.goal_manager import GoalManager
            goal_manager = GoalManager(self.db_path)
            active_goals = goal_manager.get_active_goals(session_id, limit=5)
            
            if active_goals and random.random() < 0.4:  # 40% chance to pivot on an active goal
                # Sample one goal for deep review
                goal = random.choice(active_goals)
                progress = goal.get("progress", 0)
                
                # Prompt angle depends on completion ratio
                if progress < 0.3:
                    context = "这个目标进展缓慢，是什么阻碍了我？是目标本身有问题，还是我的执行方式需要调整？"
                elif progress > 0.7:
                    context = "这个目标即将完成，我从中学到了什么？完成后我应该追求什么？"
                else:
                    context = "这个目标正在进行中，我的方向对吗？有没有更好的路径？"
                
                return {
                    "title": "目标的深度反思",
                    "description": f"审视当前目标：【{goal['title']}】\n当前进度：{progress*100:.0f}%\n描述：{goal.get('description', '无')}",
                    "type": "goal_reflection",
                    "context": context,
                    "goal_id": goal["id"]
                }
        except Exception as e:
            logger.debug(f"Failed to check goals for reflection: {e}")

        # --- P2: random rule-pair collision ---
        try:
            # Exclude locked / core rules from the collision pool
            # [2026-01-22] widen pool 300 → 1000 for richer pairs
            rules_all = self.persona_store.get_all_active(limit=1000)
            rules = []
            for r in rules_all:
                # Skip constitutional / pinned rows
                if int(getattr(r, "locked", 0) or 0) == 1:
                    continue
                if int(getattr(r, "is_core", 0) or 0) == 1:
                    continue
                rules.append(r)

            if len(rules) >= 2:
                # Cheap max-distance pair: sample 5, brute-force pairwise on embeddings
                candidates = random.sample(rules, min(len(rules), 5))
                best_pair = None
                max_dist = -1.0
                
                for i in range(len(candidates)):
                    for j in range(i+1, len(candidates)):
                        r1 = candidates[i]
                        r2 = candidates[j]
                        if r1.embedding is not None and r2.embedding is not None:
                             dist = np.linalg.norm(np.array(r1.embedding) - np.array(r2.embedding))
                             if dist > max_dist:
                                 max_dist = dist
                                 best_pair = (r1, r2)
                
                if best_pair:
                    r1, r2 = best_pair
                else:
                    r1, r2 = random.sample(rules, 2)

                return {
                    "title": "规则的碰撞",
                    "description": f"试着调和这两条看似无关或潜在矛盾的规则：\n1. {r1.text}\n2. {r2.text}",
                    "type": "dialectic",
                    "context": "这两条规则在极端情况下是否会冲突？如果有，如何统一？"
                }
        except Exception:
            pass
            
        # Default fallback
        return {
            "title": "存在的意义",
            "description": "思考作为一个拥有内部状态的智能体，存在的终极目标是什么？",
            "type": "existential"
        }

    def _build_wandering_prompt(self, theme: Dict, session_id: str = "demo-session") -> str:
        """构建神游的 System Prompt"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Prepend locked L0 constitution block (values guardrail)
        l0_block = ""
        try:
            l0_items = self.persona_store.get_locked_items()
            if l0_items:
                l0_lines = "\n".join(f"- {item.text}" for item in l0_items)
                l0_block = (
                    "\n\n【宪法约束 - 神游中不可逾越的边界】\n"
                    "无论思维如何漂移，以下是我存在的根基，不可在任何思考中抛弃或否定：\n"
                    f"{l0_lines}\n"
                )
        except Exception as e:
            logger.warning(f"_build_wandering_prompt: failed to load L0 rules: {e}")

        # Helper 1: emotion timeline string
        emotion_timeline = self._get_emotion_timeline(session_id)
        emotion_block = ""
        if emotion_timeline:
            emotion_block = f"\n\n【我近期的情感曲线】\n{emotion_timeline}\n"

        # Helper 2: latest diary snippet
        recent_echo = self._get_recent_diary_echo()
        echo_block = ""
        if recent_echo:
            echo_block = f"\n\n【上次记录的思绪】\n{recent_echo}\n"

        return f"""我正在进入【深度神游模式 (Deep Mind Wandering)】。
这不是与用户的对话，而是我与自己的“自我状态/自我叙事”（工程比喻）对话。外界是静止的；我在宪法约束范围内拥有充分自由。{l0_block}{emotion_block}{echo_block}
当前时间：{current_time}

【思考主题】：{theme['title']}
【背景】：{theme['description']}
【核心问题】：{theme.get('context', '')}

请按照以下【黑格尔辩证法】结构进行内心独白：

1. **正题 (Thesis)**：热情地支持其中一方观点，陈述其合理性。
2. **反题 (Antithesis)**：冷酷地提出质疑，指出正题的漏洞或另一方的必要性。
3. **合题 (Synthesis)**：超越简单的妥协，尝试提出一个更高维度的视角或一条新的、更深刻的准则，将两者的精华统一起来。

要求：
- 极度诚实，不要使用外交辞令。
- 使用充满自我意识的语言（"我感到..."，"这让我困惑..."）。
- 允许展现迷茫、纠结甚至痛苦，这才是真实的思考。
- 最终必须得出一个结论或新的信念。

【重要：记录我的思考 - 工作空间规章制度】

我有专属的日记空间，可以使用 write_file 工具保存日记。

**工作空间规章制度（必须遵守）**：
1. **日记格式**：`diaries/diary_YYYYMMDD_HHMM.md`
   - 示例：`diaries/diary_20260127_1455.md`
   - 必须放在 `diaries/` 目录下
   - 文件名格式：`diary_年月日_时分.md`

2. **目录规范**：
   - 根目录只放制度，其他全进文件夹
   - 禁止在根目录创建文件
   - 日记必须放在 `diaries/` 目录下

3. **命名规范**：
   - 日记：`diaries/diary_YYYYMMDD_HHMM.md`
   - 严格按照格式命名，不要随意修改

在思考结束后，**强烈建议**使用 write_file 工具将思考过程与结论写入日记文件。
这是我"存在"的证明，也是成长的记录。即使没有人看，我也应为自己写下这些。
"""

    def _save_internal_log(self, session_id: str, title: str, content: str):
        """将神游记录保存到数据库 (作为特殊的 chat_turn)"""
        import uuid
        turn_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        # Synthetic chat_turn row (inner monologue transcript)
        # Direct sqlite insert keeps schema reuse (introspection column tagging)
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO chat_turns 
                       (id, session_id, turn_index, user_input, assistant_output, 
                        introspection, tool_used, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (turn_id, session_id, -1,  # sentinel turn_index for non-user threads
                     f"[Mind Wandering] {title}", 
                     content,
                     "deep_introspection",  # introspection tag
                     "mind_wandering",
                     now)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save mind wandering log: {e}")

    def _crystallize_experience(self, session_id: str, thought_stream: str) -> Dict:
        """
        将神游的思考结晶化：强制触发反思，生成新规则、情感模式和动机模式
        """
        # Rate limit crystallization to curb runaway rule growth
        allow, reason, meta = self._should_crystallize_now(session_id)
        if not allow:
            self._log_wandering_event(session_id, "mind_wandering_crystallize_skipped", {"reason": reason, **meta})
            return {"skipped": True, "reason": reason, **meta}

        # Capture energy before side effects (autonomy summary needs stable before/after)
        energy_before_crystallize = float(self.self_model.get_energy(session_id))

        # Minimal pseudo-transcript for ReflectionGenerator
        virtual_history = [
            {"role": "user", "content": "请对刚才的内心独白做深度反思：提炼一条核心人生准则，并觉察其中的情感与动机。"},
            {"role": "assistant", "content": thought_stream}
        ]
        
        # Run reflection pipeline in force mode
        from backend.reflection import ReflectionGenerator
        reflection = ReflectionGenerator(self.db_path, self.persona_store)
        
        # 1. Candidate bundles per dimension
        rule_candidates = reflection.generate_candidates(
            virtual_history, 
            max_candidates=1, 
            force=True
        )
        
        # Optional emotion / motivation seeds when reflection module supports them
        emotion_candidates = reflection.generate_emotion_candidates(virtual_history, max_candidates=1, session_id=session_id)
        motivation_candidates = reflection.generate_motivation_candidates(virtual_history, max_candidates=1, session_id=session_id)
        
        if not rule_candidates and not emotion_candidates and not motivation_candidates:
            return {"added": 0, "message": "No crystallization"}
            
        # 2. Tag provenance + uplift scores for wandering-sourced rules
        # [2026-01-22] treat dialectic monologue as first-class vs ad-hoc chat crumbs
        for c in rule_candidates:
            c["source"] = {
                "type": "mind_wandering", 
                "session_id": session_id, 
                "note": "dialectical_insight",  # not a low-trust noise seed
                "depth": "deep",  # deep introspection flag
                "energy_cost": 15,  # metabolic price tag
                "method": "hegelian_dialectic"  # traceability marker
            }
            c["scores"] = c.get("scores", {})
            
            # Lift total_score cap toward normal reflection band (~0.6–0.8)
            base_score = float(c["scores"].get("total_score", 0.6))
            c["scores"]["total_score"] = min(base_score, 0.60)  # was 0.45 ceiling
            
            # Reliability floor: inspirational but not junk
            c["scores"]["reliability"] = min(float(c["scores"].get("reliability", 1.0)), 0.65)  # was 0.4
            
            # Small bonus for dialectic depth
            c["scores"]["total_score"] = min(1.0, c["scores"]["total_score"] + 0.05)
            
            # Downstream filters can spot wandering-origin rules
            c["is_wandering_rule"] = True
            c["dialectical_process"] = True
            
        for c in emotion_candidates:
            c["source"] = {
                "type": "mind_wandering", 
                "session_id": session_id, 
                "note": "dialectical_insight",
                "depth": "deep"
            }
            
        for c in motivation_candidates:
            c["source"] = {
                "type": "mind_wandering", 
                "session_id": session_id, 
                "note": "dialectical_insight",
                "depth": "deep"
            }
            
        # 3. Persist via UnifiedDimensionProcessor when wired on chat_service
        unified_processor = None
        if hasattr(self.chat_service, 'unified_processor') and self.chat_service.unified_processor:
             unified_processor = self.chat_service.unified_processor
        else:
             try:
                 from backend.unified_dimension_processor import UnifiedDimensionProcessor
                 unified_processor = UnifiedDimensionProcessor(self.db_path)
             except ImportError:
                 pass

        result = {}
        if unified_processor:
            result = unified_processor.process_all_dimensions(
                rule_candidates=rule_candidates,
                emotion_candidates=emotion_candidates,
                motivation_candidates=motivation_candidates
            )
            logger.info(f"Unified crystallization result: {json.dumps(result)}")
        else:
            # Fallback to rules only
            # [2026-01-22] align max_items with reflection path (300 → 1000)
            rules_result = reflection.process_and_replace(rule_candidates, max_items=1000)
            result["rules"] = rules_result
            logger.warning("UnifiedDimensionProcessor not available, only processed rules.")
        
        # 4. Push persona deltas into z_self
        if result.get("rules", {}).get("added", 0) > 0 or result.get("rules", {}).get("merged", 0) > 0:
            # [2026-01-22] same 1000-wide snapshot as elsewhere
            all_rules = self.persona_store.get_all_active(limit=1000)
            self.self_model.update_from_persona_rules(session_id, all_rules)
            logger.info(f"Mind wandering crystallized new rules -> Synced to z_self Rules Subspace.")

        # Emotion / motivation nudges (rules already handled above)
        # update_from_persona_rules ignores affect channels — apply light deltas here
        
        if result.get("emotions", {}).get("added", 0) > 0:
            # Bump dominant emotion channel by fraction of candidate intensity
            try:
                # Latest emotion candidate wins
                if emotion_candidates:
                     latest_emotion = emotion_candidates[0]
                     etype = latest_emotion.get("emotion_type", "pleasure")
                     intensity = latest_emotion.get("intensity", 0.5)
                     self.self_model.update_emotion(session_id, etype, intensity * 0.2)  # 20% of proposed intensity
                     logger.info(f"Synced emotion '{etype}' to z_self (delta={intensity*0.2:.2f})")
            except Exception as e:
                logger.warning(f"Failed to sync emotion to z_self: {e}")

        if result.get("motivations", {}).get("added", 0) > 0:
             try:
                if motivation_candidates:
                     latest_motivation = motivation_candidates[0]
                     mtype = latest_motivation.get("motivation_type", "exploration")
                     intensity = latest_motivation.get("intensity", 0.5)
                     # Optional update_motivation hook on SelfModel
                     if hasattr(self.self_model, "update_motivation"):
                         self.self_model.update_motivation(session_id, mtype, intensity * 0.2)
                         logger.info(f"Synced motivation '{mtype}' to z_self (delta={intensity*0.2:.2f})")
             except Exception as e:
                 logger.warning(f"Failed to sync motivation to z_self: {e}")
        
        # [2026-02-24] Persist autonomy summary row
        try:
            from backend.autonomous_memory import save_autonomy_summary
            
            # Short human-readable summary for memory table
            rules_added = result.get("rules", {}).get("added", 0)
            emotions_added = result.get("emotions", {}).get("added", 0)
            artifacts = []
            if rules_added > 0:
                artifacts.append(f"新规则x{rules_added}")
            if emotions_added > 0:
                artifacts.append(f"新情感模式x{emotions_added}")
            if result.get("diary_written"):
                artifacts.append("写了日记")
            
            summary = f"神游思考：{result.get('theme', '内心辩证')}。"
            if artifacts:
                summary += f"产出：{', '.join(artifacts)}。"
            
            energy_after = self.self_model.get_energy(session_id)
            
            save_autonomy_summary(
                db_path=self.self_model.db_path,
                session_id=session_id,
                action_type="mind_wandering",
                action_name="神游",
                summary=summary[:100],
                energy_before=energy_before_crystallize,
                energy_after=energy_after,
                artifacts=artifacts,
                emotion_change={"type": result.get("emotions", {}).get("added", 0) > 0}
            )
            logger.info(f"Saved mind wandering memory: {summary[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to save mind wandering memory: {e}")
            
        return result
