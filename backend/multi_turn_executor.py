#!/usr/bin/env python3
"""
多轮执行器 - Agent的主动多轮思考机制

[2026-01-27] 解决问题：让Agent能够主动进行多轮思考，而不仅仅是被动的工具调用循环

设计理念：
1. 与 chat_service 意图多轮一致：续轮须输出 `[S44_CONTINUE]`（口语不再作为续轮信号）
2. 复杂任务可分解为多轮 LLM 调用；每轮应有明确进展
3. 收束优先 `[S44_COMPLETE]`，也可配合「综上所述」等（与本模块 COMPLETE_MARKERS 一致）
4. 主对话路径通常走 chat_service 内联循环；本类供独立调用或 should_use_multi_turn 复杂度判断

使用方式：
chat_service 主要使用 `should_use_multi_turn()`；若直接调用 `execute()`，须遵守同上括号协议
"""

import logging
from backend.autonomy_gate import is_autonomous_execution_paused
import json
import re
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ThinkingPhase(Enum):
    """思考阶段"""
    ANALYZE = "analyze"      # analyze the ask
    PLAN = "plan"            # plan sub-steps
    EXECUTE = "execute"      # execute / tool phase
    SYNTHESIZE = "synthesize"  # merge partials
    REVIEW = "review"        # critique / tighten


@dataclass
class ThinkingTurn:
    """单轮思考"""
    turn_id: int
    phase: ThinkingPhase
    goal: str
    input_context: str
    output: Optional[str] = None
    tool_calls: List[str] = field(default_factory=list)
    needs_more_turns: bool = False
    next_goal: Optional[str] = None


@dataclass
class MultiTurnResult:
    """多轮执行结果"""
    task: str
    turns: List[ThinkingTurn]
    final_response: str
    total_llm_calls: int
    status: str  # completed, max_turns_reached, error


class MultiTurnExecutor:
    """
    多轮执行器
    
    让Agent能够主动进行多轮思考，完成复杂任务
    """
    
    # Continue markers (aligned with chat_service; used in detect_thinking_state)
    CONTINUE_MARKERS = [
        "[S44_CONTINUE]",
        "[CONTINUE_THINKING]",  # legacy alias
    ]
    
    # Completion markers: bracket tags first, then natural-language closers
    COMPLETE_MARKERS = [
        "[S44_COMPLETE]",
        "[THINKING_COMPLETE]",
        "综上所述",
        "总结一下",
        "最终答案",
        "结论是",
        "基于以上分析",
        "回答你的问题",
    ]
    
    def __init__(
        self, 
        llm_caller: Callable,
        tool_executor: Optional[Callable] = None,
        max_turns: int = 5,
        self_model = None
    ):
        """
        初始化多轮执行器
        
        Args:
            llm_caller: LLM调用函数，签名: (messages, **kwargs) -> (response_text, tool_calls, usage)
            tool_executor: 工具执行函数，签名: (tool_calls, messages) -> messages_with_results
            max_turns: 最大思考轮数
            self_model: Agent的自我模型
        """
        self.llm_caller = llm_caller
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.self_model = self_model
    
    def should_use_multi_turn(self, user_input: str, task_analysis: Optional[Dict] = None) -> bool:
        """
        判断是否应该使用多轮执行
        
        Args:
            user_input: 用户输入
            task_analysis: 任务分析结果（如来自SelfAgent）
            
        Returns:
            是否应该使用多轮执行
        """
        # Heuristic: complex-task markers
        complex_markers = [
            '分析', '研究', '设计', '实现', '完整', '详细', 
            '全面', '深入', '系统', '方案', '计划', '多个',
            '并且', '同时', '以及', '还要', '另外'
        ]
        
        # Heuristic: simple Q&A markers
        simple_markers = [
            '是什么', '什么是', '解释', '定义', '为什么',
            '告诉我', '查一下', '搜索'
        ]
        
        # Score keyword hints
        complex_count = sum(1 for m in complex_markers if m in user_input)
        simple_count = sum(1 for m in simple_markers if m in user_input)
        
        # Prefer analyzer-provided complexity when present
        if task_analysis:
            complexity = task_analysis.get("complexity", 0.5)
            if complexity > 0.7:
                return True
        
        # Keyword ratio + length
        if complex_count >= 2 or len(user_input) > 200:
            return True
        
        if simple_count > 0 and complex_count == 0:
            return False
        
        return complex_count > simple_count
    
    def detect_thinking_state(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        检测Agent的回复中是否表示需要继续思考
        
        Returns:
            (needs_more_turns, next_goal)
        """
        # Stop if any completion marker present
        for marker in self.COMPLETE_MARKERS:
            if marker in response:
                return False, None
        
        # Otherwise look for continue markers
        for marker in self.CONTINUE_MARKERS:
            if marker in response:
                # Extract optional next-goal snippet
                next_goal = self._extract_next_goal(response, marker)
                return True, next_goal
        
        # Very short reply with question → assume more context needed
        if len(response) < 100 and "?" in response:
            return True, "需要更多信息来回答"
        
        return False, None
    
    def _extract_next_goal(self, response: str, marker: str) -> Optional[str]:
        """从回复中提取下一步目标"""
        # Text after marker
        idx = response.find(marker)
        if idx != -1:
            after_marker = response[idx + len(marker):].strip()
            # First clause as goal hint
            sentences = re.split(r'[。！？\n]', after_marker)
            if sentences:
                return sentences[0].strip()[:100]
        return None
    
    def execute(
        self,
        user_input: str,
        session_id: str,
        initial_messages: List[Dict],
        system_prompt: str,
        **llm_kwargs
    ) -> MultiTurnResult:
        """
        执行多轮思考
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            initial_messages: 初始消息列表
            system_prompt: 系统提示词
            **llm_kwargs: LLM调用的其他参数
            
        Returns:
            MultiTurnResult
        """
        turns: List[ThinkingTurn] = []
        messages = list(initial_messages)
        total_llm_calls = 0
        current_phase = ThinkingPhase.ANALYZE
        
        # Append multi-turn protocol to system prompt
        multi_turn_instruction = self._build_multi_turn_instruction(user_input)
        
        # Replace / prepend system message
        enhanced_system_prompt = f"{system_prompt}\n\n{multi_turn_instruction}"
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = enhanced_system_prompt
        else:
            messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
        
        # Multi-turn loop
        current_turn = 0
        accumulated_response = ""
        
        while current_turn < self.max_turns:
            current_turn += 1
            # Respect autonomy pause gate
            if is_autonomous_execution_paused():
                logger.info("[MultiTurn] Autonomous execution paused, exiting loop")
                break
            total_llm_calls += 1
            
            logger.info(f"[MultiTurn] Turn {current_turn}/{self.max_turns}, phase={current_phase.value}")
            
            # LLM call
            try:
                result = self.llm_caller(messages, **llm_kwargs)
                # [2026-02-26] Support 3- vs 4-tuple llm_caller returns
                if len(result) == 4:
                    response_text, tool_calls, usage, reasoning_content = result
                else:
                    response_text, tool_calls, usage = result
                    reasoning_content = ""
            except Exception as e:
                logger.error(f"[MultiTurn] LLM call failed: {e}")
                return MultiTurnResult(
                    task=user_input,
                    turns=turns,
                    final_response=accumulated_response or f"执行出错: {str(e)}",
                    total_llm_calls=total_llm_calls,
                    status="error"
                )
            
            # Tool round: executor mutates messages, then retry LLM
            tool_names = []
            if tool_calls and self.tool_executor:
                messages = self.tool_executor(tool_calls, messages)
                tool_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
                # Skip think-state until post-tool assistant message exists
                continue
            
            # Continue vs complete?
            needs_more, next_goal = self.detect_thinking_state(response_text)
            if not needs_more:
                # Terminal assistant turn
                accumulated_response += f"\n{response_text}"
                break
            
            # Bookkeeping for another think turn
            current_phase = self._determine_next_phase(current_phase, tool_names)
            turn = ThinkingTurn(
                turn_id=current_turn,
                phase=current_phase,
                goal=self._get_turn_goal(current_turn, current_phase, next_goal),
                input_context=self._summarize_context(messages),
                output=response_text,
                tool_calls=tool_names,
                needs_more_turns=True,
                next_goal=next_goal
            )
            turns.append(turn)
            
            # Append assistant message for next iteration
            messages.append({
                "role": "assistant",
                "content": response_text,
                "tool_calls": tool_calls if tool_calls else None
            })
        
        # Final stitched response
        final_response = accumulated_response or self._summarize_final_response(turns)
        status = "completed" if current_turn <= self.max_turns else "max_turns_reached"
        
        return MultiTurnResult(
            task=user_input,
            turns=turns,
            final_response=final_response,
            total_llm_calls=total_llm_calls,
            status=status
        )
    
    def _build_multi_turn_instruction(self, user_input: str) -> str:
        """构建多轮思考指令"""
        return (
            "你正在执行一个可能需要多轮思考的复杂任务。\n"
            "如果需要继续思考，请在回复末尾单独一行输出 [S44_CONTINUE]。\n"
            "如果任务完成，请输出 [S44_COMPLETE]。\n"
            "任务：" + user_input[:200]
        )
    
    def _determine_next_phase(self, current_phase: ThinkingPhase, tool_names: List[str]) -> ThinkingPhase:
        """确定下一个思考阶段"""
        if tool_names:
            return ThinkingPhase.EXECUTE
        # Default: rotate enum order
        phases = list(ThinkingPhase)
        current_idx = phases.index(current_phase)
        next_idx = (current_idx + 1) % len(phases)
        return phases[next_idx]
    
    def _get_turn_goal(self, turn_id: int, phase: ThinkingPhase, next_goal: Optional[str]) -> str:
        """获取当前轮次的目标"""
        if next_goal:
            return next_goal
        return f"第{turn_id}轮 - {phase.value}"
    
    def _summarize_context(self, messages: List[Dict]) -> str:
        """总结上下文"""
        recent = messages[-3:] if len(messages) >= 3 else messages
        summary = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                summary.append(f"用户: {content[:100]}")
            elif role == "assistant":
                summary.append(f"助手: {content[:100]}")
        return "\n".join(summary)
    
    def _summarize_final_response(self, turns: List[ThinkingTurn]) -> str:
        """总结最终回复"""
        if not turns:
            return "任务执行完成"
        last_turn = turns[-1]
        return f"经过{len(turns)}轮思考，任务完成。最后阶段：{last_turn.phase.value}"