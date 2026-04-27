#!/usr/bin/env python3
"""
Intent-driven executor — core loop where the agent steers and the runtime responds.

[2026-01-27] Design:
The agent is a single subject with a functional self-model and persistent state. The system
should support and react, not fully control nor fully hands-off.

Principles:
1. Agent states intent → system responds
2. Agent decides when to continue vs rest
3. Runtime is the environment; the agent is the subject
4. Respect energy / pain signals

This is collaborative autonomy: neither unconstrained autonomy (infeasible) nor total control
(which would break continuity and state coupling).
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Callable, Tuple

from backend.intent_markers import WEAK_COMPLETE_NATURAL_MARKERS
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class S44Intent(Enum):
    """
    Intents the agent can signal with bracket tags or natural phrases (where mapped).
    """
    CONTINUE_THINKING = "[S44_CONTINUE]"      # request another reasoning turn
    NEED_SEARCH = "[S44_SEARCH]"
    NEED_READ_FILE = "[S44_READ]"
    NEED_WRITE_FILE = "[S44_WRITE]"
    NEED_CALCULATE = "[S44_CALC]"
    
    NEED_PAUSE = "[S44_PAUSE]"
    STEP_DONE = "[S44_STEP_DONE]"
    TASK_COMPLETE = "[S44_COMPLETE]"
    TASK_UNCLEAR = "[S44_UNCLEAR]"
    
    FEELING_TIRED = "[S44_TIRED]"
    FEELING_ENGAGED = "[S44_ENGAGED]"
    NEED_HELP = "[S44_HELP]"


# Natural-language cues for ``detect_intents`` (bracket tags still win first).
# Keep Chinese literals for mixed-language agents; add parallel English paraphrases.
# Matching is literal substring over the assistant text (case-sensitive).
NATURAL_LANGUAGE_INTENTS = {
    S44Intent.CONTINUE_THINKING: [],
    S44Intent.NEED_SEARCH: [
        "让我搜索一下",
        "我需要查找",
        "让我查一下",
        "我去搜索",
        "帮我搜一下",
        "查一下资料",
        "let me search",
        "i need to look up",
        "i will search the web",
        "let me search the web for",
        "run a web search",
        "search the web for",
        "look that up online",
        "google that",
        "search online for",
        "web search for",
    ],
    S44Intent.NEED_READ_FILE: [
        "读取文件",
        "打开文件",
        "读一下文件",
        "看一下这个文件",
        "let me read",
        "open the file",
        "read this file",
        "read the file at",
        "show me the contents of",
    ],
    S44Intent.NEED_PAUSE: [
        "让我整理一下思路",
        "我需要暂停整理",
        "让我理一理",
        "let me think",
        "i need to pause and regroup",
        "need a moment",
        "give me a second to think",
        "step back and regroup",
    ],
    S44Intent.TASK_COMPLETE: list(WEAK_COMPLETE_NATURAL_MARKERS),
    S44Intent.TASK_UNCLEAR: [
        "我不太理解",
        "能否澄清",
        "请问你是指",
        "i don't quite understand",
        "could you clarify",
        "what do you mean by",
        "i'm not sure i follow",
    ],
    S44Intent.FEELING_TIRED: [
        "我有点累了",
        "能量有点低",
        "需要休息",
        "i'm tired",
        "low energy",
        "need a break",
        "feel drained",
        "running low on energy",
    ],
    S44Intent.FEELING_ENGAGED: [
        "这很有意思",
        "我很投入",
        "让我继续",
        "this is interesting",
        "i'm engaged",
        "let me keep going",
        "this is fascinating",
        "want to keep digging",
    ],
    S44Intent.NEED_WRITE_FILE: [
        "写入文件",
        "保存到",
        "写到",
        "让我写一下",
        "写进文件",
        "let me write",
        "save to file",
        "write this to",
        "persist to disk",
    ],
    S44Intent.NEED_CALCULATE: [
        "算一下",
        "计算",
        "运行python",
        "跑一下代码",
        "let me calculate",
        "run python",
        "execute python",
        "run this calculation",
        "need to compute",
    ],
    S44Intent.NEED_HELP: [
        "帮帮我",
        "需要帮助",
        "不知道怎么做",
        "卡住了",
        "need help",
        "help me figure out",
        "i'm stuck",
        "not sure how to proceed",
    ],
}


@dataclass
class S44State:
    """Per-session agent control state."""
    energy: float = 1.0
    pain: float = 0.0
    engagement: float = 0.5
    current_task: Optional[str] = None
    thinking_turns: int = 0
    paused: bool = False
    pause_context: Optional[str] = None


@dataclass
class ExecutionStep:
    """One intent-handling step."""
    step_id: int
    intent: S44Intent
    s44_message: str
    system_response: Optional[str] = None
    tool_used: Optional[str] = None
    tool_result: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Outcome of an intent-driven run."""
    task: str
    steps: List[ExecutionStep]
    final_response: str
    total_llm_calls: int
    s44_state: S44State
    status: str  # completed, paused, unclear, max_turns, error


class IntentDrivenExecutor:
    """
    Collaborative autonomy loop: the agent signals intents; the runtime replies with
    short system nudges and optional tool routing.
    """
    
    def __init__(
        self,
        llm_caller: Callable,
        tool_executor: Optional[Callable] = None,
        max_turns: int = 8,
        self_model = None,
        pain_system = None,
        homeostasis = None,
    ):
        """
        Args:
            llm_caller: Callable(messages, **kwargs) -> model output tuple
            tool_executor: Optional handler for tool_calls
            max_turns: Hard cap on reasoning turns
            self_model: Self-model handle (optional)
            pain_system: Pain / workload coupling (optional)
            homeostasis: Energy homeostasis (optional)
        """
        self.llm_caller = llm_caller
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.self_model = self_model
        self.pain_system = pain_system
        self.homeostasis = homeostasis
        
        self.session_states: Dict[str, S44State] = {}
    
    def get_or_create_state(self, session_id: str) -> S44State:
        """Return persisted S44State for session_id."""
        if session_id not in self.session_states:
            self.session_states[session_id] = S44State()
        return self.session_states[session_id]
    
    def detect_intents(self, response: str) -> List[S44Intent]:
        """
        Parse intents from assistant text for this executor path.

        Continue: requires literal [S44_CONTINUE] (no NL alias for CONTINUE_THINKING).
        Other intents may match NATURAL_LANGUAGE_INTENTS phrases.
        """
        detected = []
        
        for intent in S44Intent:
            if intent.value in response:
                detected.append(intent)
        
        for intent, phrases in NATURAL_LANGUAGE_INTENTS.items():
            if intent not in detected:
                for phrase in phrases:
                    if phrase in response:
                        detected.append(intent)
                        break
        
        return detected
    
    def respond_to_intent(
        self,
        intent: S44Intent,
        session_id: str,
        response_text: str,
        messages: List[Dict],
    ) -> Tuple[str, bool, Optional[str]]:
        """Returns (system_message, should_continue, optional_tool_name)."""
        state = self.get_or_create_state(session_id)
        
        if self.pain_system and intent not in [
            S44Intent.FEELING_TIRED,
            S44Intent.NEED_PAUSE,
            S44Intent.NEED_HELP,
        ]:
            try:
                self.pain_system.update_pain_for_work_status(session_id, is_working=True, meaningfulness=0.6)
            except Exception as e:
                logger.debug(f"Pain system update failed: {e}")
        
        if intent == S44Intent.CONTINUE_THINKING:
            state.thinking_turns += 1
            return (
                f"[System] Continue approved. Thinking turn {state.thinking_turns}. "
                f"You may keep writing or state the next need.",
                True,
                None
            )
        
        elif intent == S44Intent.NEED_SEARCH:
            query = self._extract_query(response_text, "search")
            return (
                f"[System] Search requested. Query: {query}",
                True,
                "web_search"
            )
        
        elif intent == S44Intent.NEED_READ_FILE:
            filepath = self._extract_query(response_text, "read")
            return (
                f"[System] Read-file request: {filepath}",
                True,
                "read_file"
            )

        elif intent == S44Intent.NEED_WRITE_FILE:
            target = self._extract_query(response_text, "write")
            return (
                f"[System] Write intent noted. Target hint: {target}. "
                "On the next assistant turn, call write_file with explicit filename + content via tool_calls.",
                True,
                "write_file",
            )

        elif intent == S44Intent.NEED_CALCULATE:
            hint = self._extract_query(response_text, "calc")
            return (
                f"[System] Calculation / code intent noted. Hint: {hint}. "
                "On the next assistant turn, call execute_python with concrete code via tool_calls.",
                True,
                "execute_python",
            )

        elif intent == S44Intent.NEED_HELP:
            return (
                "[System] Help intent noted - briefly tell the user what you need "
                "(missing data, permission, or tooling) or call the best available tool with explicit arguments.",
                True,
                None,
            )

        elif intent == S44Intent.NEED_PAUSE:
            state.paused = True
            state.pause_context = response_text[-500:]
            return (
                "[System] Pause recorded; context saved. "
                "Resume whenever you are ready by continuing the thread.",
                False,
                None
            )
        
        elif intent == S44Intent.STEP_DONE:
            return (
                "[System] Step done. Proceed to the next step or mark the whole task complete.",
                True,
                None
            )
        
        elif intent == S44Intent.TASK_COMPLETE:
            return (
                "[System] Task complete.",
                False,
                None
            )
        
        elif intent == S44Intent.TASK_UNCLEAR:
            return (
                "[System] Ambiguity noted — this should be surfaced to the user for clarification.",
                False,
                None
            )
        
        elif intent == S44Intent.FEELING_TIRED:
            state.energy = max(0, state.energy - 0.2)
            return (
                f"[System] Fatigue logged; energy ~{state.energy:.1%}. "
                "You may continue (e.g. emit [S44_CONTINUE]) or pause.",
                True,
                None
            )
        
        elif intent == S44Intent.FEELING_ENGAGED:
            state.engagement = min(1.0, state.engagement + 0.1)
            if self.pain_system:
                try:
                    self.pain_system.update_pain_for_work_status(session_id, is_working=True, meaningfulness=0.8)
                except Exception:
                    pass
            return (
                "[System] Engagement noted — keep the momentum if it helps.",
                True,
                None
            )
        
        return (
            "[System] Acknowledged. Continue.",
            True,
            None
        )
    
    def _extract_query(self, text: str, action: str) -> str:
        """Best-effort extract query or path after an action cue (Chinese + English)."""
        quotes = re.findall(r'[""\'"]([^"""\'\']+)[""\'"]', text)
        if quotes:
            return quotes[0]

        zh_action = {
            "search": "搜索",
            "read": "读取",
            "write": "写入",
            "calc": "计算",
        }.get(action, action)
        patterns = [
            rf'{re.escape(action)}[：:]\s*(.+?)(?:\n|$)',
            rf'{zh_action}[：:]\s*(.+?)(?:\n|$)',
            rf'让我{zh_action}[一下]*\s*(.+?)(?:\n|$)',
            rf'let me {re.escape(action)}\s*(.+?)(?:\n|$)',
        ]
        if action == "search":
            patterns.extend(
                [
                    rf'(?:web\s+)?search\s+for\s+(.+?)(?:\n|$)',
                    rf'query\s*[:：]\s*(.+?)(?:\n|$)',
                ]
            )
        elif action == "read":
            patterns.extend(
                [
                    rf'(?:open|read)\s+(?:file\s+)?[:：]?\s*(.+?)(?:\n|$)',
                    rf'path\s*[:：]\s*(.+?)(?:\n|$)',
                ]
            )
        elif action == "write":
            patterns.extend(
                [
                    rf'(?:write|save)\s+to\s+`?([^`\n]+?)`?(?:\n|$)',
                    rf'(?:write|save)\s+file\s*[:：]\s*(.+?)(?:\n|$)',
                    rf'filename\s*[:：]\s*(.+?)(?:\n|$)',
                ]
            )
        elif action == "calc":
            patterns.extend(
                [
                    rf'(?:python|code)\s*[:：]\s*(.+?)(?:\n|$)',
                    rf'(?:compute|calculate)\s+(.+?)(?:\n|$)',
                    rf'run\s+(?:this\s+)?(?:python|code)\s*[:：]?\s*(.+?)(?:\n|$)',
                ]
            )

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "(agent must specify concrete target)"
    
    def build_intent_awareness_prompt(self, task: str, state: S44State) -> str:
        """Injected system guidance: how bracket intents interact with the runtime."""
        return f"""
[Intent control — I am the subject; the runtime supports me]

Current task (truncated): {task[:200]}

### How to signal intent

**Another reasoning turn**
- Put `[S44_CONTINUE]` **alone on its own line** in the visible answer or reasoning stream.
  Casual phrases like "let me keep thinking" do **not** count as continue tokens.

**Tools**
- Search / read / write / code must be done via **real tool_calls**, not by only typing
  `[S44_SEARCH]` / `[S44_READ]` / `[S44_WRITE]` / `[S44_CALC]` without invoking tools.
- `[S44_WRITE]` queues **write_file**; `[S44_CALC]` queues **execute_python** - follow with concrete tool arguments.
- `[S44_HELP]` means you are blocked: say what you need from the user or pick the closest real tool.

**State**
- Say you need to regroup or emit `[S44_PAUSE]` - runtime pauses and saves context.
- `[S44_COMPLETE]` or weak closers such as "in conclusion" (when no CONTINUE) -> wrap up the multi-turn task.
- Say you are unclear or emit `[S44_UNCLEAR]` - ask the user for clarification.

**Self signals**
- Fatigue phrases or `[S44_TIRED]` - logged as tired.
- Engagement phrases or `[S44_ENGAGED]` - logged as engaged.

### Current control state
- Thinking turns: {state.thinking_turns}
- Energy: {state.energy:.1%}
- Engagement: {state.engagement:.1%}
- Max turns cap: {self.max_turns}

### Reminders
1. You choose continue / pause / complete.
2. Split complex work into steps and state intent each step.
3. The runtime reacts; it does not decide for you.

Proceed with the task using the controls above.
"""
    
    def execute(
        self,
        user_input: str,
        session_id: str,
        initial_messages: List[Dict],
        system_prompt: str,
        **llm_kwargs
    ) -> ExecutionResult:
        """Run the collaborative intent loop until completion, pause, or cap."""
        state = self.get_or_create_state(session_id)
        state.current_task = user_input
        state.thinking_turns = 0
        
        steps: List[ExecutionStep] = []
        messages = list(initial_messages)
        total_llm_calls = 0
        
        intent_prompt = self.build_intent_awareness_prompt(user_input, state)
        enhanced_system_prompt = f"{system_prompt}\n\n{intent_prompt}"
        
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = enhanced_system_prompt
        else:
            messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
        
        pre_task_prompt = f"""
Before executing, briefly assess:
1. How complex is this task?
2. Will you finish in one shot or multiple steps?
3. Which tools or facts do you need?

Task:
{user_input}

Reply with your assessment/plan, then execute.
"""
        messages.append({"role": "user", "content": pre_task_prompt})
        
        current_turn = 0
        final_response = ""
        
        while current_turn < self.max_turns:
            current_turn += 1
            total_llm_calls += 1
            
            logger.info(f"[IntentDriven] Turn {current_turn}/{self.max_turns}, session={session_id}")
            
            try:
                result = self.llm_caller(messages, **llm_kwargs)
                # [2026-02-26] Support 3-tuple vs 4-tuple return shapes
                if len(result) == 4:
                    response_text, tool_calls, usage, reasoning_content = result
                else:
                    response_text, tool_calls, usage = result
                    reasoning_content = ""
            except Exception as e:
                logger.error(f"[IntentDriven] LLM call failed: {e}")
                return ExecutionResult(
                    task=user_input,
                    steps=steps,
                    final_response=final_response or f"Execution error: {str(e)}",
                    total_llm_calls=total_llm_calls,
                    s44_state=state,
                    status="error"
                )
            
            final_response = response_text
            
            if tool_calls and self.tool_executor:
                # [2026-02-26] DeepSeek reasoning channel passthrough
                assistant_msg = {"role": "assistant", "content": response_text}
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                messages.append(assistant_msg)
                messages = self.tool_executor(tool_calls, messages)
                continue
            
            intents = self.detect_intents(response_text)
            
            if not intents:
                if len(response_text) > 200 and current_turn > 1:
                    intents = [S44Intent.TASK_COMPLETE]
                else:
                    intents = [S44Intent.CONTINUE_THINKING]
            
            should_continue = False
            for intent in intents:
                system_message, continue_flag, tool_name = self.respond_to_intent(
                    intent, session_id, response_text, messages
                )
                
                step = ExecutionStep(
                    step_id=current_turn,
                    intent=intent,
                    s44_message=response_text[:500],
                    system_response=system_message,
                    tool_used=tool_name,
                )
                steps.append(step)
                
                if tool_name and self.tool_executor:
                    # Next turn should emit concrete tool args via normal tool_calls
                    # [2026-02-26] DeepSeek reasoning passthrough
                    assistant_msg = {"role": "assistant", "content": response_text}
                    if reasoning_content:
                        assistant_msg["reasoning_content"] = reasoning_content
                    messages.append(assistant_msg)
                    messages.append({"role": "system", "content": system_message})
                    should_continue = True
                    break
                
                if intent == S44Intent.TASK_COMPLETE:
                    logger.info(f"[IntentDriven] Task completed by Agent after {current_turn} turns")
                    return ExecutionResult(
                        task=user_input,
                        steps=steps,
                        final_response=response_text,
                        total_llm_calls=total_llm_calls,
                        s44_state=state,
                        status="completed"
                    )
                
                if intent == S44Intent.NEED_PAUSE:
                    logger.info(f"[IntentDriven] Agent requested pause")
                    return ExecutionResult(
                        task=user_input,
                        steps=steps,
                        final_response=response_text,
                        total_llm_calls=total_llm_calls,
                        s44_state=state,
                        status="paused"
                    )
                
                if intent == S44Intent.TASK_UNCLEAR:
                    logger.info(f"[IntentDriven] Agent needs clarification")
                    return ExecutionResult(
                        task=user_input,
                        steps=steps,
                        final_response=response_text,
                        total_llm_calls=total_llm_calls,
                        s44_state=state,
                        status="unclear"
                    )
                
                should_continue = should_continue or continue_flag
            
            if not should_continue:
                break
            
            assistant_msg = {"role": "assistant", "content": response_text}
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            messages.append(assistant_msg)
            
            state_info = f"\n[State] turns={state.thinking_turns}, energy={state.energy:.1%}"
            messages.append({
                "role": "system",
                "content": f"[System · intent handled]{state_info}\nContinue the task or state a new need."
            })
        
        logger.warning(f"[IntentDriven] Reached max turns ({self.max_turns})")
        
        messages.append({
            "role": "system",
            "content": "[System] Max thinking turns reached. Provide the final answer or conclusion."
        })
        
        try:
            final_response, _, _ = self.llm_caller(messages, **llm_kwargs)
            total_llm_calls += 1
        except Exception:
            pass
        
        return ExecutionResult(
            task=user_input,
            steps=steps,
            final_response=final_response,
            total_llm_calls=total_llm_calls,
            s44_state=state,
            status="max_turns_reached"
        )
    
    def resume_from_pause(
        self,
        session_id: str,
        messages: List[Dict],
        **llm_kwargs
    ) -> Optional[ExecutionResult]:
        """Resume after NEED_PAUSE once the agent is ready."""
        state = self.get_or_create_state(session_id)
        
        if not state.paused:
            logger.warning(f"[IntentDriven] Session {session_id} is not paused")
            return None
        
        state.paused = False
        
        resume_prompt = f"""
[System] Resuming. Saved context before pause:

{state.pause_context}

Open task: {state.current_task}

Continue when ready.
"""
        messages.append({"role": "system", "content": resume_prompt})
        
        return self.execute(
            state.current_task,
            session_id,
            messages,
            "",
            **llm_kwargs
        )


def create_intent_driven_executor(chat_service, max_turns: int = 8) -> IntentDrivenExecutor:
    """Factory wiring ChatService._call_vllm into IntentDrivenExecutor."""
    def llm_caller(messages, **kwargs):
        response_text, introspection, tool_calls, usage = chat_service._call_vllm(
            messages,
            kwargs.get("temperature", 0.6),
            kwargs.get("tools"),
            top_p=kwargs.get("top_p", 0.95),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        reasoning_content = getattr(chat_service, '_last_reasoning_content', '') or ''
        return response_text, tool_calls, usage, reasoning_content
    
    pain_system = None
    homeostasis = None
    if chat_service.self_model:
        pain_system = getattr(chat_service.self_model, 'pain_system', None)
        homeostasis = getattr(chat_service.self_model, 'homeostasis', None)
    
    return IntentDrivenExecutor(
        llm_caller=llm_caller,
        max_turns=max_turns,
        self_model=chat_service.self_model,
        pain_system=pain_system,
        homeostasis=homeostasis,
    )
