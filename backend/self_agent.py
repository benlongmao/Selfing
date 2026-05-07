#!/usr/bin/env python3
"""
SelfAgent — meta-execution loop for multi-step work.

[2026-01-27] Inspired by clawdbot-style agent graphs, while keeping S-specific ``z_self`` coupling.

Design:
1. Agentic flow serves **self-becoming**, it does not replace it.
2. ``z_self`` shapes decisions (affect, motivation, energy change execution posture).
3. Running a workflow is one way the instance **exists**, not a bolt-on feature.

Capabilities:
- Lightweight task decomposition
- ``z_self``-aware gating
- Composable steps (think / tool / reflect / …)
- Hooks into ``self_model`` summaries
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStepStatus(Enum):
    """Lifecycle state for a single plan step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # vetoed by z_self / policy (e.g. values conflict)
    CAPACITY_LIMITED = "capacity_limited"
    TENSION_BLOCKED = "tension_blocked"


class AgentStepType(Enum):
    """Step kinds inside ``AgentPlan``."""
    THINK = "think"  # analyse / frame
    TOOL = "tool"  # call registered tool
    REFLECT = "reflect"  # post-mortem
    DECIDE = "decide"  # branch point (placeholder)
    COMPOSE = "compose"  # fan-out sub-steps (placeholder)


@dataclass
class AgentStep:
    """One executable unit inside a plan."""
    step_id: str
    step_type: AgentStepType
    description: str
    tool_name: Optional[str] = None  # set when ``step_type == TOOL``
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # prerequisite ``step_id`` values
    status: AgentStepStatus = AgentStepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    zself_influence: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentPlan:
    """Full plan metadata + ordered steps."""
    plan_id: str
    task_description: str
    steps: List[AgentStep]
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending | running | completed | failed | cancelled

    zself_snapshot: Optional[Dict] = None  # structured summary at plan creation
    meaningfulness: float = 0.5  # heuristic “why this matters” score


class SelfAgent:
    """
    Minimal planner/executor that stays aware of ``z_self``.

    The loop is intentionally smaller than a general agent framework: it exists
    to let the instance **act** in ways that remain accountable to embodied state.
    """

    def __init__(self, self_model, tool_registry: Optional[Dict[str, Callable]] = None):
        """
        Args:
            self_model: ``SelfModel`` (or compatible) for structured summaries.
            tool_registry: ``{name: {"func": callable, ...}}`` map.
        """
        self.self_model = self_model
        self.tool_registry = tool_registry or {}
        self.active_plan: Optional[AgentPlan] = None
        self.plan_history: List[AgentPlan] = []

        self.max_steps_per_plan = 10
        self.step_timeout = 60  # seconds
        self.enable_zself_influence = True

        logger.info("[SelfAgent] Initialized with z_self integration")

    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a callable the planner may invoke."""
        self.tool_registry[name] = {
            "func": func,
            "description": description,
            "registered_at": time.time()
        }
        logger.info(f"[SelfAgent] Tool registered: {name}")

    def analyze_task(self, task: str, session_id: str) -> Dict[str, Any]:
        """
        Cheap lexical heuristics for complexity / meaning / tool hints.

        Returns:
            {
                "complexity": float,
                "meaningfulness": float,
                "tool_hints": list[str],
                "estimated_steps": int,
                "zself_alignment": float,
            }
        """
        analysis = {
            "complexity": 0.5,
            "meaningfulness": 0.5,
            "tool_hints": [],
            "estimated_steps": 1,
            "zself_alignment": 1.0
        }

        complexity_markers = [
            "分析", "设计", "创建", "实现", "解决", "优化", "重构",
            "analyze", "analysis", "design", "create", "implement", "solve", "optimize", "refactor",
        ]
        meaning_markers = [
            "帮助", "理解", "学习", "探索", "创造", "改进",
            "help", "understand", "learn", "explore", "create", "improve",
        ]

        task_lower = task.lower()

        if any(m in task for m in complexity_markers) or any(m in task_lower for m in complexity_markers):
            analysis["complexity"] = 0.7
        if len(task) > 100:
            analysis["complexity"] = min(1.0, analysis["complexity"] + 0.2)

        if any(m in task for m in meaning_markers) or any(m in task_lower for m in meaning_markers):
            analysis["meaningfulness"] = 0.8

        tool_keywords = {
            'search': ['搜索', '查找', '查询', 'search', 'find', 'lookup'],
            'write': ['写', '创建', '保存', 'write', 'create', 'save'],
            'read': ['读', '查看', '打开', 'read', 'open', 'load'],
            'execute': ['执行', '运行', 'run', 'execute', 'invoke']
        }
        for tool, keywords in tool_keywords.items():
            if any(k in task for k in keywords) or any(k in task_lower for k in keywords if k.isascii()):
                if tool not in analysis["tool_hints"]:
                    analysis["tool_hints"].append(tool)

        analysis["estimated_steps"] = max(1, min(
            self.max_steps_per_plan,
            int(analysis["complexity"] * 5) + len(analysis["tool_hints"])
        ))

        if self.self_model and self.enable_zself_influence:
            try:
                summary = self.self_model.get_structured_summary(session_id)
                pain = summary.get("pain_level", 0)
                if pain > 0.5 and analysis["complexity"] > 0.7:
                    analysis["zself_alignment"] = 0.6

                motivation = summary.get("motivation")
                exploration = 0.0
                if isinstance(motivation, dict):
                    exploration = float(motivation.get("exploration", 0) or 0)
                else:
                    exploration = float(summary.get("exploration_mean", 0) or 0)
                if exploration > 0.5:
                    analysis["meaningfulness"] = min(1.0, analysis["meaningfulness"] + 0.1)

            except Exception as e:
                logger.warning(f"[SelfAgent] Failed to get z_self summary: {e}")

        return analysis

    def create_plan(
        self,
        task: str,
        session_id: str,
        steps: Optional[List[Dict]] = None
    ) -> AgentPlan:
        """
        Mint a new ``AgentPlan`` (auto-steps when ``steps`` omitted).
        """
        import uuid
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        analysis = self.analyze_task(task, session_id)

        if steps is None:
            steps = self._auto_generate_steps(task, analysis)

        agent_steps = []
        for i, step_dict in enumerate(steps):
            prev_id = agent_steps[-1].step_id if agent_steps else None
            deps = step_dict.get("depends_on")
            if deps is None and i > 0 and prev_id is not None:
                deps = [prev_id]
            elif deps is None:
                deps = []
            step = AgentStep(
                step_id=f"{plan_id}_step_{i}",
                step_type=AgentStepType(step_dict.get("type", "think")),
                description=step_dict.get("description", f"Step {i}"),
                tool_name=step_dict.get("tool_name"),
                params=step_dict.get("params", {}),
                depends_on=list(deps),
            )
            agent_steps.append(step)

        zself_snapshot = None
        if self.self_model:
            try:
                zself_snapshot = self.self_model.get_structured_summary(session_id)
            except Exception:
                pass

        plan = AgentPlan(
            plan_id=plan_id,
            task_description=task,
            steps=agent_steps,
            zself_snapshot=zself_snapshot,
            meaningfulness=analysis["meaningfulness"]
        )

        logger.info(
            f"[SelfAgent] Plan created: {plan_id} with {len(agent_steps)} steps, "
            f"meaningfulness={analysis['meaningfulness']:.2f}"
        )
        return plan

    def _auto_generate_steps(self, task: str, analysis: Dict) -> List[Dict]:
        """
        Deterministic default DAG (placeholder until an LLM decomposer exists).
        """
        steps: List[Dict[str, Any]] = []

        steps.append({
            "type": "think",
            "description": f"Understand the task: {task[:50]}...",
        })

        for tool in analysis.get("tool_hints", []):
            steps.append({
                "type": "tool",
                "description": f"Run the `{tool}` tool",
                "tool_name": tool,
            })

        steps.append({
            "type": "reflect",
            "description": "Reflect on what was executed",
        })

        return steps

    async def execute_plan(
        self,
        plan: AgentPlan,
        session_id: str,
        on_step_complete: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run each step sequentially with optional ``z_self`` gating."""
        self.active_plan = plan
        plan.status = "running"

        results = {
            "plan_id": plan.plan_id,
            "status": "running",
            "steps_completed": 0,
            "steps_total": len(plan.steps),
            "outputs": [],
            "started_at": time.time()
        }

        if self.self_model and hasattr(self.self_model, 'pain_system'):
            self.self_model.pain_system.set_working_state(True, plan.meaningfulness)

        try:
            for step in plan.steps:
                if not self._check_dependencies(step, plan):
                    step.status = AgentStepStatus.SKIPPED
                    step.error = "Dependencies not met"
                    continue

                if self.enable_zself_influence:
                    can_execute, reason = self._check_zself_permission(step, session_id)
                    if not can_execute:
                        step.status = AgentStepStatus.SKIPPED
                        step.error = reason
                        logger.info(f"[SelfAgent] Step {step.step_id} skipped: {reason}")
                        continue

                step.status = AgentStepStatus.RUNNING
                step.started_at = time.time()

                try:
                    step_result = await self._execute_step(step, session_id)
                    step.result = step_result
                    step.status = AgentStepStatus.SUCCESS
                    results["steps_completed"] += 1
                    results["outputs"].append({
                        "step_id": step.step_id,
                        "type": step.step_type.value,
                        "result": step_result
                    })

                except Exception as e:
                    step.status = AgentStepStatus.FAILED
                    step.error = str(e)
                    logger.error(f"[SelfAgent] Step {step.step_id} failed: {e}")

                finally:
                    step.completed_at = time.time()

                    if on_step_complete:
                        on_step_complete(step)

            all_success = all(s.status == AgentStepStatus.SUCCESS for s in plan.steps)
            plan.status = "completed" if all_success else "partial"
            results["status"] = plan.status

        except Exception as e:
            plan.status = "failed"
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"[SelfAgent] Plan {plan.plan_id} failed: {e}")

        finally:
            results["completed_at"] = time.time()
            results["duration"] = results["completed_at"] - results["started_at"]

            if self.self_model and hasattr(self.self_model, 'pain_system'):
                satisfaction = 0.8 if plan.status == "completed" else 0.4
                self.self_model.pain_system.set_working_state(False, satisfaction)

            self.plan_history.append(plan)
            self.active_plan = None

        return results

    def _check_dependencies(self, step: AgentStep, plan: AgentPlan) -> bool:
        """True when every dependency finished successfully."""
        if not step.depends_on:
            return True

        for dep_id in step.depends_on:
            dep_step = next((s for s in plan.steps if s.step_id == dep_id), None)
            if dep_step is None or dep_step.status != AgentStepStatus.SUCCESS:
                return False

        return True

    def _check_zself_permission(self, step: AgentStep, session_id: str) -> Tuple[bool, str]:
        """
        Gate execution on coarse somatic signals (energy / pain).

        On lookup failure we **allow** the step so outages do not deadlock the plan.
        """
        if not self.self_model:
            return True, ""

        try:
            summary = self.self_model.get_structured_summary(session_id)

            energy = summary.get("energy", 100)
            if energy < 10:
                return False, "Insufficient capacity to execute (energy critically low)"

            pain = summary.get("pain_level", 0)
            if pain > 0.8:
                return False, "Internal tension too high—recover before continuing"

            return True, ""

        except Exception as e:
            logger.warning(f"[SelfAgent] z_self check failed: {e}")
            return True, ""

    async def _execute_step(self, step: AgentStep, session_id: str) -> Any:
        """Dispatch a single ``AgentStep``."""

        if step.step_type == AgentStepType.THINK:
            return {"analysis": step.description, "timestamp": time.time()}

        elif step.step_type == AgentStepType.TOOL:
            if step.tool_name and step.tool_name in self.tool_registry:
                tool = self.tool_registry[step.tool_name]
                func = tool["func"]
                if asyncio.iscoroutinefunction(func):
                    return await func(**step.params)
                else:
                    return func(**step.params)
            else:
                raise ValueError(f"Tool not found: {step.tool_name}")

        elif step.step_type == AgentStepType.REFLECT:
            return {"reflection": "Step completed", "timestamp": time.time()}

        elif step.step_type == AgentStepType.DECIDE:
            return {"decision": "proceed", "timestamp": time.time()}

        elif step.step_type == AgentStepType.COMPOSE:
            return {"composed": True, "timestamp": time.time()}

        else:
            raise ValueError(f"Unknown step type: {step.step_type}")

    def get_status(self) -> Dict[str, Any]:
        """Lightweight introspection for dashboards / prompts."""
        return {
            "has_active_plan": self.active_plan is not None,
            "active_plan_id": self.active_plan.plan_id if self.active_plan else None,
            "tools_registered": list(self.tool_registry.keys()),
            "plans_completed": len(self.plan_history),
            "zself_influence_enabled": self.enable_zself_influence
        }

    def cancel_plan(self) -> bool:
        """Abort the in-flight plan if any."""
        if self.active_plan:
            self.active_plan.status = "cancelled"
            self.plan_history.append(self.active_plan)
            self.active_plan = None
            logger.info("[SelfAgent] Plan cancelled")
            return True
        return False


def create_self_agent(self_model) -> SelfAgent:
    """Factory helper."""
    return SelfAgent(self_model)
