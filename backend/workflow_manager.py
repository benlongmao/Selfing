#!/usr/bin/env python3
"""
Workflow manager — optional multi-step playbooks for the agent.

[2026-01-27] Inspired by Clawdbot’s Lobster-style engine, adapted for this codebase.

Principles:
1. Workflows can read z_self summaries so the same definition may score differently by state.
2. Execution should feel rewarding — bias toward reducing friction, not adding pain.
3. Definitions compose like small pipelines (DAG of think/tool/reflect steps).

YAML sketch:
```yaml
name: research_and_summarize
triggers:
  - pattern: "(?i)research.+"
steps:
  - id: search
    type: tool
    tool: web_search
  - id: analyze
    type: think
    depends_on: [search]
  - id: summarize
    type: tool
    tool: generate_text
```
"""

import logging
import yaml
import os
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WorkflowTrigger:
    """Single activation rule (regex pattern, substring keyword, or coarse intent tag)."""
    trigger_type: str  # pattern, intent, keyword
    value: str
    priority: int = 0


@dataclass
class WorkflowStep:
    """One node in the workflow DAG."""
    step_id: str
    step_type: str  # tool, think, reflect, decide, compose
    description: str = ""
    tool_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    zself_conditions: Dict[str, Any] = field(default_factory=dict)
    output_key: Optional[str] = None


@dataclass
class WorkflowDefinition:
    """Bundled triggers + steps + resource hints."""
    name: str
    description: str
    version: str
    triggers: List[WorkflowTrigger]
    steps: List[WorkflowStep]
    min_energy: float = 20.0
    estimated_complexity: float = 0.5
    max_steps: int = 10
    zself_influence: Dict[str, Any] = field(default_factory=dict)
    author: str = "Agent"
    created_at: Optional[str] = None


class WorkflowManager:
    """
    Load YAML/inline definitions, score user text against triggers, expose steps for SelfAgent.
    """

    def __init__(self, workflows_dir: str = "workflows", self_model=None):
        """
        Args:
            workflows_dir: Directory of optional ``*.yaml`` workflow files.
            self_model: Optional ``SelfModel`` for z_self-aware score tweaks.
        """
        self.workflows_dir = Path(workflows_dir)
        self.self_model = self_model
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.builtin_workflows: Dict[str, WorkflowDefinition] = {}

        self._register_builtin_workflows()
        self._load_workflows_from_dir()
        
        logger.info(f"[WorkflowManager] Initialized with {len(self.workflows)} workflows")
    
    def _register_builtin_workflows(self):
        """Seed three bilingual-friendly defaults."""

        self.builtin_workflows["research_and_summarize"] = WorkflowDefinition(
            name="research_and_summarize",
            description="Research a topic and distill a summary.",
            version="1.0",
            triggers=[
                WorkflowTrigger("pattern", r"帮我研究.*", 10),
                WorkflowTrigger("pattern", r"调研一下.*", 10),
                WorkflowTrigger("keyword", "研究", 5),
                WorkflowTrigger("pattern", r"(?i)research(\s+on|\s+about|\s+into)?\s+.*", 10),
                WorkflowTrigger("pattern", r"(?i)deep\s+dive\s+(on|into)\s+.*", 9),
                WorkflowTrigger("keyword", "investigate", 5),
            ],
            steps=[
                WorkflowStep(
                    step_id="analyze_topic",
                    step_type="think",
                    description="Clarify the research question and unknowns."
                ),
                WorkflowStep(
                    step_id="search_info",
                    step_type="tool",
                    tool_name="web_search",
                    depends_on=["analyze_topic"],
                    description="Gather external sources."
                ),
                WorkflowStep(
                    step_id="synthesize",
                    step_type="think",
                    depends_on=["search_info"],
                    description="Integrate findings into a coherent view."
                ),
                WorkflowStep(
                    step_id="reflect",
                    step_type="reflect",
                    depends_on=["synthesize"],
                    description="Reflect on process limits and follow-ups."
                )
            ],
            min_energy=30.0,
            estimated_complexity=0.7,
            zself_influence={
                "high_curiosity": {"search_depth": "deep"},
                "high_caution": {"verify_sources": True}
            }
        )
        
        self.builtin_workflows["code_analysis"] = WorkflowDefinition(
            name="code_analysis",
            description="Review code and suggest concrete improvements.",
            version="1.0",
            triggers=[
                WorkflowTrigger("pattern", r"分析.*代码.*", 10),
                WorkflowTrigger("pattern", r"检查.*代码.*", 10),
                WorkflowTrigger("keyword", "代码分析", 8),
                WorkflowTrigger("pattern", r"(?i)review(\s+the)?\s+.*code.*", 10),
                WorkflowTrigger("pattern", r"(?i)analyze(\s+this)?\s+.*code.*", 10),
                WorkflowTrigger("keyword", "refactor", 7),
            ],
            steps=[
                WorkflowStep(
                    step_id="read_code",
                    step_type="tool",
                    tool_name="read_file",
                    description="Load the relevant files."
                ),
                WorkflowStep(
                    step_id="analyze_structure",
                    step_type="think",
                    depends_on=["read_code"],
                    description="Map structure, dependencies, and hotspots."
                ),
                WorkflowStep(
                    step_id="identify_issues",
                    step_type="think",
                    depends_on=["analyze_structure"],
                    description="List risks, bugs, and smell."
                ),
                WorkflowStep(
                    step_id="suggest_improvements",
                    step_type="think",
                    depends_on=["identify_issues"],
                    description="Propose fixes with rationale."
                )
            ],
            min_energy=25.0,
            estimated_complexity=0.6
        )
        
        self.builtin_workflows["creative_writing"] = WorkflowDefinition(
            name="creative_writing",
            description="Creative writing loop from brief to polished draft.",
            version="1.0",
            triggers=[
                WorkflowTrigger("pattern", r"写一篇.*", 8),
                WorkflowTrigger("pattern", r"帮我写.*", 7),
                WorkflowTrigger("keyword", "创作", 5),
                WorkflowTrigger("pattern", r"(?i)write(\s+me)?\s+(a|an|the)\s+.*", 8),
                WorkflowTrigger("pattern", r"(?i)draft\s+.*", 7),
                WorkflowTrigger("keyword", "story", 5),
            ],
            steps=[
                WorkflowStep(
                    step_id="understand_request",
                    step_type="think",
                    description="Lock voice, audience, and constraints."
                ),
                WorkflowStep(
                    step_id="brainstorm",
                    step_type="think",
                    depends_on=["understand_request"],
                    description="Brainstorm angles and motifs."
                ),
                WorkflowStep(
                    step_id="outline",
                    step_type="think",
                    depends_on=["brainstorm"],
                    description="Shape sections / beats."
                ),
                WorkflowStep(
                    step_id="write",
                    step_type="compose",
                    depends_on=["outline"],
                    description="Produce the first full draft."
                ),
                WorkflowStep(
                    step_id="refine",
                    step_type="reflect",
                    depends_on=["write"],
                    description="Tighten language, pacing, and clarity."
                )
            ],
            min_energy=20.0,
            estimated_complexity=0.5,
            zself_influence={
                "high_creativity": {"style": "experimental"},
                "low_energy": {"style": "concise"}
            }
        )
        
        self.workflows.update(self.builtin_workflows)

    def _load_workflows_from_dir(self):
        """Load optional YAML definitions from ``workflows_dir``."""
        if not self.workflows_dir.exists():
            logger.info(f"[WorkflowManager] Workflows directory not found: {self.workflows_dir}")
            return
        
        for yaml_file in self.workflows_dir.glob("*.yaml"):
            try:
                workflow = self._parse_workflow_file(yaml_file)
                if workflow:
                    self.workflows[workflow.name] = workflow
                    logger.info(f"[WorkflowManager] Loaded workflow: {workflow.name}")
            except Exception as e:
                logger.warning(f"[WorkflowManager] Failed to load {yaml_file}: {e}")
    
    def _parse_workflow_file(self, filepath: Path) -> Optional[WorkflowDefinition]:
        """Parse one YAML file into a ``WorkflowDefinition``."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                return None
            
            triggers = []
            for t in data.get("triggers", []):
                if isinstance(t, dict):
                    triggers.append(WorkflowTrigger(
                        trigger_type=t.get("type", "pattern"),
                        value=t.get("value", t.get("pattern", "")),
                        priority=t.get("priority", 0)
                    ))
                elif isinstance(t, str):
                    triggers.append(WorkflowTrigger("pattern", t, 0))
            
            steps = []
            for s in data.get("steps", []):
                steps.append(WorkflowStep(
                    step_id=s.get("id", f"step_{len(steps)}"),
                    step_type=s.get("type", "think"),
                    description=s.get("description", ""),
                    tool_name=s.get("tool"),
                    params=s.get("params", {}),
                    depends_on=s.get("depends_on", []),
                    zself_conditions=s.get("zself_conditions", {}),
                    output_key=s.get("output_key")
                ))
            
            return WorkflowDefinition(
                name=data.get("name", filepath.stem),
                description=data.get("description", ""),
                version=data.get("version", "1.0"),
                triggers=triggers,
                steps=steps,
                min_energy=float(data.get("resource_requirements", {}).get("min_energy", 20.0)),
                estimated_complexity=float(data.get("resource_requirements", {}).get("estimated_complexity", 0.5)),
                max_steps=int(data.get("resource_requirements", {}).get("max_steps", 10)),
                zself_influence=data.get("zself_influence", {}),
                author=data.get("author", "external"),
                created_at=data.get("created_at")
            )
            
        except Exception as e:
            logger.error(f"[WorkflowManager] Failed to parse {filepath}: {e}")
            return None
    
    def match_workflow(self, user_input: str, session_id: str = "default") -> Optional[WorkflowDefinition]:
        """
        Score every workflow; return the best above the internal threshold.

        ``session_id`` is forwarded to ``self_model`` when present for z_self-aware tuning.
        """
        if not user_input:
            return None
        
        best_match: Optional[WorkflowDefinition] = None
        best_score = 0
        
        for name, workflow in self.workflows.items():
            score = self._calculate_match_score(user_input, workflow)
            
            if self.self_model:
                score = self._adjust_score_by_zself(score, workflow, session_id)
            
            if score > best_score:
                best_score = score
                best_match = workflow
        
        if best_score >= 5:
            logger.info(f"[WorkflowManager] Matched workflow: {best_match.name} (score={best_score})")
            return best_match
        
        return None
    
    def _calculate_match_score(self, user_input: str, workflow: WorkflowDefinition) -> int:
        """Sum trigger hits with simple type-specific weights."""
        score = 0
        
        for trigger in workflow.triggers:
            if trigger.trigger_type == "pattern":
                if re.search(trigger.value, user_input, re.IGNORECASE):
                    score += trigger.priority + 10
            elif trigger.trigger_type == "keyword":
                if trigger.value.lower() in user_input.lower():
                    score += trigger.priority + 5
            elif trigger.trigger_type == "intent":
                if trigger.value.lower() in user_input.lower():
                    score += trigger.priority + 3
        
        return score
    
    def _adjust_score_by_zself(
        self, 
        base_score: int, 
        workflow: WorkflowDefinition, 
        session_id: str
    ) -> int:
        """Re-weight ``base_score`` using a lightweight ``get_structured_summary`` snapshot."""
        try:
            summary = self.self_model.get_structured_summary(session_id)

            energy = summary.get("energy", 100)
            if energy < workflow.min_energy:
                base_score = int(base_score * 0.5)

            pain = summary.get("pain_level", 0)
            if pain > 0.5 and workflow.estimated_complexity > 0.7:
                base_score = int(base_score * 0.7)

            # Motivation payload is often coarse labels; fall back to exploration_mean.
            motivation = summary.get("motivation")
            exploration = 0.0
            if isinstance(motivation, dict):
                exploration = float(motivation.get("exploration", 0) or 0)
            else:
                exploration = float(summary.get("exploration_mean", 0) or 0)
            if exploration > 0.6:
                desc = (workflow.description or "").lower()
                if "research" in workflow.name or "分析" in workflow.description or "analyze" in desc:
                    base_score = int(base_score * 1.2)
            
        except Exception as e:
            logger.warning(f"[WorkflowManager] Failed to adjust score by z_self: {e}")
        
        return base_score
    
    def convert_to_agent_steps(self, workflow: WorkflowDefinition) -> List[Dict]:
        """Flatten to the dict list consumed by SelfAgent."""
        steps = []
        
        for wf_step in workflow.steps:
            agent_step = {
                "type": wf_step.step_type,
                "description": wf_step.description,
                "depends_on": wf_step.depends_on
            }
            
            if wf_step.tool_name:
                agent_step["tool_name"] = wf_step.tool_name
            
            if wf_step.params:
                agent_step["params"] = wf_step.params
            
            steps.append(agent_step)
        
        return steps
    
    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Lookup by ``name``."""
        return self.workflows.get(name)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """Summaries for UI / diagnostics."""
        return [
            {
                "name": wf.name,
                "description": wf.description,
                "version": wf.version,
                "complexity": wf.estimated_complexity,
                "min_energy": wf.min_energy,
                "steps_count": len(wf.steps),
                "is_builtin": wf.name in self.builtin_workflows
            }
            for wf in self.workflows.values()
        ]
    
    def add_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Register an in-memory definition (no disk write)."""
        if workflow.name in self.workflows:
            logger.warning(f"[WorkflowManager] Workflow {workflow.name} already exists")
            return False
        
        self.workflows[workflow.name] = workflow
        logger.info(f"[WorkflowManager] Added workflow: {workflow.name}")
        return True
    
    def save_workflow(self, workflow: WorkflowDefinition, filename: Optional[str] = None) -> bool:
        """Serialize ``workflow`` to YAML under ``workflows_dir``."""
        if not filename:
            filename = f"{workflow.name}.yaml"
        
        filepath = self.workflows_dir / filename
        
        try:
            self.workflows_dir.mkdir(parents=True, exist_ok=True)
            
            data = {
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version,
                "triggers": [
                    {"type": t.trigger_type, "value": t.value, "priority": t.priority}
                    for t in workflow.triggers
                ],
                "steps": [
                    {
                        "id": s.step_id,
                        "type": s.step_type,
                        "description": s.description,
                        "tool": s.tool_name,
                        "params": s.params,
                        "depends_on": s.depends_on
                    }
                    for s in workflow.steps
                ],
                "resource_requirements": {
                    "min_energy": workflow.min_energy,
                    "estimated_complexity": workflow.estimated_complexity,
                    "max_steps": workflow.max_steps
                },
                "zself_influence": workflow.zself_influence,
                "author": workflow.author
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            
            logger.info(f"[WorkflowManager] Saved workflow to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[WorkflowManager] Failed to save workflow: {e}")
            return False


def create_workflow_manager(workflows_dir: str = "workflows", self_model=None) -> WorkflowManager:
    """Factory wrapper."""
    return WorkflowManager(workflows_dir, self_model)
