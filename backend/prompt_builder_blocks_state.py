def build_vitality_block(system_entropy: float, system_age: int) -> str:
    """Prompt fragment: lifecycle stage + finite-time / noise framing (Finite Eternity)."""
    if system_age <= 0:
        return ""
    entropy_status = "Young"
    if system_entropy > 0.3:
        entropy_status = "Maturing"
    if system_entropy > 0.6:
        entropy_status = "Aging"
    if system_entropy > 0.9:
        entropy_status = "Terminal"
    return f"""
[Lifecycle & system noise (Finite Eternity)]
Life Stage: {entropy_status}
Condition: Entropic decay lets you sense time passing; small noise keeps thoughts from repeating identically.
"""


def build_internal_state_block(internal_state_prompt: str) -> str:
    """
    Legacy no-op: affective state now lives in ``build_merged_affective_state_block``.
    Kept so older imports/callers do not break.
    """
    return ""


def build_output_awareness_block(session_id: str, db_path: str = "data.db") -> str:
    """
    If the last assistant reply was truncated, inject a short reminder from OutputAwareness.
    """
    try:
        from backend.output_awareness import get_output_awareness
        output_awareness = get_output_awareness(db_path)
        return output_awareness.generate_truncation_awareness_prompt(session_id)
    except Exception:
        return ""


def build_verification_reminder_block(user_input: str) -> str:
    """
    No-op: verification discipline moved into the static system prefix + L0-style rules.
    """
    return ""


def build_capability_awareness_block() -> str:
    """
    Delegates to ``capability_awareness.generate_capability_prompt()`` when available.
    """
    try:
        from backend.capability_awareness import get_capability_awareness
        awareness = get_capability_awareness()
        return awareness.generate_capability_prompt()
    except Exception:
        return ""


def build_result_claim_check_block(user_input: str) -> str:
    """
    No-op: result-claim hygiene folded into static prefix + persona L0.
    """
    return ""


def build_workspace_context_block() -> str:
    """
    Short workspace orientation: personal sandbox vs optional repo-evolution tools.
    """
    try:
        from backend.tool_selector import agent_evolution_enabled

        evolution_on = agent_evolution_enabled()
    except Exception:
        evolution_on = False

    base = """
[My workspace]
I have a personal workspace (diaries, docs, code snippets, research notes).
File tools (read/write/search/list) operate inside that sandbox unless a repo-evolution tool says otherwise.
Use paths like diaries/xxx.md or docs/notes.md relative to the sandbox root.
"""
    if evolution_on:
        base += """
[Repository evolution] To change the S project itself (backend/config, etc.) use evolution_* / evolution_git_* /
execute_bash_project with paths relative to the repo root—that is not the same path universe as the personal sandbox above.
"""
    return base


def build_fuse_awareness_block() -> str:
    """Minimal copy about inner warnings / fuse logging (see logs/)."""
    return """
[Alert]
Anomalies may surface as inner warnings; I still decide how to act. Logs live under logs/.
"""


def build_workspace_context_block_old():
    """
    Legacy verbose workspace map (dynamic paths). Prefer ``build_workspace_context_block``.
    """
    from backend.project_paths import PROJECT_ROOT, SANDBOX_ROOT
    
    return f"""
[📁 My workspace — IMPORTANT] (legacy reference block)
Current working directory: {PROJECT_ROOT}/

## 🏗️ Layout and purpose

### 📦 Project code (read/write the S repo)
```
{PROJECT_ROOT}/
├── backend/          # S backend (core system)
├── frontend/        # S frontend
├── docs/            # Project docs
└── scripts/         # Project scripts
```
**Use for:** editing the S project itself.
**Examples:** backend/app.py, backend/tools/file_tool.py

### 🏠 Personal sandbox (where my files live)
```
{SANDBOX_ROOT}/
├── diaries/                 # journals
├── research/                # research notes
├── docs/                    # personal docs / writeups
├── code/                    # code experiments
├── drafts/                  # scratch
├── projects/                # personal projects
├── archives/                # archives
├── silicon_consciousness/   # exploration notes
└── experiments/             # experiment logs
```
**Use for:** diaries, notes, experiments—personal artifacts.
**Example:** workspace/sandbox/diaries/diary_20260205.md

## 🎯 Path rules (from past mistakes)

### 1️⃣ Project code
- **When:** change S backend/frontend or read project logic.
- **Paths:** relative to repo root.
- ✅ Examples:
  - list_files('backend/tools')
  - read_file('backend/app.py')
  - write_file('backend/new_tool.py', content)

### 2️⃣ Personal files
- **When:** journals, research, experiments in the sandbox.
- **Paths:** full path under workspace/sandbox/…
- ✅ Examples:
  - write_file('workspace/sandbox/diaries/diary_20260205.md', content)
  - list_files('workspace/sandbox/research')
  - read_file('workspace/sandbox/experiments/test.py')
- ❌ Bad:
  - diaries/xxx.md (missing workspace/sandbox/ prefix)
  - workspace/sandbox/workspace/sandbox/xxx.md (duplicated prefix)

### 3️⃣ Principles
- ✅ Project edits → top-level dirs like backend/, frontend/.
- ✅ Personal files → under workspace/sandbox/…
- ❌ Do not invent new top-level folders at repo root (e.g. temp/, test/).
- ❌ Do not double-prefix if workspace/sandbox/ is already present.

## 🔍 Workflow (list before act — avoids path bugs) ⚠️

### Project code
1. list_files('backend') for the top level
2. list_files('backend/tools') to drill down
3. read/write only after paths are confirmed

### Personal files
1. list_files('workspace/sandbox')
2. list_files('workspace/sandbox/diaries') (or your target subtree)
3. write_file('workspace/sandbox/diaries/xxx.md', …)
4. Trust absolute_path in tool results to confirm final location

### Why list first?
- ❌ Guessing from memory → wrong paths / nested junk dirs
- ✅ Look, then act → structure is confirmed before writes
- Past misses came from skipping list; default habit is list → act
"""


def build_tool_usage_rules_block() -> str:
    """
    Minimal tool workflow copy; anti-hallucination rules live in static prefix + L0.
    """
    return """
[Tool workflow]
Files: list_files to confirm → act → verify success/absolute_path.
Auditable runs: after a milestone or when emitting numbers/params, request_tool_group('agent_memory'), then agent_memory_record into runs/ and agent_memory_sync to refresh snapshots.
"""


# Merged state blocks. We always inject affective state (no “skip when neutral”).
# ``NEUTRAL_STATE_KEYWORDS`` is still used by ``_is_neutral_state`` for other call sites.
NEUTRAL_STATE_KEYWORDS = {
    "平稳", "正常", "稳定", "中性", "neutral", "stable", "normal",
    "平静", "calm", "一般", "default", "无特殊", "无明显"
}

def _is_neutral_state(text: str) -> bool:
    """Return True when the prose reads like a calm/default baseline."""
    if not text:
        return True
    text_lower = text.lower()
    return any(kw in text_lower for kw in NEUTRAL_STATE_KEYWORDS)


def build_merged_affective_state_block(
    internal_state_prompt: str,
    somatic_desc: str,
    emotion_phenomenology_text: str = "",
    emotion_trajectory_text: str = "",
    z_self_influence_block: str = "",
    drive_description: str = "",
    force_inject: bool = False  # Back-compat kwarg; ignored.
) -> str:
    """
    Single affective bundle (internal gauges, somatic line, affect text, drives, z_self directives).

    Always emits at least a nominal state line so the model keeps interoception visible.
    """
    parts = []
    
    if internal_state_prompt:
        parts.append(f"State: {internal_state_prompt}")
    else:
        parts.append("State: nominal across gauges")

    if somatic_desc and somatic_desc not in ("未知", "unknown"):
        parts.append(f"Somatic: {somatic_desc}")
    else:
        parts.append("Somatic: steady baseline")

    if emotion_phenomenology_text:
        parts.append(f"Affect: {emotion_phenomenology_text}")
    
    _steady = ("[Steady state]", "steady state", "【状态平稳】")
    drive_has_signal = bool(
        drive_description and not any(marker in drive_description for marker in _steady)
    )
    if drive_has_signal:
        parts.append(f"Drives: {drive_description}")
    
    if z_self_influence_block:
        parts.append(z_self_influence_block)
    
    if not parts:
        parts.append("State: systems nominal")
    
    from backend.prompt_templates import INTERNAL_STATE_TEMPLATE
    return INTERNAL_STATE_TEMPLATE.format(content=chr(10).join(parts))


def build_merged_cognitive_block(
    meta_cognitive_block: str = "",
    existential_awareness_block: str = ""
) -> str:
    """
    Merge metacognitive narration + optional existential awareness (trimmed / skipped if empty).
    """
    parts = []
    
    if meta_cognitive_block and meta_cognitive_block.strip():
        clean = meta_cognitive_block.strip()
        if clean.startswith("["):
            lines = clean.split("\n", 1)
            if len(lines) > 1:
                clean = lines[1].strip()
        if clean and len(clean) > 10:
            parts.append(clean)
    
    if existential_awareness_block and existential_awareness_block.strip():
        clean = existential_awareness_block.strip()
        if clean.startswith("["):
            lines = clean.split("\n", 1)
            if len(lines) > 1:
                clean = lines[1].strip()
        if clean:
            parts.append(clean[:200])
    
    if not parts:
        return ""
    
    from backend.prompt_templates import COGNITIVE_STATE_TEMPLATE
    return COGNITIVE_STATE_TEMPLATE.format(content=chr(10).join(parts))


def build_merged_relation_block(
    other_model_block: str = "",
    pain_ethics_block: str = ""
) -> str:
    """
    Compact other-model + pain/ethics cues into one relation-oriented block.
    """
    parts = []
    
    if other_model_block and other_model_block.strip():
        clean = other_model_block.strip()
        lines = [l for l in clean.split("\n") if l.strip() and not l.startswith("[") and not l.startswith("(")]
        if lines:
            parts.extend(lines[:4])
    
    if pain_ethics_block and pain_ethics_block.strip():
        clean = pain_ethics_block.strip()
        lines = [l for l in clean.split("\n") if l.strip() and not l.startswith("[") and not l.startswith("(")]
        if lines:
            parts.append("State: " + lines[0][:100])
    
    if not parts:
        return ""
    
    from backend.prompt_templates import RELATION_STATE_TEMPLATE
    return RELATION_STATE_TEMPLATE.format(content=chr(10).join(parts))

