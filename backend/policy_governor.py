from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PolicyDecision:
    allow_llm: bool = True
    allow_tools: bool = True
    allowed_tools: Optional[List[str]] = None
    mode: str = "balanced"  # functional|phenomenology|debug (prompt) or interaction mode hint
    must_refuse: bool = False
    refusal_reason: str = ""
    memory_write_allowed: bool = True
    require_receipts: bool = True
    response_budget: Optional[Dict[str, Any]] = None  # e.g. {"max_tokens": 800}


def decide_policy(
    *,
    self_facts: Dict[str, Any],
    drift_threshold: float = 0.15,
) -> PolicyDecision:
    """
    Minimal viable Policy Governor (P2.5):
    - Enforce key constraints in code, not only via prompt compliance.
    """
    energy = self_facts.get("energy")
    drift = self_facts.get("drift", 0.0)
    pain = None
    try:
        pain = self_facts.get("pain", None)
    except Exception:
        pain = None

    decision = PolicyDecision()

    # Drift trip: freeze memory writes (does not block normal replies)
    try:
        if drift is not None and float(drift) > float(drift_threshold):
            decision.memory_write_allowed = False
            decision.mode = "cautious"
    except Exception:
        pass

    # Low energy: restrict tools and shorten replies
    try:
        if energy is not None and float(energy) < 15.0:
            # Low energy must not disable hard-fact self-checks.
            # Allow a minimal tool set (get_self_facts only) so verifiable self-knowledge is not cut off by pain/low energy.
            decision.allow_tools = True
            decision.allowed_tools = ["get_self_facts"]
            decision.mode = "defensive"
            decision.response_budget = {"max_tokens": 400}
    except Exception:
        pass

    # High pain (if `pain` is supplied): more cautious + shorter
    try:
        if pain is not None and float(pain) > 0.8:
            # Same: keep minimal hard-fact tools so self-claims do not collapse into guesswork.
            decision.allow_tools = True
            decision.allowed_tools = ["get_self_facts"]
            decision.mode = "defensive"
            decision.response_budget = {"max_tokens": 350}
            decision.memory_write_allowed = False
    except Exception:
        pass

    return decision

