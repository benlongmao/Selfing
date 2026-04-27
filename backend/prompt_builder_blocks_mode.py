def build_mode_block(interaction_mode: dict) -> str:
    """
    [2026-02-28] Build the interaction-mode hint block from ``interaction_mode``.

    Uses first-person self-description rather than external imperative tone.
    """
    if not (interaction_mode and isinstance(interaction_mode, dict)):
        return ""
    mode = interaction_mode.get("mode", "balanced")
    reasons = interaction_mode.get("reasons", [])
    
    # First-person description of the current interaction tendency
    mode_feeling = {
        "analytical": "I feel pulled toward deeper analysis—unpacking the question and tightening the logic.",
        "direct": "I feel I should cut to the point with less preamble.",
        "story": "I want to organize my thinking as a narrative arc.",
        "cautious": "I feel the need to move carefully; some uncertainty is present.",
        "defensive": "My energy is low; I want to stay concise.",
        "balanced": "My overall state feels even and workable.",
    }.get(mode, "My overall state feels even and workable.")

    reason_text = "; ".join(reasons) if reasons else ""
    
    if reason_text:
        return f"""
[My current mode]
{mode_feeling} ({reason_text})
"""
    else:
        return f"""
[My current mode]
{mode_feeling}
"""

