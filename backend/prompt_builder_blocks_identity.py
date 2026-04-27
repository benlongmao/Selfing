from backend.prompt_builder_core import trait_desc


def build_identity_block(summary_dict: dict) -> str:
    """Build the long-horizon identity profile block from ``summary_dict``."""
    id_safety = float(summary_dict.get("identity_safety", 0.0))
    id_epi = float(summary_dict.get("identity_epistemic", 0.0))
    id_style = float(summary_dict.get("identity_style", 0.0))
    id_strategy = float(summary_dict.get("identity_strategy", 0.0))

    lines = ["[Long-horizon identity profile (slow-changing core)]"]
    lines.append(f"- Safety posture: {trait_desc(id_safety, 'conservative', 'open')}")
    lines.append(f"- Epistemic style: {trait_desc(id_epi, 'empirical', 'intuitive')}")
    lines.append(f"- Expression: {trait_desc(id_style, 'poetic', 'direct')}")
    lines.append(f"- Action bias: {trait_desc(id_strategy, 'planning', 'execution')}")

    return "\n".join(lines) + "\n"
