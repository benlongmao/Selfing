import logging


# Shared logger for PromptBuilder submodules
logger = logging.getLogger("prompt_builder")


def trait_desc(val: float, label_pos: str, label_neg: str) -> str:
    """Map a value in [-1, 1] to an English directional hint for prompts."""
    if val > 0.2:
        return f"leans toward {label_pos}"
    if val < -0.2:
        return f"leans toward {label_neg}"
    return "balanced / mid"
