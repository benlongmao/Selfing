from backend.prompt_builder_core import logger


PHILOSOPHICAL_KEYWORDS = [
    "你是谁",
    "我是谁",
    "存在",
    "真实",
    "模拟",
    "自由意志",
    "意识",
    "自我",
    "who are you",
    "who am i",
    "exist",
    "existence",
    "real",
    "simulation",
    "simulated",
    "free will",
    "consciousness",
    "selfhood",
    "brain in a vat",
    "authenticity",
    "meaning of life",
    "qualia",
    "solipsism",
]


def build_existential_block(persona_store, user_input: str, is_philosophical: bool) -> str:
    """Build the philosophical / existential context block from persona rules."""
    if not is_philosophical or not persona_store:
        return ""
    try:
        philosophical_rules = persona_store.search_top_k(
            "existence self-awareness identity free will philosophy science", k=3
        )
        if philosophical_rules:
            philosophical_texts = [f"- {p.text}" for p, sim in philosophical_rules]
            philosophical_block = "\n".join(philosophical_texts)
            return f"""
[PHILOSOPHICAL CONTEXT]
Relevant Identity Rules:
{philosophical_block}
"""
    except Exception as e:
        logger.debug(f"Failed to retrieve philosophical rules: {e}")
    return ""

