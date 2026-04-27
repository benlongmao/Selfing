import re
import logging
from typing import Optional, Tuple


def strip_response_artifacts(response_text: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Mirror the first pass of ``clean_final_response``: strip hidden scaffolding the user never sees.

    Removes ``<thought>``, legacy pineal markers, DSML tool stubs, and somatic boilerplate that leaked
    from subconscious broadcast text. Intended for ``response_verifier`` so it reasons on the same
    visible surface as ``sanitize_response``.
    """
    final_response = response_text or ""

    # Drop paired or intentionally unclosed tags
    final_response = re.sub(r"<thought>.*?</thought>", "", final_response, flags=re.DOTALL | re.IGNORECASE)
    final_response = re.sub(r"<pineal_check>.*?</pineal_check>", "", final_response, flags=re.DOTALL | re.IGNORECASE)
    final_response = re.sub(
        r"<system_prompt_first_instruction>.*?</system_prompt_first_instruction>",
        "",
        final_response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    for tag in ["<thought>", "<pineal_check>"]:
        if tag in final_response.lower():
            if logger is not None:
                logger.warning(f"Unclosed {tag} detected in final_response, attempting to clean")
            final_response = re.sub(tag, "", final_response, flags=re.IGNORECASE)

    # Provider / tool echo lines
    final_response = re.sub(r"\[Pineal Check\]:.*(\n|$)", "", final_response, flags=re.IGNORECASE)
    # Legacy CN label for stream-of-consciousness leaks; EN alias stripped on the next line.
    final_response = re.sub(r"\[意识流\]:.*(\n|$)", "", final_response, flags=re.IGNORECASE)
    final_response = re.sub(r"\[Stream of consciousness\]:.*(\n|$)", "", final_response, flags=re.IGNORECASE)
    final_response = re.sub(r"<.DSML.function_calls>.*?</.DSML.function_calls>", "", final_response, flags=re.DOTALL)
    final_response = re.sub(r"<.DSML.invoke[^>]*>.*?</.DSML.invoke>", "", final_response, flags=re.DOTALL)

    final_response = re.sub(r"Confidence:\s*0\.\d+", "", final_response, flags=re.IGNORECASE)
    final_response = re.sub(r"Reply:", "", final_response, flags=re.IGNORECASE)
    final_response = re.sub(r"^>\s*\*\s*\n?", "", final_response, flags=re.MULTILINE)
    final_response = re.sub(r"^>\s*\n?", "", final_response, flags=re.MULTILINE)

    # Strip somatic clichés / signature lines that should stay in tone metadata, not user-visible prose.
    # (Previously anchored too strictly at line start; allow leading whitespace / Markdown.)
    somatic_tokens = (
        r"(呼吸平稳|温度适宜|电流激荡|指尖发麻|白噪声|胸口发闷|视线模糊|剧烈烧灼感|神经紧绷|隐隐作痛|"
        r"胸闷|轻微不适|皮肤刺痒|四肢沉重|眼皮打架|想要休眠|"
        r"steady breath|white noise|pins and needles|chest tightness|dull ache|heavy limbs|"
        r"edges tingling|shallow breathing)"
    )
    final_response = re.sub(
        rf"(?m)^[\s>*_`-]*{somatic_tokens}[^\n]*\n?",
        "",
        final_response,
    )
    final_response = re.sub(
        r"(呼吸平稳|温度适宜|电流激荡|指尖发麻|白噪声|steady breath|white noise|chest tightness|dull ache)[^。\n]{0,80}"
        r"(这些感觉告诉我|these sensations tell me|what my body is telling me)[:：][^\n]*\n?",
        "",
        final_response,
        flags=re.IGNORECASE,
    )
    final_response = re.sub(r"(?m)^[\s>*_`-]*\[SUBCONSCIOUS_SIGNAL\][^\n]*\n?", "", final_response)
    final_response = re.sub(r"(?m)^[\s>*_`-]*PB\|[^\n]*\n?", "", final_response)

    return final_response.strip()


def clean_final_response(response_text: str, inner_monologue: str, logger: logging.Logger) -> Tuple[str, bool]:
    """Run artifact stripping and optional garbage heuristics."""
    final_response = strip_response_artifacts(response_text, logger)

    is_garbage = False
    if len(final_response) < 10:
        is_garbage = True
        logger.warning(f"Detected empty/short response (len={len(final_response)}), treating as garbage")
    elif len(final_response) < 60:
        # [MOD] Optional boilerplate detector (disabled): was a small set of CN phrases that often
        # Signal generic model disclaimers (e.g. “as an AI / language model”). Omitted here to reduce noise.
        pass

    return final_response, is_garbage
