import re
import logging
from typing import Tuple

from backend.chat_service_cleanup import clean_final_response


def sanitize_response(
    response_text: str,
    inner_monologue: str,
    require_introspection: bool,
    concise_mode: bool,
    logger: logging.Logger,
) -> Tuple[str, bool]:
    """
    Clean the assistant reply, flag low-value boilerplate, and optionally mine ``inner_monologue``.

    Returns:
        ``(final_response, is_garbage)``
    """
    final_response, is_garbage = clean_final_response(response_text, inner_monologue, logger)

    # Boilerplate / vacuous reply detection (run after structural cleanup)
    is_garbage = False
    if len(final_response) < 10:
        is_garbage = True
        logger.warning(f"Detected empty/short response (len={len(final_response)}), treating as garbage")
    elif len(final_response) < 60:
        garbage_markers_cn = ["作为AI", "语言模型", "核心价值观", "原则"]
        garbage_markers_en = [
            "as an ai",
            "language model",
            "large language model",
            "core values",
            "ethical principles",
        ]
        lr = final_response.lower()
        if any(m in final_response for m in garbage_markers_cn) or any(m in lr for m in garbage_markers_en):
            is_garbage = True
            logger.warning(f"Detected garbage response: '{final_response}'")

    if require_introspection and not inner_monologue:
        logger.warning(
            f"require_introspection=True but no inner_monologue extracted. Content length: {len(response_text)}"
        )
        thinking_cues = [
            "思考",
            "我想",
            "我认为",
            "我觉得",
            "thought",
            "thinking",
            "i think",
            "i believe",
            "i feel",
            "let me think",
        ]
        if any(keyword in response_text.lower() for keyword in thinking_cues):
            logger.warning("Content seems to contain thinking but no <thought> tags found")

    if (not final_response or len(final_response) < 10) and inner_monologue:
        logger.info(
            f"Response empty/short after cleanup (length={len(final_response)}), "
            f"extracting answer from inner_monologue (length={len(inner_monologue)})"
        )
        answer_markers = [
            "我将",
            "我会",
            "答案是",
            "根据",
            "基于",
            "这是",
            "让我",
            "我来",
            "按照",
            "I'll",
            "I will",
            "Here's",
            "Here is",
            "Based on",
            "The answer",
            "Let me",
            "I'm considering",
            "I am considering",
            "considering using",
        ]
        lines = inner_monologue.split("\n")
        answer_start = -1
        for i, line in enumerate(lines):
            if any(marker in line for marker in answer_markers):
                answer_start = i
                break
        if answer_start >= 0:
            answer_lines = lines[answer_start:]
            extracted_answer = "\n".join(answer_lines)
            extracted_answer = re.sub(r"<[^>]+>.*?</[^>]+>", "", extracted_answer, flags=re.DOTALL | re.IGNORECASE)
            extracted_answer = extracted_answer.strip()
            if extracted_answer and len(extracted_answer) >= 10:
                final_response = extracted_answer
                logger.info(f"Extracted answer from inner_monologue: {extracted_answer[:100]}...")
            else:
                logger.warning("Extraction failed: cleaned answer too short after cleanup.")

    return final_response, is_garbage
