import re
import logging
from typing import Dict, Optional, List, Tuple, Callable


def parse_introspection(
    content: str,
    tools_enabled: bool,
    parse_xml_tool_calls: Callable[[str], Optional[List[Dict]]],
    logger: logging.Logger,
) -> Tuple[Dict, Optional[List[Dict]]]:
    """
    Parse ``<thought>`` / legacy ``<introspection>`` blocks and optionally recover XML tool calls.

    When the model omits native ``tool_calls``, ``parse_xml_tool_calls`` may synthesize them.
    """
    introspection: Dict = {}
    tool_calls_override: Optional[List[Dict]] = None

    # Step 1 — Prefer a closed <thought> block (fallback: legacy <introspection>).
    thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL | re.IGNORECASE)
    if thought_match:
        introspection["inner_monologue"] = thought_match.group(1).strip()
        logger.debug(f"Extracted inner_monologue (closed): {introspection['inner_monologue'][:100]}...")
    else:
        alt_match = re.search(r"<introspection>(.*?)</introspection>", content, re.DOTALL | re.IGNORECASE)
        if alt_match:
            introspection["inner_monologue"] = alt_match.group(1).strip()
            logger.debug(f"Extracted inner_monologue (<introspection>): {introspection['inner_monologue'][:100]}...")
        else:
            thought_match_open = re.search(r"<thought>(.*)", content, re.DOTALL | re.IGNORECASE)
            if thought_match_open:
                introspection["inner_monologue"] = thought_match_open.group(1).strip()
                logger.warning(f"Extracted inner_monologue (open tag): {introspection['inner_monologue'][:100]}...")
            else:
                logger.debug(f"No <thought> tag found in content (length={len(content)})")

    # [2026-04-07] <pineal_check> parsing removed (broadcast path unused)

    # Step 3 — XML tool-call salvage when tools are enabled but the gateway omitted tool_calls.
    if tools_enabled:
        xml_tool_calls = parse_xml_tool_calls(content)
        if xml_tool_calls:
            logger.info(f"Parsed {len(xml_tool_calls)} tool calls from XML content in chat.")
            tool_calls_override = xml_tool_calls

    # Step 5 — Confidence hint inside the monologue (legacy CN labels + EN; supports metrics / UI).
    try:
        im = introspection.get("inner_monologue", "") if isinstance(introspection, dict) else ""
        if isinstance(im, str) and im:
            m = re.search(
                r"(?:自省置信度|置信度|confidence|introspection confidence)\s*[:：]\s*([01](?:\.\d+)?)",
                im,
                re.IGNORECASE,
            )
            if m:
                v = float(m.group(1))
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0
                introspection["confidence"] = v
    except Exception:
        pass

    return introspection, tool_calls_override
