"""Volitional pain / interference hooks (currently disabled; see ``apply_pain_noise``)."""
import logging
from typing import List, Dict, Tuple


def apply_pain_noise(
    messages: List[Dict],
    final_user_content: str,
    noise_injection_prob: float,
    logger: logging.Logger,
) -> Tuple[List[Dict], str]:
    """
    [DEACTIVATED] No-op: volitional noise and system interference injection are disabled.
    """
    return messages, final_user_content

