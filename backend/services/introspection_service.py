import re
from typing import Any, Dict, Tuple


class IntrospectionService:
    """
    Parse optional ``<thought>...</thought>`` blocks from model output (v1.1).

    Strips the block from the user-visible reply and exposes lightweight key/value
    hints when the model uses ``Key: value`` lines inside the thought.
    """

    @staticmethod
    def extract_introspection(text: str) -> Tuple[str, Dict[str, Any], str]:
        """
        Split ``text`` into public reply + parsed introspection dict + raw thought body.

        Args:
            text: Full model completion (may include ``<thought>``).

        Returns:
            ``(final_response, introspection_data, raw_thought)``
            - ``final_response``: text with ``<thought>`` removed.
            - ``introspection_data``: optional flags such as ``conflict_detected``, ``confidence``.
            - ``raw_thought``: inner ``<thought>`` body or empty string.
        """
        match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)

        introspection_data: Dict[str, Any] = {}
        raw_thought = ""
        final_response = text

        if match:
            raw_thought = match.group(1).strip()
            final_response = re.sub(
                r"<thought>.*?</thought>", "", text, flags=re.DOTALL
            ).strip()

            for line in raw_thought.split("\n"):
                if ":" not in line:
                    continue
                parts = line.split(":", 1)
                key = parts[0].strip().lower()
                value = parts[1].strip()
                # Bilingual keys: models may emit EN or legacy CN labels.
                if "conflict" in key or "冲突" in key:
                    introspection_data["conflict_detected"] = True
                if "confidence" in key or "置信度" in key:
                    try:
                        nums = re.findall(r"0\.\d+|[01]", value)
                        if nums:
                            introspection_data["confidence"] = float(nums[0])
                    except (ValueError, TypeError):
                        pass

            introspection_data["raw_thought"] = raw_thought

        return final_response, introspection_data, raw_thought
