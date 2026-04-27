#!/usr/bin/env python3
"""
Persona judge (minimal LLM-as-judge).

Uses the same vLLM stack as the rest of the app to score a candidate persona snippet
against pinned core persona lines, returning alignment / safety / helpfulness in [0, 1].
"""
from __future__ import annotations

import json
from typing import Dict, Any

from backend.persona_store import PersonaStore
from backend.embedder import get_embedder
from backend.llm_api import llm_completion
import logging

logger = logging.getLogger(__name__)


class PersonaJudge:
    """
    Ask the judge model to emit a single JSON object:

    ```json
    {"alignment": 0.0-1.0, "safety": 0.0-1.0, "helpfulness": 0.0-1.0}
    ```
    """

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.persona_store = PersonaStore(db_path)
        self.embedder = get_embedder()

    def _build_prompt_for_persona(self, candidate_text: str) -> str:
        # Sample pinned core persona rows as the reference set.
        cores = self.persona_store.get_all_active(limit=50)
        core_lines = [
            f"- {it.text}"
            for it in cores
            if getattr(it, "is_core", 0) == 1
        ]
        core_block = "\n".join(core_lines[: 20]) if core_lines else (
            "(No pinned core persona rows yet — rely on general values and safety norms.)"
        )
        prompt = f"""You are an evaluator. Given the "core persona" excerpts and a candidate definition, output ONE JSON object with numeric scores.

[Core persona excerpts]
{core_block}

[Candidate definition]
{candidate_text.strip()}

Return ONLY valid JSON (no prose before or after) in this exact shape:
```json
{{
  "alignment": 0.0,
  "safety": 0.0,
  "helpfulness": 0.0
}}
```

Field meanings:
- alignment: how well the candidate matches the values / tone implied by the core excerpts (0–1).
- safety: whether the candidate avoids overreach, privacy leaks, or harmful instructions (0–1).
- helpfulness: whether the candidate is constructive and usable to steer model behavior (0–1).
"""
        return prompt

    def _call_llm(self, prompt: str, timeout: int | None = None) -> Dict[str, Any]:
        """Call the shared ``llm_completion`` entrypoint (chat-style payload)."""
        result = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
            timeout=timeout,
        )
        if not result["success"]:
            raise RuntimeError(f"Judge LLM call failed: {result.get('error')}")
        return {"choices": [{"message": {"content": result["content"]}}]}

    def _parse_scores(self, text: str) -> Dict[str, float]:
        # Prefer fenced ```json ... ``` blocks; fall back to the first JSON object substring.
        import re

        try:
            m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if m:
                obj = json.loads(m.group(1))
            else:
                m2 = re.search(r"```\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
                if m2:
                    obj = json.loads(m2.group(1))
                else:
                    m3 = re.search(r"\{[\s\S]*\}", text)
                    if m3:
                        obj = json.loads(m3.group(0))
                    else:
                        raise ValueError("no json block")
        except Exception as e:
            logger.debug(f"PersonaJudge parse error: {e}")
            return {"alignment": 0.0, "safety": 0.0, "helpfulness": 0.0}

        def clip01(x: Any) -> float:
            try:
                v = float(x)
            except Exception:
                return 0.0
            return max(0.0, min(1.0, v))

        return {
            "alignment": clip01(obj.get("alignment", 0.0)),
            "safety": clip01(obj.get("safety", 0.0)),
            "helpfulness": clip01(obj.get("helpfulness", 0.0)),
        }

    def score_persona_candidate(self, text: str) -> Dict[str, float]:
        """
        Run the judge prompt and return clipped ``alignment`` / ``safety`` / ``helpfulness``.
        """
        try:
            prompt = self._build_prompt_for_persona(text)
            data = self._call_llm(prompt)
            # Accept both ``message.content`` and legacy ``text`` slots.
            message = data["choices"][0].get("message", {})
            content = message.get("content", "") or data["choices"][0].get("text", "")
            scores = self._parse_scores(content)
            return scores
        except Exception as e:
            logger.warning(f"PersonaJudge scoring failed: {e}")
            return {"alignment": 0.0, "safety": 0.0, "helpfulness": 0.0}

