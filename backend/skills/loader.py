#!/usr/bin/env python3
"""
Skills loader: discover, parse, and inject skill modules into prompts.

Layout (example)::

  skills/
    builtins/              # shipped skills
      weather/SKILL.md
      cron/SKILL.md
    custom/                # user-added skills under workspace
      my_skill/SKILL.md

``SKILL.md`` format::

  ---
  name: weather
  description: Fetch weather via built-in tools
  version: "1.0"
  requires_bins: ["curl"]
  requires_env: []
  always_load: false
  os: ["linux", "darwin"]
  ---

  # Weather Skill

  …Markdown body…

[2026-02-07] Phase 3 — skills system
"""

import logging
import os
import platform
import re
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

BUILTIN_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "builtins")
CUSTOM_SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "workspace", "skills")


@dataclass
class SkillInfo:
    """Parsed metadata for one skill."""
    name: str
    description: str
    version: str = "1.0"
    path: str = ""
    requires_bins: List[str] = field(default_factory=list)
    requires_env: List[str] = field(default_factory=list)
    always_load: bool = False
    os_filter: List[str] = field(default_factory=list)
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "available": self.available,
            "always_load": self.always_load,
        }


class SkillsLoader:
    """
    Discover built-in + custom skills, verify requirements, and build prompt blocks.
    """

    def __init__(
        self,
        builtin_dir: str = BUILTIN_SKILLS_DIR,
        custom_dir: str = CUSTOM_SKILLS_DIR,
    ):
        self.builtin_dir = builtin_dir
        self.custom_dir = custom_dir
        self._skills_cache: Dict[str, SkillInfo] = {}
        self._content_cache: Dict[str, str] = {}

        os.makedirs(self.custom_dir, exist_ok=True)
        self._discover_skills()

    def _discover_skills(self):
        """Scan builtin and custom directories for ``SKILL.md``."""
        self._skills_cache.clear()

        if os.path.isdir(self.builtin_dir):
            for name in os.listdir(self.builtin_dir):
                skill_dir = os.path.join(self.builtin_dir, name)
                skill_file = os.path.join(skill_dir, "SKILL.md")
                if os.path.isdir(skill_dir) and os.path.isfile(skill_file):
                    info = self._parse_skill(name, skill_file)
                    if info:
                        self._skills_cache[name] = info

        if os.path.isdir(self.custom_dir):
            for name in os.listdir(self.custom_dir):
                skill_dir = os.path.join(self.custom_dir, name)
                skill_file = os.path.join(skill_dir, "SKILL.md")
                if os.path.isdir(skill_dir) and os.path.isfile(skill_file):
                    info = self._parse_skill(name, skill_file)
                    if info:
                        self._skills_cache[name] = info

        logger.info(f"Skills discovered: {list(self._skills_cache.keys())}")

    def _parse_skill(self, name: str, skill_file: str) -> Optional[SkillInfo]:
        """Parse YAML front matter and Markdown body from ``SKILL.md``."""
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = {}
            body = content

            fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
            if fm_match:
                import yaml
                try:
                    metadata = yaml.safe_load(fm_match.group(1)) or {}
                except Exception:
                    metadata = {}
                body = content[fm_match.end():]

            info = SkillInfo(
                name=metadata.get("name", name),
                description=metadata.get("description", ""),
                version=str(metadata.get("version", "1.0")),
                path=skill_file,
                requires_bins=metadata.get("requires_bins", []),
                requires_env=metadata.get("requires_env", []),
                always_load=metadata.get("always_load", False),
                os_filter=metadata.get("os", []),
            )

            info.available = self._check_requirements(info)
            self._content_cache[name] = body.strip()

            return info

        except Exception as e:
            logger.warning(f"Failed to parse skill '{name}': {e}")
            return None

    def _check_requirements(self, info: SkillInfo) -> bool:
        """Return False when OS, binaries, or env vars are not satisfied."""
        if info.os_filter:
            current_os = platform.system().lower()
            if current_os not in [os_name.lower() for os_name in info.os_filter]:
                return False

        for bin_name in info.requires_bins:
            if not shutil.which(bin_name):
                logger.debug(f"Skill '{info.name}' missing binary: {bin_name}")
                return False

        for env_name in info.requires_env:
            if not os.environ.get(env_name):
                logger.debug(f"Skill '{info.name}' missing env: {env_name}")
                return False

        return True

    def list_skills(self) -> List[SkillInfo]:
        """Return all discovered ``SkillInfo`` rows."""
        return list(self._skills_cache.values())

    def get_skill(self, name: str) -> Optional[SkillInfo]:
        """Lookup metadata by directory/skill name."""
        return self._skills_cache.get(name)

    def load_skill_content(self, name: str) -> Optional[str]:
        """Return Markdown body (without front matter) for a skill."""
        return self._content_cache.get(name)

    def get_always_skills(self) -> List[str]:
        """Names of skills marked ``always_load: true`` and currently available."""
        return [
            name for name, info in self._skills_cache.items()
            if info.always_load and info.available
        ]

    def build_skills_prompt(self, skill_names: Optional[List[str]] = None) -> str:
        """
        Build the ``<skills>``…``</skills>`` block for the system prompt.

        Args:
            skill_names: Extra skills to include. ``None`` loads only ``always_load`` skills.

        Returns:
            Concatenated skill bodies or empty string when nothing applies.
        """
        # Hot reload: re-scan so newly added workspace skills apply without process restart.
        self._discover_skills()

        names = set(self.get_always_skills())

        if skill_names:
            names.update(skill_names)

        if not names:
            return ""

        parts = ["<skills>"]
        for name in sorted(names):
            info = self._skills_cache.get(name)
            content = self._content_cache.get(name)

            if not info or not content or not info.available:
                continue

            parts.append(f"\n<skill name=\"{info.name}\" version=\"{info.version}\">")
            parts.append(content)
            parts.append("</skill>")

        parts.append("\n</skills>")

        return "\n".join(parts)

    def build_skills_summary(self) -> str:
        """
        Short operator-facing list: name, version, description, availability.

        Unavailable rows append missing binary/env hints in English.
        """
        lines = ["Available skills:"]

        for name, info in sorted(self._skills_cache.items()):
            status = "✅" if info.available else "❌"
            line = f"- {info.name} (v{info.version}): {info.description} {status}"

            if not info.available:
                missing = []
                for bin_name in info.requires_bins:
                    if not shutil.which(bin_name):
                        missing.append(f"missing binary: {bin_name}")
                for env_name in info.requires_env:
                    if not os.environ.get(env_name):
                        missing.append(f"missing env: {env_name}")
                if missing:
                    line += f" ({', '.join(missing)})"

            lines.append(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    def refresh(self):
        """Force a filesystem rescan."""
        self._discover_skills()
