"""
Knowledge wiki compiler.

Materializes ``knowledge_items`` and curated notes into Markdown pages under
``workspace/sandbox/wiki``. The SQLite DB remains the fast runtime retrieval
layer; the wiki is the readable, governable long-term organization layer.

No LLM is used for normal indexing, catalog generation, or search. LLM calls are
limited to ``deep_compile()``, which consolidates append-heavy pages.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join("workspace", "sandbox", "wiki")
INDEX_FILE = os.path.join(WIKI_DIR, "index.md")
LINT_LOG = os.path.join(WIKI_DIR, "_lint_log.md")
CATALOG_FILE = os.path.join(WIKI_DIR, "_catalog.json")
MAP_FILE = os.path.join(WIKI_DIR, "map.md")
PERSONAL_EXPERIENCE_README = os.path.join(WIKI_DIR, "个人经验", "README.md")

# On-disk directory names mirror ``KnowledgeBase.VALID_CATEGORIES`` literals.
_CATEGORY_DIR_MAP = {
    "技术": "技术",
    "科学": "科学",
    "常识": "常识",
    "个人经验": "个人经验",
    "项目相关": "项目相关",
    "用户偏好": "用户偏好",
    "世界知识": "世界知识",
    "用户身份": "用户身份",
    "哲学": "哲学",
    "数学": "科学",
}

_FILENAME_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _sanitize_filename(name: str, max_len: int = 60) -> str:
    name = _FILENAME_UNSAFE.sub("_", name).strip().strip(".")
    if len(name) > max_len:
        name = name[:max_len].rstrip("_")
    return name or "untitled"


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _extract_backlinks(text: str) -> List[str]:
    """Return link titles inside ``[[wikilink]]`` markers."""
    return re.findall(r"\[\[([^\]]+)\]\]", text)


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [v.strip() for v in re.split(r"[,，]", text) if v.strip()]
    text = str(value).strip()
    return [text] if text else []


def _make_frontmatter(
    title: str,
    category: str,
    sources: List[str],
    confidence: float = 0.5,
    summary: str = "",
    tags: Optional[List[str]] = None,
    kind: str = "",
    use_when: str = "",
    status: str = "active",
    priority: str = "normal",
    aliases: Optional[List[str]] = None,
    supersedes: Optional[List[str]] = None,
    valid_until: str = "",
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fields: List[Tuple[str, Any]] = [
        ("title", title),
        ("category", category),
        ("kind", kind or _infer_kind(category, title, "")),
        ("status", status),
        ("priority", priority),
        ("summary", summary),
        ("use_when", use_when),
        ("tags", tags or []),
        ("aliases", aliases or []),
        ("supersedes", supersedes or []),
        ("sources", sources or ["unknown"]),
        ("confidence", confidence),
        ("last_compiled", now),
    ]
    if valid_until:
        fields.insert(7, ("valid_until", valid_until))

    lines = ["---"]
    for key, value in fields:
        if value in ("", [], None):
            continue
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            for item in _normalize_list(value) or ["unknown"]:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse a minimal YAML-like front matter block."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm_text = content[3:end].strip()
    body = content[end + 4:].lstrip("\n")
    meta: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for line in fm_text.split("\n"):
        if line.startswith("  - ") and current_key:
            current = meta.get(current_key)
            if not isinstance(current, list):
                current = [] if current in ("", None) else [current]
            current.append(line[4:].strip())
            meta[current_key] = current
            continue
        if ":" in line and not line.startswith("  "):
            key, val = line.split(":", 1)
            current_key = key.strip()
            meta[current_key] = val.strip()
    return meta, body


def _is_wiki_page(fname: str) -> bool:
    return (
        fname.endswith(".md")
        and not fname.startswith("_")
        and fname not in {"index.md", "README.md", "map.md"}
    )


def _strip_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"[*_>#-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_summary(body: str, title: str = "", limit: int = 140) -> str:
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        cleaned = _strip_markdown(line)
        if cleaned and cleaned != title:
            return cleaned[:limit]
    return ""


def _extract_heading_title(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _infer_kind(category: str, title: str, body: str) -> str:
    text = f"{title} {body[:500]}"
    if category == "个人经验":
        if re.search(r"method|strategy|framework|方法|策略|框架|技巧|方案|路径|构建|写作|治理", text, re.I):
            return "methodology"
        if re.search(r"lesson|mistake|verify|avoid|原则|教训|核查|查库|避免|必须|边界|准确|重复|证据", text, re.I):
            return "lesson"
        if re.search(r"completed|status|milestone|daily|完成|更新|状态|里程碑|每日|本次|这次对话|本轮", text, re.I):
            return "operation_record"
        if re.search(r"identity|self|conscious|自我|身份|意识|存在|认知|数字生命|主体", text, re.I):
            return "self_reflection"
        return "lesson"
    mapping = {
        "世界知识": "world_event",
        "技术": "technical",
        "常识": "reference",
        "项目相关": "project",
        "用户偏好": "preference",
        "用户身份": "identity",
        "哲学": "philosophy",
        "科学": "reference",
    }
    return mapping.get(category, "reference")


def _infer_status(category: str, title: str, kind: str) -> str:
    if kind == "operation_record":
        return "archive_candidate"
    if category == "世界知识":
        return "historical"
    if re.search(r"completed|status|milestone|daily|完成|已完成|状态|里程碑|每日", title, re.I):
        return "archive_candidate"
    return "active"


def _infer_priority(category: str, title: str, kind: str, status: str) -> str:
    if status in {"archived", "archive_candidate", "historical"}:
        return "low"
    if category in {"用户偏好", "项目相关"}:
        return "high"
    if kind in {"methodology", "lesson", "preference", "project"}:
        return "high"
    return "normal"


def _infer_tags(category: str, title: str, kind: str) -> List[str]:
    tags = {category, kind}
    keyword_tags = {
        "writing": ["writing"],
        "research": ["research"],
        "project": ["project"],
        "memory": ["memory"],
        "identity": ["identity"],
        "rule": ["rules"],
        "写作": ["writing"],
        "推广": ["promotion"],
        "记忆": ["memory"],
        "身份": ["identity"],
        "规则": ["rules"],
        "查库": ["verification"],
        "进度": ["progress"],
    }
    for keyword, values in keyword_tags.items():
        if keyword in title:
            tags.update(values)
    return sorted(tags)


def _infer_use_when(category: str, title: str, kind: str) -> str:
    if kind == "methodology":
        return "Use when you need a reusable method, framework, or writing technique."
    if kind == "lesson":
        return "Use when avoiding a known failure mode or calibrating behavior."
    if kind == "self_reflection":
        return "Use for identity, continuity, or self-description questions."
    if kind == "operation_record":
        return "Use only to verify a concrete past action, status, or progress record."
    if category == "用户偏好":
        return "Use to confirm durable user preferences, names, boundaries, or collaboration style."
    if category == "项目相关":
        return "Use when checking project mechanics, paths, permissions, or architecture facts."
    if category == "世界知识":
        return "Use as historical or external background; check freshness before relying on it."
    return "Use when this topic is relevant to the current task."


def _query_keywords(query: str) -> set:
    """Tokenize English and Chinese queries; Chinese gets 2-4 char windows."""
    keywords = set(re.findall(r"[a-zA-Z]{2,}", query.lower()))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", query):
        if len(chunk) <= 4:
            keywords.add(chunk)
        for size in (2, 3, 4):
            if len(chunk) >= size:
                keywords.update(chunk[i:i + size] for i in range(len(chunk) - size + 1))
    return {kw for kw in keywords if kw.strip()}


def _extract_knowledge_ids(text: str) -> List[str]:
    return sorted(set(re.findall(r"K-\d{14}-[0-9a-f]{6}", text)))


class KnowledgeCompiler:
    """Compile SQLite knowledge rows into a governed sandbox wiki."""

    def __init__(self, wiki_dir: str = WIKI_DIR):
        self.wiki_dir = wiki_dir
        self.index_file = os.path.join(self.wiki_dir, "index.md")
        self.lint_log = os.path.join(self.wiki_dir, "_lint_log.md")
        self.catalog_file = os.path.join(self.wiki_dir, "_catalog.json")
        self.map_file = os.path.join(self.wiki_dir, "map.md")
        self.personal_experience_readme = os.path.join(self.wiki_dir, "个人经验", "README.md")
        os.makedirs(self.wiki_dir, exist_ok=True)

    # --- Level 1: quick append (no LLM) ---

    def quick_append(
        self,
        title: str,
        content: str,
        category: str,
        source_id: str = "",
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Append one fact to the wiki and refresh catalog/index files."""
        cat_dir = _CATEGORY_DIR_MAP.get(category, category)
        page_dir = os.path.join(self.wiki_dir, cat_dir)
        os.makedirs(page_dir, exist_ok=True)

        existing_page = self._find_similar_page(page_dir, title)
        if existing_page:
            self._append_to_page(existing_page, title, content, source_id)
            action = "appended"
            page_path = existing_page
        else:
            page_path = self._create_page(page_dir, title, content, category, source_id, confidence)
            action = "created"

        rel_path = os.path.relpath(page_path, self.wiki_dir)
        self.rebuild_index()
        entry = self._entry_from_page(page_path, rel_path) or {}
        logger.info(f"[WIKI] {action}: {rel_path} (title={title[:40]})")
        return {"success": True, "action": action, "page": rel_path, "entry": entry}

    def _find_similar_page(self, page_dir: str, title: str, threshold: float = 0.55) -> Optional[str]:
        if not os.path.isdir(page_dir):
            return None
        best_path, best_sim = None, 0.0
        for fname in os.listdir(page_dir):
            if not _is_wiki_page(fname):
                continue
            page_title = fname[:-3].replace("_", " ")
            try:
                with open(os.path.join(page_dir, fname), "r", encoding="utf-8") as f:
                    first_600 = f.read(600)
                meta, body = _parse_frontmatter(first_600)
                page_title = meta.get("title") or _extract_heading_title(body) or page_title
            except Exception:
                pass
            sim = _title_similarity(title, page_title)
            if sim > best_sim:
                best_sim = sim
                best_path = os.path.join(page_dir, fname)
        return best_path if best_sim >= threshold else None

    def _create_page(
        self,
        page_dir: str,
        title: str,
        content: str,
        category: str,
        source_id: str,
        confidence: float,
    ) -> str:
        page_path = os.path.join(page_dir, _sanitize_filename(title) + ".md")
        sources = [source_id] if source_id else []
        kind = _infer_kind(category, title, content)
        status = _infer_status(category, title, kind)
        priority = _infer_priority(category, title, kind, status)
        fm = _make_frontmatter(
            title,
            category,
            sources,
            confidence=confidence,
            summary=_extract_summary(content, title),
            tags=_infer_tags(category, title, kind),
            kind=kind,
            use_when=_infer_use_when(category, title, kind),
            status=status,
            priority=priority,
        )

        body = f"\n# {title}\n\n{content}\n"
        links = _extract_backlinks(content)
        if links:
            body += "\n## Related topics\n"
            for link in links:
                body += f"- [[{link}]]\n"

        with open(page_path, "w", encoding="utf-8") as f:
            f.write(fm + "\n" + body)
        return page_path

    def _append_to_page(self, page_path: str, title: str, content: str, source_id: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        append_block = (
            f"\n\n---\n### Supplement ({now})\n"
            f"**Source**: {source_id}\n\n"
            f"{content}\n"
        )
        with open(page_path, "a", encoding="utf-8") as f:
            f.write(append_block)

    # --- Catalog / index generation ---

    def _iter_page_paths(self) -> List[Tuple[str, str]]:
        pages: List[Tuple[str, str]] = []
        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in sorted(files):
                if not _is_wiki_page(fname):
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.wiki_dir)
                pages.append((fpath, rel))
        return pages

    def _entry_from_page(self, fpath: str, rel: str) -> Optional[Dict[str, Any]]:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None

        meta, body = _parse_frontmatter(content)
        category = str(meta.get("category") or os.path.basename(os.path.dirname(fpath)))
        title = str(meta.get("title") or _extract_heading_title(body) or os.path.basename(fpath)[:-3])
        kind = str(meta.get("kind") or _infer_kind(category, title, body))
        status = str(meta.get("status") or _infer_status(category, title, kind))
        priority = str(meta.get("priority") or _infer_priority(category, title, kind, status))
        summary = str(meta.get("summary") or _extract_summary(body, title))
        use_when = str(meta.get("use_when") or _infer_use_when(category, title, kind))
        tags = _normalize_list(meta.get("tags")) or _infer_tags(category, title, kind)
        aliases = _normalize_list(meta.get("aliases"))
        supersedes = _normalize_list(meta.get("supersedes"))
        sources = _normalize_list(meta.get("sources"))
        knowledge_ids = _extract_knowledge_ids(content)

        return {
            "title": title,
            "path": rel,
            "full_path": os.path.join(self.wiki_dir, rel),
            "category": category,
            "kind": kind,
            "status": status,
            "priority": priority,
            "summary": summary,
            "use_when": use_when,
            "tags": tags,
            "aliases": aliases,
            "supersedes": supersedes,
            "sources": sources,
            "knowledge_ids": knowledge_ids,
            "last_compiled": str(meta.get("last_compiled", "")),
            "size_bytes": len(content.encode("utf-8")),
            "supplement_count": content.count("### 补充") + content.count("### Supplement"),
        }

    def rebuild_catalog(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for fpath, rel in self._iter_page_paths():
            entry = self._entry_from_page(fpath, rel)
            if entry:
                entries.append(entry)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(entries),
            "entries": entries,
        }
        with open(self.catalog_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return entries

    def _load_catalog(self) -> List[Dict[str, Any]]:
        try:
            with open(self.catalog_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            entries = payload.get("entries", [])
            return entries if isinstance(entries, list) else []
        except Exception:
            return []

    def get_catalog_entry(self, path: str) -> Optional[Dict[str, Any]]:
        normalized = path.replace("\\", "/")
        entries = self._load_catalog() or self.rebuild_catalog()
        for entry in entries:
            if str(entry.get("path", "")).replace("\\", "/") == normalized:
                return entry
        return None

    def rebuild_index(self):
        """Rewrite ``index.md``, ``_catalog.json``, and generated navigation files."""
        entries = self.rebuild_catalog()
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for entry in entries:
            categories.setdefault(entry["category"], []).append(entry)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# Knowledge wiki\n",
            f"> {len(entries)} page(s) | Last updated: {now}\n",
            "Use [map.md](map.md) for scenario routing; use this index for full browsing.\n",
        ]
        for cat in sorted(categories.keys()):
            pages = sorted(categories[cat], key=lambda x: (x.get("priority") != "high", x.get("title", "")))
            lines.append(f"\n## {cat} ({len(pages)})\n")
            for entry in pages:
                label = " / ".join(
                    bit for bit in [
                        str(entry.get("kind", "")),
                        str(entry.get("status", "")),
                        str(entry.get("priority", "")),
                    ] if bit
                )
                summary = f" — {entry.get('summary')}" if entry.get("summary") else ""
                use_when = f"; use when: {entry.get('use_when')}" if entry.get("use_when") else ""
                lines.append(f"- [{entry['title']}]({entry['path']}) `[{label}]`{summary}{use_when}")

        if os.path.exists(self.lint_log):
            try:
                with open(self.lint_log, "r", encoding="utf-8") as f:
                    lint_content = f.read()
                issue_count = lint_content.count("- [")
                if issue_count:
                    lines.append(f"\n## Open lint issues ({issue_count})\n")
                    lines.append("See [_lint_log.md](_lint_log.md)")
            except Exception:
                pass

        lines.append("")
        with open(self.index_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self._write_personal_experience_readme(entries)
        self._write_map(entries)
        logger.info(f"[WIKI] Index rebuilt: {len(entries)} pages in {len(categories)} categories")

    def _write_personal_experience_readme(self, entries: List[Dict[str, Any]]):
        groups = {
            "methodology": ("Methodology", "Reusable methods, frameworks, and techniques."),
            "lesson": ("Lessons", "Failure modes and behavior calibration rules."),
            "self_reflection": ("Self Reflection", "Identity, continuity, boundary, and self-description notes."),
            "operation_record": ("Operation Records", "Concrete historical actions or status records."),
        }
        items = [entry for entry in entries if entry.get("category") == "个人经验"]
        if not items:
            return
        lines = [
            "# Personal Experience Sections",
            "",
            "Generated from wiki catalog metadata. Source pages are not moved; `kind` drives this view.",
            "",
        ]
        for kind, (label, desc) in groups.items():
            grouped = [entry for entry in items if entry.get("kind") == kind]
            lines.append(f"## {label} ({len(grouped)})")
            lines.append("")
            lines.append(desc)
            lines.append("")
            for entry in sorted(grouped, key=lambda x: (x.get("priority") != "high", x.get("title", ""))):
                local_path = entry["path"].split("/", 1)[-1]
                lines.append(f"- [{entry['title']}]({local_path}) — {entry.get('summary') or entry.get('use_when') or ''}")
            lines.append("")
        os.makedirs(os.path.dirname(self.personal_experience_readme), exist_ok=True)
        with open(self.personal_experience_readme, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")

    def _write_map(self, entries: List[Dict[str, Any]]):
        """Generate a generic scenario router without shipping private content."""
        def pick(predicate, limit: int = 6) -> List[Dict[str, Any]]:
            selected = [entry for entry in entries if predicate(entry)]
            return sorted(selected, key=lambda x: (x.get("priority") != "high", x.get("title", "")))[:limit]

        scenarios = [
            (
                "Writing, Research, Or Reports",
                "Start here when drafting reports, public writing, research notes, or evidence-backed explanations.",
                pick(lambda e: e.get("kind") in {"methodology", "lesson"} or "writing" in (e.get("tags") or [])),
            ),
            (
                "Project Mechanics",
                "Start here when checking paths, runtime mechanisms, permissions, scheduler behavior, or architecture facts.",
                pick(lambda e: e.get("category") == "项目相关" or e.get("kind") == "project"),
            ),
            (
                "User Preferences",
                "Start here when confirming durable user preferences, naming, boundaries, or collaboration style.",
                pick(lambda e: e.get("category") in {"用户偏好", "用户身份"}),
            ),
            (
                "Identity And Reflection",
                "Start here for identity, continuity, self-description, and philosophical framing.",
                pick(lambda e: e.get("kind") in {"self_reflection", "philosophy"} or e.get("category") == "哲学"),
            ),
            (
                "Historical Or Archived Context",
                "Start here only when verifying past status, events, or records; do not treat these as current rules.",
                pick(lambda e: e.get("status") in {"historical", "archive_candidate", "archived"}),
            ),
        ]

        lines = [
            "---",
            "title: Knowledge Map",
            "category: Navigation",
            "kind: router",
            "status: active",
            "priority: high",
            "summary: Scenario router for selecting a small set of wiki pages before reading details.",
            "use_when: Use before broad or long-running tasks to avoid opening the full index as context.",
            "---",
            "",
            "# Knowledge Map",
            "",
            "This generated page is a route map, not a full directory. Pick a scenario, read a few linked pages, then inspect details as needed.",
            "",
        ]
        for title, description, items in scenarios:
            lines.append(f"## {title}")
            lines.append("")
            lines.append(description)
            lines.append("")
            if not items:
                lines.append("- No matching pages yet.")
            else:
                for entry in items:
                    lines.append(f"- [[{entry['title']}]] — {entry.get('use_when') or entry.get('summary') or ''}")
            lines.append("")

        with open(self.map_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")

    # --- Lint ---

    def lint(self) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        all_pages: Dict[str, str] = {}
        all_backlinks: Dict[str, List[str]] = {}
        page_dates: Dict[str, str] = {}
        referenced_titles: set = set()

        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in files:
                if not _is_wiki_page(fname):
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.wiki_dir)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue
                meta, body = _parse_frontmatter(content)
                title = meta.get("title") or _extract_heading_title(body) or fname[:-3]
                all_pages[str(title).lower()] = rel
                page_dates[rel] = str(meta.get("last_compiled", ""))
                for link_target in _extract_backlinks(body):
                    referenced_titles.add(link_target.lower())
                    all_backlinks.setdefault(link_target.lower(), []).append(rel)

        for ref_title in referenced_titles:
            if ref_title not in all_pages:
                issues.append({
                    "type": "gap",
                    "description": f"[[{ref_title}]] is linked but no page exists",
                    "pages": all_backlinks.get(ref_title, [])[:3],
                })

        for title_lower, rel in all_pages.items():
            if title_lower not in referenced_titles:
                issues.append({
                    "type": "orphan",
                    "description": f"Orphan page {rel} (no inbound wikilinks)",
                    "pages": [rel],
                })

        try:
            now = datetime.now(timezone.utc)
            for rel, date_str in page_dates.items():
                if not date_str:
                    continue
                try:
                    compiled = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if (now - compiled).days > 30:
                        issues.append({
                            "type": "stale",
                            "description": f"Page {rel} stale for {(now - compiled).days} days",
                            "pages": [rel],
                        })
                except ValueError:
                    pass
        except Exception:
            pass

        if issues:
            self._write_lint_log(issues)
        elif os.path.exists(self.lint_log):
            os.remove(self.lint_log)
        logger.info(f"[WIKI-LINT] Found {len(issues)} issues")
        return issues

    def _write_lint_log(self, issues: List[Dict[str, Any]]):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        labels = {
            "gap": "Missing pages (broken wikilinks)",
            "orphan": "Orphan pages",
            "stale": "Stale pages",
            "contradiction": "Contradictions",
        }
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for issue in issues:
            grouped.setdefault(issue["type"], []).append(issue)
        lines = ["# Wiki lint report\n", f"> Generated: {now}\n"]
        for issue_type, label in labels.items():
            items = grouped.get(issue_type, [])
            if not items:
                continue
            lines.append(f"\n## {label} ({len(items)})\n")
            for item in items:
                pages = ", ".join(item.get("pages", []))
                lines.append(f"- [{issue_type}] {item['description']} (pages: {pages})")
        lines.append("")
        with open(self.lint_log, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # --- Level 2: deep compile (LLM) ---

    def deep_compile(self, max_pages: int = 5) -> Dict[str, Any]:
        candidates = self._find_compile_candidates(max_pages)
        if not candidates:
            return {"success": True, "compiled": 0, "message": "Nothing to compile"}
        compiled = 0
        for page_path in candidates:
            try:
                if self._compile_single_page(page_path):
                    compiled += 1
            except Exception as e:
                logger.warning(f"[WIKI] Deep compile failed for {page_path}: {e}")
        if compiled:
            self.rebuild_index()
        return {"success": True, "compiled": compiled, "total_candidates": len(candidates)}

    def _find_compile_candidates(self, max_pages: int) -> List[str]:
        scored: List[Tuple[int, str]] = []
        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in files:
                if not _is_wiki_page(fname):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    meta, _body = _parse_frontmatter(content)
                    if meta.get("status") == "archived":
                        continue
                    supplement_count = content.count("### 补充") + content.count("### Supplement")
                    score = supplement_count * 3 + (len(content) // 2000)
                    if supplement_count >= 2 or len(content) >= 6000:
                        scored.append((score, fpath))
                except Exception:
                    continue
        scored.sort(key=lambda item: -item[0])
        return [path for _score, path in scored[:max_pages]]

    def _compile_single_page(self, page_path: str) -> bool:
        try:
            with open(page_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return False
        if len(content) < 200:
            return False
        meta, body = _parse_frontmatter(content)

        try:
            from backend.llm_api import llm_completion
        except ImportError:
            logger.warning("[WIKI] llm_api not available, skipping deep compile")
            return False

        prompt = (
            "You are a knowledge-base editor. The wiki page below contains multiple appended snippets.\n"
            "Merge them into one coherent Markdown article:\n"
            "1. Remove duplication while keeping facts accurate.\n"
            "2. If you spot contradictions, mark them with ⚠️ without deleting either side.\n"
            "3. Preserve every materially useful detail.\n"
            "4. Add [[wikilinks]] for related concepts.\n"
            "5. Output Markdown starting with a single top-level # title.\n"
            "6. Do not emit YAML front matter.\n\n"
            f"Page body:\n\n{body}"
        )
        result = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3,
        )
        if not result.get("success"):
            logger.warning(f"[WIKI] LLM compile failed: {result.get('error')}")
            return False
        new_body = result.get("content", "").strip()
        if not new_body or len(new_body) < 50:
            return False

        title = str(meta.get("title") or os.path.basename(page_path)[:-3])
        category = str(meta.get("category", ""))
        sources = _normalize_list(meta.get("sources"))
        kind = str(meta.get("kind") or _infer_kind(category, title, new_body))
        status = str(meta.get("status") or _infer_status(category, title, kind))
        priority = str(meta.get("priority") or _infer_priority(category, title, kind, status))
        fm = _make_frontmatter(
            title,
            category,
            sources,
            summary=str(meta.get("summary") or _extract_summary(new_body, title)),
            tags=_normalize_list(meta.get("tags")) or _infer_tags(category, title, kind),
            kind=kind,
            use_when=str(meta.get("use_when") or _infer_use_when(category, title, kind)),
            status=status,
            priority=priority,
            aliases=_normalize_list(meta.get("aliases")),
            supersedes=_normalize_list(meta.get("supersedes")),
        )
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + new_body + "\n")
        logger.info(f"[WIKI] Deep compiled: {os.path.relpath(page_path, self.wiki_dir)}")
        return True

    # --- Wiki search (recall_memory helper) ---

    def search_wiki(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search the machine catalog first; do not scan full wiki bodies per query."""
        query_lower = query.lower()
        query_keywords = _query_keywords(query)
        if not query_keywords:
            return []
        entries = self._load_catalog() or self.rebuild_catalog()

        priority_weight = {"high": 1.25, "normal": 1.0, "low": 0.65}
        status_weight = {"active": 1.0, "historical": 0.7, "archive_candidate": 0.55, "archived": 0.35}
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry in entries:
            primary = " ".join(
                str(part) for part in [
                    entry.get("title", ""),
                    entry.get("category", ""),
                    entry.get("kind", ""),
                    entry.get("summary", ""),
                    " ".join(entry.get("tags", []) or []),
                    " ".join(entry.get("aliases", []) or []),
                ]
            ).lower()
            secondary = str(entry.get("use_when", "")).lower()
            primary_hits = sum(1 for kw in query_keywords if kw in primary)
            secondary_hits = sum(1 for kw in query_keywords if kw in secondary)
            if primary_hits == 0 and secondary_hits == 0:
                continue
            score = (primary_hits + secondary_hits * 0.3) / max(len(query_keywords), 1)
            if query_lower in str(entry.get("title", "")).lower():
                score += 0.5
            if query_lower in secondary:
                score += 0.2
            score *= priority_weight.get(str(entry.get("priority", "normal")), 1.0)
            score *= status_weight.get(str(entry.get("status", "active")), 0.8)
            result = dict(entry)
            result["score"] = round(score, 2)
            scored.append((score, result))
        scored.sort(key=lambda item: -item[0])
        return [item for _score, item in scored[:limit]]

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        total = 0
        categories = set()
        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in files:
                if _is_wiki_page(fname):
                    total += 1
                    categories.add(os.path.basename(root))
        return {
            "total_pages": total,
            "categories": len(categories),
            "wiki_dir": self.wiki_dir,
            "catalog_file": self.catalog_file,
        }


_compiler_instance: Optional[KnowledgeCompiler] = None


def get_compiler() -> KnowledgeCompiler:
    """Return the process-wide ``KnowledgeCompiler`` singleton."""
    global _compiler_instance
    if _compiler_instance is None:
        _compiler_instance = KnowledgeCompiler()
    return _compiler_instance
