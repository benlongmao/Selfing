"""
S-44 knowledge wiki compiler.

Materializes ``knowledge_items`` (and related research notes) into Markdown pages
under ``workspace/sandbox/wiki``. On-disk folder names use the same **Chinese category literals**
as ``KnowledgeBase.VALID_CATEGORIES`` (see ``_CATEGORY_DIR_MAP``); do not rename those keys
without a coordinated DB/wiki migration. Pages use YAML front matter and ``[[wikilinks]]``.

Tiers:
- **Level 1 — ``quick_append``:** immediate, no LLM, zero tokens; appends or creates pages.
- **Level 2 — ``deep_compile``:** deferred consolidation via ``llm_api`` (e.g. heartbeat).

Initial version: 2026-04-07.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join("workspace", "sandbox", "wiki")
INDEX_FILE = os.path.join(WIKI_DIR, "index.md")
LINT_LOG = os.path.join(WIKI_DIR, "_lint_log.md")

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
    return re.findall(r'\[\[([^\]]+)\]\]', text)


def _make_frontmatter(title: str, category: str, sources: List[str],
                      confidence: float = 0.5) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    src_yaml = "\n".join(f"  - {s}" for s in sources) if sources else "  - unknown"
    return f"""---
title: {title}
category: {category}
sources:
{src_yaml}
confidence: {confidence}
last_compiled: {now}
---"""


def _parse_frontmatter(content: str) -> Tuple[Dict, str]:
    """Parse a minimal ``---`` / ``---`` block; return ``(meta_dict, body)``."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm_text = content[3:end].strip()
    body = content[end + 4:].lstrip("\n")
    meta = {}
    for line in fm_text.split("\n"):
        if ":" in line and not line.startswith("  "):
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()
    return meta, body


class KnowledgeCompiler:
    """Compile SQLite knowledge rows into sandbox wiki Markdown."""

    def __init__(self, wiki_dir: str = WIKI_DIR):
        self.wiki_dir = wiki_dir
        os.makedirs(self.wiki_dir, exist_ok=True)

    # --- Level 1: quick append (no LLM) ---

    def quick_append(self, title: str, content: str, category: str,
                     source_id: str = "", confidence: float = 0.5) -> Dict:
        """
        Append one fact to the wiki without calling an LLM.

        Resolves the category folder, finds a similar page title (fuzzy match),
        appends or creates the page, then refreshes ``index.md``.
        """
        cat_dir = _CATEGORY_DIR_MAP.get(category, category)
        page_dir = os.path.join(self.wiki_dir, cat_dir)
        os.makedirs(page_dir, exist_ok=True)

        existing_page = self._find_similar_page(page_dir, title)

        if existing_page:
            self._append_to_page(existing_page, title, content, source_id)
            action = "appended"
            page_path = existing_page
        else:
            page_path = self._create_page(
                page_dir, title, content, category, source_id, confidence
            )
            action = "created"

        self.rebuild_index()

        rel_path = os.path.relpath(page_path, self.wiki_dir)
        logger.info(f"[WIKI] {action}: {rel_path} (title={title[:40]})")
        return {"success": True, "action": action, "page": rel_path}

    def _find_similar_page(self, page_dir: str, title: str,
                           threshold: float = 0.55) -> Optional[str]:
        """Return the path of the most similar ``*.md`` page under ``page_dir``, if any."""
        if not os.path.isdir(page_dir):
            return None
        best_path, best_sim = None, 0.0
        for fname in os.listdir(page_dir):
            if not fname.endswith(".md") or fname.startswith("_"):
                continue
            page_title = fname[:-3].replace("_", " ")
            try:
                with open(os.path.join(page_dir, fname), "r", encoding="utf-8") as f:
                    first_400 = f.read(400)
                meta, _ = _parse_frontmatter(first_400)
                if meta.get("title"):
                    page_title = meta["title"]
            except Exception:
                pass
            sim = _title_similarity(title, page_title)
            if sim > best_sim:
                best_sim = sim
                best_path = os.path.join(page_dir, fname)
        return best_path if best_sim >= threshold else None

    def _create_page(self, page_dir: str, title: str, content: str,
                     category: str, source_id: str,
                     confidence: float) -> str:
        fname = _sanitize_filename(title) + ".md"
        page_path = os.path.join(page_dir, fname)

        sources = [source_id] if source_id else []
        fm = _make_frontmatter(title, category, sources, confidence)

        body = f"\n# {title}\n\n{content}\n"

        links = _extract_backlinks(content)
        if links:
            body += "\n## Related topics\n"
            for lk in links:
                body += f"- [[{lk}]]\n"

        with open(page_path, "w", encoding="utf-8") as f:
            f.write(fm + "\n" + body)
        return page_path

    def _append_to_page(self, page_path: str, title: str, content: str,
                        source_id: str):
        try:
            with open(page_path, "r", encoding="utf-8") as f:
                existing = f.read()
        except Exception:
            existing = ""

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        append_block = (
            f"\n\n---\n### Supplement ({now})\n"
            f"**Source**: {source_id}\n\n"
            f"{content}\n"
        )

        if source_id and source_id not in existing:
            meta, body = _parse_frontmatter(existing)
            if meta:
                existing = existing.replace(
                    f"last_compiled: {meta.get('last_compiled', '')}",
                    f"last_compiled: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                )

        with open(page_path, "a", encoding="utf-8") as f:
            f.write(append_block)

    # --- Index rebuild ---

    def rebuild_index(self):
        """Rewrite ``wiki/index.md`` with every non-hidden Markdown page."""
        categories: Dict[str, List[Tuple[str, str, str]]] = {}
        total = 0

        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in sorted(files):
                if not fname.endswith(".md") or fname.startswith("_") or fname == "index.md":
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.wiki_dir)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        head = f.read(500)
                    meta, body = _parse_frontmatter(head)
                except Exception:
                    meta, body = {}, ""

                cat = meta.get("category", os.path.basename(root))
                title = meta.get("title", fname[:-3].replace("_", " "))
                summary = body.strip().split("\n")[0][:80] if body.strip() else ""

                categories.setdefault(cat, []).append((title, rel, summary))
                total += 1

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"# Knowledge wiki\n",
            f"> {total} page(s) | Last updated: {now}\n",
        ]

        for cat in sorted(categories.keys()):
            pages = categories[cat]
            lines.append(f"\n## {cat} ({len(pages)})\n")
            for title, rel, summary in pages:
                summary_part = f" — {summary}" if summary else ""
                lines.append(f"- [{title}]({rel}){summary_part}")

        if os.path.exists(LINT_LOG):
            try:
                with open(LINT_LOG, "r", encoding="utf-8") as f:
                    lint_content = f.read()
                issue_count = lint_content.count("- [")
                if issue_count:
                    lines.append(f"\n## Open lint issues ({issue_count})\n")
                    lines.append("See [_lint_log.md](_lint_log.md)")
            except Exception:
                pass

        lines.append("")
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"[WIKI] Index rebuilt: {total} pages in {len(categories)} categories")

    # --- Lint (gaps / orphans / staleness) ---

    def lint(self) -> List[Dict]:
        """
        Cheap structural checks without an LLM:

        1. **Gap:** wikilink target has no backing page.
        2. **Orphan:** page is never linked from another page.
        3. **Stale:** ``last_compiled`` older than 30 days.
        """
        all_pages: Dict[str, str] = {}  # title_lower → rel_path
        all_backlinks: Dict[str, List[str]] = {}  # target_title_lower → [source_page]
        page_dates: Dict[str, str] = {}  # rel_path → last_compiled
        referenced_titles: set = set()
        issues: List[Dict] = []

        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in files:
                if not fname.endswith(".md") or fname.startswith("_") or fname == "index.md":
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.wiki_dir)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue

                meta, body = _parse_frontmatter(content)
                title = meta.get("title", fname[:-3])
                all_pages[title.lower()] = rel
                page_dates[rel] = meta.get("last_compiled", "")

                for link_target in _extract_backlinks(body):
                    referenced_titles.add(link_target.lower())
                    all_backlinks.setdefault(link_target.lower(), []).append(rel)

        for ref_title in referenced_titles:
            if ref_title not in all_pages:
                sources = all_backlinks.get(ref_title, [])
                issues.append({
                    "type": "gap",
                    "description": f"[[{ref_title}]] is linked but no page exists",
                    "pages": sources[:3],
                })

        for title_lower, rel in all_pages.items():
            if title_lower not in referenced_titles:
                issues.append({
                    "type": "orphan",
                    "description": f"Orphan page {rel} (no inbound wikilinks)",
                    "pages": [rel],
                })

        # Stale: last_compiled older than 30 days
        try:
            now = datetime.now(timezone.utc)
            for rel, date_str in page_dates.items():
                if not date_str:
                    continue
                try:
                    compiled = datetime.strptime(date_str, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
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
        elif os.path.exists(LINT_LOG):
            os.remove(LINT_LOG)

        logger.info(f"[WIKI-LINT] Found {len(issues)} issues")
        return issues

    def _write_lint_log(self, issues: List[Dict]):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"# Wiki lint report\n", f"> Generated: {now}\n"]

        by_type = {}
        for issue in issues:
            by_type.setdefault(issue["type"], []).append(issue)

        type_labels = {
            "gap": "Missing pages (broken wikilinks)",
            "orphan": "Orphan pages",
            "stale": "Stale pages",
            "contradiction": "Contradictions",
        }
        for t, label in type_labels.items():
            items = by_type.get(t, [])
            if not items:
                continue
            lines.append(f"\n## {label} ({len(items)})\n")
            for item in items:
                pages = ", ".join(item.get("pages", []))
                lines.append(f"- [{t}] {item['description']} (pages: {pages})")

        lines.append("")
        with open(LINT_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # --- Level 2: deep compile (LLM) ---

    def deep_compile(self, max_pages: int = 5) -> Dict:
        """
        Ask the LLM to merge append-heavy wiki pages.

        Goals: dedupe facts, flag suspected contradictions with ``⚠️``, enrich
        ``[[wikilinks]]``. Each call processes at most ``max_pages`` files to cap tokens.

        Priority: pages with multiple supplement sections (legacy ``### 补充`` or
        new ``### Supplement``) indicating fragmented notes.
        """
        candidates = self._find_compile_candidates(max_pages)
        if not candidates:
            return {"success": True, "compiled": 0, "message": "Nothing to compile"}

        compiled = 0
        for page_path in candidates:
            try:
                ok = self._compile_single_page(page_path)
                if ok:
                    compiled += 1
            except Exception as e:
                logger.warning(f"[WIKI] Deep compile failed for {page_path}: {e}")

        if compiled:
            self.rebuild_index()

        return {"success": True, "compiled": compiled, "total_candidates": len(candidates)}

    def _find_compile_candidates(self, max_pages: int) -> List[str]:
        """Prefer pages with two or more supplement blocks (ZH or EN heading)."""
        scored = []
        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in files:
                if not fname.endswith(".md") or fname.startswith("_") or fname == "index.md":
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    supplement_count = content.count("### 补充") + content.count("### Supplement")
                    if supplement_count >= 2:
                        scored.append((supplement_count, fpath))
                except Exception:
                    continue
        scored.sort(key=lambda x: -x[0])
        return [path for _, path in scored[:max_pages]]

    def _compile_single_page(self, page_path: str) -> bool:
        """Rewrite a single wiki page body via ``llm_completion``."""
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
            "You are a knowledge-base editor. The wiki page below contains multiple "
            "appended snippets.\n"
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

        title = meta.get("title", os.path.basename(page_path)[:-3])
        category = meta.get("category", "")
        sources = [s.strip("- ") for s in meta.get("sources", "").split("\n") if s.strip("- ")]
        fm = _make_frontmatter(title, category, sources)

        with open(page_path, "w", encoding="utf-8") as f:
            f.write(fm + "\n\n" + new_body + "\n")

        logger.info(f"[WIKI] Deep compiled: {os.path.relpath(page_path, self.wiki_dir)}")
        return True

    # --- Wiki search (recall_memory helper) ---

    def search_wiki(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Lightweight keyword scan over wiki Markdown (no embeddings).

        Returns title, relative path, short summary, and a heuristic score.
        """
        query_lower = query.lower()
        query_keywords = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}', query_lower))
        if not query_keywords:
            return []

        scored: List[Tuple[float, Dict]] = []

        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in files:
                if not fname.endswith(".md") or fname.startswith("_") or fname == "index.md":
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.wiki_dir)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read(2000)
                except Exception:
                    continue

                meta, body = _parse_frontmatter(content)
                title = meta.get("title", fname[:-3])
                searchable = (title + " " + body[:500]).lower()

                hits = sum(1 for kw in query_keywords if kw in searchable)
                if hits == 0:
                    continue

                score = hits / max(len(query_keywords), 1)
                if query_lower in title.lower():
                    score += 0.5

                summary = body.strip().split("\n")[0][:120] if body.strip() else ""
                scored.append((score, {
                    "title": title,
                    "path": rel,
                    "full_path": os.path.join(self.wiki_dir, rel),
                    "category": meta.get("category", ""),
                    "summary": summary,
                    "score": round(score, 2),
                }))

        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:limit]]

    # --- Stats ---

    def get_stats(self) -> Dict:
        """Return page counts grouped by on-disk category folder."""
        total = 0
        categories = set()
        for root, _dirs, files in os.walk(self.wiki_dir):
            for fname in files:
                if fname.endswith(".md") and not fname.startswith("_") and fname != "index.md":
                    total += 1
                    categories.add(os.path.basename(root))
        return {
            "total_pages": total,
            "categories": len(categories),
            "wiki_dir": self.wiki_dir,
        }


_compiler_instance: Optional[KnowledgeCompiler] = None


def get_compiler() -> KnowledgeCompiler:
    """Return the process-wide ``KnowledgeCompiler`` singleton."""
    global _compiler_instance
    if _compiler_instance is None:
        _compiler_instance = KnowledgeCompiler()
    return _compiler_instance
