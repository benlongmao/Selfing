#!/usr/bin/env python3
"""
Deep research workflow: multi-query search, dedupe, light synthesis, optional Markdown report.

[2026-02-22] Replaces the old state-only research_engine.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DeepResearchTool:
    """Tavily-backed multi-step research and reporting."""

    def __init__(self, tavily_client=None, llm_client=None, workspace_dir: str = "workspace/sandbox"):
        self.tavily = tavily_client
        self.llm_client = llm_client
        self.workspace_dir = Path(workspace_dir)
        self.research_dir = self.workspace_dir / "research"
        self.research_dir.mkdir(parents=True, exist_ok=True)

    def research(
        self,
        topic: str,
        depth: str = "medium",
        perspectives: Optional[List[str]] = None,
        max_sources: int = 10,
        generate_report: bool = True,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run multi-query search, dedupe, extract bullets, optional LLM report, optional save to disk.
        """
        logger.info(f"[RESEARCH] Starting deep research on: {topic}")

        result = {
            "topic": topic,
            "depth": depth,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "searches": [],
            "sources": [],
            "key_findings": [],
            "report": None,
            "report_path": None,
            "success": False
        }

        try:
            queries = self._generate_queries(topic, depth, perspectives)
            result["queries"] = queries

            all_results = []
            for query in queries:
                search_result = self._search(query)
                if search_result:
                    result["searches"].append({
                        "query": query,
                        "results_count": len(search_result.get("results", []))
                    })
                    all_results.extend(search_result.get("results", []))

            unique_sources = self._deduplicate_sources(all_results, max_sources)
            result["sources"] = unique_sources
            result["sources_count"] = len(unique_sources)

            key_findings = self._extract_key_findings(topic, unique_sources)
            result["key_findings"] = key_findings

            if generate_report and self.llm_client:
                report = self._generate_report(topic, unique_sources, key_findings, depth)
                result["report"] = report

                if save_report and report:
                    report_path = self._save_report(topic, report)
                    result["report_path"] = str(report_path)

            result["success"] = True
            logger.info(f"[RESEARCH] Completed: {len(unique_sources)} sources, {len(key_findings)} findings")

        except Exception as e:
            logger.error(f"[RESEARCH] Failed: {e}")
            result["error"] = str(e)

        return result

    def _generate_queries(
        self,
        topic: str,
        depth: str,
        perspectives: Optional[List[str]]
    ) -> List[str]:
        """Build query variants; include CN/EN n-grams to match mixed corpora."""
        queries = [topic]

        if depth in ("medium", "deep"):
            queries.extend([
                f"{topic} latest updates",
                f"{topic} pros and cons",
            ])

        if depth == "deep":
            queries.extend([
                f"{topic} expert opinions",
                f"{topic} case studies",
                f"{topic} future trends",
            ])

        if perspectives:
            for p in perspectives[:3]:
                queries.append(f"{topic} {p}")

        return queries[:8]

    def _search(self, query: str) -> Optional[Dict]:
        if not self.tavily:
            logger.warning("[RESEARCH] Tavily client not available")
            return None

        try:
            return self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
        except Exception as e:
            logger.warning(f"[RESEARCH] Search failed for '{query}': {e}")
            return None

    def _deduplicate_sources(self, results: List[Dict], max_sources: int) -> List[Dict]:
        seen_urls = set()
        unique = []

        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "content": (r.get("content", ""))[:500],
                    "score": r.get("score", 0)
                })

        unique.sort(key=lambda x: x.get("score", 0), reverse=True)
        return unique[:max_sources]

    def _extract_key_findings(self, topic: str, sources: List[Dict]) -> List[str]:
        findings = []
        for source in sources[:5]:
            content = source.get("content", "")
            if content:
                sentences = re.split(r'[。.!！?？]', content)
                for sent in sentences:
                    if 20 < len(sent) < 200:
                        topic_words = set(topic.lower().split())
                        sent_words = set(sent.lower().split())
                        if topic_words & sent_words:
                            findings.append(sent.strip())
                            break

        return findings[:10]

    def _generate_report(
        self,
        topic: str,
        sources: List[Dict],
        findings: List[str],
        depth: str
    ) -> str:
        if not self.llm_client:
            return self._generate_simple_report(topic, sources, findings)

        sources_text = "\n".join([
            f"- [{s['title']}]({s['url']}): {s['content'][:200]}..."
            for s in sources[:8]
        ])

        findings_text = "\n".join([f"- {f}" for f in findings[:8]])

        length_hint = (
            "about 200–400 words" if depth == "quick" else
            "about 400–800 words" if depth == "medium" else
            "about 800–1200 words"
        )

        prompt = f"""Write a concise research memo on the topic: "{topic}".

## Sources
{sources_text}

## Initial bullets
{findings_text}

Requirements:
1. Use Markdown. Sections: Overview, Key points, Conclusion.
2. Length: {length_hint}
3. Stay factual; cite implications cautiously.
4. Primary language: English. You may keep short quoted phrases in their original language when needed.

Output the memo:
"""

        try:
            response = self.llm_client.call(prompt, temperature=0.5, max_tokens=1500)
            return response.get("content", "") if isinstance(response, dict) else str(response)
        except Exception as e:
            logger.warning(f"[RESEARCH] LLM report generation failed: {e}")
            return self._generate_simple_report(topic, sources, findings)

    def _generate_simple_report(
        self,
        topic: str,
        sources: List[Dict],
        findings: List[str]
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        report = f"""# Research memo: {topic}

**Generated**: {now}
**Sources**: {len(sources)}

## References

"""
        for s in sources[:5]:
            report += f"- [{s['title']}]({s['url']})\n"

        report += "\n## Key bullets\n\n"
        for i, f in enumerate(findings[:5], 1):
            report += f"{i}. {f}\n"

        report += "\n---\n_Auto-generated by the deep_research tool._\n"

        return report

    def _save_report(self, topic: str, report: str) -> Path:
        safe_topic = re.sub(r'[^\w\u4e00-\u9fff]', '_', topic)[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{safe_topic}_{timestamp}.md"

        filepath = self.research_dir / filename
        filepath.write_text(report, encoding="utf-8")

        logger.info(f"[RESEARCH] Report saved to: {filepath}")
        return filepath

    def get_tool_definitions(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "deep_research",
                    "description": """Multi-source research: Tavily search, dedupe, key bullets, optional Markdown report saved under workspace/.../research/.

depth: quick | medium | deep. Optional ``perspectives`` add query axes (e.g. technology, market, risk).""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic or question to investigate"
                            },
                            "depth": {
                                "type": "string",
                                "enum": ["quick", "medium", "deep"],
                                "description": "Search breadth: quick (fewer queries) … deep (more)"
                            },
                            "perspectives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Extra query lenses appended to the topic (research angles or sub-questions)."
                            }
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_research_reports",
                    "description": "List saved research_*.md files under the research directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Max files to return (default 10)"
                            }
                        }
                    }
                }
            }
        ]

    def list_reports(self, limit: int = 10) -> List[Dict]:
        reports = []

        try:
            for f in sorted(self.research_dir.glob("research_*.md"), reverse=True)[:limit]:
                stat = f.stat()
                reports.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        except Exception as e:
            logger.error(f"[RESEARCH] Failed to list reports: {e}")

        return reports


_instance = None

def get_deep_research_tool(tavily_client=None, llm_client=None) -> DeepResearchTool:
    global _instance
    if _instance is None:
        _instance = DeepResearchTool(tavily_client, llm_client)
    return _instance
