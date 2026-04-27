#!/usr/bin/env python3
"""
Tavily search client — optimized build with cache, error handling, and dynamic options.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = os.environ.get("TAVILY_ENDPOINT", "https://api.tavily.com/search")


class TavilyClient:
    """
    Tavily search client (optimized).

    Features:
    - Caching to reduce duplicate searches and save quota
    - Timeouts, 429 handling, and other HTTP edge cases
    - Configurable search_depth: basic / advanced
    - Full snippets (not aggressively truncated)
    - Optional AI-generated answer from Tavily
    - Usage stats
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        timeout: int = 30,
        max_snippet_length: int = 2000,
    ):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.max_results = max_results
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.max_snippet_length = max_snippet_length

        # cache: {cache_key: (result, timestamp)}
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

        self.stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0,
            "total_cost_estimate": 0.0,  # rough $0.01 per call
        }

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get_cache_key(self, query: str, max_results: int, search_depth: str) -> str:
        """Build cache key."""
        key = f"{query}:{max_results}:{search_depth}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Return cached result if still valid."""
        if not self.cache_enabled:
            return None

        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.stats["cache_hits"] += 1
                logger.info(f"Tavily cache hit (key: {cache_key[:8]}...)")
                return result
            del self._cache[cache_key]

        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Store result in cache."""
        if self.cache_enabled:
            self._cache[cache_key] = (result, time.time())
            logger.debug(f"Tavily result cached (key: {cache_key[:8]}...)")

    def search(
        self,
        query: str,
        max_results: int | None = None,
        search_depth: str = "advanced",
        include_answer: bool = True,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a Tavily search.

        Args:
            query: Search string
            max_results: Max items (default from client)
            search_depth: "basic" | "advanced" (speed vs depth)
            include_answer: Ask Tavily for a synthesized answer
            use_cache: Read/write cache

        Returns:
            {"results": [...], "answer": "..."} when include_answer
        """
        if not self.enabled:
            raise RuntimeError("TAVILY_API_KEY is not set; search cannot run.")

        self.stats["total_searches"] += 1
        max_results = max_results or self.max_results

        cache_key = self._get_cache_key(query, max_results, search_depth)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": False,
        }

        logger.info(f"Tavily search [depth={search_depth}]: {query}")

        try:
            resp = requests.post(TAVILY_ENDPOINT, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            self.stats["api_calls"] += 1
            self.stats["total_cost_estimate"] += 0.01

            if not data.get("results"):
                logger.warning(f"Tavily returned no results: {query}")
                data = {
                    "results": [],
                    "answer": "No relevant results. Try different keywords or a wider query.",
                }

            if use_cache:
                self._save_to_cache(cache_key, data)

            return data

        except requests.exceptions.Timeout:
            self.stats["errors"] += 1
            logger.error(f"Tavily search timed out (>{self.timeout}s): {query}")
            return {
                "results": [],
                "answer": f"Search timed out (>{self.timeout}s). Retry later or simplify the query.",
            }

        except requests.exceptions.HTTPError as e:
            self.stats["errors"] += 1

            if e.response.status_code == 429:
                logger.error("Tavily API rate limit (429 Too Many Requests)")
                return {
                    "results": [],
                    "answer": "Search quota exhausted. Try again later or upgrade the Tavily plan.",
                }
            if e.response.status_code == 401:
                logger.error("Tavily API key invalid (401 Unauthorized)")
                return {
                    "results": [],
                    "answer": "API key invalid. Check TAVILY_API_KEY.",
                }
            logger.error(f"Tavily HTTP error ({e.response.status_code}): {e}")
            raise

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Tavily search failed: {e}")
            raise

    @staticmethod
    def format_results(
        data: Dict[str, Any],
        max_items: int = 5,
        max_snippet_length: int = 2000,
        show_scores: bool = True,
    ) -> str:
        """
        Format search results for display.

        Args:
            data: Raw Tavily response
            max_items: How many results to show
            max_snippet_length: Max snippet length (default 2000)
            show_scores: Show relevance score when available
        """
        results = data.get("results") or []
        answer = data.get("answer")

        if not results and not answer:
            return "No reliable results were retrieved."

        lines: List[str] = []

        if answer:
            lines.append(f"**AI summary**\n{answer}\n")
            lines.append("---\n")

        if results:
            lines.append(f"**Search results** ({len(results)})\n")

            for idx, item in enumerate(results[:max_items], start=1):
                title = item.get("title") or "(no title)"
                url = item.get("url") or ""
                snippet = item.get("content") or item.get("snippet") or ""
                score = item.get("score", 0)

                result_text = f"{idx}. **{title}**"

                if show_scores and score > 0:
                    result_text += f" `[score: {score:.2f}]`"

                result_text += f"\n   {url}"

                if snippet:
                    snippet_display = snippet.strip()[:max_snippet_length]
                    if len(snippet.strip()) > max_snippet_length:
                        snippet_display += "..."
                    result_text += f"\n   {snippet_display}"

                lines.append(result_text)

        return "\n\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        cache_hit_rate = (
            self.stats["cache_hits"] / self.stats["total_searches"]
            if self.stats["total_searches"] > 0
            else 0.0
        )

        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.1%}",
            "cache_size": len(self._cache),
            "cache_enabled": self.cache_enabled,
        }

    def clear_cache(self):
        """Drop all cache entries."""
        self._cache.clear()
        logger.info("Tavily cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Cache inspection for debugging."""
        now = time.time()
        valid_entries = 0
        expired_entries = 0

        for cache_key, (result, timestamp) in self._cache.items():
            if now - timestamp < self.cache_ttl:
                valid_entries += 1
            else:
                expired_entries += 1

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "ttl_seconds": self.cache_ttl,
        }
