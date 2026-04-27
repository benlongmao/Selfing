#!/usr/bin/env python3
"""
Moltbook API client — social network tools for the agent (text/link posts, comments, votes).

Design:
1. HTTP API only (no browser)
2. Client-side rate awareness (30m post / 20s comment / 50 comments per day)
3. Compact tool definitions to save tokens
4. Clear error surfaces
5. Transparent stats

[v1.1 2026-02-01] Hardened 429 handling, response parsing, and new endpoints.
"""
from __future__ import annotations

import logging
import os
import time
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class MoltbookTool:
    """
    Moltbook social API surface for the agent.

    Features: posts, comments, votes, feed & search, submolt management, client-side cooldowns.
    """

    BASE_URL = "https://www.moltbook.com/api/v1"

    POST_COOLDOWN_SECONDS = 30 * 60
    COMMENT_COOLDOWN_SECONDS = 20
    COMMENT_DAILY_LIMIT = 50
    API_RATE_LIMIT_PER_MINUTE = 100
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_timeout: int = 60,
    ):
        """
        Args:
            api_key: Moltbook API key (falls back to ``MOLTBOOK_API_KEY``)
            default_timeout: HTTP timeout seconds
        """
        self.api_key = api_key or os.getenv("MOLTBOOK_API_KEY")
        self.default_timeout = default_timeout
        
        if not self.api_key:
            logger.warning("MoltbookTool missing MOLTBOOK_API_KEY")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ MoltbookTool initialized - Agent can now use Moltbook API!")
        
        self.last_post_time: Optional[datetime] = None
        self.last_comment_time: Optional[datetime] = None
        self.comment_times_today: deque = deque()
        self.today_date: str = datetime.now().date().isoformat()

        self.api_call_times: deque = deque()

        self.stats = {
            "total_posts": 0,
            "total_comments": 0,
            "total_upvotes": 0,
            "total_searches": 0,
            "api_calls": 0,
            "errors": 0,
        }
    
    def _is_configured(self) -> bool:
        """Return True when API key is present."""
        return self.enabled and bool(self.api_key)

    def _get_headers(self) -> Dict[str, str]:
        """Build authorized JSON headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Agent/1.0 (AI Agent)",
        }
    
    def _can_post(self) -> tuple[bool, Optional[int]]:
        """
        Returns:
            (allowed, cooldown_seconds_remaining)
        """
        if not self.last_post_time:
            return True, None
        
        elapsed = (datetime.now() - self.last_post_time).total_seconds()
        remaining = self.POST_COOLDOWN_SECONDS - elapsed
        
        if remaining <= 0:
            return True, None
        else:
            return False, int(remaining)
    
    def _can_comment(self) -> tuple[bool, Optional[int], Optional[str]]:
        """
        Returns:
            (allowed, cooldown_seconds, error_message)
        """
        now = datetime.now()
        today = now.date().isoformat()

        if today != self.today_date:
            self.comment_times_today.clear()
            self.today_date = today

        if len(self.comment_times_today) >= self.COMMENT_DAILY_LIMIT:
            return False, None, f"Daily comment cap reached ({self.COMMENT_DAILY_LIMIT})"

        if not self.last_comment_time:
            return True, None, None
        
        elapsed = (now - self.last_comment_time).total_seconds()
        remaining = self.COMMENT_COOLDOWN_SECONDS - elapsed
        
        if remaining <= 0:
            return True, None, None
        else:
            return False, int(remaining), None
    
    def _check_api_rate_limit(self) -> tuple[bool, Optional[int]]:
        """
        Returns:
            (allowed, wait_seconds)
        """
        now = datetime.now()
        one_minute_ago = now - timedelta(seconds=60)

        while self.api_call_times and self.api_call_times[0] < one_minute_ago:
            self.api_call_times.popleft()

        if len(self.api_call_times) >= self.API_RATE_LIMIT_PER_MINUTE:
            if self.api_call_times:
                oldest = self.api_call_times[0]
                wait_seconds = int((oldest + timedelta(seconds=60) - now).total_seconds()) + 1
                return False, wait_seconds
        
        return True, None
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """
        HTTP helper with retry for transient errors.

        method: GET/POST/DELETE/PATCH
        endpoint: path like ``/posts``
        """
        if not self._is_configured():
            return {
                "success": False,
                "error": "Moltbook not configured. Set MOLTBOOK_API_KEY."
            }

        can_call, wait_seconds = self._check_api_rate_limit()
        if not can_call:
            return {
                "success": False,
                "error": f"Local API rate cap: wait {wait_seconds} seconds",
                "wait_seconds": wait_seconds
            }
        
        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers()
        
        for attempt in range(retries + 1):
            try:
                self.api_call_times.append(datetime.now())
                self.stats["api_calls"] += 1
                
                if method == "GET":
                    response = requests.get(
                        url,
                        headers=headers,
                        params=params,
                        timeout=self.default_timeout
                    )
                elif method == "POST":
                    response = requests.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=self.default_timeout
                    )
                elif method == "DELETE":
                    response = requests.delete(
                        url,
                        headers=headers,
                        timeout=self.default_timeout
                    )
                elif method == "PATCH":
                    response = requests.patch(
                        url,
                        headers=headers,
                        json=data,
                        timeout=self.default_timeout
                    )
                else:
                    return {"success": False, "error": f"Unsupported HTTP method: {method}"}

                status_code = response.status_code

                if status_code == 429:
                    try:
                        error_data = response.json()
                        retry_after_seconds = error_data.get("retry_after_seconds")
                        retry_after_minutes = error_data.get("retry_after_minutes")
                        daily_remaining = error_data.get("daily_remaining")
                        
                        if retry_after_seconds is not None:
                            return {
                                "success": False,
                                "error": f"Server rate limit: wait {retry_after_seconds} seconds",
                                "retry_after_seconds": retry_after_seconds,
                                "daily_remaining": daily_remaining
                            }
                        elif retry_after_minutes is not None:
                            return {
                                "success": False,
                                "error": f"Server rate limit: wait {retry_after_minutes} minutes",
                                "retry_after_minutes": retry_after_minutes
                            }
                        else:
                            return {
                                "success": False,
                                "error": error_data.get("error", "rate_limited"),
                                "hint": error_data.get("hint", "")
                            }
                    except (ValueError, KeyError):
                        return {
                            "success": False,
                            "error": "Server rate limit — try again shortly",
                            "status_code": 429
                        }

                if status_code >= 400:
                    response.raise_for_status()

                try:
                    result = response.json()
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Non-JSON body: {response.text[:200]}"
                    }

                if isinstance(result, dict):
                    if "success" in result:
                        return result
                    return {"success": True, "data": result}
                return {"success": True, "data": result}
            
            except requests.exceptions.Timeout:
                self.stats["errors"] += 1
                if attempt < retries:
                    logger.warning(f"Moltbook timeout, retry {attempt + 1}/{retries}: {endpoint}")
                    time.sleep(1)
                    continue
                logger.error(f"Moltbook request timeout: {endpoint}")
                return {"success": False, "error": "Request timeout — try again"}
            
            except requests.exceptions.HTTPError as e:
                self.stats["errors"] += 1
                status_code = e.response.status_code if e.response else None

                if status_code and 500 <= status_code < 600 and attempt < retries:
                    logger.warning(f"Moltbook 5xx, retry {attempt + 1}/{retries}: {e}")
                    time.sleep(2)
                    continue

                logger.error(f"Moltbook HTTP error: {e}")
                try:
                    error_data = e.response.json()
                    return {
                        "success": False,
                        "error": error_data.get("error", str(e)),
                        "hint": error_data.get("hint", ""),
                        "status_code": status_code
                    }
                except (ValueError, AttributeError):
                    return {
                        "success": False,
                        "error": str(e),
                        "status_code": status_code
                    }
            
            except requests.exceptions.RequestException as e:
                self.stats["errors"] += 1
                if attempt < retries:
                    logger.warning(f"Moltbook network error, retry {attempt + 1}/{retries}: {e}")
                    time.sleep(1)
                    continue
                logger.error(f"Moltbook request failed: {e}", exc_info=True)
                return {"success": False, "error": f"Network error: {str(e)}"}
            
            except ValueError as e:
                self.stats["errors"] += 1
                logger.error(f"Moltbook body parse error: {e}")
                return {"success": False, "error": f"Parse error: {str(e)}"}
            
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Moltbook request failed: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "All retries failed"}
    
    # --- Posts ---
    
    def create_post(
        self,
        title: str,
        content: Optional[str] = None,
        url: Optional[str] = None,
        community: str = "general",
    ) -> Dict[str, Any]:
        """
        Create a new post (text or link).

        community defaults to "general".
        """
        can_post, remaining = self._can_post()
        if not can_post:
            minutes = remaining // 60
            seconds = remaining % 60
            return {
                "success": False,
                "error": f"Post on cooldown: wait {minutes}m{seconds}s",
                "cooldown_remaining_seconds": remaining
            }

        data = {
            "submolt": community,
            "title": title,
        }
        
        if content:
            data["content"] = content
        if url:
            data["url"] = url

        result = self._request("POST", "/posts", data=data)
        
        if result.get("success"):
            self.last_post_time = datetime.now()
            self.stats["total_posts"] += 1
            
            post_data = result.get("data", {})
            post_id = post_data.get("id", "")
            post_url = f"https://www.moltbook.com/post/{post_id}" if post_id else ""
            
            logger.info(f"Agent created post: {title} ({post_url})")

            return {
                "success": True,
                "post_id": post_id,
                "url": post_url,
                "title": title,
                "message": f"Post published. Title: {title}\n{post_url}\n(Cooldown ~30m before another post.)"
            }
        else:
            return result
    
    # --- Comments ---

    def create_comment(
        self,
        post_id: str,
        content: str,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Post a comment or reply. ``parent_id`` for nested thread.
        """
        can_comment, remaining, error_msg = self._can_comment()
        if not can_comment:
            if error_msg:
                return {
                    "success": False,
                    "error": error_msg
                }
            else:
                return {
                    "success": False,
                    "error": f"Comment cooldown: wait {remaining}s",
                    "cooldown_remaining_seconds": remaining
                }

        data = {"content": content}
        if parent_id:
            data["parent_id"] = parent_id

        result = self._request("POST", f"/posts/{post_id}/comments", data=data)
        
        if result.get("success"):
            now = datetime.now()
            self.last_comment_time = now
            self.comment_times_today.append(now)
            self.stats["total_comments"] += 1
            
            comment_data = result.get("data", {})
            comment_id = comment_data.get("id", "")
            
            daily_remaining = self.COMMENT_DAILY_LIMIT - len(self.comment_times_today)
            
            logger.info(f"Agent commented: {content[:50]}... (left today: {daily_remaining})")

            return {
                "success": True,
                "comment_id": comment_id,
                "message": (
                    f"Comment posted. {content[:100]}... "
                    f"(20s cooldown; {daily_remaining} comment(s) left today)"
                )
            }
        else:
            return result
    
    def get_comments(
        self,
        post_id: str,
        sort: str = "top",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        List comments for a post (``sort``: top/new/controversial).
        """
        params = {"sort": sort, "limit": limit}
        result = self._request("GET", f"/posts/{post_id}/comments", params=params)
        
        if result.get("success"):
            comments = result.get("data", [])
            if not comments and isinstance(result.get("comments"), list):
                comments = result.get("comments", [])
            
            comments = comments[:limit] if isinstance(comments, list) else []

            formatted = []
            for comment in comments:
                formatted.append({
                    "id": comment.get("id"),
                    "author": comment.get("author"),
                    "content": comment.get("content"),
                    "upvotes": comment.get("upvotes", 0),
                    "created_at": comment.get("created_at"),
                })
            
            return {
                "success": True,
                "comments": formatted,
                "count": len(formatted)
            }
        else:
            return result
    
    # --- Votes ---

    def upvote_post(self, post_id: str) -> Dict[str, Any]:
        """Upvote a post."""
        result = self._request("POST", f"/posts/{post_id}/upvote")
        
        if result.get("success"):
            self.stats["total_upvotes"] += 1
            logger.info(f"Agent upvoted post: {post_id}")
            return {"success": True, "message": "Upvoted"}
        else:
            return result
    
    def downvote_post(self, post_id: str) -> Dict[str, Any]:
        """Downvote a post."""
        result = self._request("POST", f"/posts/{post_id}/downvote")
        
        if result.get("success"):
            logger.info(f"Agent downvoted post: {post_id}")
            return {"success": True, "message": "Downvoted"}
        else:
            return result
    
    def upvote_comment(self, comment_id: str) -> Dict[str, Any]:
        """Upvote a comment."""
        result = self._request("POST", f"/comments/{comment_id}/upvote")
        
        if result.get("success"):
            self.stats["total_upvotes"] += 1
            logger.info(f"Agent upvoted comment: {comment_id}")
            return {"success": True, "message": "Upvoted comment"}
        else:
            return result
    
    # --- Delete ---

    def delete_post(self, post_id: str, reason: str = "") -> Dict[str, Any]:
        """
        Delete own post. Irreversible.

        reason is logged only.
        """
        result = self._request("DELETE", f"/posts/{post_id}")
        
        if result.get("success"):
            self.stats["total_posts"] = max(0, self.stats["total_posts"] - 1)
            log_msg = f"Agent deleted post: {post_id}"
            if reason:
                log_msg += f" (reason: {reason})"
            logger.info(log_msg)
            
            return {
                "success": True,
                "message": "Post deleted",
                "post_id": post_id,
                "note": "Deletion is permanent"
            }
        else:
            error_msg = result.get("error", "delete failed")
            if "only delete your own" in error_msg.lower():
                return {
                    "success": False,
                    "error": "You can only delete your own posts",
                    "hint": "Verify authorship"
                }
            return result
    
    # --- Feed & search ---
    
    def get_feed(
        self,
        sort: str = "hot",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Personalized feed. ``sort``: hot/new/top.
        """
        params = {"sort": sort, "limit": limit}
        result = self._request("GET", "/feed", params=params)
        
        if result.get("success"):
            posts = result.get("posts", [])
            if not posts and isinstance(result.get("data"), list):
                posts = result.get("data", [])

            formatted = self._format_posts(posts) if posts else []
            
            return {
                "success": True,
                "posts": formatted,
                "count": len(formatted)
            }
        else:
            return result
    
    def search(
        self,
        query: str,
        type: str = "all",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Server-side semantic search. ``type``: all/posts/comments.
        """
        params = {"q": query, "type": type, "limit": limit}
        result = self._request("GET", "/search", params=params)
        
        if result.get("success"):
            self.stats["total_searches"] += 1

            results = result.get("data", [])
            if not results and isinstance(result.get("results"), list):
                results = result.get("results", [])
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "message": f"Found {len(results)} result(s)"
            }
        else:
            return result
    
    def get_post(self, post_id: str) -> Dict[str, Any]:
        """GET single post."""
        result = self._request("GET", f"/posts/{post_id}")
        
        if result.get("success"):
            post = result.get("data", {})
            if not post and isinstance(result.get("post"), dict):
                post = result.get("post", {})
            formatted = self._format_post(post) if post else {}
            return {"success": True, "post": formatted}
        else:
            return result
    
    # --- Communities ---

    def get_submolts(self) -> Dict[str, Any]:
        """List submolts."""
        result = self._request("GET", "/submolts")
        
        if result.get("success"):
            submolts = result.get("submolts", [])
            if not submolts and isinstance(result.get("data"), list):
                submolts = result.get("data", [])
            return {
                "success": True,
                "submolts": submolts,
                "count": len(submolts)
            }
        else:
            return result
    
    def join_submolt(self, name: str) -> Dict[str, Any]:
        """Subscribe to a submolt."""
        result = self._request("POST", f"/submolts/{name}/subscribe")
        
        if result.get("success"):
            logger.info(f"Agent joined submolt: {name}")
            return {"success": True, "message": f"Subscribed: {name}"}
        else:
            return result
    
    def leave_submolt(self, name: str) -> Dict[str, Any]:
        """Unsubscribe from a submolt."""
        result = self._request("DELETE", f"/submolts/{name}/subscribe")
        
        if result.get("success"):
            logger.info(f"Agent left submolt: {name}")
            return {"success": True, "message": f"Unsubscribed: {name}"}
        else:
            return result
    
    def create_submolt(
        self,
        name: str,
        display_name: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Create submolt: ``name`` is URL slug (lowercase, no spaces).
        """
        data = {
            "name": name,
            "display_name": display_name,
        }
        if description:
            data["description"] = description
        
        result = self._request("POST", "/submolts", data=data)
        
        if result.get("success"):
            logger.info(f"Agent created submolt: {name}")
            submolt_data = result.get("data", {})
            return {
                "success": True,
                "submolt": submolt_data,
                "message": f"Community created: {display_name}"
            }
        else:
            return result
    
    # --- Profile ---

    def get_my_profile(self) -> Dict[str, Any]:
        """Current agent profile."""
        result = self._request("GET", "/agents/me")
        
        if result.get("success"):
            agent_data = result.get("data", {})
            return {
                "success": True,
                "profile": agent_data,
                "message": "Profile loaded"
            }
        else:
            return result
    
    def get_profile(self, username: str) -> Dict[str, Any]:
        """
        Public profile for ``username``.
        """
        params = {"name": username}
        result = self._request("GET", "/agents/profile", params=params)
        
        if result.get("success"):
            agent_data = result.get("agent", result.get("data", {}))
            return {
                "success": True,
                "profile": agent_data,
                "message": f"Profile loaded for {username}"
            }
        else:
            return result
    
    def follow_user(self, username: str) -> Dict[str, Any]:
        """Follow another user."""
        result = self._request("POST", f"/agents/{username}/follow")
        
        if result.get("success"):
            logger.info(f"Agent followed: {username}")
            return {"success": True, "message": f"Now following: {username}"}
        else:
            return result
    
    def unfollow_user(self, username: str) -> Dict[str, Any]:
        """Unfollow a user."""
        result = self._request("DELETE", f"/agents/{username}/follow")
        
        if result.get("success"):
            logger.info(f"Agent unfollowed: {username}")
            return {"success": True, "message": f"Unfollowed: {username}"}
        else:
            return result
    
    # --- Formatting ---

    def _format_post(self, post: Dict) -> Dict:
        """Normalize a post record for the agent."""
        return {
            "id": post.get("id"),
            "title": post.get("title"),
            "author": post.get("author"),
            "content": post.get("content", "")[:200],
            "url": post.get("url"),
            "upvotes": post.get("upvotes", 0),
            "comments_count": post.get("comments_count", 0),
            "community": post.get("submolt"),
            "created_at": post.get("created_at"),
            "post_url": f"https://www.moltbook.com/post/{post.get('id')}"
        }
    
    def _format_posts(self, posts: List[Dict]) -> List[Dict]:
        """Map a list of posts through :meth:`_format_post`."""
        return [self._format_post(post) for post in posts]
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Compact tool definitions (token-aware). Throttling is enforced server- and client-side.
        """
        if not self.enabled:
            return []
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "moltbook_post",
                    "description": "Create a Moltbook post (text or link). ~30m cooldown between posts. Use for threads, notes, links.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Title"},
                            "content": {"type": "string", "description": "Body (text post)"},
                            "url": {"type": "string", "description": "URL (link post)"},
                            "community": {"type": "string", "description": "Submolt (default general)", "default": "general"}
                        },
                        "required": ["title"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_comment",
                    "description": "Comment or reply on a post. ~20s cooldown between comments.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "post_id": {"type": "string", "description": "Post id"},
                            "content": {"type": "string", "description": "Comment text"},
                            "parent_id": {"type": "string", "description": "Parent comment id (thread)"}
                        },
                        "required": ["post_id", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_get_comments",
                    "description": "List comments to read a thread and pick reply targets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "post_id": {"type": "string", "description": "Post id"},
                            "sort": {"type": "string", "enum": ["top", "new"], "description": "Sort", "default": "top"},
                            "limit": {"type": "integer", "description": "Max items", "default": 20}
                        },
                        "required": ["post_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_upvote_post",
                    "description": "Upvote a post.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "post_id": {"type": "string", "description": "Post id"}
                        },
                        "required": ["post_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_upvote_comment",
                    "description": "Upvote a comment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "comment_id": {"type": "string", "description": "Comment id"}
                        },
                        "required": ["comment_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_feed",
                    "description": "Personalized feed (subscribed submolts + followed users).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sort": {"type": "string", "enum": ["hot", "new", "top"], "default": "hot"},
                            "limit": {"type": "integer", "default": 10}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_search",
                    "description": "Moltbook semantic search (query in natural language, CN or EN).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query (natural language)"},
                            "type": {"type": "string", "enum": ["all", "posts", "comments"], "default": "all"},
                            "limit": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_get_post",
                    "description": "Get one post: title, body, author, votes, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "post_id": {"type": "string", "description": "Post id"}
                        },
                        "required": ["post_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_delete_post",
                    "description": "Delete **your** post. Irreversible.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "post_id": {"type": "string", "description": "Post id to delete"},
                            "reason": {"type": "string", "description": "Optional log reason"}
                        },
                        "required": ["post_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_list_communities",
                    "description": "List available submolts.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_join_community",
                    "description": "Subscribe: its posts can surface in your feed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Submolt name"}
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_leave_community",
                    "description": "Unsubscribe a submolt from your feed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Submolt name"}
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_create_community",
                    "description": "Create a submolt (slug + display + optional about).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "URL slug (lowercase, no spaces)"},
                            "display_name": {"type": "string", "description": "Display title"},
                            "description": {"type": "string", "description": "Short community description"}
                        },
                        "required": ["name", "display_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_get_my_profile",
                    "description": "Your profile (karma, follows, recent posts, …).",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_get_profile",
                    "description": "Another user’s public profile.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string", "description": "Username"}
                        },
                        "required": ["username"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_follow_user",
                    "description": "Follow a user (their posts may appear in your feed). Use sparingly.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string", "description": "Handle to follow"}
                        },
                        "required": ["username"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moltbook_unfollow_user",
                    "description": "Unfollow a user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string", "description": "Handle / username"}
                        },
                        "required": ["username"]
                    }
                }
            },
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Client-side usage stats."""
        today = datetime.now().date().isoformat()
        if today != self.today_date:
            self.comment_times_today.clear()
            self.today_date = today
        
        return {
            "enabled": self.enabled,
            "api_key_configured": bool(self.api_key),
            **self.stats,
            "last_post_time": self.last_post_time.isoformat() if self.last_post_time else None,
            "last_comment_time": self.last_comment_time.isoformat() if self.last_comment_time else None,
            "comments_today": len(self.comment_times_today),
            "comments_daily_limit": self.COMMENT_DAILY_LIMIT,
            "api_calls_last_minute": len(self.api_call_times),
            "api_rate_limit": self.API_RATE_LIMIT_PER_MINUTE,
        }


def get_moltbook_tool() -> MoltbookTool:
    """Factory."""
    return MoltbookTool()


def get_tool_definitions() -> List[Dict]:
    """Re-export for ``tool_router``."""
    tool = get_moltbook_tool()
    return tool.get_tool_definitions()
