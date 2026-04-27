"""
Agent learning tool: goals, knowledge capture, and recall.

Created 2026-02-07, v1.0.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Knowledge base imports (optional in some test environments)
try:
    from backend.knowledge_base import (
        get_knowledge_base,
        get_learning_manager,
        KnowledgeBase,
    )
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False


class LearningTool:
    """
    Active learning helpers: set goals, store distilled knowledge, search/update/delete,
    link entries, and complete goals with a summary.
    """

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.knowledge_base = None
        self.learning_manager = None

        if KNOWLEDGE_BASE_AVAILABLE:
            self.knowledge_base = get_knowledge_base(db_path)
            self.learning_manager = get_learning_manager(db_path)
            logger.info("LearningTool: Knowledge base connected")

    def get_tool_definitions(self) -> List[Dict]:
        """Tool definitions for the tool router."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_learning_goal",
                    "description": "Set a learning goal when you want to study a topic in depth.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to learn",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this topic matters",
                            },
                        },
                        "required": ["topic", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_learned_knowledge",
                    "description": (
                        "Save **distilled** knowledge to the knowledge base. "
                        "**Rules:** "
                        "(1) Title must be a short summary phrase (about 10–30 characters or words), not a raw user-quote fragment; "
                        "(2) Body must be your synthesized insight, not a verbatim copy of chat or user text; "
                        "(3) Do not store pure questions, chit-chat, imperative instructions, or drill problems as “knowledge”; "
                        "(4) Keep body concise (roughly within ~200 Chinese characters or a short English paragraph)—overlong text means it was not distilled."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short summary title (e.g. “Task management: fewer, sharper goals”). Do not prefix with “Definition:”.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Distilled knowledge (about 50–200 words or a tight Chinese paragraph). Paraphrase; do not paste raw dialogue.",
                            },
                            "source": {
                                "type": "string",
                                "enum": ["web_search", "user_teach", "self_discovery", "reading", "experience", "reflection"],
                                "description": "Origin: web_search, user_teach, self_discovery, reading, experience, reflection",
                            },
                            "category": {
                                "type": "string",
                                "enum": ["技术", "科学", "常识", "个人经验", "项目相关", "用户偏好", "世界知识", "哲学"],
                                "description": (
                                    "Category literal expected by the knowledge store (must match enum exactly). "
                                    "Rough mapping: tech, science, general knowledge, personal experience, project, "
                                    "user preference, world knowledge, philosophy."
                                ),
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence in [0, 1]",
                            },
                        },
                        "required": ["title", "content", "source", "category"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_my_knowledge",
                    "description": "Search your knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional category filter (Chinese category token)",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Max results (default 5)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_my_knowledge",
                    "description": "Update or correct an existing knowledge row (wrong body, bad title, etc.).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "knowledge_id": {
                                "type": "string",
                                "description": "Knowledge row id",
                            },
                            "new_title": {
                                "type": "string",
                                "description": "Optional new title",
                            },
                            "new_content": {
                                "type": "string",
                                "description": "New body text",
                            },
                            "new_confidence": {
                                "type": "number",
                                "description": "Optional new confidence",
                            },
                            "add_tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tags to append",
                            },
                        },
                        "required": ["knowledge_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_my_knowledge",
                    "description": (
                        "Permanently delete one knowledge row. You need the exact knowledge_id "
                        "(use search_my_knowledge or get_knowledge_stats first). Deletion cannot be undone."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "knowledge_id": {
                                "type": "string",
                                "description": "Knowledge id to delete",
                            },
                        },
                        "required": ["knowledge_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "link_knowledge",
                    "description": "Create a link between two knowledge rows.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "Source knowledge id",
                            },
                            "target_id": {
                                "type": "string",
                                "description": "Target knowledge id",
                            },
                            "relation_type": {
                                "type": "string",
                                "enum": ["related", "prerequisite", "extends", "contradicts"],
                                "description": "Relation type",
                            },
                        },
                        "required": ["source_id", "target_id", "relation_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_my_learning_goals",
                    "description": "List active learning goals.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_knowledge_stats",
                    "description": "Knowledge base statistics.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_learning_goal",
                    "description": "Mark a learning goal complete and attach a written summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal_id": {
                                "type": "string",
                                "description": "Goal id",
                            },
                            "summary": {
                                "type": "string",
                                "description": "Completion summary",
                            },
                        },
                        "required": ["goal_id", "summary"],
                    },
                },
            },
        ]

    def route_tool_call(self, func_name: str, args: Dict, session_id: str = "selfing-session") -> Dict:
        """Dispatch a tool call by name."""
        if func_name == "set_learning_goal":
            return self.set_learning_goal(
                topic=args.get("topic"),
                reason=args.get("reason"),
                session_id=session_id,
            )
        if func_name == "add_learned_knowledge":
            return self.add_learned_knowledge(
                title=args.get("title"),
                content=args.get("content"),
                source=args.get("source"),
                category=args.get("category"),
                tags=args.get("tags"),
                confidence=args.get("confidence", 0.5),
                session_id=session_id,
            )
        if func_name == "search_my_knowledge":
            return self.search_my_knowledge(
                query=args.get("query"),
                category=args.get("category"),
                top_k=args.get("top_k", 5),
                session_id=session_id,
            )
        if func_name == "update_my_knowledge":
            return self.update_my_knowledge(
                knowledge_id=args.get("knowledge_id"),
                new_content=args.get("new_content"),
                new_confidence=args.get("new_confidence"),
                add_tags=args.get("add_tags"),
                new_title=args.get("new_title"),
                session_id=session_id,
            )
        if func_name == "delete_my_knowledge":
            return self.delete_my_knowledge(
                knowledge_id=args.get("knowledge_id"),
                session_id=session_id,
            )
        if func_name == "link_knowledge":
            return self.link_knowledge(
                source_id=args.get("source_id"),
                target_id=args.get("target_id"),
                relation_type=args.get("relation_type", "related"),
            )
        if func_name == "get_my_learning_goals":
            return self.get_my_learning_goals(session_id=session_id)
        if func_name == "get_knowledge_stats":
            return self.get_knowledge_stats(session_id=session_id)
        if func_name == "complete_learning_goal":
            return self.complete_learning_goal(
                goal_id=args.get("goal_id"),
                summary=args.get("summary"),
                session_id=session_id,
            )
        return {"error": f"Unknown function: {func_name}"}

    def set_learning_goal(
        self,
        topic: str,
        reason: str,
        session_id: str = "selfing-session",
    ) -> Dict[str, Any]:
        """Create a learning goal."""
        if not self.learning_manager:
            return {"success": False, "error": "Learning manager not available"}

        result = self.learning_manager.create_goal(topic, reason, session_id)

        if result.get("success"):
            self.learning_manager.log_action(
                goal_id=result["goal_id"],
                action="create_goal",
                content=f"Learning goal set: {topic}",
                result="success",
                session_id=session_id,
            )

        return result

    def add_learned_knowledge(
        self,
        title: str,
        content: str,
        source: str,
        category: str,
        tags: List[str] = None,
        confidence: float = 0.5,
        session_id: str = "selfing-session",
    ) -> Dict[str, Any]:
        """Insert distilled knowledge."""
        if not self.knowledge_base:
            return {"success": False, "error": "Knowledge base not available"}

        return self.knowledge_base.add_knowledge(
            title=title,
            content=content,
            source=source,
            category=category,
            tags=tags,
            confidence=confidence,
            session_id=session_id,
        )

    def search_my_knowledge(
        self,
        query: str,
        category: str = None,
        top_k: int = 5,
        session_id: str = "selfing-session",
    ) -> Dict[str, Any]:
        """Semantic / keyword search over stored knowledge."""
        if not self.knowledge_base:
            return {"success": False, "error": "Knowledge base not available"}

        results = self.knowledge_base.search_knowledge(
            query=query,
            top_k=top_k,
            category=category,
            session_id=session_id,
        )

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }

    def update_my_knowledge(
        self,
        knowledge_id: str,
        new_content: str = None,
        new_confidence: float = None,
        add_tags: List[str] = None,
        new_title: str = None,
        session_id: str = "selfing-session",
    ) -> Dict[str, Any]:
        """Patch an existing knowledge row."""
        if not self.knowledge_base:
            return {"success": False, "error": "Knowledge base not available"}

        return self.knowledge_base.update_knowledge(
            knowledge_id=knowledge_id,
            new_content=new_content,
            new_confidence=new_confidence,
            add_tags=add_tags,
            new_title=new_title,
            session_id=session_id,
        )

    def delete_my_knowledge(
        self,
        knowledge_id: str,
        session_id: str = "selfing-session",
    ) -> Dict[str, Any]:
        """Delete one knowledge row (scoped to the session rules in KnowledgeBase)."""
        if not self.knowledge_base:
            return {"success": False, "error": "Knowledge base not available"}
        if not knowledge_id or not str(knowledge_id).strip():
            return {"success": False, "error": "knowledge_id is required"}
        return self.knowledge_base.delete_knowledge(
            str(knowledge_id).strip(),
            session_id=session_id,
        )

    def link_knowledge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "related",
    ) -> Dict[str, Any]:
        """Link two knowledge rows."""
        if not self.knowledge_base:
            return {"success": False, "error": "Knowledge base not available"}

        return self.knowledge_base.link_knowledge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
        )

    def get_my_learning_goals(self, session_id: str = "selfing-session") -> Dict[str, Any]:
        """Return active goals."""
        if not self.learning_manager:
            return {"success": False, "error": "Learning manager not available"}

        goals = self.learning_manager.get_active_goals(session_id)

        return {
            "success": True,
            "active_goals": goals,
            "count": len(goals),
        }

    def get_knowledge_stats(self, session_id: str = "selfing-session") -> Dict[str, Any]:
        """Aggregate knowledge stats."""
        if not self.knowledge_base:
            return {"success": False, "error": "Knowledge base not available"}

        return self.knowledge_base.get_knowledge_stats(session_id)

    def complete_learning_goal(
        self,
        goal_id: str,
        summary: str,
        session_id: str = "selfing-session",
    ) -> Dict[str, Any]:
        """Mark a goal finished and persist a summary row."""
        if not self.learning_manager:
            return {"success": False, "error": "Learning manager not available"}

        result = self.learning_manager.update_progress(goal_id, 1.0)

        if result.get("success"):
            self.learning_manager.log_action(
                goal_id=goal_id,
                action="complete",
                content=summary,
                result="completed",
                session_id=session_id,
            )

            if self.knowledge_base:
                self.knowledge_base.add_knowledge(
                    title=f"Learning summary: {goal_id}",
                    content=summary,
                    source="reflection",
                    category="个人经验",
                    tags=["learning_summary"],
                    confidence=0.8,
                    session_id=session_id,
                )

        return {
            "success": True,
            "goal_id": goal_id,
            "message": "Learning goal marked complete",
            "summary_saved": True,
        }


_learning_tool: Optional[LearningTool] = None


def get_learning_tool(db_path: str = "data.db") -> LearningTool:
    """Singleton accessor."""
    global _learning_tool
    if _learning_tool is None:
        _learning_tool = LearningTool(db_path)
    return _learning_tool
