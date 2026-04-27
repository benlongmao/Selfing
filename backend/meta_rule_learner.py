#!/usr/bin/env python3
"""
Meta-rule learning: rules about *how* to learn or revise persona rules.

Examples (stored as plain text for later prompting):
- "When I face a novel situation, I should observe before acting."
- "When a rule keeps getting negative feedback, revise it instead of deleting it."
- "When several rules overlap, compress them into a more abstract rule."
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timezone
import logging
from backend.persona_store import PersonaStore, PersonaItem
from backend.embedder import get_embedder
from backend.scoring import ScoringSystem
from backend.config import config
from backend.llm_api import llm_completion

logger = logging.getLogger(__name__)

# Prefer ``settings.yaml`` / ``config``; ``META_RULE_MIN_EVIDENCE`` env remains a fallback default.
META_RULE_ENABLED = bool(config.get("system.meta_rule_enabled", True))
META_RULE_MIN_EVIDENCE = int(config.get("parameters.thresholds.meta_rule_min_evidence", os.environ.get("META_RULE_MIN_EVIDENCE", "3")) or 3)


class MetaRuleLearner:
    """Learn and retrieve meta-rules that steer future persona-rule edits."""
    
    def __init__(self, db_path: str = "data.db", persona_store: Optional[PersonaStore] = None):
        self.db_path = db_path
        self.persona_store = persona_store or PersonaStore(db_path)
        self.embedder = get_embedder()
        self.scoring = ScoringSystem(db_path, self.persona_store)
        self._ensure_meta_rules_table()
    
    def _ensure_meta_rules_table(self):
        """Create ``meta_rules`` (+ index) if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS meta_rules (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        category TEXT NOT NULL,  -- "learning", "modification", "compression", "selection"
                        embedding BLOB,
                        evidence_count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,  -- empirical success rate
                        created_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        status TEXT DEFAULT 'active'
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_meta_rules_category 
                    ON meta_rules(category, status)
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure meta_rules table: {e}")
    
    def learn_from_experience(
        self,
        learning_event: Dict
    ) -> Optional[PersonaItem]:
        """
        Turn a structured learning event into a stored meta-rule (or update an existing one).

        Args:
            learning_event: Dict with ``type``, ``context``, ``outcome``, ``feedback``.

        Returns:
            A ``PersonaItem`` representing the meta-rule, or ``None`` if learning is skipped.
        """
        if not META_RULE_ENABLED:
            return None
        
        event_type = learning_event.get("type")
        context = learning_event.get("context", {})
        outcome = learning_event.get("outcome", "unknown")
        feedback = learning_event.get("feedback", {})

        try:
            conv_turns = int(context.get("conversation_turns", 0) or 0)
        except Exception:
            conv_turns = 0
        if META_RULE_MIN_EVIDENCE > 0 and conv_turns > 0 and conv_turns < META_RULE_MIN_EVIDENCE:
            return None
        
        meta_rule_text = None
        category = None
        
        if event_type == "rule_creation":
            if outcome == "success" and feedback.get("positive", False):
                meta_rule_text = self._generate_creation_meta_rule(context, feedback)
                category = "learning"
        
        elif event_type == "rule_modification":
            if outcome == "success":
                meta_rule_text = self._generate_modification_meta_rule(context, feedback)
                category = "modification"
        
        elif event_type == "rule_compression":
            if outcome == "success":
                meta_rule_text = self._generate_compression_meta_rule(context, feedback)
                category = "compression"
        
        elif event_type == "rule_selection":
            if outcome == "success":
                meta_rule_text = self._generate_selection_meta_rule(context, feedback)
                category = "selection"
        
        if not meta_rule_text or not category:
            return None
        
        if len(meta_rule_text.strip()) < 10 or len(meta_rule_text.strip()) > 200:
            logger.debug(f"Meta-rule text length invalid: {len(meta_rule_text)}")
            return None
        
        existing = self._find_similar_meta_rule(meta_rule_text, category)
        if existing:
            existing.evidence_count += 1
            existing.last_seen_at = datetime.now(timezone.utc).isoformat()
            if outcome == "success":
                existing.success_rate = (existing.success_rate * (existing.evidence_count - 1) + 1.0) / existing.evidence_count
            else:
                existing.success_rate = (existing.success_rate * (existing.evidence_count - 1) + 0.0) / existing.evidence_count
            self._save_meta_rule(existing)
            logger.debug(f"Updated existing meta-rule: {existing.text[:50]}... (evidence={existing.evidence_count}, success_rate={existing.success_rate:.2f})")
            return existing
        
        meta_rule = PersonaItem(
            id=f"meta-{datetime.now(timezone.utc).timestamp()}",
            text=meta_rule_text,
            embedding=self.embedder.encode(meta_rule_text),
            score=0.5,
            importance=0.5,
            novelty=1.0,
            reliability=0.3,
            evidence_count=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            last_seen_at=datetime.now(timezone.utc).isoformat(),
            status="active"
        )
        
        meta_rule.category = category
        meta_rule.success_rate = 1.0 if outcome == "success" else 0.0
        
        self._save_meta_rule(meta_rule)
        
        logger.info(f"Learned new meta-rule: {meta_rule_text[:50]}... (category: {category})")
        
        return meta_rule
    
    def _generate_creation_meta_rule(self, context: Dict, feedback: Dict) -> Optional[str]:
        """Ask the LLM (via ``llm_api``) for a meta-rule after a successful rule creation."""
        try:

            prompt = f"""From this successful rule-creation episode, write ONE concise meta-rule
about *how I should learn or update persona rules in the future*.

Context:
- Conversation turns: {context.get('conversation_turns', 'N/A')}
- Rule type: {context.get('rule_type', 'N/A')}
- Feedback note: {feedback.get('comment', 'N/A')}

Requirements:
1. Capture the success pattern in plain language.
2. Output exactly one sentence (roughly 20–50 words).
3. Prefer first person: "When …, I should …" or "I should …".

Meta-rule:"""

            result = llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You distill learning episodes into a single actionable meta-rule "
                            "for an autonomous agent. Answer in English only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.3,
            )
            if not result["success"]:
                logger.debug("LLM meta-rule generation failed: %s", result.get("error"))
                return self._fallback_creation_meta_rule(context, feedback)
            content = result["content"]

            content = content.strip('"').strip("'").strip()
            if content.startswith("-"):
                content = content[1:].strip()
            
            logger.info(f"[META-RULE] LLM generated: {content[:50]}...")
            return content if len(content) >= 10 else None
            
        except Exception as e:
            logger.debug(f"LLM meta-rule generation failed: {e}")
            return self._fallback_creation_meta_rule(context, feedback)

    def _fallback_creation_meta_rule(self, context: Dict, feedback: Dict) -> str:
        """Deterministic template when the LLM path is unavailable."""
        added = feedback.get('added', 0)
        merged = feedback.get('merged', 0)
        turns = context.get('conversation_turns', 0)

        if merged > added:
            return "When I notice overlapping rules, I should merge instead of adding duplicates."
        elif turns > 5:
            return f"After {turns} dialogue turns, I should summarize feedback to refine how I learn."
        else:
            return "When a new values pattern appears in chat, I should capture it as a fresh rule."

    def _generate_modification_meta_rule(self, context: Dict, feedback: Dict) -> Optional[str]:
        """Template meta-rule after successful modifications."""
        if feedback.get("negative_count", 0) >= 3:
            return "When a rule accumulates negative feedback, I should revise it instead of deleting it."
        return "When a rule needs updating, I should keep prior versions for traceability."

    def _generate_compression_meta_rule(self, context: Dict, feedback: Dict) -> Optional[str]:
        """Template meta-rule after successful compressions."""
        return "When several rules exceed a similarity threshold, I should compress them into an abstract rule."

    def _generate_selection_meta_rule(self, context: Dict, feedback: Dict) -> Optional[str]:
        """Template meta-rule after successful selections."""
        return "When choosing among rules, I should prioritize refinement quality and importance."
    
    def _find_similar_meta_rule(self, text: str, category: str, similarity_threshold: float = 0.85) -> Optional[PersonaItem]:
        """
        Embedding-based dedupe: return an existing row if cosine similarity is high.

        Args:
            text: Candidate meta-rule body.
            category: Bucket (``learning``, ``modification``, …).
            similarity_threshold: Cosine similarity cutoff (default 0.85).

        Returns:
            Matching ``PersonaItem`` or ``None``.
        """
        text_emb = self.embedder.encode(text)
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, text, embedding, evidence_count, success_rate FROM meta_rules WHERE category=? AND status='active'",
                (category,)
            )
            rows = cur.fetchall()
        
        best_similarity = 0.0
        best_match = None
        
        for row in rows:
            if row[2]:
                existing_emb = np.frombuffer(row[2], dtype=np.float32)
                existing_emb = existing_emb / (np.linalg.norm(existing_emb) + 1e-8)
                similarity = np.dot(text_emb, existing_emb)

                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    item = PersonaItem(
                        id=row[0],
                        text=row[1],
                        embedding=existing_emb,
                        evidence_count=row[3],
                        status="active"
                    )
                    item.category = category
                    item.success_rate = row[4]
                    best_match = item
        
        if best_match:
            logger.debug(f"Found similar meta-rule (similarity={best_similarity:.3f}): {best_match.text[:50]}...")
        
        return best_match
    
    def _save_meta_rule(self, meta_rule: PersonaItem):
        """Upsert a ``meta_rules`` row."""
        emb_blob = meta_rule.embedding.astype(np.float32).tobytes() if meta_rule.embedding is not None else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO meta_rules (id, text, category, embedding, evidence_count, success_rate, created_at, last_seen_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     evidence_count=excluded.evidence_count,
                     success_rate=excluded.success_rate,
                     last_seen_at=excluded.last_seen_at
                """,
                (
                    meta_rule.id,
                    meta_rule.text,
                    getattr(meta_rule, "category", "learning"),
                    emb_blob,
                    meta_rule.evidence_count,
                    getattr(meta_rule, "success_rate", 0.0),
                    meta_rule.created_at,
                    meta_rule.last_seen_at,
                    meta_rule.status
                )
            )
            conn.commit()
    
    def get_meta_rules(self, category: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Return active meta-rules ordered by success rate and evidence."""
        with sqlite3.connect(self.db_path) as conn:
            if category:
                cur = conn.execute(
                    "SELECT id, text, category, evidence_count, success_rate, created_at FROM meta_rules WHERE category=? AND status='active' ORDER BY success_rate DESC, evidence_count DESC LIMIT ?",
                    (category, limit)
                )
            else:
                cur = conn.execute(
                    "SELECT id, text, category, evidence_count, success_rate, created_at FROM meta_rules WHERE status='active' ORDER BY success_rate DESC, evidence_count DESC LIMIT ?",
                    (limit,)
                )
            rows = cur.fetchall()
        
        meta_rules = []
        for row in rows:
            meta_rules.append({
                "id": row[0],
                "text": row[1],
                "category": row[2],
                "evidence_count": row[3],
                "success_rate": row[4],
                "created_at": row[5]
            })
        
        return meta_rules
    
    def apply_meta_rules(
        self,
        action_type: str,
        context: Dict
    ) -> Dict:
        """
        Surface the strongest meta-rule suggestion for a downstream action.

        Args:
            action_type: ``create`` | ``modify`` | ``compress`` | ``select``.
            context: Optional diagnostics (currently unused but kept for API symmetry).

        Returns:
            Dict with ``applied``, ``suggestion``, and supporting metadata.
        """
        category_map = {
            "create": "learning",
            "modify": "modification",
            "compress": "compression",
            "select": "selection"
        }
        
        category = category_map.get(action_type, "learning")
        meta_rules = self.get_meta_rules(category=category, limit=5)
        
        if not meta_rules:
            return {
                "applied": False,
                "suggestion": None,
                "meta_rules": []
            }
        
        min_success = float(config.get("parameters.thresholds.meta_rule_apply_min_success_rate", 0.3) or 0.3)
        min_evidence = int(config.get("parameters.thresholds.meta_rule_apply_min_evidence", 3) or 3)
        filtered_rules = [
            rule for rule in meta_rules
            if rule["success_rate"] >= min_success and rule["evidence_count"] >= min_evidence
        ]
        
        if not filtered_rules:
            return {
                "applied": False,
                "suggestion": None,
                "meta_rules": [],
                "reason": f"No qualified meta-rules (success_rate >= {min_success} and evidence_count >= {min_evidence})"
            }
        
        best_meta_rule = max(filtered_rules, key=lambda x: x["success_rate"])
        
        return {
            "applied": True,
            "suggestion": best_meta_rule["text"],
            "meta_rules": filtered_rules,
            "best_meta_rule": best_meta_rule
        }
    
    def evaluate_and_cleanup_meta_rules(self, min_success_rate: float = 0.2, min_evidence: int = 5) -> Dict:
        """
        Archive weak meta-rules based on success rate and evidence counts.

        Args:
            min_success_rate: Archive when success rate falls below this (with caveats below).
            min_evidence: Minimum samples before trusting low success rates.

        Returns:
            ``{"archived": int, "kept": int, "details": [...]}``.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, text, category, evidence_count, success_rate FROM meta_rules WHERE status='active'"
            )
            rows = cur.fetchall()
        
        archived_count = 0
        kept_count = 0
        details = []
        
        for row in rows:
            rule_id, text, category, evidence_count, success_rate = row
            
            should_archive = False
            reason = ""
            
            if success_rate < min_success_rate:
                if evidence_count >= min_evidence:
                    should_archive = True
                    reason = f"Low success rate ({success_rate:.2f} < {min_success_rate}) with sufficient evidence"
                elif evidence_count < 3:
                    should_archive = True
                    reason = f"Low success rate ({success_rate:.2f}) and insufficient evidence ({evidence_count})"

            if should_archive:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE meta_rules SET status='archived' WHERE id=?",
                        (rule_id,)
                    )
                    conn.commit()
                archived_count += 1
                details.append({
                    "id": rule_id,
                    "text": text[:50] + "...",
                    "category": category,
                    "action": "archived",
                    "reason": reason
                })
            else:
                kept_count += 1
        
        logger.info(f"Meta-rule cleanup: archived {archived_count}, kept {kept_count}")
        
        return {
            "archived": archived_count,
            "kept": kept_count,
            "details": details
        }

