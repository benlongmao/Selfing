#!/usr/bin/env python3
"""
Rule compression: merge several concrete persona rules into fewer, more abstract ones.

Aligns with “keep only the most essential, distilled rules” storage policy.
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import logging
from backend.config import config
from backend.persona_store import PersonaStore, PersonaItem
from backend.embedder import get_embedder
from backend.scoring import ScoringSystem
from backend.llm_api import llm_completion

logger = logging.getLogger(__name__)

# Compression tuning
COMPRESSION_SIMILARITY_THRESHOLD = float(os.environ.get("COMPRESSION_SIMILARITY_THRESHOLD", "0.85"))
COMPRESSION_MIN_GROUP_SIZE = int(os.environ.get("COMPRESSION_MIN_GROUP_SIZE", "3"))
COMPRESSION_ENABLED = os.environ.get("COMPRESSION_ENABLED", "true").lower() == "true"

class RuleCompressor:
    """Cluster similar persona rules and mint higher-level abstractions."""

    def __init__(self, db_path: str = "data.db", persona_store: Optional[PersonaStore] = None, meta_rule_learner=None):
        self.db_path = db_path
        self.persona_store = persona_store or PersonaStore(db_path)
        self.embedder = get_embedder()
        self.scoring = ScoringSystem(db_path, self.persona_store)
        self.meta_rule_learner = meta_rule_learner  # optional MetaRuleLearner hook

    def _group_contains_l0(self, group: List[PersonaItem]) -> bool:
        """Return True when a compression group contains constitutional L0 rows."""
        return any(int(getattr(item, "locked", 0) or 0) == 1 for item in group)

    def _archive_old_rule_after_compression(self, old_item: PersonaItem, new_item: PersonaItem) -> bool:
        """
        Archive a source rule only when this cannot demote L0 content.

        L0 rules may be consolidated, but their replacement must remain active L0
        (``is_core=1`` and ``locked=1``), so constitutional content is still injected
        every round after compression.
        """
        old_is_l0 = int(getattr(old_item, "locked", 0) or 0) == 1
        new_is_l0 = (
            int(getattr(new_item, "locked", 0) or 0) == 1
            and int(getattr(new_item, "is_core", 0) or 0) == 1
        )
        if old_is_l0 and not new_is_l0:
            logger.warning(
                "[L0-PROTECT] Skip archiving locked rule %s because replacement %s is not L0",
                old_item.id,
                new_item.id,
            )
            return False

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE persona_items SET status='archived', last_seen_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), old_item.id)
            )
            conn.commit()
        return True
    
    def compress_rules(
        self,
        max_items: int = 100,
        similarity_threshold: float = None
    ) -> Dict:
        """
        Compress: cluster similar active rules and mint abstract replacements.

        Args:
            max_items: Soft cap for active persona memory (used for limit math).
            similarity_threshold: Cosine threshold override (defaults to env/config).

        Returns:
            ``{"compressed": int, "merged_groups": [...], "new_rules": [...]}``.
        """
        if not COMPRESSION_ENABLED:
            logger.info("Rule compression is disabled")
            return {"compressed": 0, "merged_groups": [], "new_rules": []}
        
        threshold = similarity_threshold or COMPRESSION_SIMILARITY_THRESHOLD
        
        # Capacity note: we no longer bail when "full" — compression is how we recover space.
        current_count = len(self.persona_store.get_all_active(limit=max_items + 10))
        # if current_count >= max_items:
        #     logger.info(f"Rule count already at limit ({current_count} >= {max_items}), skipping compression")
        #     return {"compressed": 0, "merged_groups": [], "new_rules": []}
        
        # 1) Load active rules
        items = self.persona_store.get_all_active(limit=max_items * 2)
        if len(items) < COMPRESSION_MIN_GROUP_SIZE:
            logger.info(f"Not enough rules to compress: {len(items)} < {COMPRESSION_MIN_GROUP_SIZE}")
            return {"compressed": 0, "merged_groups": [], "new_rules": []}
        
        # 2) Cluster
        groups = self._find_similar_groups(items, threshold)
        
        # 3) Abstract each eligible cluster
        merged_groups = []
        new_rules = []
        compressed_count = 0
        
        # Track projected row growth vs max_items
        estimated_new_rules = 0
        
        for group in groups:
            if len(group) < COMPRESSION_MIN_GROUP_SIZE:
                continue
            
            # Skip if this merge would exceed max_items budget
            if current_count - len(group) + estimated_new_rules + 1 > max_items:
                logger.debug(f"Skipping compression group: would exceed limit ({current_count} - {len(group)} + {estimated_new_rules} + 1 > {max_items})")
                continue
            
            # Generate abstraction
            abstract_rule = self._generate_abstract_rule(group)
            if not abstract_rule:
                continue
            
            # Dedup near-identical text
            if self._rule_exists(abstract_rule, items):
                logger.debug(f"Abstract rule already exists: {abstract_rule[:50]}...")
                continue
            
            estimated_new_rules += 1
            
            # Score the new rule
            abstract_emb = self.embedder.encode(abstract_rule)
            scores = self.scoring.score_candidate(abstract_rule, abstract_emb, evidence_count=len(group))
            
            # Pool evidence / importance from the cluster
            total_evidence = sum(item.evidence_count for item in group)
            max_importance = max(item.importance for item in group)
            scores["importance"] = max_importance
            scores["reliability"] = min(1.0, total_evidence / 10.0)
            scores["total_score"] = (
                0.35 * scores["importance"] +
                0.25 * scores["novelty"] +
                0.15 * scores["reliability"] +
                0.25 * scores.get("coreness", 0.5)
            )
            
            group_has_l0 = self._group_contains_l0(group)
            source = {
                "type": "l0_compression" if group_has_l0 else "rule_compression",
                "source_rule_ids": [item.id for item in group],
                "preserves_l0_injection": bool(group_has_l0),
            }

            # Mint PersonaItem. If any source row is L0, the abstraction must stay L0.
            new_item = PersonaItem(
                id=f"compressed-{datetime.now(timezone.utc).timestamp()}",
                text=abstract_rule,
                embedding=abstract_emb,
                score=max(2.0, scores["total_score"]) if group_has_l0 else scores["total_score"],
                importance=scores["importance"],
                novelty=scores["novelty"],
                reliability=scores["reliability"],
                evidence_count=total_evidence,
                created_at=datetime.now(timezone.utc).isoformat(),
                last_seen_at=datetime.now(timezone.utc).isoformat(),
                status="active",
                is_core=1 if group_has_l0 else 0,
                core_version=max((getattr(item, "core_version", 0) or 0) for item in group) if group_has_l0 else 0,
                locked=1 if group_has_l0 else 0,
                source=source,
            )
            
            # Persist
            self.persona_store.add_or_update(new_item, update_embedding=True)
            
            # Archive superseded rows (history retained)
            for old_item in group:
                archived = self._archive_old_rule_after_compression(old_item, new_item)
                
                # Audit event
                if archived:
                    self._record_compression_event(old_item.id, new_item.id, abstract_rule)
            
            merged_groups.append({
                "group_size": len(group),
                "old_rules": [item.text for item in group],
                "new_rule": abstract_rule,
                "new_rule_id": new_item.id
            })
            new_rules.append(abstract_rule)
            compressed_count += len(group)
        
        logger.info(f"Rule compression completed: {compressed_count} rules compressed into {len(new_rules)} abstract rules")
        
        return {
            "compressed": compressed_count,
            "merged_groups": merged_groups,
            "new_rules": new_rules
        }
    
    def _find_similar_groups(self, items: List[PersonaItem], threshold: float) -> List[List[PersonaItem]]:
        """
        Greedy clustering by embedding cosine similarity (enhanced pass).

        Slightly looser than ``threshold`` so borderline siblings can be merged,
        then ``_generate_abstract_rule`` does stricter inductive synthesis.
        """
        groups = []
        used = set()

        clustering_threshold = max(0.65, threshold - 0.1)
        
        for i, item1 in enumerate(items):
            if i in used or item1.embedding is None:
                continue
            
            group = [item1]
            used.add(i)
            
            for j, item2 in enumerate(items[i+1:], start=i+1):
                if j in used or item2.embedding is None:
                    continue
                
                # Cosine similarity
                similarity = np.dot(item1.embedding, item2.embedding) / (
                    np.linalg.norm(item1.embedding) * np.linalg.norm(item2.embedding) + 1e-8
                )
                
                # Strategy 1: high embedding similarity
                if similarity >= clustering_threshold:
                    group.append(item2)
                    used.add(j)
                    continue
                    
                # Strategy 2 (implicit): borderline pairs may still co-cluster via the relaxed threshold.

            if len(group) >= COMPRESSION_MIN_GROUP_SIZE:
                groups.append(group)
        
        return groups
    
    def _generate_abstract_rule(self, group: List[PersonaItem]) -> Optional[str]:
        """
        Produce one abstract rule for a cluster.

        Prefer LLM inductive summarization; fall back to lightweight keyword heuristics.
        """
        if not group:
            return None
            
        texts = [item.text for item in group]
        
        try:
            abstract_rule = self._llm_generate_inductive_rule(texts)
            if abstract_rule:
                return abstract_rule
        except Exception as e:
            logger.debug(f"LLM inductive generation failed: {e}")

        # Fallback: token overlap (Chinese + ASCII punctuation as separators)
        words: List[str] = []
        for text in texts:
            t = text.replace("，", " ").replace("。", " ").replace("、", " ")
            t = t.replace(",", " ").replace(".", " ")
            words.extend(t.split())
        
        # Frequent tokens shared across the cluster
        from collections import Counter
        word_freq = Counter(words)
        common_words = [word for word, freq in word_freq.most_common(5) if freq >= len(texts) * 0.5]
        
        # Single text => nothing to compress
        if len(texts) == 1:
            return None
        
        # Second LLM attempt (alias)
        abstract_rule = self._llm_generate_abstract(texts)
        if abstract_rule:
            return abstract_rule
        
        base_rule = texts[0]  # noqa: F841 — reserved for richer merge heuristics

        if len(texts) <= 3:
            return min(texts, key=len)

        # Keyword overlap fallback
        all_words = []
        for text in texts:
            t = text.replace("，", " ").replace("。", " ").replace("、", " ")
            t = t.replace(",", " ").replace(".", " ")
            words = t.split()
            all_words.extend([w for w in words if len(w) > 1])

        # Shared keywords
        from collections import Counter
        word_freq = Counter(all_words)
        common_words = [word for word, freq in word_freq.most_common(5) if freq >= len(texts) * 0.6]
        
        if common_words:
            # Bilingual stems — persona rules are first-person in both ZH/EN installs
            structure_words = [
                "我", "应该", "必须", "优先", "避免", "确保", "保护",
                "I", "should", "must", "prioritize", "avoid", "ensure", "protect",
            ]
            found_structure = [w for w in structure_words if any(w in text for text in texts)]

            if found_structure:
                abstract = found_structure[0] + " " + " ".join(common_words[:3])
                if 10 <= len(abstract) <= 50:
                    return abstract

        # Prefer short text that still hits shared keywords
        scored_texts = []
        for text in texts:
            keyword_count = sum(1 for kw in common_words if kw in text)
            score = keyword_count - len(text) / 100.0
            scored_texts.append((score, text))
        
        if scored_texts:
            return max(scored_texts, key=lambda x: x[0])[1]
        
        return min(texts, key=len)
    
    def _llm_generate_inductive_rule(self, texts: List[str]) -> Optional[str]:
        """
        Ask the LLM for a single inductive principle (not a concatenation).

        Example pattern: several specific preferences → one covering principle, still first-person.
        """

        prompt = f"""Perform **logical induction** on the concrete rules below and output **one** higher-level principle.

Rules:
{chr(10).join(f"- {text}" for text in texts[:10])}

Requirements:
1. Surface the shared pattern or motive behind the rules.
2. If they are facets of one class of behavior, unify them.
3. If there is an implicit causal structure, state it as one principle.
4. The new rule MUST start in the first person (use **I** in English, or **我** in Chinese—match the dominant language of the inputs).
5. Keep it between 20 and 50 characters **or** roughly 12–30 English words (stay concise).
6. Output **only** the induced rule—no labels such as “Induced rule:”, no markdown fences, no explanation.

Induced rule:"""

        try:
            result = llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You specialize in careful logical induction and distilling stable "
                            "first-person principles from noisy rule lists."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.4,
            )
            if not result["success"]:
                logger.warning("Inductive rule generation failed: %s", result.get("error"))
                return None
            content = result["content"]
            content = content.strip('"').strip("'").strip()
            if content.startswith("-"): content = content[1:].strip()
            if len(content) < 5: return None
            return content
        except Exception as e:
            logger.warning(f"Inductive rule generation failed: {e}")
            return None

    def _llm_generate_abstract(self, texts: List[str]) -> Optional[str]:
        """Deprecated alias — forwards to ``_llm_generate_inductive_rule``."""
        return self._llm_generate_inductive_rule(texts)
    
    def _rule_exists(self, rule_text: str, existing_items: List[PersonaItem]) -> bool:
        """True if an active item is already extremely similar in embedding space."""
        rule_emb = self.embedder.encode(rule_text)
        for item in existing_items:
            if item.embedding is None:
                continue
            similarity = np.dot(rule_emb, item.embedding) / (
                np.linalg.norm(rule_emb) * np.linalg.norm(item.embedding) + 1e-8
            )
            if similarity >= 0.90:
                return True
        return False
    
    def _record_compression_event(self, old_rule_id: str, new_rule_id: str, abstract_rule: str):
        """Append a persona_events audit row."""
        import uuid
        event_id = str(uuid.uuid4())
        detail = {
            "type": "rule_compression",
            "old_rule_id": old_rule_id,
            "new_rule_id": new_rule_id,
            "abstract_rule": abstract_rule,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO persona_events (id, ts, type, persona_id, detail) VALUES (?, ?, ?, ?, ?)",
                (event_id, datetime.now(timezone.utc).isoformat(), "rule_compression", old_rule_id, json.dumps(detail, ensure_ascii=False))
            )
            conn.commit()

    def _process_episodic_memory(self, session_id: str, sensory_buffer=None) -> Dict:
        """
        Episodic → semantic metabolism: mine durable first-person rules from dialogue.

        Prefers ``sensory_buffer`` text when provided, else falls back to ``chat_turns``.
        """
        if sensory_buffer:
            try:
                buffer_text = sensory_buffer.get_all_for_metabolism(session_id)
                if buffer_text:
                    new_rules = self._llm_extract_rules(buffer_text)
                    if new_rules:
                        added_count = 0
                        for rule_text in new_rules:
                            if self._rule_exists(rule_text, self.persona_store.get_all_active(limit=500)):
                                continue
                            emb = self.embedder.encode(rule_text)
                            new_item = PersonaItem(
                                id=f"learned-{datetime.now(timezone.utc).timestamp()}-{added_count}",
                                text=rule_text,
                                embedding=emb,
                                score=0.6,
                                importance=0.5,
                                novelty=0.6,
                                reliability=0.5,
                                evidence_count=1,
                                created_at=datetime.now(timezone.utc).isoformat(),
                                last_seen_at=datetime.now(timezone.utc).isoformat(),
                                status="active",
                                source={"type": "sensory_buffer_extraction", "session_id": session_id}
                            )
                            self.persona_store.add_or_update(new_item, update_embedding=True)
                            added_count += 1
                            logger.info(f"Extracted from sensory buffer: {rule_text}")
                        
                        if added_count > 0:
                            return {
                                "extracted": added_count,
                                "processed_turns": len(sensory_buffer.get_recent_inputs(session_id)),
                                "source": "sensory_buffer"
                            }
            except Exception as e:
                logger.warning(f"Failed to process sensory buffer: {e}, falling back to chat_turns")
        
        # Fallback: unread chat_turns rows
        # [P1] PRAGMA-guard ``metabolized`` column to avoid silent OperationalError
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("PRAGMA table_info(chat_turns)")
            cols = [r[1] for r in cur.fetchall()]
        if "metabolized" not in cols:
            logger.warning(
                "Column 'metabolized' not found in chat_turns. "
                "Run schema migrations (app.run_schema_migrations) to add it. Skipping episodic processing."
            )
            return {"status": "skipped", "reason": "schema mismatch"}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT id, user_input, assistant_output, created_at 
                   FROM chat_turns 
                   WHERE session_id=? AND (metabolized IS NULL OR metabolized=0)
                   ORDER BY created_at ASC LIMIT 50""",
                (session_id,)
            )
            rows = cursor.fetchall()
            
        if not rows:
            return {"extracted": 0, "processed_turns": 0}
            
        # Need a minimal span of turns before paying LLM cost
        if len(rows) < 3:
            return {"extracted": 0, "processed_turns": 0, "message": "not enough history"}

        # Build transcript
        history_text = ""
        turn_ids = []
        for row in rows:
            history_text += f"User: {row['user_input']}\nAI: {row['assistant_output']}\n\n"
            turn_ids.append(row['id'])

        # LLM extraction
        new_rules = self._llm_extract_rules(history_text)
        
        added_count = 0
        for rule_text in new_rules:
            if self._rule_exists(rule_text, self.persona_store.get_all_active(limit=500)):
                continue

            emb = self.embedder.encode(rule_text)

            new_item = PersonaItem(
                id=f"learned-{datetime.now(timezone.utc).timestamp()}-{added_count}",
                text=rule_text,
                embedding=emb,
                score=0.6,
                importance=0.5,
                novelty=0.6,
                reliability=0.5,
                evidence_count=1,
                created_at=datetime.now(timezone.utc).isoformat(),
                last_seen_at=datetime.now(timezone.utc).isoformat(),
                status="active",
                source={"type": "episodic_extraction", "session_id": session_id}
            )
            self.persona_store.add_or_update(new_item, update_embedding=True)
            added_count += 1
            logger.info(f"Extracted semantic memory: {rule_text}")

        # Mark turns metabolized (raw rows kept for audit; semantic signal lifted)
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' for _ in turn_ids)
            conn.execute(
                f"UPDATE chat_turns SET metabolized=1 WHERE id IN ({placeholders})",
                turn_ids
            )
            conn.commit()
            
        return {
            "extracted": added_count,
            "processed_turns": len(turn_ids)
        }

    def _llm_extract_rules(self, history_text: str) -> List[str]:
        """Extract 1–3 durable first-person rules via ``llm_completion``."""
        _NO_RULE_MARKERS = frozenset(
            {
                "无",
                "没有",
                "none",
                "nothing",
                "n/a",
                "na",
                "nil",
                "no rules",
            }
        )
        try:
            
            prompt = f"""Read the dialogue below and extract **1 to 3** long-lived behavioral rules or memories about **me** (the assistant).

Requirements:
1. Ignore small talk unless it encodes a durable preference.
2. Prioritize user preferences, explicit commitments I made, or stable personality signals I displayed.
3. Each line must start in the first person (**I …** in English, or **我…** in Chinese—match the dialogue language).
4. Keep each rule under ~30 Chinese characters **or** ~24 English words.
5. If nothing durable exists, reply with a single line: **none**.

Dialogue:
{history_text}

Rules (one per line):"""

            result = llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You consolidate episodic chat into a tiny set of first-person rules "
                            "suitable for a persistent persona store."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            if not result["success"]:
                logger.error("Failed to extract rules: %s", result.get("error"))
                return []
            content = result["content"]
            
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            rules = []
            for line in lines:
                clean_line = line.lstrip('1234567890.- ').strip()
                low = clean_line.lower()
                if len(clean_line) <= 5:
                    continue
                if clean_line in _NO_RULE_MARKERS or low in _NO_RULE_MARKERS:
                    continue
                if "我" in clean_line or low.startswith("i ") or low.startswith("i'm") or low.startswith("i am "):
                    rules.append(clean_line)
            
            return rules[:3]
        except Exception as e:
            logger.error(f"Failed to extract rules from history: {e}")
            return []

    def sleep_consolidation(self, session_id: Optional[str] = None, sensory_buffer=None) -> Dict:
        """
        Sleep-cycle consolidation (memory metabolism).

        1. Episodic → semantic extraction (dialogue / sensory buffer).
        2. Merge low-scoring persona shards into higher-level rules.
        3. Archive irrecoverable low-value clutter (controlled forgetting).

        Args:
            session_id: When set, run episodic metabolism for this session.
            sensory_buffer: Optional buffer instance read before ``chat_turns``.
        """
        logger.info("Starting Sleep Consolidation (Memory Metabolism)...")
        results = {}

        # 1) Episodic metabolism
        if session_id:
            try:
                episodic_res = self._process_episodic_memory(session_id, sensory_buffer=sensory_buffer)
                results["episodic_metabolism"] = episodic_res
                
                # Trim stale sensory buffer rows after successful extraction
                if sensory_buffer and episodic_res.get("extracted", 0) > 0:
                    sensory_buffer.clear_old_inputs(session_id, keep_last_n=5)
            except Exception as e:
                logger.error(f"Episodic metabolism failed: {e}")
                results["episodic_error"] = str(e)
        
        # 2) Low-score persona items
        low_score_items = self.persona_store.get_low_score_items(threshold=0.4, limit=20)
        if len(low_score_items) < 2:
            # Still report success if episodic pass already ran
            if session_id and "episodic_metabolism" in results:
                results["status"] = "completed"
                results["consolidated"] = 0
                results["excreted"] = 0
                return results
            return {"status": "skipped", "reason": "not enough low score items"}
            
        groups = self._find_similar_groups(low_score_items, threshold=0.75)
        
        consolidated_count = 0
        excreted_count = 0
        
        for group in groups:
            if len(group) < 2: continue
            
            abstract_rule = self._generate_abstract_rule(group)
            if abstract_rule:
                abstract_emb = self.embedder.encode(abstract_rule)

                max_novelty = max(item.novelty for item in group)
                new_score = 0.5 + (max_novelty * 0.3)
                
                new_item = PersonaItem(
                    id=f"consolidated-{datetime.now(timezone.utc).timestamp()}",
                    text=abstract_rule,
                    embedding=abstract_emb,
                    score=new_score,
                    importance=0.6,
                    novelty=max_novelty,
                    reliability=0.8,
                    evidence_count=sum(i.evidence_count for i in group),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    last_seen_at=datetime.now(timezone.utc).isoformat(),
                    status="active"
                )
                self.persona_store.add_or_update(new_item, update_embedding=True)
                
                # Archive fused shards
                for old_item in group:
                    archived = self._archive_old_rule_after_compression(old_item, new_item)
                    if archived:
                        self._record_compression_event(old_item.id, new_item.id, abstract_rule)
                
                consolidated_count += len(group)
                logger.info(f"Consolidated {len(group)} items into: {abstract_rule[:30]}...")

        # 4) Archive stale low-novelty debris (second pass — prior step may have freed slots)
        remaining_items = self.persona_store.get_low_score_items(threshold=0.3, limit=20)
        for item in remaining_items:
            if int(getattr(item, "locked", 0) or 0) == 1:
                logger.warning("[L0-PROTECT] Skip excreting locked rule %s", item.id)
                continue
            if item.novelty < 0.2 and item.evidence_count < 3:
                # Low novelty + weak evidence => safe to archive
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE persona_items SET status='archived', last_seen_at=? WHERE id=?",
                        (datetime.now(timezone.utc).isoformat(), item.id)
                    )
                    conn.commit()
                excreted_count += 1
                logger.info(f"Excreted metabolic waste: {item.text[:20]}...")

        results.update({
            "status": "completed",
            "consolidated": consolidated_count,
            "excreted": excreted_count
        })
        return results

