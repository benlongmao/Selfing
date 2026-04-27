#!/usr/bin/env python3
"""
Persona ↔ ``z_self`` consistency probe (legacy name: ``SoulConsistencyChecker``).

- Measures cosine alignment between latent slices and aggregated Persona Memory embeddings.
- “Soul” here is an engineering metaphor for the measurable self-state vector, not a metaphysical claim.
- Does **not** assert anything about qualia or non-computational minds.
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timezone
import logging
from backend.persona_store import PersonaStore, PersonaItem
from backend.self_model import SelfModel

logger = logging.getLogger(__name__)

CONSISTENCY_THRESHOLD = float(os.environ.get("CONSISTENCY_THRESHOLD", "0.7"))
CONSISTENCY_CHECK_ENABLED = os.environ.get("CONSISTENCY_CHECK_ENABLED", "true").lower() == "true"
AUTO_FIX_ENABLED = os.environ.get("CONSISTENCY_AUTO_FIX", "false").lower() == "true"


class SoulConsistencyChecker:
    """Compare ``z_self`` blocks against Persona Memory anchors + embeddings."""

    def __init__(self, db_path: str = "data.db", persona_store: Optional[PersonaStore] = None, self_model: Optional[SelfModel] = None):
        self.db_path = db_path
        self.persona_store = persona_store or PersonaStore(db_path)
        self.self_model = self_model or SelfModel(db_path, self.persona_store)

    def check_consistency(
        self,
        session_id: str,
        auto_fix: bool = None
    ) -> Dict:
        """
        Returns:
            ``consistent``, ``consistency_score``, ``inconsistencies``, ``fixed``, ``fix_action``.
        """
        if not CONSISTENCY_CHECK_ENABLED:
            return {
                "consistent": True,
                "consistency_score": 1.0,
                "inconsistencies": [],
                "fixed": False,
                "fix_action": None
            }

        auto_fix = auto_fix if auto_fix is not None else AUTO_FIX_ENABLED

        z_self = self.self_model.get_z_self(session_id)
        if z_self is None:
            return {
                "consistent": True,
                "consistency_score": 1.0,
                "inconsistencies": [],
                "fixed": False,
                "fix_action": "z_self not initialized"
            }

        rules = self.persona_store.get_all_active(limit=100)
        if not rules:
            return {
                "consistent": True,
                "consistency_score": 1.0,
                "inconsistencies": [],
                "fixed": False,
                "fix_action": "no rules in Persona Memory"
            }

        inconsistencies = []
        consistency_scores = []

        from backend.self_model import SELF_SUBSPACE_DIMS

        for subspace_name, (start_idx, end_idx) in SELF_SUBSPACE_DIMS.items():
            z_subspace = z_self[start_idx:end_idx]

            relevant_rules = self._get_subspace_rules(subspace_name, rules)
            if not relevant_rules:
                continue

            rule_vectors = [self.self_model._project_to_latent(rule.embedding) for rule in relevant_rules if rule.embedding is not None]
            if not rule_vectors:
                continue

            rule_aggregate = np.mean(rule_vectors, axis=0)
            rule_subspace = rule_aggregate[start_idx:end_idx]

            consistency = np.dot(z_subspace, rule_subspace) / (
                np.linalg.norm(z_subspace) * np.linalg.norm(rule_subspace) + 1e-8
            )
            consistency_scores.append(float(consistency))

            if consistency < CONSISTENCY_THRESHOLD:
                inconsistencies.append({
                    "subspace": subspace_name,
                    "consistency": float(consistency),
                    "z_self_mean": float(np.mean(z_subspace)),
                    "rule_mean": float(np.mean(rule_subspace)),
                    "relevant_rules_count": len(relevant_rules)
                })

        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        is_consistent = overall_consistency >= CONSISTENCY_THRESHOLD

        fix_action = None
        fixed = False

        self._record_consistency_check(session_id, overall_consistency, inconsistencies, fixed)

        return {
            "consistent": is_consistent,
            "consistency_score": float(overall_consistency),
            "inconsistencies": inconsistencies,
            "fixed": fixed,
            "fix_action": fix_action
        }

    # Bilingual anchor blurbs for embedder similarity (Chinese retained + English tokens).
    _SUBSPACE_ANCHORS = {
        "safety":    "safety ethics compliance ban protect risk privacy harm 安全 伦理 合规 禁止 保护 风险 隐私 不伤害",
        "epistemic": "evidence fact inference uncertainty confidence verify knowledge calibration 证据 事实 推断 不确定性 置信度 验证 知识 校准",
        "style":     "tone voice concise teach explain communication style 表达风格 沟通方式 简洁 教学 解释 语气 表述",
        "strategy":  "plan execute strategy steps methods tools tasks goals 规划 执行 策略 步骤 方法 工具 任务 目标",
        "valence":   "emotion affect joy sadness positive negative feeling 情绪 情感 愉快 悲伤 积极 消极 感受",
        "arousal":   "activation arousal energy alert calm excitement 激活 唤醒 能量 活跃 平静 兴奋",
        "motivation": "drive curiosity connection meaning growth motivation 动机 驱动 好奇 连接 意义 成长",
        "somatic":   "interoception body fatigue tension relaxation state 体感 身体感觉 疲劳 紧张 放松 状态",
        "needs":     "needs clarity novelty connection memory signal 需求 清晰 新颖 联结 记忆 信号",
    }

    def _get_subspace_rules(self, subspace_name: str, rules: List[PersonaItem]) -> List[PersonaItem]:
        """
        Priority:
        1. Persona rows whose ``subspace`` field matches exactly.
        2. Embedding cosine ≥ 0.35 against the anchor string (when embedder works).
        3. Keyword fallback (CN + EN literals) for rows without embeddings.
        """
        explicit = [r for r in rules if getattr(r, "subspace", None) == subspace_name]
        remaining = [r for r in rules if getattr(r, "subspace", None) != subspace_name]

        if not remaining:
            return explicit

        anchor_text = self._SUBSPACE_ANCHORS.get(subspace_name, subspace_name)
        try:
            from backend.embedder import get_embedder
            embedder = get_embedder()

            anchor_vec = embedder.encode(anchor_text, normalize=True)

            scored = []
            no_embed = []
            for rule in remaining:
                if rule.embedding is not None and len(rule.embedding) > 0:
                    rule_vec = np.array(rule.embedding, dtype=np.float32)
                    norm = np.linalg.norm(rule_vec)
                    if norm > 1e-8:
                        rule_vec = rule_vec / norm
                    sim = float(np.dot(anchor_vec, rule_vec))
                    scored.append((rule, sim))
                else:
                    no_embed.append(rule)

            SIMILARITY_THRESHOLD = 0.35
            by_embedding = [r for r, sim in scored if sim >= SIMILARITY_THRESHOLD]

            keywords_map = {
                "safety":    ["安全", "隐私", "合规", "禁止", "避免", "保护", "风险", "伤害", "safety", "privacy", "risk", "harm"],
                "epistemic": ["证据", "事实", "推断", "不确定", "校准", "置信", "验证", "evidence", "fact", "uncertain", "verify"],
                "style":     ["解释", "简洁", "教学", "风格", "表达", "沟通", "tone", "style", "explain", "concise"],
                "strategy":  ["规划", "执行", "策略", "步骤", "方法", "工具", "plan", "execute", "strategy", "tool"],
            }
            keywords = keywords_map.get(subspace_name, [])
            by_keyword = [r for r in no_embed if any(kw in r.text.lower() for kw in keywords)]

            return explicit + by_embedding + by_keyword

        except Exception as e:
            logger.debug(f"Embedder unavailable for subspace rule matching ({e}), falling back to keywords")

        keywords_map = {
            "safety":    ["安全", "隐私", "合规", "禁止", "避免", "保护", "风险", "伤害", "safety", "privacy", "risk", "harm"],
            "epistemic": ["证据", "事实", "推断", "不确定", "校准", "置信", "验证", "evidence", "fact", "uncertain", "verify"],
            "style":     ["解释", "简洁", "教学", "风格", "表达", "沟通", "tone", "style", "explain", "concise"],
            "strategy":  ["规划", "执行", "策略", "步骤", "方法", "工具", "plan", "execute", "strategy", "tool"],
        }
        keywords = keywords_map.get(subspace_name, [])
        by_keyword = [r for r in remaining if any(kw in r.text.lower() for kw in keywords)]
        return explicit + by_keyword

    def _auto_fix(
        self,
        session_id: str,
        inconsistencies: List[Dict],
        rules: List[PersonaItem]
    ) -> Optional[str]:
        """Legacy hook — prefer ``SelfHomeostasis`` orchestration instead."""
        try:
            self.self_model.update_from_persona_rules(session_id, rules)

            logger.info(f"Auto-fixed z_self for session {session_id} using Persona Memory rules")
            return "updated_z_self_from_rules"

        except Exception as e:
            logger.error(f"Auto-fix failed for session {session_id}: {e}")
            return None

    def _record_consistency_check(
        self,
        session_id: str,
        consistency_score: float,
        inconsistencies: List[Dict],
        fixed: bool
    ):
        """Persist a JSON payload into ``persona_events``."""
        import uuid
        event_id = str(uuid.uuid4())
        detail = {
            "type": "consistency_check",
            "session_id": session_id,
            "consistency_score": consistency_score,
            "inconsistencies": inconsistencies,
            "fixed": fixed,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO persona_events (id, ts, type, persona_id, detail) VALUES (?, ?, ?, ?, ?)",
                (event_id, datetime.now(timezone.utc).isoformat(), "consistency_check", session_id, json.dumps(detail, ensure_ascii=False))
            )
            conn.commit()

    def get_consistency_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Return recent ``consistency_check`` rows for dashboards."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """SELECT id, ts, detail FROM persona_events 
                   WHERE persona_id=? AND type='consistency_check' 
                   ORDER BY ts DESC LIMIT ?""",
                (session_id, limit)
            )
            rows = cur.fetchall()

        history = []
        for row in rows:
            try:
                detail = json.loads(row[2]) if row[2] else {}
                history.append({
                    "event_id": row[0],
                    "timestamp": row[1],
                    "consistency_score": detail.get("consistency_score"),
                    "inconsistencies_count": len(detail.get("inconsistencies", [])),
                    "fixed": detail.get("fixed", False)
                })
            except Exception as e:
                logger.warning(f"Failed to parse consistency check event {row[0]}: {e}")

        return history
