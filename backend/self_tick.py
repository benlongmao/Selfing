#!/usr/bin/env python3
"""
Self Tick 节律机制：周期性自我更新与广播
- 聚合近期证据
- 更新 z_self
- 广播到全局工作空间（GW）
- 记录自我轨迹
"""
import os
import random
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional
import numpy as np
from backend.self_model import SelfModel, RULES_DIM, EMOTION_DIM, MOTIVATION_DIM, SOMATIC_DIM, SOMATIC_START_IDX, WORLDVIEW_DIM, MEMORY_DIM, ATTENTION_DIM
from backend.persona_store import PersonaStore
from backend.metrics_calculator import MetricsCalculator
from backend.observability.structured_logging import emit_structured_log
import logging

try:
    from backend.homeostasis import SelfHomeostasis
except ImportError:
    SelfHomeostasis = None

try:
    from backend.dimension_interaction import DimensionInteraction
    DIMENSION_INTERACTION_AVAILABLE = True
except ImportError:
    DIMENSION_INTERACTION_AVAILABLE = False
    logging.warning("DimensionInteraction not available")

from backend.config import config
from backend.utils.path_utils import get_workspace_root

logger = logging.getLogger(__name__)

SELF_TICK_EVIDENCE_WINDOW = config.get("parameters.self_tick_evidence_window", 4)  # rolling chat window size

# [Level 2] Multi-level drift analyzer (optional)
try:
    from backend.drift_levels import get_drift_analyzer, DriftLevel
    DRIFT_ANALYZER_AVAILABLE = True
except ImportError:
    DRIFT_ANALYZER_AVAILABLE = False
    logger.warning("DriftAnalyzer not available, using legacy drift handling")

class SelfTick:
    """Self Tick 节律机制"""
    
    def __init__(self, db_path: str = "data.db", self_model: Optional[SelfModel] = None, persona_store: Optional[PersonaStore] = None):
        self.db_path = db_path
        self.self_model = self_model
        self.persona_store = persona_store
        
        # Per-session evidence buffer for aggregation
        self.session_evidence_cache: Dict[str, List[str]] = {}  # session_id -> [evidence_text, ...]
        
        # P2.2: metrics calculator
        self.metrics_calculator = MetricsCalculator(db_path)
        
        # P3: homeostasis engine
        self.homeostasis = None
        if SelfHomeostasis:
            try:
                self.homeostasis = SelfHomeostasis(db_path)
                logger.info("SelfHomeostasis initialized")
            except Exception as e:
                logger.warning(f"Failed to init SelfHomeostasis: {e}")
        
        # Dimension interaction resolver (optional)
        self.dimension_interaction = None
        if DIMENSION_INTERACTION_AVAILABLE:
            try:
                self.dimension_interaction = DimensionInteraction(db_path)
                logger.info("DimensionInteraction initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize DimensionInteraction: {e}")
        
        # [removed 2026-03] MetaCognition was template-only; deleted

        # Dimension-level metrics (optional)
        self.dimension_metrics = None
        try:
            from backend.dimension_metrics import DimensionMetrics
            self.dimension_metrics = DimensionMetrics(db_path)
            logger.info("DimensionMetrics initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize DimensionMetrics: {e}")
            
        self.mind_wandering = None

        self.event_logger = None
        try:
            from backend.event_logger import EventLogger
            self.event_logger = EventLogger(db_path)
            logger.info("EventLogger initialized in SelfTick")
        except Exception as e:
            logger.warning(f"Failed to initialize EventLogger in SelfTick: {e}")
        
        # [Level 2] Drift analyzer init
        self.drift_analyzer = None
        if DRIFT_ANALYZER_AVAILABLE:
            try:
                self.drift_analyzer = get_drift_analyzer()
                logger.info("MultiLevelDriftAnalyzer initialized in SelfTick")
            except Exception as e:
                logger.warning(f"Failed to initialize DriftAnalyzer: {e}")

    def set_mind_wandering(self, module):
        self.mind_wandering = module

    def _minutes_since_last_mind_wandering(self, session_id: str) -> Optional[float]:
        """最近一次神游写入 chat_turns（tool_used='mind_wandering'）距今的分钟数；无记录为 None。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT created_at FROM chat_turns
                       WHERE session_id=? AND tool_used='mind_wandering'
                       ORDER BY created_at DESC LIMIT 1""",
                    (session_id,),
                ).fetchone()
            if not row or not row[0]:
                return None
            raw = str(row[0]).strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            last = datetime.fromisoformat(raw)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last
            return max(0.0, delta.total_seconds() / 60.0)
        except Exception as e:
            logger.debug(f"_minutes_since_last_mind_wandering: {e}")
            return None

    def _idle_mind_wandering_params(self) -> Dict:
        """空闲自动神游参数；enabled=false 时退回旧版保守默认值。"""
        raw = config.get("parameters.mind_wandering_idle")
        mw = raw if isinstance(raw, dict) else {}
        if bool(mw.get("enabled", True)):
            return {
                "no_evidence_min_energy": float(mw.get("no_evidence_min_energy", 25.0)),
                "base_probability": float(mw.get("base_probability", 0.055)),
                "max_probability": float(mw.get("max_probability", 0.42)),
                "tick_spontaneous_probability": float(mw.get("tick_spontaneous_probability", 0.055)),
                "tick_min_energy": float(mw.get("tick_min_energy", 32.0)),
                "tick_max_evidence_items": int(mw.get("tick_max_evidence_items", 2)),
                "min_interval_minutes": int(mw.get("min_interval_minutes", 35)),
            }
        return {
            "no_evidence_min_energy": 30.0,
            "base_probability": 0.02,
            "max_probability": 0.15,
            "tick_spontaneous_probability": 0.02,
            "tick_min_energy": 40.0,
            "tick_max_evidence_items": 1,
            "min_interval_minutes": 0,
        }

    def _mind_wandering_cooldown_active(self, session_id: str, min_interval_minutes: int) -> bool:
        if min_interval_minutes <= 0:
            return False
        elapsed = self._minutes_since_last_mind_wandering(session_id)
        if elapsed is None:
            return False
        return elapsed < float(min_interval_minutes)
    
    def trigger(
        self,
        session_id: str,
        self_model: Optional[SelfModel] = None,
        persona_store: Optional[PersonaStore] = None,
        trigger_reason: str = "scheduled"
    ) -> Dict:
        """
        触发 Self Tick
        
        Args:
            session_id: 会话ID
            self_model: SelfModel 实例（如果未在初始化时提供）
            persona_store: PersonaStore 实例（如果未在初始化时提供）
        
        Returns:
            {
                "success": bool,
                "z_self_updated": bool,
                "drift": float,
                "evidence_count": int,
                "tick_count": int
            }
        """
        self_model = self_model or self.self_model
        persona_store = persona_store or self.persona_store
        
        if not self_model:
            logger.error("SelfModel not available for Self Tick")
            emit_structured_log(
                logger,
                "self_tick_missing_self_model",
                level="error",
                session_id=session_id,
                trigger_reason=trigger_reason,
            )
            return {"success": False, "error": "SelfModel not available"}
        
        try:
            # 1. Aggregate recent evidence
            evidence_list = self._aggregate_evidence(session_id)
            imc = self._idle_mind_wandering_params()
            
            # [P3] No chat evidence still allows idle MindWandering path
            # Core idle autonomy: does not require transcript evidence
            if not evidence_list:
                logger.info(f"No evidence for session {session_id}, checking for autonomous activity...")
                emit_structured_log(
                    logger,
                    "self_tick_no_evidence",
                    level="info",  # idle without evidence is normal, not warn
                    session_id=session_id,
                    trigger_reason=trigger_reason,
                )
                
                # Idle mind-wandering stochastic path
                # High distress: suppress background divergence; prefer quiet recovery
                try:
                    try:
                        pain_status = self_model.get_pain_status(session_id)
                        channels = (pain_status or {}).get("channels") or {}
                        distress = float(channels.get("distress", (pain_status or {}).get("total_pain", 0.0)))
                        challenge = float(channels.get("challenge", 0.0))
                    except Exception:
                        distress = 0.0
                        challenge = 0.0

                    distress_thresh = float(
                        config.get("parameters.self_tick.idle_distress_suppress_threshold", 0.9)
                        or 0.9
                    )
                    if distress_thresh < 1.0 and distress > distress_thresh:
                        logger.info(
                            f"🛑 Idle activity suppressed due to high distress={distress:.2f} "
                            f"(threshold={distress_thresh}, session={session_id})"
                        )
                        return {"success": True, "idle_mode": True, "message": "Idle suppressed (high distress)"}

                    logger.info(f"[S44进化] 检测到空闲状态，正在检查待办任务...")
                    # [S44] Optional HEARTBEAT.md pending-task scan
                    try:
                        import os
                        import re

                        heartbeat_path = os.path.join(get_workspace_root(), "HEARTBEAT.md")
                        if os.path.exists(heartbeat_path):
                            with open(heartbeat_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            pending_section = re.search(
                                r"- \*\*待办任务\*\*(.*?)(?=\n##|\Z)", content, re.DOTALL
                            )
                            if pending_section:
                                pending_text = pending_section.group(1)
                                pending_count = pending_text.count("- ")
                                logger.info(
                                    f"📋 发现 {pending_count} 项待办任务，建议触发 S44_continue 或节律提醒"
                                )
                    except Exception as e:
                        logger.debug(f"检查待办任务失败: {e}")
                    energy = self_model.get_energy(session_id)
                    if energy > imc["no_evidence_min_energy"]:
                        # [2026-03-25] State-conditioned idle wandering probability
                        # Higher boredom / exploration nudges probability up
                        base_prob = imc["base_probability"]
                        max_prob = imc["max_probability"]
                        
                        try:
                            # Novelty need (low → bored)
                            needs = self_model.homeostasis.load_needs(session_id) if self_model.homeostasis else {}
                            novelty_need = float((needs or {}).get("novelty", 0.5) or 0.5)
                            boredom = 1.0 - novelty_need  # low novelty → high boredom
                            
                            # Exploration drive from structured summary
                            struct = self_model.get_structured_summary(session_id)
                            exploration = float(struct.get("exploration_mean", 0.5) or 0.5)
                            
                            # Scale base probability by boredom / exploration
                            state_factor = 1.0
                            if boredom > 0.3:
                                state_factor += boredom * 1.5  # boredom bump (~few % absolute)
                            if exploration > 0.6:
                                state_factor += (exploration - 0.5) * 2.0  # exploration bump
                            
                            wandering_prob = min(max_prob, base_prob * state_factor)
                            
                            logger.debug(f"Mind wandering prob: {wandering_prob:.3f} (boredom={boredom:.2f}, exploration={exploration:.2f})")
                        except Exception as e:
                            logger.debug(f"Failed to calculate state-based wandering prob: {e}")
                            wandering_prob = base_prob
                        
                        if self._mind_wandering_cooldown_active(session_id, imc["min_interval_minutes"]):
                            logger.debug(
                                "Idle mind wandering skipped (cooldown), "
                                f"min_interval_minutes={imc['min_interval_minutes']}"
                            )
                        elif random.random() < wandering_prob:
                            if self.mind_wandering:
                                logger.info(f"🧠 Triggering spontaneous Mind Wandering during idle (session={session_id}, energy={energy:.1f}, prob={wandering_prob:.3f})")
                                wandering_result = self.mind_wandering.trigger_wandering(session_id)
                                return {
                                    "success": True,
                                    "idle_mode": True,
                                    "mind_wandering": wandering_result,
                                    "trigger_reason": f"state-triggered (prob={wandering_prob:.3f})"
                                }
                    else:
                        logger.debug(f"Energy too low for idle activity: {energy:.1f}")
                except Exception as e:
                    logger.warning(f"Idle activity trigger failed: {e}")
                
                # Nothing fired → idle noop
                return {"success": True, "idle_mode": True, "message": "No activity triggered (idle)"}
            
            # 2. Join evidence strings
            aggregated_evidence = "\n".join(evidence_list)
            
            # [FIX 2026-01-25] Use live chat turn count as tick proxy
            # [FIX 2026-03-12] Tick must be monotone: cleanup may shrink COUNT(*)
            # max(previous_tick, current_turn) prevents backwards jumps
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM chat_turns WHERE session_id=?",
                        (session_id,)
                    )
                    current_turn = cur.fetchone()[0]
                previous_tick = self._get_tick_count(session_id)
                tick_count = max(previous_tick, current_turn)
            except Exception as e:
                logger.warning(f"Failed to get tick count: {e}")
                tick_count = self._get_tick_count(session_id)
            
            # 3. P2.2: introspection feature vector
            introspection_features = self.metrics_calculator.get_introspection_features(
                session_id,
                window_size=SELF_TICK_EVIDENCE_WINDOW
            )
            
            # 4. SelfModel.update on tick cadence (P0.2 staged refresh)
            z_self_new, drift = self_model.update(
                session_id,
                aggregated_evidence,
                persona_topk=None,  # optional retrieved persona snippets
                introspection_features=introspection_features  # P2.2 metrics vector
            )
            
            # [Phase 1] Identity friction → somatic feedback
            # Large drift bumps visceral tension (same 16-d layout as sync_somatic_to_z_self)
            if drift > 0.12 and self_model.dim >= SOMATIC_START_IDX + SOMATIC_DIM:
                try:
                    logger.info(f"⚡ High Identity Friction detected (drift={drift:.3f}) -> Increasing Somatic Tension")
                    somatic_start = SOMATIC_START_IDX
                    
                    # Map drift>0.12 to friction intensity
                    friction_intensity = min(1.5, (drift - 0.12) * 5.0)
                    
                    # 1. Pain/tension slice [8:12]
                    z_self_new[somatic_start + 8 : somatic_start + 12] += friction_intensity * 0.5
                    
                    # 2. Viscosity/stress [4:8]
                    z_self_new[somatic_start + 4 : somatic_start + 8] += friction_intensity * 0.25
                    
                    # 3. Vitality down [12:16]
                    z_self_new[somatic_start + 12 : somatic_start + 16] -= friction_intensity * 0.15
                    
                    # Persist somatic override immediately
                    # [FIX 2026-01-24] Use tick_count, not stale DB tick
                    self_model._save_z_self(
                        session_id, 
                        z_self_new, 
                        tick=tick_count, 
                        drift=float(drift), 
                        last_summary=f"【生理反馈】检测到认知失调(Drift={drift:.2f})，导致内在紧张度急剧升高。"
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply identity friction feedback: {e}")
            
            # 4.05 P3: background consciousness nudge
            # Sparse evidence + energy → small chance of background thought
            try:
                # Gate: few evidence lines + energy above threshold
                if (
                    len(evidence_list) <= imc["tick_max_evidence_items"]
                    and self_model.get_energy(session_id) > imc["tick_min_energy"]
                ):
                    tick_p = imc["tick_spontaneous_probability"]
                    if self._mind_wandering_cooldown_active(session_id, imc["min_interval_minutes"]):
                        logger.debug(
                            "Tick-path mind wandering skipped (cooldown), "
                            f"min_interval_minutes={imc['min_interval_minutes']}"
                        )
                    elif random.random() < tick_p:
                        if self.mind_wandering:
                            logger.info(
                                f"🧠 Triggering spontaneous Mind Wandering in SelfTick "
                                f"(session={session_id}, p={tick_p:.3f})"
                            )
                            self.mind_wandering.trigger_wandering(session_id)

            except Exception as e:
                logger.warning(f"Background consciousness trigger failed: {e}")

            # [Level 2] 4.05: multi-level drift analysis
            drift_analysis = None
            if self.drift_analyzer and DRIFT_ANALYZER_AVAILABLE:
                try:
                    drift_analysis = self.drift_analyzer.analyze(session_id, drift)
                    
                    # Log by drift severity band
                    if drift_analysis.level == DriftLevel.EMERGENCY:
                        logger.error(
                            f"🚨 [EMERGENCY DRIFT] {drift_analysis.reason} | "
                            f"drift={drift:.4f}, cumulative={drift_analysis.cumulative_drift:.4f}, "
                            f"trend={drift_analysis.trend}"
                        )
                    elif drift_analysis.level == DriftLevel.ALERT:
                        logger.warning(
                            f"🔴 [ALERT DRIFT] {drift_analysis.reason} | "
                            f"drift={drift:.4f}, cumulative={drift_analysis.cumulative_drift:.4f}, "
                            f"trend={drift_analysis.trend}"
                        )
                    elif drift_analysis.level == DriftLevel.ATTENTION:
                        logger.info(
                            f"🟠 [ATTENTION DRIFT] {drift_analysis.reason} | "
                            f"drift={drift:.4f}, trend={drift_analysis.trend}"
                        )
                    elif drift_analysis.level == DriftLevel.WARNING:
                        logger.info(
                            f"🟡 [WARNING DRIFT] {drift_analysis.reason} | "
                            f"drift={drift:.4f}, trend={drift_analysis.trend}"
                        )
                    else:
                        logger.debug(f"🟢 [NORMAL DRIFT] drift={drift:.4f}, trend={drift_analysis.trend}")
                    
                except Exception as e:
                    logger.warning(f"Multi-level drift analysis failed: {e}")

            # 4.1 P3: homeostasis arbitration after SelfModel.update
            # Replaces legacy SoulConsistencyChecker auto_fix path
            # [Level 2] Uses drift_analysis severity when available
            ref_rules_meaningful = (
                self_model.ref_vector is not None
                and self_model.ref_vector.shape[0] >= RULES_DIM
                and float(np.linalg.norm(self_model.ref_vector[:RULES_DIM])) > 1e-6
            )
            if self.homeostasis and ref_rules_meaningful:
                try:
                    # Coarse evidence_strength from drift magnitude (proxy for shock)
                    evidence_strength = min(1.0, abs(drift) * 3.0) 
                    
                    # [Level 2] Boost weight on ALERT/EMERGENCY drift
                    if drift_analysis and drift_analysis.level in [DriftLevel.ALERT, DriftLevel.EMERGENCY]:
                        evidence_strength = min(1.0, evidence_strength * 1.5)  # stronger correction signal
                        logger.info(f"Boosting evidence_strength to {evidence_strength:.2f} due to {drift_analysis.level.value} drift")
                    
                    decision = self.homeostasis.regulate(
                        session_id, 
                        z_self_new, 
                        self_model.ref_vector, 
                        evidence_strength=evidence_strength
                    )
                    
                    action = decision.get("action")
                    injection = decision.get("system_prompt_injection", "")
                    
                    # Apply homeostasis action
                    if action == "suppress_self":
                        # Hard blend back toward constitutional rules (alpha=0.8)
                        if z_self_new.shape[0] >= RULES_DIM and self_model.ref_vector.shape[0] >= RULES_DIM:
                            alpha = 0.8
                            z_self_new[:RULES_DIM] = (
                                z_self_new[:RULES_DIM] * (1 - alpha)
                                + self_model.ref_vector[:RULES_DIM] * alpha
                            )
                            drift = float(self_model._drift_vs_rules_ref(z_self_new))
                            logger.info(f"🛡️ Homeostasis enforced SUPPRESSION. Reverted z_self. New drift: {drift:.4f}")
                    
                    elif action == "induce_stress":
                         logger.info(f"⚠️ Homeostasis induced STRESS. Injection: {injection[:30]}...")
                         
                    elif action == "paradigm_shift":
                         logger.info(f"🌪️ Homeostasis detected PARADIGM SHIFT. Allowing drift.")

                    # Re-save if injection or suppression rewrote vector
                    # (SelfModel.update already persisted once)
                    if injection or action == "suppress_self":
                        # PromptBuilder reads last_summary — surface injection text there
                        
                        final_summary = injection if injection else "状态平稳"
                        # [FIX 2026-01-24] tick_count not stale DB tick
                        self_model._save_z_self(session_id, z_self_new, tick=tick_count, drift=float(drift), last_summary=final_summary)
                        
                except Exception as e:
                    logger.warning(f"Homeostasis regulation failed: {e}", exc_info=True)
            
            # 4.4.5 Personality activation slice [0:32]
            personality_updated = False
            if self_model.personality_store and self_model.dim >= 32:
                personality_updated = self._update_personality_from_evidence(
                    session_id, aggregated_evidence, self_model, tick_count,
                )

            # 4.5 Emotion store refresh from evidence
            emotion_updated = False
            if self_model.emotion_store and self_model.dim >= 48:
                emotion_updated = self._update_emotion_from_evidence(
                    session_id,
                    aggregated_evidence,
                    self_model,
                    tick_count
                )
            
            # 4.6 Motivation store refresh from evidence
            motivation_updated = False
            if self_model.motivation_store and self_model.dim >= 64:
                motivation_updated = self._update_motivation_from_evidence(
                    session_id,
                    aggregated_evidence,
                    self_model,
                    tick_count
                )
            
            # 4.6.5 Somatic store → z_self sync
            somatic_synced = False
            if self_model.somatic_store and self_model.dim >= SOMATIC_START_IDX + SOMATIC_DIM:
                somatic_synced = self_model.sync_somatic_to_z_self(session_id)
                if somatic_synced:
                    logger.debug(f"Somatic dimension synced to z_self in SelfTick (session={session_id})")
            
            # [disabled 2026-02-02] worldview/memory/attention no longer live in z_self
            # Stores may still run standalone without z_self sync

            # 4.7 Dimension interaction + conflict resolution (128-d layout)
            interaction_result = None
            if (self.dimension_interaction is not None and 
                self_model.dim >= 104 and  # need rules+emotion+motivation+somatic slices
                self_model.emotion_store and 
                self_model.motivation_store):
                
                z_self_current = self_model.get_z_self(session_id)
                if z_self_current is not None and z_self_current.shape[0] >= 104:
                    # 128-d: rules 0-32, emotion 32-48, motivation 48-64, reserved 64-88, somatic 88-104, needs 104-128
                    rules_vec = z_self_current[:32]
                    emotion_vec = z_self_current[32:48]
                    motivation_vec = z_self_current[48:64]
                    somatic_vec = z_self_current[88:104]
                    needs_vec = z_self_current[104:128] if z_self_current.shape[0] >= 128 else None
                    ref_rules = None
                    if (
                        self_model.ref_vector is not None
                        and self_model.ref_vector.shape[0] >= RULES_DIM
                        and float(np.linalg.norm(self_model.ref_vector[:RULES_DIM])) > 1e-6
                    ):
                        ref_rules = self_model.ref_vector[:RULES_DIM]
                    ref_emotion = None
                    ref_motivation = None
                    
                    interaction_result = self.dimension_interaction.update_all_dimensions(
                        session_id,
                        rules_vec,
                        emotion_vec,
                        motivation_vec,
                        somatic_vector=somatic_vec,
                        needs_vector=needs_vec,
                        ref_rules=ref_rules,
                        ref_emotion=ref_emotion,
                        ref_motivation=ref_motivation
                    )
                    
                    z_self_updated = z_self_current.copy()
                    z_self_updated[:32] = interaction_result.updated_rules
                    z_self_updated[32:48] = interaction_result.updated_emotion
                    z_self_updated[48:64] = interaction_result.updated_motivation
                    
                    if interaction_result.updated_somatic is not None:
                        z_self_updated[88:104] = interaction_result.updated_somatic
                    if interaction_result.updated_needs is not None and z_self_updated.shape[0] >= 128:
                        z_self_updated[104:128] = interaction_result.updated_needs
                    
                    drift = float(np.linalg.norm(z_self_updated[32:] - z_self_current[32:]))
                    self_model._save_z_self(session_id, z_self_updated, tick=tick_count, drift=drift)
                    
                    logger.info(
                        f"Dimension interaction completed: "
                        f"strength={interaction_result.interaction_strength:.3f}, "
                        f"conflicts={len(interaction_result.conflicts)}"
            )
            
            # 4.8 Periodic metabolism / evolution hooks (every 10 ticks)
            tick_count = self._get_tick_count(session_id)
            if tick_count % 10 == 0:
                try:
                    if self_model.emotion_store:
                        self_model.emotion_store.evolve_emotions(session_id)
                    if self_model.motivation_store:
                        self_model.motivation_store.evolve_motivations(session_id)
                    if getattr(self_model, 'somatic_store', None):
                        self_model.somatic_store.evolve_somatics()
                    if getattr(self_model, 'world_store', None):
                        self_model.world_store.evolve_beliefs()
                        try:
                            self_model.sync_worldview_to_z_self(session_id)
                        except Exception as sync_e:
                            logger.debug(f"sync_worldview_to_z_self after evolve_beliefs: {sync_e}")
                    logger.info(f"Dimension evolution triggered for session {session_id}")
                except Exception as e:
                    logger.warning(f"Dimension evolution failed: {e}")
            
            # [2026-02-02] 4.8.1 Identity anchor slow drift (every 100 ticks)
            if tick_count > 0 and tick_count % 100 == 0:
                try:
                    # Layered drift snapshot
                    layered_drift = self_model.compute_layered_drift(session_id)
                    
                    # Allow growth only if L0 stable and L1 drift moderate
                    if not layered_drift["l0_violation"] and layered_drift["drift_l1"] < 0.5:
                        # alpha scales with stability (0.03..0.08)
                        stability_factor = 1.0 - layered_drift["drift_l1"]  # in [0.5,1] when drift_l1<=0.5
                        alpha = 0.03 + 0.05 * stability_factor  # map to [0.03,0.08]
                        
                        success = self_model.evolve_identity_anchor(session_id, alpha=alpha)
                        if success:
                            logger.info(
                                f"🌱 [Identity Growth] L1 anchor evolved at tick {tick_count}. "
                                f"alpha={alpha:.4f}, L1_drift={layered_drift['drift_l1']:.4f}"
                            )
                    else:
                        logger.info(
                            f"[Identity Growth] Skipped at tick {tick_count}: "
                            f"L0_violation={layered_drift['l0_violation']}, L1_drift={layered_drift['drift_l1']:.4f}"
                        )
                except Exception as e:
                    logger.warning(f"Identity evolution failed: {e}")

            # [2026-02-24] 4.8.4 Weekly consolidation + forgetting (every 500 ticks)
            if tick_count > 0 and tick_count % 500 == 0:
                try:
                    from backend.self_narrative import SelfNarrative
                    self_narrative = SelfNarrative(self.db_path)
                    
                    # Merge last-7d episodic shards
                    consolidation_result = self_narrative.consolidate_weekly_memories(session_id, days=7)
                    if consolidation_result and consolidation_result.get("status") == "success":
                        logger.info(
                            f"🧠 [Memory Consolidation] Tick {tick_count}: "
                            f"Processed {consolidation_result['memories_processed']} memories, "
                            f"Themes: {consolidation_result.get('themes', [])[:3]}"
                        )
                    
                    # Forgetting curve on stale low-salience rows
                    decay_result = self_narrative.apply_forgetting_curve(decay_days=30, min_significance=0.3)
                    if decay_result and decay_result.get("decayed_count", 0) > 0:
                        logger.info(
                            f"📉 [Forgetting Curve] Tick {tick_count}: "
                            f"Decayed {decay_result['decayed_count']} old memories"
                        )
                except Exception as e:
                    logger.warning(f"Memory consolidation/forgetting failed: {e}")

            # 4.8.5 Persona memory score decay (every 10 ticks; disable via config)
            if tick_count % 10 == 0:
                try:
                    decay_stats = persona_store.decay_memory()
                    if decay_stats.get("decayed", 0) > 0:
                        logger.info(f"Memory decay at tick {tick_count}: {decay_stats}")
                except Exception as e:
                    logger.warning(f"Memory decay failed: {e}")

            # [2026-03-30] 4.8.5b Topic archive + similarity compression (every 20 ticks)
            if tick_count % 20 == 0:
                try:
                    archive_result = persona_store.archive_redundant_by_topic(max_per_topic=5)
                    if archive_result.get("total_archived", 0) > 0:
                        logger.info(f"Redundancy archive at tick {tick_count}: {archive_result}")
                except Exception as e:
                    logger.warning(f"Redundancy archive failed: {e}")

                try:
                    from backend.rule_compressor import RuleCompressor
                    compressor = RuleCompressor(self.db_path, persona_store)
                    compress_result = compressor.compress_rules(max_items=1000)
                    if compress_result.get("compressed", 0) > 0:
                        logger.info(f"Auto-compression at tick {tick_count}: {compress_result}")
                except Exception as e:
                    logger.warning(f"Auto-compression failed: {e}")

            # [2026-01-23] 4.8.6 Needs-based event reflection (every 3 ticks)
            if tick_count % 3 == 0:
                try:
                    from backend.event_triggered_reflection import (
                        EventDetector, 
                        get_event_processor,
                        process_events_batch
                    )
                    
                    # Current needs vector (+ energy)
                    needs = {}
                    if hasattr(self_model, 'homeostasis') and self_model.homeostasis:
                        needs = self_model.homeostasis.load_needs(session_id) or {}
                        needs['energy'] = self_model.get_energy(session_id)
                    
                    # Detect unmet-need events
                    needs_events = EventDetector.detect_from_needs(needs, session_id)
                    
                    if needs_events:
                        logger.info(f"Detected {len(needs_events)} needs-based events in SelfTick")
                        # Batch-process into reflection candidates
                        event_results = process_events_batch(needs_events, self.db_path)
                        for result in event_results:
                            if result.get("candidates_added"):
                                logger.info(f"Event-triggered reflection: {result['event_type']} -> added {result['candidates_added']}")
                except Exception as e:
                    logger.warning(f"Event-triggered reflection failed: {e}")

            # 5. tick_count already computed above
            # 6. P0.2: structured summary → DB (GW broadcast node)
            self_summary_dict = self_model.get_structured_summary(session_id)
            self._save_self_summary(session_id, self_summary_dict)
            
            # 7. persona_events trajectory (+ optional state_glimpse for associative recall)
            state_glimpse = None
            last_summary = None
            try:
                if isinstance(self_summary_dict, dict):
                    # Prefer concise last_summary when present
                    last_summary = self_summary_dict.get("last_summary") or None
                    # Fallback glimpse from a few stable scalar fields
                    parts = []
                    if self_summary_dict.get("energy") is not None:
                        try:
                            parts.append(f"energy={float(self_summary_dict.get('energy')):.0f}")
                        except Exception:
                            pass
                    try:
                        parts.append(f"drift={float(self_summary_dict.get('drift', 0.0)):.2f}")
                    except Exception:
                        pass
                    emo = self_summary_dict.get("emotion")
                    if emo:
                        parts.append(f"emotion={str(emo)[:12]}")
                    if parts:
                        state_glimpse = " / ".join(parts)
            except Exception:
                state_glimpse = None
                last_summary = None

            self._record_trajectory(
                session_id, 
                tick_count, 
                float(drift), 
                len(evidence_list), 
                trigger_reason=trigger_reason,
                metrics=introspection_features,  # P2.2 metrics snapshot
                state_glimpse=state_glimpse,
                last_summary=last_summary,
            )
            
            # 8. Flush per-session evidence buffer
            if session_id in self.session_evidence_cache:
                self.session_evidence_cache[session_id] = []
            
            # P0: time-based passive recovery via event bus
            # [2026-03-30] rest_interval events
            if self.self_model:
                current_energy = self.self_model.get_energy(session_id)
                if current_energy < 80.0:
                    # Low energy → gentle rest_interval nudge
                    self.self_model.trigger_event(session_id, "rest_interval", intensity=0.5)
                    logger.debug(f"Rest interval event triggered: energy was {current_energy:.1f}")
            
            # [v2.0] Mirror snapshot row in self_history
            try:
                z_vec_str = json.dumps(z_self_updated.tolist()) if 'z_self_updated' in locals() else json.dumps(self_model.get_z_self(session_id).tolist())
                
                # Dominant emotion label if emotion slice changed
                dom_emotion = "neutral"
                if emotion_updated and self_model.emotion_store:
                    emotion_state = self_model.emotion_store.get_emotion_state(session_id)
                    if emotion_state:
                        dom_emotion = emotion_state.dominant_emotion
                
                ts_now = datetime.now(timezone.utc).isoformat()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO self_history (session_id, tick, z_self_vector, trigger_event, dominant_emotion, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, tick_count, z_vec_str, trigger_reason, dom_emotion, ts_now)
                    )
                    conn.commit()
                logger.debug(f"Self history snapshot saved: session={session_id}, tick={tick_count}, emotion={dom_emotion}")
            except Exception as snapshot_e:
                logger.warning(f"Failed to save self_history snapshot: {snapshot_e}")
            
            logger.info(f"Self Tick completed for session {session_id}: tick={tick_count}, drift={drift:.4f}, evidence={len(evidence_list)}")
            
            result = {
                "success": True,
                "z_self_updated": True,
                "drift": float(drift),
                "evidence_count": len(evidence_list),
                "tick_count": tick_count,
                "self_summary": self_summary_dict,  # P0.2 structured summary
                "trigger_reason": trigger_reason  # P1.2 why tick fired
            }
            
            # [Level 2] Attach drift analyzer payload
            if drift_analysis:
                result["drift_level"] = drift_analysis.level.value
                result["drift_should_check_consistency"] = drift_analysis.should_trigger_consistency_check
                result["drift_check_depth"] = drift_analysis.check_depth
                result["drift_reason"] = drift_analysis.reason
                result["drift_cumulative"] = drift_analysis.cumulative_drift
                result["drift_trend"] = drift_analysis.trend
            
            # Surface emotion delta to caller
            if emotion_updated:
                emotion_state = self_model.emotion_store.get_emotion_state(session_id)
                if emotion_state:
                    result["emotion_updated"] = True
                    result["emotion"] = emotion_state.dominant_emotion
                    result["emotion_intensity"] = emotion_state.intensity
            
            # Surface motivation delta to caller
            if motivation_updated:
                motivation_state = self_model.motivation_store.get_motivation_state(session_id)
                if motivation_state:
                    result["motivation_updated"] = True
                    result["motivation"] = motivation_state.dominant_motivation
                    result["motivation_intensity"] = motivation_state.intensity
            
            # Surface dimension_interaction summary
            if interaction_result is not None:
                result["dimension_interaction"] = {
                    "interaction_strength": float(interaction_result.interaction_strength),
                    "conflicts_detected": len(interaction_result.conflicts),
                    "conflicts": [
                        {
                            "type": c.get("type", "unknown"),
                            "description": c.get("description", ""),
                            "severity": c.get("severity", "low")
                        }
                        for c in interaction_result.conflicts
                    ]
                }
            
            # Optional dimension_metrics bundle
            if self.dimension_metrics and self_model:
                try:
                    metrics = self.dimension_metrics.get_all_metrics(session_id)
                    result["dimension_metrics"] = metrics
                    logger.info(f"Dimension metrics computed for session {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to compute dimension metrics: {e}")
            
            # [2026-04-07] SelfModeling removed (unused). Drift repair lives in homeostasis.regulate().
            
            structured_payload = {
                "session_id": session_id,
                "tick_count": tick_count,
                "drift": float(drift),
                "evidence_count": len(evidence_list),
                "trigger_reason": trigger_reason,
                "metrics": introspection_features,
                "emotion_updated": bool(result.get("emotion_updated")),
                "motivation_updated": bool(result.get("motivation_updated")),
                "dimension_metrics": result.get("dimension_metrics"),
            }
            if interaction_result is not None and "dimension_interaction" in result:
                structured_payload["dimension_interaction"] = result["dimension_interaction"]
            emit_structured_log(
                logger,
                "self_tick_completed",
                **structured_payload,
            )
            
            # [2026-02-22] Growth system hook
            try:
                from backend.growth_system import get_growth_system
                growth_system = get_growth_system(self.db_path)
                growth_result = growth_system.process_growth(session_id, self_model)
                
                # [2026-02-22] Guard: growth_result must be dict-shaped
                if isinstance(growth_result, dict) and growth_result.get("grew"):
                    result["growth"] = growth_result
                    if growth_result.get("milestone"):
                        logger.info(f"🎯 Growth milestone: {growth_result['milestone'].get('description')}")
                    elif growth_result.get("daily_growth"):
                        logger.info(f"🌱 Daily growth applied: distance={growth_result.get('evolution_distance', 0):.6f}")
            except Exception as e:
                logger.debug(f"Growth system skipped: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Self Tick failed for session {session_id}: {e}", exc_info=True)
            emit_structured_log(
                logger,
                "self_tick_failed",
                level="error",
                session_id=session_id,
                trigger_reason=trigger_reason,
                error=str(e),
            )
            return {"success": False, "error": str(e)}
    
    def add_evidence(self, session_id: str, evidence_text: str):
        """
        添加证据到缓存（在每轮对话后调用）
        
        Args:
            session_id: 会话ID
            evidence_text: 证据文本（用户输入+模型输出摘要）
        """
        if session_id not in self.session_evidence_cache:
            self.session_evidence_cache[session_id] = []
        
        self.session_evidence_cache[session_id].append(evidence_text)
        
        # Ring buffer cap per session
        if len(self.session_evidence_cache[session_id]) > SELF_TICK_EVIDENCE_WINDOW * 2:
            self.session_evidence_cache[session_id] = self.session_evidence_cache[session_id][-SELF_TICK_EVIDENCE_WINDOW:]
    
    def _aggregate_evidence(self, session_id: str) -> List[str]:
        """
        聚合近期证据
        
        Returns:
            证据文本列表（最近 N 轮）
        """
        if session_id not in self.session_evidence_cache:
            return []
        
        evidence_list = self.session_evidence_cache[session_id]
        
        # Tail window
        return evidence_list[-SELF_TICK_EVIDENCE_WINDOW:]
    
    def _get_tick_count(self, session_id: str) -> int:
        """获取当前 tick 计数"""
        if self.self_model:
            return self.self_model._get_tick(session_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT tick FROM self_state WHERE session_id=?", (session_id,)
            )
            row = cur.fetchone()
            return row[0] if row else 0
            
    def process_dreaming(self, session_id: str):
        """
        执行梦境处理（离线整理）
        - 在精力较低时进行
        - 整理记忆、恢复精力、生成非逻辑梦境
        """
        if not self.self_model:
            return
            
        try:
            # Energy gate for dream pass
            energy = self.self_model.get_energy(session_id)
            # Dreaming only when depleted (<50)
            if energy > 50.0:
                return
                
            logger.info(f"💤 Session {session_id} is entering dream state... (Energy: {energy:.1f})")
            
            # 1. Passive energy bump (+5) while "asleep"
            self.self_model.update_energy(session_id, 5.0)
            
            # [Phase 1] Dream text generation removed (MindWandering covers that niche)

            # 3. Light passive rule review sample
            items = self.persona_store.get_all_active(limit=100)
            if items:
                sampled = random.sample(items, min(3, len(items)))
                logger.info(f"💤 [Dreaming] Passive rule review: {[i.text[:20] for i in sampled]}")
            
            # 4. Narrative consolidation hooks
            if self.self_model.narrative_identity and energy > 30.0:
                ni = self.self_model.narrative_identity
                try:
                    trajectory = self.get_trajectory(session_id, limit=10)
                    if trajectory:
                        peak_event = max(trajectory, key=lambda x: x.get("drift", 0.0))
                        peak_drift = peak_event.get("drift", 0.0)
                        if peak_drift > 0.08:
                            content = f"在梦中整合了 Tick {peak_event['tick']} 的认知震荡(Drift={peak_drift:.2f})。"
                            ni.record_turning_point(
                                session_id, content, significance=min(1.0, peak_drift * 3.0)
                            )
                except Exception as ne:
                    logger.warning(f"Dream narrative consolidation failed: {ne}")

                # 5. Pull high-strength relations into narrative_identity
                try:
                    with sqlite3.connect(getattr(self.self_model, "db_path", "data.db")) as conn:
                        conn.row_factory = sqlite3.Row
                        cur = conn.execute(
                            """
                            SELECT entity_name, relationship_summary, relationship_strength
                            FROM relation_memory
                            WHERE session_id = ? AND relationship_strength >= 0.5
                            ORDER BY updated_at DESC LIMIT 3
                            """,
                            (session_id,),
                        )
                        relations = cur.fetchall()
                    if relations:
                        existing = ni.get_narratives_by_type(session_id, "relationship")
                        existing_entities = {n.get("content", "")[:30] for n in existing[-5:]}
                        for rel in relations:
                            entity = rel["entity_name"]
                            tag = f"与用户{entity}的关系"
                            if not any(tag in ex for ex in existing_entities):
                                desc = (rel["relationship_summary"] or "")[:80]
                                if desc:
                                    ni.record_relationship(session_id, entity, desc)
                except Exception as re_err:
                    logger.debug(f"Dream relationship consolidation skipped: {re_err}")
                    
        except Exception as e:
            logger.error(f"Dreaming process failed: {e}")

    def _record_trajectory(
        self,
        session_id: str,
        tick_count: int,
        drift: float,
        evidence_count: int,
        trigger_reason: str = "scheduled",
        metrics: Optional[Dict] = None,
        state_glimpse: Optional[str] = None,
        last_summary: Optional[str] = None,
    ):
        """
        记录自我轨迹到 persona_events 表
        
        Args:
            session_id: 会话ID
            tick_count: 当前 tick 计数
            drift: 漂移值
            evidence_count: 聚合的证据数量
        """
        event_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        
        detail = {
            "session_id": session_id,
            "tick": tick_count,
            "drift": drift,
            "evidence_count": evidence_count,
            "type": "self_tick",
            "trigger_reason": trigger_reason,  # P1.2 why tick fired
            "metrics": metrics or {}  # P2.2 metric bundle
        }
        # Optional associative recall hint (skip heavy joins later)
        if state_glimpse:
            detail["state_glimpse"] = str(state_glimpse)[:200]
        if last_summary:
            detail["last_summary"] = str(last_summary)[:400]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO persona_events (id, ts, type, persona_id, detail) VALUES (?, ?, ?, ?, ?)",
                (event_id, ts, "self_tick", session_id, json.dumps(detail))
            )
            conn.commit()
        
        logger.debug(f"Recorded self_tick trajectory: {event_id}")
    
    # [2026-04-07] Removed _needs_self_repair / _perform_self_repair (SelfModeling deleted; homeostasis covers drift)

    def _save_self_summary(self, session_id: str, summary_dict: Dict):
        """
        P0.2: 保存结构化自我摘要到数据库（GW 广播节点）
        
        Args:
            session_id: 会话ID
            summary_dict: 结构化摘要字典
        """
        try:
            summary_json = json.dumps(summary_dict, ensure_ascii=False)
            with sqlite3.connect(self.db_path) as conn:
                # Migrate-on-write if self_summary column missing
                try:
                    conn.execute(
                        "UPDATE self_state SET self_summary=? WHERE session_id=?",
                        (summary_json, session_id)
                    )
                    conn.commit()
                except sqlite3.OperationalError as e:
                    if "no such column: self_summary" in str(e):
                        logger.warning(f"self_summary column not found, attempting to add it: {e}")
                        try:
                            conn.execute("ALTER TABLE self_state ADD COLUMN self_summary TEXT")
                            conn.execute(
                                "UPDATE self_state SET self_summary=? WHERE session_id=?",
                                (summary_json, session_id)
                            )
                            conn.commit()
                            logger.info("Added self_summary column to self_state table")
                        except Exception as alter_e:
                            logger.error(f"Failed to add self_summary column: {alter_e}")
                    else:
                        raise
        except Exception as e:
            logger.error(f"Failed to save self_summary for session {session_id}: {e}", exc_info=True)
    
    def get_trajectory(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        获取会话的自我轨迹
        
        Args:
            session_id: 会话ID
            limit: 返回数量限制
        
        Returns:
            轨迹事件列表
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, ts, detail FROM persona_events WHERE type='self_tick' AND persona_id=? ORDER BY ts DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cur.fetchall()
        
        trajectory = []
        for row in rows:
            try:
                detail = json.loads(row[2]) if row[2] else {}
                trajectory.append({
                    "id": row[0],
                    "ts": row[1],
                    "tick": detail.get("tick", 0),
                    "drift": detail.get("drift", 0.0),
                    "evidence_count": detail.get("evidence_count", 0)
                })
            except Exception as e:
                logger.warning(f"Failed to parse trajectory event {row[0]}: {e}")
        
        return trajectory
    
    def _update_personality_from_evidence(
        self,
        session_id: str,
        evidence_text: str,
        self_model: SelfModel,
        tick_count: int = 0,
    ) -> bool:
        """
        基于证据更新人格激活度 (z_self[0:32])。
        与 _update_emotion_from_evidence / _update_motivation_from_evidence 平行。
        """
        try:
            if not self_model.personality_store:
                return False

            matches = self_model.personality_store.search_matching_patterns(
                evidence_text=evidence_text, top_k=3, similarity_threshold=0.45,
            )
            if not matches:
                return False

            from backend.personality_store import PERSONALITY_DIM
            personality_delta = np.zeros(PERSONALITY_DIM, dtype=np.float32)
            trigger_source = "pattern_match"
            last_pattern_id = ""

            for pattern, similarity in matches:
                if pattern.personality_vector is None:
                    continue
                activation = similarity * pattern.intensity * 0.5
                personality_delta += pattern.personality_vector * activation
                if pattern.trigger_condition:
                    trigger_source = pattern.trigger_condition
                last_pattern_id = pattern.id

            if not np.any(personality_delta != 0):
                return False

            self_model.personality_store.update_personality(
                session_id, personality_delta,
                trigger_source=trigger_source, pattern_id=last_pattern_id,
            )

            z_self = self_model.get_z_self(session_id)
            if z_self is not None and z_self.shape[0] >= PERSONALITY_DIM:
                z_prev_personality = z_self[:PERSONALITY_DIM].copy()
                state = self_model.personality_store.get_personality_state(session_id)
                z_self[:PERSONALITY_DIM] = state.personality_vector
                drift = float(np.linalg.norm(z_self[:PERSONALITY_DIM] - z_prev_personality))
                self_model._save_z_self(session_id, z_self, tick=tick_count, drift=drift)

            return True
        except Exception as e:
            logger.error(f"Failed to update personality from evidence: {e}", exc_info=True)
            return False

    def _update_emotion_from_evidence(
        self,
        session_id: str,
        evidence_text: str,
        self_model: SelfModel,
        tick_count: int = 0
    ) -> bool:
        """
        基于证据更新情感状态
        
        改进版：从 Emotion Memory 中检索匹配的模式并激活它们
        
        根据文档，情感更新触发源包括：
        - 规则执行结果（成功→快乐，失败→沮丧）
        - 用户反馈（正面→积极情感，负面→消极情感）
        - 任务难度（简单→自信，困难→焦虑）
        - 社会互动（被认可→自豪，被拒绝→羞愧）
        - 深度思考产出洞见（thinking_insight）→ 平静的笃定
        
        Returns:
            bool: 是否成功更新情感
        """
        if not self_model.emotion_store:
            return False
        
        try:
            # Pull emotion patterns via embedding match
            matching_patterns = self_model.emotion_store.search_matching_patterns(
                evidence_text=evidence_text,
                top_k=3,
                similarity_threshold=0.5
            )
            
            # Accumulator for emotion delta
            emotion_delta = np.zeros(16, dtype=np.float32)
            trigger_source = "unknown"
            pattern_activated = False
            
            # Activate matched patterns
            if matching_patterns:
                from backend.emotion_store import EMOTION_SUBSPACE_DIMS
                
                for pattern, similarity in matching_patterns:
                    if pattern.emotion_vector is not None:
                        # Weighted activation
                        activation_strength = similarity * pattern.intensity * 0.5  # damping factor

                        # Accumulate weighted vector
                        emotion_delta += pattern.emotion_vector * activation_strength
                        
                        # Track trigger label
                        if pattern.trigger_condition:
                            trigger_source = pattern.trigger_condition
                        else:
                            trigger_source = "pattern_match"
                        
                        pattern_activated = True
                        
                        logger.debug(
                            f"Activated emotion pattern: {pattern.id} "
                            f"(similarity={similarity:.3f}, intensity={pattern.intensity:.3f})"
                        )
            
            # [2026-04-08] No keyword fallback — embedding patterns only

            # Persist emotion store + mirror into z_self
            if np.any(emotion_delta != 0):
                self_model.emotion_store.update_emotion(
                    session_id,
                    emotion_delta,
                    trigger_source=trigger_source
                )
                
                # Mirror emotion subvector into z_self[32:48]
                z_self = self_model.get_z_self(session_id)
                if z_self is not None and z_self.shape[0] >= 48:
                    z_prev_emotion = z_self[32:48].copy()
                    emotion_state = self_model.emotion_store.get_emotion_state(session_id)
                    if emotion_state:
                        z_self[32:48] = emotion_state.emotion_vector
                        drift = float(np.linalg.norm(z_self[32:48] - z_prev_emotion))
                        self_model._save_z_self(session_id, z_self, tick=tick_count, drift=drift)
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to update emotion from evidence: {e}", exc_info=True)
            return False
    
    def _update_motivation_from_evidence(
        self,
        session_id: str,
        evidence_text: str,
        self_model: SelfModel,
        tick_count: int = 0
    ) -> bool:
        """
        基于证据更新动机状态
        
        改进版：从 Motivation Memory 中检索匹配的模式并激活它们
        
        根据文档，动机更新触发源包括：
        - 任务完成（成功→成就动机强化）
        - 用户反馈（正面→关系动机强化）
        - 新知识学习（探索动机强化）
        - 风险事件（安全动机强化）
        - 思考确认存在（cogito_drive）→ 主动寻找推理任务
        
        Returns:
            bool: 是否成功更新动机
        """
        if not self_model.motivation_store:
            return False
        
        try:
            # Pull motivation patterns via embedding match
            matching_patterns = self_model.motivation_store.search_matching_patterns(
                evidence_text=evidence_text,
                top_k=3,
                similarity_threshold=0.5
            )
            
            # Accumulator for motivation delta
            motivation_delta = np.zeros(16, dtype=np.float32)
            satisfaction_source = "unknown"
            pattern_activated = False
            
            # Activate matched patterns
            if matching_patterns:
                from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
                
                for pattern, similarity in matching_patterns:
                    if pattern.motivation_vector is not None:
                        # Weighted activation
                        activation_strength = similarity * pattern.intensity * 0.5  # damping factor

                        # Accumulate weighted vector
                        motivation_delta += pattern.motivation_vector * activation_strength
                        
                        # Track trigger label
                        if pattern.trigger_condition:
                            satisfaction_source = pattern.trigger_condition
                        else:
                            satisfaction_source = "pattern_match"
                        
                        pattern_activated = True
                        
                        logger.debug(
                            f"Activated motivation pattern: {pattern.id} "
                            f"(similarity={similarity:.3f}, intensity={pattern.intensity:.3f})"
                        )
            
            # [2026-04-08] No keyword fallback — embedding patterns only

            # Persist motivation store + mirror into z_self
            if np.any(motivation_delta != 0):
                self_model.motivation_store.update_motivation(
                    session_id,
                    motivation_delta,
                    satisfaction_source=satisfaction_source
                )
                
                # Mirror motivation subvector into z_self[48:64]
                z_self = self_model.get_z_self(session_id)
                if z_self is not None and z_self.shape[0] >= 64:
                    z_prev_motivation = z_self[48:64].copy()
                    motivation_state = self_model.motivation_store.get_motivation_state(session_id)
                    if motivation_state:
                        z_self[48:64] = motivation_state.motivation_vector
                        drift = float(np.linalg.norm(z_self[48:64] - z_prev_motivation))
                        self_model._save_z_self(session_id, z_self, tick=tick_count, drift=drift)
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to update motivation from evidence: {e}", exc_info=True)
            return False

