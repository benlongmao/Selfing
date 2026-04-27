#!/usr/bin/env python3
"""
Meaning generation layer: subjective readouts from ``z_self`` drift.

Turns numeric state deltas into short first-person experiential lines suitable
for prompts and operator-facing summaries.

Pipeline:
1. **Change detection** — compare current vs previous ``z_self``.
2. **Meaning extraction** — map deltas to coarse subspace semantics.
3. **Narrative generation** — stitch interpretation into a brief experiential line.
"""

import logging
import numpy as np
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class MeaningGenerationLayer:
    """Rule-based experiential narrative from ``z_self`` vectors (optional embedder hook)."""

    def __init__(self, db_path: str = "data.db", embedder = None):
        self.db_path = db_path
        self.embedder = embedder
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure ``meaning_records`` exists and run the legacy ``timestamp`` migration if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS meaning_records (
                        session_id TEXT,
                        timestamp TEXT,
                        z_self_snapshot TEXT, -- JSON
                        meaning_data TEXT,    -- JSON (extracted meaning struct)
                        narrative_text TEXT,  -- first-person narrative line
                        PRIMARY KEY (session_id, timestamp)
                    )
                """)
                
                # Legacy DBs: ensure ``timestamp`` column exists (one-time reshape).
                try:
                    cursor = conn.execute("PRAGMA table_info(meaning_records)")
                    columns = [info[1] for info in cursor.fetchall()]
                    if "timestamp" not in columns:
                        logger.info("Migrating meaning_records table...")
                        conn.execute("ALTER TABLE meaning_records RENAME TO meaning_records_old")
                        conn.execute("""
                            CREATE TABLE meaning_records (
                                session_id TEXT,
                                timestamp TEXT,
                                z_self_snapshot TEXT,
                                meaning_data TEXT,
                                narrative_text TEXT,
                                PRIMARY KEY (session_id, timestamp)
                            )
                        """)
                        now = datetime.now(timezone.utc).isoformat()
                        conn.execute(f"""
                            INSERT INTO meaning_records (session_id, timestamp, z_self_snapshot, meaning_data, narrative_text)
                            SELECT session_id, '{now}', z_self_snapshot, meaning_data, narrative_text 
                            FROM meaning_records_old
                        """)
                        conn.execute("DROP TABLE meaning_records_old")
                        logger.info("Migration completed.")
                except Exception as e:
                    logger.warning(f"Migration check failed: {e}")

                conn.commit()
        except Exception as e:
            logger.error(f"Failed to init/migrate meaning_records table: {e}")

    def extract_meaning(
        self, 
        current_z: np.ndarray, 
        prev_z: Optional[np.ndarray], 
        active_personas: List,
        recent_events: List[Dict]
    ) -> Dict:
        """Summarize how ``current_z`` diverges from ``prev_z`` plus a coarse event hook."""
        meaning = {
            "significant_change": False,
            "change_type": "stable",
            "affected_subspaces": [],
            "context_event": None
        }
        
        if prev_z is None:
            meaning["change_type"] = "birth"
            return meaning
            
        # Overall drift on the first 32 rule dimensions.
        dim = min(current_z.shape[0], prev_z.shape[0])
        rules_dim = 32
        
        curr_core = current_z[:rules_dim]
        prev_core = prev_z[:rules_dim]
        
        drift = np.linalg.norm(curr_core - prev_core)
        meaning["drift_magnitude"] = float(drift)
        
        if drift > 0.05:
            meaning["significant_change"] = True
            
        # Coarse subspace buckets (same 32-d slice).
        subspaces = {
            "safety": (0, 8),
            "epistemic": (8, 16),
            "style": (16, 24),
            "strategy": (24, 32)
        }
        
        max_delta = 0.0
        primary_subspace = None
        
        for name, (start, end) in subspaces.items():
            if end <= dim:
                delta = float(np.mean(current_z[start:end]) - np.mean(prev_z[start:end]))
                if abs(delta) > 0.02:
                    meaning["affected_subspaces"].append({
                        "name": name,
                        "delta": delta,
                        "direction": "increase" if delta > 0 else "decrease"
                    })
                    if abs(delta) > abs(max_delta):
                        max_delta = delta
                        primary_subspace = name
                        
        meaning["primary_subspace"] = primary_subspace
        
        if recent_events:
            # Caller supplies newest-first events; keep the latest type as context.
            meaning["context_event"] = recent_events[0].get("type", "unknown")
            
        return meaning

    def interpret(self, meaning: Dict, current_z: np.ndarray) -> str:
        """Turn structured ``meaning`` into a single first-person sentence."""
        if not meaning["significant_change"]:
            return "My internal state feels steady."

        subspace = meaning.get("primary_subspace")

        interpretation = ""

        if subspace == "safety":
            interpretation = "My sense of safety in the environment has shifted."
        elif subspace == "epistemic":
            interpretation = "My way of judging evidence and forming beliefs is adjusting."
        elif subspace == "style":
            interpretation = "My urge to express myself in a particular tone is changing."
        elif subspace == "strategy":
            interpretation = "My approach to tackling problems is evolving."
        else:
            interpretation = "I feel a subtle inner shift that is hard to name."

        return interpretation

    def generate_narrative(self, interpretation: str, current_z: np.ndarray, context: Dict) -> str:
        """Wrap ``interpretation`` with a light random prefix (template-only for now)."""
        import random
        prefixes = [
            "I sense that ",
            "A feeling surfaces: ",
            "My inner monitor notes ",
            "Right now, ",
        ]
        
        prefix = random.choice(prefixes)
        
        return f"{prefix}{interpretation}"

    def save_meaning_record(self, session_id: str, z_snapshot: Dict, meaning: Dict, narrative: str):
        """
        Persist a row to ``meaning_records`` only when drift is material.

        Skips steady-state lines such as ``My internal state feels steady`` to limit DB growth.
        """
        try:
            drift_magnitude = meaning.get("drift_magnitude", 0.0)
            significant_change = meaning.get("significant_change", False)

            # Record when drift exceeds 0.08 or ``significant_change`` is true.
            if drift_magnitude < 0.08 and not significant_change:
                logger.debug(f"Skipping meaning record: drift={drift_magnitude:.4f}, no significant change")
                return
            
            timestamp = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO meaning_records (session_id, timestamp, z_self_snapshot, meaning_data, narrative_text)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id, 
                    timestamp, 
                    json.dumps(z_snapshot), 
                    json.dumps(meaning), 
                    narrative
                ))
                conn.commit()
                logger.debug(f"Meaning record saved: drift={drift_magnitude:.4f}, narrative={narrative[:30]}...")
        except Exception as e:
            logger.error(f"Failed to save meaning record: {e}")
