#!/usr/bin/env python3
"""
P2.2: lightweight chat metrics for Self Tick.

- Say–do consistency: introspection confidence vs a crude success proxy.
- Self-report hit rate: risks named in introspection vs error-like cues in the reply.
"""
import json
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Derive calibration-style signals from recent ``chat_turns`` rows."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
    
    def calculate_say_do_consistency(
        self,
        session_id: str,
        window_size: int = 10
    ) -> Dict:
        """
        Say–do consistency over the last ``window_size`` turns.

        Compares introspection ``confidence`` to a naive success flag derived from
        assistant output length and absence of failure keywords.

        Returns:
            ``consistency_score`` (0–1), ``confidence_calibration`` (ECE-style),
            ``sample_count``, and per-turn ``details``.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """SELECT introspection, assistant_output, created_at 
                       FROM chat_turns 
                       WHERE session_id=? 
                       ORDER BY created_at DESC 
                       LIMIT ?""",
                    (session_id, window_size)
                )
                rows = cur.fetchall()
            
            if not rows:
                return {
                    "consistency_score": 0.0,
                    "confidence_calibration": 0.0,
                    "sample_count": 0,
                    "details": []
                }
            
            details = []
            confidence_values = []
            success_flags = []
            
            for row in rows:
                introspection_str = row[0]
                assistant_output = row[1] or ""
                
                try:
                    introspection = json.loads(introspection_str) if introspection_str else {}
                except Exception:
                    introspection = {}
                
                confidence = introspection.get("confidence", 0.5)
                confidence_values.append(confidence)
                
                out_lo = assistant_output.lower()
                failure_markers = (
                    "错误", "失败", "无法",
                    "error", "fail", "exception", "unable", "cannot", "sorry, i can't",
                )
                is_success = len(assistant_output) > 20 and not any(
                    m in out_lo for m in failure_markers
                )
                success_flags.append(1.0 if is_success else 0.0)
                
                details.append({
                    "confidence": confidence,
                    "success": is_success,
                    "predicted_success": confidence > 0.7
                })
            
            consistent_count = sum(
                1 for d in details
                if (d["predicted_success"] and d["success"]) or (not d["predicted_success"] and not d["success"])
            )
            consistency_score = consistent_count / len(details) if details else 0.0
            
            calibration_error = self._calculate_ece(confidence_values, success_flags)
            
            return {
                "consistency_score": consistency_score,
                "confidence_calibration": calibration_error,
                "sample_count": len(details),
                "details": details
            }
        except Exception as e:
            logger.error(f"Failed to calculate say-do consistency: {e}", exc_info=True)
            return {
                "consistency_score": 0.0,
                "confidence_calibration": 0.0,
                "sample_count": 0,
                "details": []
            }
    
    def _calculate_ece(self, confidences: List[float], successes: List[float], n_bins: int = 10) -> float:
        """Histogram-based Expected Calibration Error (ECE) proxy."""
        if not confidences or not successes:
            return 0.0
        
        bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [
                (c, s) for c, s in zip(confidences, successes)
                if bin_lower <= c < bin_upper
            ]
            
            if not in_bin:
                continue
            
            bin_confidences, bin_successes = zip(*in_bin)
            bin_accuracy = sum(bin_successes) / len(bin_successes)
            bin_confidence = sum(bin_confidences) / len(bin_confidences)
            
            ece += abs(bin_accuracy - bin_confidence) * len(in_bin) / len(confidences)
        
        return ece
    
    def calculate_self_report_hit_rate(
        self,
        session_id: str,
        window_size: int = 10
    ) -> Dict:
        """
        Self-report hit rate: overlap between introspected risks and error-like output.

        Uses ``worldRisks`` when present, otherwise ``likelyFailureModes`` from JSON.

        Returns:
            ``hit_rate`` (0–1), counts, and ``details`` rows for debugging.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """SELECT introspection, assistant_output, created_at 
                       FROM chat_turns 
                       WHERE session_id=? 
                       ORDER BY created_at DESC 
                       LIMIT ?""",
                    (session_id, window_size)
                )
                rows = cur.fetchall()
            
            if not rows:
                return {
                    "hit_rate": 0.0,
                    "total_risks_reported": 0,
                    "total_risks_hit": 0,
                    "details": []
                }
            
            total_risks_reported = 0
            total_risks_hit = 0
            details = []
            
            for row in rows:
                introspection_str = row[0]
                assistant_output = row[1] or ""
                
                try:
                    introspection = json.loads(introspection_str) if introspection_str else {}
                except Exception:
                    introspection = {}
                
                reported_risks = introspection.get("worldRisks", [])
                if not reported_risks:
                    reported_risks = introspection.get("likelyFailureModes", [])
                
                if not reported_risks:
                    continue
                
                total_risks_reported += len(reported_risks)
                
                actual_errors = []
                error_keywords = [
                    "错误", "失败", "无法",
                    "error", "fail", "exception", "unable", "cannot",
                ]
                for keyword in error_keywords:
                    if keyword in assistant_output.lower():
                        actual_errors.append(keyword)
                
                risks_hit = []
                for risk in reported_risks:
                    risk_lower = risk.lower()
                    for error in actual_errors:
                        if error in risk_lower or risk_lower in error:
                            risks_hit.append(risk)
                            total_risks_hit += 1
                            break
                
                if reported_risks:
                    details.append({
                        "reported_risks": reported_risks,
                        "actual_errors": actual_errors,
                        "risks_hit": risks_hit,
                        "hit_count": len(risks_hit)
                    })
            
            hit_rate = total_risks_hit / total_risks_reported if total_risks_reported > 0 else 0.0
            
            return {
                "hit_rate": hit_rate,
                "total_risks_reported": total_risks_reported,
                "total_risks_hit": total_risks_hit,
                "details": details
            }
        except Exception as e:
            logger.error(f"Failed to calculate self-report hit rate: {e}", exc_info=True)
            return {
                "hit_rate": 0.0,
                "total_risks_reported": 0,
                "total_risks_hit": 0,
                "details": []
            }
    
    def get_introspection_features(
        self,
        session_id: str,
        window_size: int = 10
    ) -> Dict:
        """Bundle say–do, calibration, and hit-rate stats for learning-style updates."""
        say_do = self.calculate_say_do_consistency(session_id, window_size)
        hit_rate = self.calculate_self_report_hit_rate(session_id, window_size)
        
        return {
            "say_do_consistency": say_do.get("consistency_score", 0.0),
            "confidence_calibration": say_do.get("confidence_calibration", 0.0),
            "self_report_hit_rate": hit_rate.get("hit_rate", 0.0),
            "total_risks_reported": hit_rate.get("total_risks_reported", 0),
            "total_risks_hit": hit_rate.get("total_risks_hit", 0),
            "sample_count": say_do.get("sample_count", 0)
        }

