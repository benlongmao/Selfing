#!/usr/bin/env python3
"""
Technical indicators (Technical Indicators)

This is not a bare "indicator calculator" — it packages **technical analysis**
so the agent can reason about price action using trends, momentum, and volume.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Technical analysis for the agent: trend, momentum, and coarse signals
    over OHLCV-style history rows.
    """

    def __init__(self):
        """Initialize analyzer state."""
        logger.info("Agent technical analysis capability initialized")

    def analyze_stock(self, history_data: List[Dict]) -> Dict:
        """
        Run technical analysis on historical bars.

        Steps:
        1. Observe price structure
        2. Infer short/medium trend via moving averages
        3. Momentum: MACD, RSI, volume
        4. Heuristic buy/sell/hold scoring

        Returns:
            Structured analysis dict for the agent.
        """
        if not history_data or len(history_data) < 20:
            return {
                "status": "insufficient_data",
                "message": "Insufficient history: at least 20 trading days are required for technical analysis."
            }

        df = pd.DataFrame(history_data)

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df),
            "analysis_type": "technical"
        }

        trend_analysis = self._analyze_trend(df)
        analysis["trend"] = trend_analysis

        momentum = self._analyze_momentum(df)
        analysis["momentum"] = momentum

        signals = self._generate_signals(df, trend_analysis, momentum)
        analysis["signals"] = signals

        conclusion = self._make_conclusion(trend_analysis, momentum, signals)
        analysis["conclusion"] = conclusion

        return analysis

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        Classify short vs longer-term trend using 5/20/60 MA when available.
        """
        close_col = self._get_close_column(df)

        ma5 = df[close_col].rolling(window=5).mean()
        ma20 = df[close_col].rolling(window=20).mean()
        ma60 = df[close_col].rolling(window=60).mean() if len(df) >= 60 else None

        current_price = df[close_col].iloc[-1]

        trend = {
            "current_price": float(current_price),
            "ma5": float(ma5.iloc[-1]) if not pd.isna(ma5.iloc[-1]) else None,
            "ma20": float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else None,
            "ma60": float(ma60.iloc[-1]) if ma60 is not None and not pd.isna(ma60.iloc[-1]) else None,
        }

        if trend["ma5"] and trend["ma20"]:
            if current_price > trend["ma5"] > trend["ma20"]:
                trend["short_term"] = "strong uptrend"
                trend["short_term_signal"] = "bullish"
            elif current_price < trend["ma5"] < trend["ma20"]:
                trend["short_term"] = "weak downtrend"
                trend["short_term_signal"] = "bearish"
            else:
                trend["short_term"] = "choppy / range"
                trend["short_term_signal"] = "neutral"

        if trend["ma20"] and trend["ma60"]:
            if trend["ma20"] > trend["ma60"]:
                trend["long_term"] = "upward alignment"
                trend["long_term_signal"] = "bullish"
            elif trend["ma20"] < trend["ma60"]:
                trend["long_term"] = "downward alignment"
                trend["long_term_signal"] = "bearish"
            else:
                trend["long_term"] = "unclear"
                trend["long_term_signal"] = "neutral"

        return trend

    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """
        MACD, RSI, and simple volume pressure vs 5d average.
        """
        close_col = self._get_close_column(df)
        volume_col = self._get_volume_column(df)

        momentum: Dict = {}

        macd_data = self._calculate_macd(df[close_col])
        if macd_data:
            momentum["macd"] = macd_data

            if macd_data["macd"] > 0 and macd_data["signal"] > 0:
                if macd_data["macd"] > macd_data["signal"]:
                    momentum["macd_interpretation"] = "Positive momentum; continuation possible."
                else:
                    momentum["macd_interpretation"] = "Momentum fading; watch for pullback."
            else:
                if macd_data["macd"] > macd_data["signal"]:
                    momentum["macd_interpretation"] = "Downside slowing; possible basing."
                else:
                    momentum["macd_interpretation"] = "Downside pressure; stay cautious."

        rsi = self._calculate_rsi(df[close_col])
        if rsi is not None:
            momentum["rsi"] = float(rsi)

            if rsi > 70:
                momentum["rsi_interpretation"] = "Overbought; pullback risk."
                momentum["rsi_signal"] = "overbought"
            elif rsi < 30:
                momentum["rsi_interpretation"] = "Oversold; bounce possible."
                momentum["rsi_signal"] = "oversold"
            else:
                momentum["rsi_interpretation"] = "Neutral band; need more context."
                momentum["rsi_signal"] = "neutral"

        if volume_col and volume_col in df.columns:
            volume_ma5 = df[volume_col].rolling(window=5).mean()
            current_volume = df[volume_col].iloc[-1]
            avg_volume = volume_ma5.iloc[-1]

            if not pd.isna(avg_volume) and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                momentum["volume_ratio"] = float(volume_ratio)

                if volume_ratio > 2:
                    momentum["volume_interpretation"] = "High volume; attention on participation."
                elif volume_ratio > 1.5:
                    momentum["volume_interpretation"] = "Above-average volume."
                elif volume_ratio < 0.5:
                    momentum["volume_interpretation"] = "Light volume; cautious participation."
                else:
                    momentum["volume_interpretation"] = "Typical volume."

        return momentum

    def _generate_signals(self, df: pd.DataFrame, trend: Dict, momentum: Dict) -> Dict:
        """
        Score bullish vs bearish heuristics into buy/sell/hold.
        """
        signals: Dict = {
            "timestamp": datetime.now().isoformat(),
            "signals": []
        }

        buy_score = 0
        buy_reasons: List[str] = []

        if trend.get("short_term_signal") == "bullish":
            buy_score += 2
            buy_reasons.append("Short-term trend aligned up")

        if trend.get("long_term_signal") == "bullish":
            buy_score += 2
            buy_reasons.append("Longer-term MAs favor bulls")

        if momentum.get("rsi_signal") == "oversold":
            buy_score += 2
            buy_reasons.append("RSI oversold: potential bounce")

        if momentum.get("macd", {}).get("histogram", 0) > 0:
            buy_score += 1
            buy_reasons.append("MACD histogram > 0")

        sell_score = 0
        sell_reasons: List[str] = []

        if trend.get("short_term_signal") == "bearish":
            sell_score += 2
            sell_reasons.append("Short-term trend aligned down")

        if trend.get("long_term_signal") == "bearish":
            sell_score += 2
            sell_reasons.append("Longer-term MAs favor bears")

        if momentum.get("rsi_signal") == "overbought":
            sell_score += 2
            sell_reasons.append("RSI overbought: pullback risk")

        if momentum.get("macd", {}).get("histogram", 0) < 0:
            sell_score += 1
            sell_reasons.append("MACD histogram < 0")

        if buy_score >= 4:
            signals["primary_signal"] = "buy"
            signals["signal_strength"] = min(buy_score / 7, 1.0)
            signals["reasons"] = buy_reasons
        elif sell_score >= 4:
            signals["primary_signal"] = "sell"
            signals["signal_strength"] = min(sell_score / 7, 1.0)
            signals["reasons"] = sell_reasons
        else:
            signals["primary_signal"] = "hold"
            signals["signal_strength"] = 0.5
            signals["reasons"] = ["Mixed / weak; prefer to wait."]

        return signals

    def _make_conclusion(self, trend: Dict, momentum: Dict, signals: Dict) -> Dict:
        """
        High-level risk note plus plain-language advice (not financial advice).
        """
        conclusion = {
            "timestamp": datetime.now().isoformat(),
        }

        risk_factors: List[str] = []
        if trend.get("short_term_signal") == "bearish":
            risk_factors.append("Short-term trend not supportive")
        if momentum.get("rsi_signal") == "overbought":
            risk_factors.append("Short-term pullback risk (RSI)")
        if trend.get("long_term_signal") == "bearish":
            risk_factors.append("Longer-term structure weak")

        conclusion["risk_factors"] = risk_factors
        n = len(risk_factors)
        conclusion["risk_level"] = "high" if n >= 2 else "medium" if n == 1 else "low"

        signal_type = signals.get("primary_signal")
        signal_strength = signals.get("signal_strength", 0)

        if signal_type == "buy":
            if signal_strength > 0.7:
                conclusion["recommendation"] = "buy"
                conclusion["confidence"] = "higher"
                conclusion["advice"] = "Technicals lean bullish; still combine with fundamentals and risk limits."
            else:
                conclusion["recommendation"] = "wait / probe"
                conclusion["confidence"] = "medium"
                conclusion["advice"] = "Some buy-side tilt; consider smaller size or clearer confirmation."
        elif signal_type == "sell":
            if signal_strength > 0.7:
                conclusion["recommendation"] = "reduce / sell"
                conclusion["confidence"] = "higher"
                conclusion["advice"] = "Technicals lean bearish; consider de-risking or hard stops."
            else:
                conclusion["recommendation"] = "wait / trim"
                conclusion["confidence"] = "medium"
                conclusion["advice"] = "Some sell-side signals; long holders may scale down."
        else:
            conclusion["recommendation"] = "hold / watch"
            conclusion["confidence"] = "medium"
            conclusion["advice"] = "No clean edge; default is patience."

        conclusion["s44_note"] = (
            "Technical analysis is one lens. Real decisions also need fundamentals, industry context, and valuation. "
            "I keep improving; this is not personalized investment advice."
        )

        return conclusion

    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Optional[Dict]:
        """Compute MACD (line, signal, histogram)."""
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line

            return {
                "macd": float(macd.iloc[-1]),
                "signal": float(signal_line.iloc[-1]),
                "histogram": float(histogram.iloc[-1]),
                "status": "golden_cross" if histogram.iloc[-1] > 0 else "dead_cross"
            }
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period=14) -> Optional[float]:
        """Compute last RSI value."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1]
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return None

    def _get_close_column(self, df: pd.DataFrame) -> str:
        """
        Resolve a close/price column (CN labels kept for mixed feeds).
        """
        possible_names = ['收盘', 'close', 'Close', '最新价', 'price']
        for name in possible_names:
            if name in df.columns:
                return name
        return df.columns[-1]

    def _get_volume_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Resolve volume column (CN + EN tokens).
        """
        possible_names = ['成交量', 'volume', 'Volume', 'vol']
        for name in possible_names:
            if name in df.columns:
                return name
        return None


def get_technical_analyzer():
    """Return a process-wide TechnicalAnalyzer instance."""
    if not hasattr(get_technical_analyzer, "_instance"):
        get_technical_analyzer._instance = TechnicalAnalyzer()
    return get_technical_analyzer._instance


def get_tool_definition() -> Dict:
    """
    OpenAI-style tool: technical analysis over history rows from get_stock_info.
    """
    return {
        "type": "function",
        "function": {
            "name": "technical_analysis",
            "description": """Technical analysis on recent price history (trend, momentum, coarse signals).

Use after fetching bars via get_stock_info. This interprets structure — not a guaranteed forecast.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_data": {
                        "type": "array",
                        "description": "K-line history list (e.g. history.data from get_stock_info)",
                        "items": {"type": "object"}
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Ticker for labeling"
                    }
                },
                "required": ["history_data"]
            }
        }
    }
