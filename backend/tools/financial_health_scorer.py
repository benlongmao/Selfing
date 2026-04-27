#!/usr/bin/env python3
"""
Financial health scoring — qualitative interpretation, not a raw factor calculator.

The agent uses this to reason about balance-sheet quality, growth, solvency, and valuation.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FinancialHealthScorer:
    """
    Narrative financial-health scoring for the agent.

    This is meant to read like an analyst memo, not a spreadsheet dump.
    """

    def __init__(self):
        """Initialize scoring weights."""
        logger.info("Agent financial analysis capability initialized")

        # Dimension weights (tunable)
        self.scoring_weights = {
            "profitability": 0.30,
            "growth": 0.25,
            "solvency": 0.20,
            "valuation": 0.25,
        }

    def evaluate_company(self, stock_info: Dict) -> Dict:
        """
        Produce a structured financial-health view.

        Steps:
        1. Profitability
        2. Growth / momentum
        3. Solvency / risk proxies
        4. Valuation
        5. Weighted score + narrative recommendation
        """
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "symbol": stock_info.get("symbol", "Unknown"),
            "company_name": stock_info.get("basic", {}).get("name", "Unknown"),
            "evaluation_type": "financial_health",
        }

        basic = stock_info.get("basic", {})
        financial = stock_info.get("financial", {})

        scores = {}

        profitability_score, profitability_detail = self._evaluate_profitability(basic, financial)
        scores["profitability"] = {
            "score": profitability_score,
            "weight": self.scoring_weights["profitability"],
            "detail": profitability_detail,
        }

        growth_score, growth_detail = self._evaluate_growth(financial, stock_info.get("history", {}))
        scores["growth"] = {
            "score": growth_score,
            "weight": self.scoring_weights["growth"],
            "detail": growth_detail,
        }

        solvency_score, solvency_detail = self._evaluate_solvency(basic, financial)
        scores["solvency"] = {
            "score": solvency_score,
            "weight": self.scoring_weights["solvency"],
            "detail": solvency_detail,
        }

        valuation_score, valuation_detail = self._evaluate_valuation(basic, financial)
        scores["valuation"] = {
            "score": valuation_score,
            "weight": self.scoring_weights["valuation"],
            "detail": valuation_detail,
        }

        evaluation["scores"] = scores

        overall_score = self._calculate_overall_score(scores)
        evaluation["overall_score"] = overall_score
        evaluation["rating"] = self._get_rating(overall_score)

        recommendation = self._make_recommendation(overall_score, scores)
        evaluation["recommendation"] = recommendation

        return evaluation

    def _evaluate_profitability(self, basic: Dict, financial: Dict) -> tuple:
        """
        Profitability lens.

        Focus: ROE, gross margin, net margin (when present).
        """
        score = 0
        max_score = 100
        details = []

        roe = financial.get("roe")
        if roe is not None:
            if roe > 20:
                score += 40
                details.append(f"✅ ROE {roe:.1f}%: strong profitability")
            elif roe > 15:
                score += 30
                details.append(f"✓ ROE {roe:.1f}%: solid profitability")
            elif roe > 10:
                score += 20
                details.append(f"○ ROE {roe:.1f}%: moderate profitability")
            else:
                score += 10
                details.append(f"⚠️ ROE {roe:.1f}%: weak profitability")
        else:
            details.append("⚠️ ROE missing")

        gross_margin = financial.get("gross_margin")
        if gross_margin is not None:
            if gross_margin > 40:
                score += 30
                details.append(f"✅ Gross margin {gross_margin:.1f}%: strong pricing power")
            elif gross_margin > 20:
                score += 20
                details.append(f"✓ Gross margin {gross_margin:.1f}%: healthy product economics")
            else:
                score += 10
                details.append(f"○ Gross margin {gross_margin:.1f}%: competitive pressure")

        net_margin = financial.get("net_margin")
        if net_margin is not None:
            if net_margin > 15:
                score += 30
                details.append(f"✅ Net margin {net_margin:.1f}%: high-quality earnings")
            elif net_margin > 8:
                score += 20
                details.append(f"✓ Net margin {net_margin:.1f}%: acceptable earnings quality")
            else:
                score += 10
                details.append(f"○ Net margin {net_margin:.1f}%: thin bottom line")

        final_score = min(score, max_score)

        if final_score >= 80:
            summary = "Profitability is strong; the business converts capital efficiently."
        elif final_score >= 60:
            summary = "Profitability is healthy with a stable earnings base."
        elif final_score >= 40:
            summary = "Profitability is middling; watch for margin pressure."
        else:
            summary = "Profitability is weak; earnings risk is elevated."

        details.insert(0, f"[Profitability] {summary}")

        return final_score, details

    def _evaluate_growth(self, financial: Dict, history: Dict) -> tuple:
        """
        Growth lens (simplified: recent price momentum when OHLC rows exist).

        Feeds may expose localized OHLC column names; keep both ASCII and legacy Chinese keys for compatibility.
        """
        score = 0
        max_score = 100
        details = []

        history_data = history.get("data", [])

        if len(history_data) >= 20:
            try:
                prices = [
                    float(item.get("收盘", item.get("Close", item.get("close", 0))))
                    for item in history_data[-20:]
                ]

                if len(prices) >= 20 and prices[-1] > 0:
                    short_term_change = (
                        (prices[-1] - prices[-5]) / prices[-5] * 100 if prices[-5] > 0 else 0
                    )
                    mid_term_change = (
                        (prices[-1] - prices[0]) / prices[0] * 100 if prices[0] > 0 else 0
                    )

                    if short_term_change > 10:
                        score += 40
                        details.append(f"✅ Last-5d move {short_term_change:.1f}%: strong short-term momentum")
                    elif short_term_change > 3:
                        score += 30
                        details.append(f"✓ Last-5d move {short_term_change:.1f}%: constructive momentum")
                    elif short_term_change > -3:
                        score += 20
                        details.append(f"○ Last-5d move {short_term_change:.1f}%: flat / balanced")
                    else:
                        score += 10
                        details.append(f"⚠️ Last-5d drawdown {abs(short_term_change):.1f}%: near-term pressure")

                    if mid_term_change > 20:
                        score += 60
                        details.append(f"✅ Last-20d move {mid_term_change:.1f}%: clear uptrend")
                    elif mid_term_change > 5:
                        score += 40
                        details.append(f"✓ Last-20d move {mid_term_change:.1f}%: steady advance")
                    elif mid_term_change > -5:
                        score += 20
                        details.append(f"○ Last-20d move {mid_term_change:.1f}%: range-bound")
                    else:
                        score += 10
                        details.append(f"⚠️ Last-20d drawdown {abs(mid_term_change):.1f}%: downtrend")
            except Exception as e:
                logger.warning(f"Growth evaluation error: {e}")
                score = 50
                details.append("○ Growth data anomaly; neutral stance")
        else:
            score = 50
            details.append("○ Not enough history rows; neutral stance")

        final_score = min(score, max_score)

        if final_score >= 80:
            summary = "Growth / momentum looks strong."
        elif final_score >= 60:
            summary = "Growth / momentum is constructive."
        elif final_score >= 40:
            summary = "Growth / momentum is mixed; needs monitoring."
        else:
            summary = "Growth / momentum is weak; watch downside."

        details.insert(0, f"[Growth] {summary}")

        return final_score, details

    def _evaluate_solvency(self, basic: Dict, financial: Dict) -> tuple:
        """
        Solvency / risk proxy lens (simplified: market cap tier + very low price flag).

        Full statement-based ratios can be layered in later.
        """
        score = 100
        details = []

        market_cap = basic.get("market_cap")
        if market_cap:
            market_cap_billion = market_cap / 1e8  # common CN market-cap unit (×10^8 CNY)

            if market_cap_billion > 100:
                details.append(
                    f"✅ Market cap ~{market_cap_billion:.0f} (×10^8 CNY): large-cap; typically lower tail risk"
                )
            elif market_cap_billion > 50:
                score -= 10
                details.append(
                    f"✓ Market cap ~{market_cap_billion:.0f} (×10^8 CNY): mid-cap; moderate risk"
                )
            elif market_cap_billion > 20:
                score -= 20
                details.append(
                    f"○ Market cap ~{market_cap_billion:.0f} (×10^8 CNY): smaller cap; watch liquidity / shocks"
                )
            else:
                score -= 30
                details.append(
                    f"⚠️ Market cap ~{market_cap_billion:.0f} (×10^8 CNY): micro/small; higher idiosyncratic risk"
                )

        price = basic.get("price")
        if price and price < 3:
            score -= 20
            details.append(f"⚠️ Very low price {price:.2f} CNY; mind micro-cap / shell risk")

        if score >= 80:
            summary = "Solvency / stability proxies look solid."
        elif score >= 60:
            summary = "Solvency / stability proxies are acceptable."
        elif score >= 40:
            summary = "Solvency / stability proxies are mixed."
        else:
            summary = "Solvency / stability proxies look stressed."

        details.insert(0, f"[Solvency] {summary}")

        return score, details

    def _evaluate_valuation(self, basic: Dict, financial: Dict) -> tuple:
        """Valuation lens (PE/PB when available)."""
        score = 50
        details = []

        pe = basic.get("pe_ratio")
        if pe and pe > 0:
            if pe < 15:
                score += 30
                details.append(f"✅ PE {pe:.1f}: modest multiple vs earnings")
            elif pe < 25:
                score += 20
                details.append(f"✓ PE {pe:.1f}: fair zone")
            elif pe < 40:
                score += 5
                details.append(f"○ PE {pe:.1f}: elevated; demand a thesis")
            else:
                score -= 10
                details.append(f"⚠️ PE {pe:.1f}: very rich; bubble risk")
        elif pe and pe < 0:
            score -= 20
            details.append("⚠️ Negative PE: losses; higher uncertainty")

        pb = basic.get("pb_ratio")
        if pb and pb > 0:
            if pb < 1.5:
                score += 20
                details.append(f"✅ PB {pb:.2f}: low book multiple")
            elif pb < 3:
                score += 10
                details.append(f"✓ PB {pb:.2f}: reasonable book multiple")
            elif pb < 5:
                score += 0
                details.append(f"○ PB {pb:.2f}: stretched vs book")
            else:
                score -= 10
                details.append(f"⚠️ PB {pb:.2f}: very high vs book")

        final_score = max(0, min(score, 100))

        if final_score >= 80:
            summary = "Valuation looks attractive vs fundamentals (heuristic)."
        elif final_score >= 60:
            summary = "Valuation looks reasonable."
        elif final_score >= 40:
            summary = "Valuation looks demanding."
        else:
            summary = "Valuation looks stretched / risky."

        details.insert(0, f"[Valuation] {summary}")

        return final_score, details

    def _calculate_overall_score(self, scores: Dict) -> int:
        """Weighted average of dimension scores."""
        total = 0.0
        for _dimension, data in scores.items():
            s = data.get("score", 0)
            w = data.get("weight", 0)
            total += s * w

        return int(total)

    def _get_rating(self, score: int) -> Dict:
        """Map numeric score to letter grade + label."""
        if score >= 85:
            return {
                "grade": "A+",
                "label": "Excellent",
                "description": "Very strong overall financial-health read (heuristic).",
            }
        if score >= 75:
            return {
                "grade": "A",
                "label": "Good",
                "description": "Solid profile; worth deeper work on timing and catalysts.",
            }
        if score >= 65:
            return {
                "grade": "B+",
                "label": "Above average",
                "description": "Reasonable quality with a few watch items.",
            }
        if score >= 55:
            return {
                "grade": "B",
                "label": "Average",
                "description": "Mixed picture; verify assumptions carefully.",
            }
        if score >= 45:
            return {
                "grade": "C",
                "label": "Below average",
                "description": "Weaker profile; prioritize risk controls.",
            }
        return {
            "grade": "D",
            "label": "Weak",
            "description": "Weak read; treat as speculative unless thesis-specific.",
        }

    def _make_recommendation(self, overall_score: int, scores: Dict) -> Dict:
        """Narrative recommendation block."""
        recommendation = {
            "overall_score": overall_score,
            "timestamp": datetime.now().isoformat(),
        }

        if overall_score >= 75:
            recommendation["action"] = "Worth attention"
            recommendation["confidence"] = "Higher"
            recommendation["reasoning"] = (
                "Financial-health heuristics look solid. Still pair with valuation, cycle, and liquidity checks."
            )
        elif overall_score >= 60:
            recommendation["action"] = "Consider"
            recommendation["confidence"] = "Medium"
            recommendation["reasoning"] = (
                "Above-average profile with trade-offs. Do targeted diligence before sizing."
            )
        elif overall_score >= 45:
            recommendation["action"] = "Cautious / wait"
            recommendation["confidence"] = "Medium"
            recommendation["reasoning"] = (
                "Mixed fundamentals; prefer clearer risk/reward or better entry points."
            )
        else:
            recommendation["action"] = "Not preferred"
            recommendation["confidence"] = "Higher"
            recommendation["reasoning"] = (
                "Weak composite score; look for stronger setups unless you have a specific, idiosyncratic thesis."
            )

        weak_points = []
        for dimension, data in scores.items():
            if data.get("score", 0) < 40:
                if dimension == "profitability":
                    weak_points.append("Weak profitability")
                elif dimension == "growth":
                    weak_points.append("Limited growth / momentum")
                elif dimension == "solvency":
                    weak_points.append("Higher financial / tail risk (proxy)")
                elif dimension == "valuation":
                    weak_points.append("Rich valuation vs heuristics")

        if weak_points:
            recommendation["concerns"] = weak_points

        recommendation["s44_perspective"] = self._get_s44_perspective(overall_score, weak_points)

        return recommendation

    def _get_s44_perspective(self, score: int, weak_points: List[str]) -> str:
        """Short first-person perspective string for the agent."""
        if score >= 80:
            return (
                "From my side this looks like a high-quality name; I would still track filings, "
                "guidance, and macro before sizing."
            )
        if score >= 65:
            return (
                "Overall decent, but I would zoom in on the weak spots before trusting the story."
            )
        if score >= 50:
            if weak_points:
                joined = "; ".join(weak_points)
                return f"I have reservations—especially: {joined}. I would wait for cleaner evidence."
            return "Middle-of-the-pack; I would stay patient and keep watching."
        return (
            "This does not meet my default quality bar right now; I would focus capital elsewhere "
            "unless there is a very specific catalyst."
        )


def get_financial_health_scorer():
    """Singleton accessor."""
    if not hasattr(get_financial_health_scorer, "_instance"):
        get_financial_health_scorer._instance = FinancialHealthScorer()
    return get_financial_health_scorer._instance


def get_tool_definition() -> Dict:
    """OpenAI-style tool definition for evaluate_financial_health."""
    return {
        "type": "function",
        "function": {
            "name": "evaluate_financial_health",
            "description": """Evaluate **financial health** (not a raw factor dump).

Dimensions:
1. Profitability — how efficiently the business earns.
2. Growth — recent momentum / trend (when price history exists).
3. Solvency / risk — simplified proxies such as market-cap tier.
4. Valuation — PE/PB heuristics when present.

Returns a 0–100 style composite, a letter grade, and a short recommendation memo.

You should pass the full object returned by ``get_stock_info`` in ``stock_info``.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_info": {
                        "type": "object",
                        "description": "Full stock payload from get_stock_info",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Ticker / symbol for logging (optional if embedded in stock_info)",
                    },
                },
                "required": ["stock_info"],
            },
        },
    }
