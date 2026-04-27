#!/usr/bin/env python3
"""
Stock data tool — infrastructure for the agent’s investment analysis.

Data source order:
1. Tushare — CN / US / HK (requires token + points)
2. BaoStock — A-share fallback (free, no signup)

The agent is the decision-maker; these feeds only supply market context.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, date
import json
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _convert_to_json_serializable(obj):
    """Convert values to JSON-serializable forms."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj


def _dataframe_to_json_records(df: pd.DataFrame) -> List[Dict]:
    """Serialize a DataFrame to record dicts safe for JSON."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
        elif df_copy[col].dtype == 'object':
            try:
                df_copy[col] = df_copy[col].apply(
                    lambda x: x.isoformat() if isinstance(x, (date, datetime)) else x
                )
            except Exception:
                pass
    return df_copy.to_dict('records')


# Tushare
TUSHARE_AVAILABLE = False
ts = None
try:
    import tushare as ts
    tushare_token = os.environ.get("TUSHARE_TOKEN")
    if tushare_token:
        ts.set_token(tushare_token)
        TUSHARE_AVAILABLE = True
        logger.info("Tushare is available with token")
    else:
        logger.warning("Tushare installed but no TUSHARE_TOKEN set.")
except ImportError:
    logger.warning("Tushare not installed.")

# BaoStock — free A-share data
BAOSTOCK_AVAILABLE = False
bs = None
try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
    logger.info("BaoStock is available (free A-share data)")
except ImportError:
    logger.warning("BaoStock not installed. Run: pip install baostock")


class StockDataTool:
    """
    Stock data provider.

    Policy:
    - A-shares: Tushare first, then BaoStock if Tushare is unavailable
    - US / HK: Tushare only (points / permissions)
    """

    def __init__(self):
        self.use_tushare = TUSHARE_AVAILABLE
        self.use_baostock = BAOSTOCK_AVAILABLE
        self.ts_pro = None
        self._bs_logged_in = False

        if self.use_tushare:
            try:
                self.ts_pro = ts.pro_api()
            except Exception as e:
                logger.error(f"Failed to init Tushare pro_api: {e}")
                self.use_tushare = False

        if not self.use_tushare and not self.use_baostock:
            logger.error("No stock data source available!")

    def _ensure_baostock_login(self):
        """Ensure BaoStock session is logged in before queries."""
        if not self.use_baostock:
            return False
        if not self._bs_logged_in:
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    self._bs_logged_in = True
                    logger.debug("BaoStock login success")
                else:
                    logger.error(f"BaoStock login failed: {lg.error_msg}")
                    return False
            except Exception as e:
                logger.error(f"BaoStock login error: {e}")
                return False
        return True

    def get_stock_info(
        self,
        symbol: str,
        market: str = "CN",
        include_history: bool = True,
        include_financial: bool = True,
        include_news: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch a consolidated quote and optional history / financials.

        Args:
            symbol: e.g. A-share 000001, US AAPL, HK 00700
            market: "CN" / "US" / "HK"
            include_history: include recent k-line
            include_financial: include financials where available
            include_news: not supported
        """
        logger.info(f"Getting stock info for {symbol} (market={market})")

        if market == "CN":
            return self._get_cn_stock(symbol, include_history, include_financial)
        elif market == "US":
            return self._get_us_stock(symbol, include_history)
        elif market == "HK":
            return self._get_hk_stock(symbol, include_history, include_financial)
        else:
            return {
                "success": False,
                "error": f"Unsupported market: {market}. Supported: CN, US, HK.",
                "symbol": symbol,
                "market": market
            }

    def _get_cn_stock(
        self,
        symbol: str,
        include_history: bool,
        include_financial: bool
    ) -> Dict[str, Any]:
        """A-shares: Tushare first, then BaoStock."""
        # Try Tushare
        if self.use_tushare:
            result = self._get_cn_stock_tushare(symbol, include_history, include_financial)
            if result.get("success"):
                return result
            tushare_error = result.get("error", "")
            logger.warning(f"Tushare failed for {symbol}: {tushare_error}, trying BaoStock...")

        if self.use_baostock:
            result = self._get_cn_stock_baostock(symbol, include_history, include_financial)
            if result.get("success"):
                return result

        return {
            "success": False,
            "error": "No data source available. Tushare may need points/permissions, or BaoStock is missing/offline.",
            "symbol": symbol,
            "market": "CN"
        }

    def _get_cn_stock_tushare(
        self,
        symbol: str,
        include_history: bool,
        include_financial: bool
    ) -> Dict[str, Any]:
        """A-shares via Tushare."""
        result = {
            "symbol": symbol, "market": "CN", "source": "tushare",
            "timestamp": datetime.now().isoformat(), "success": False
        }

        ts_code = self._normalize_cn_code(symbol)

        try:
            daily_df = self.ts_pro.daily(ts_code=ts_code, start_date=datetime.now().strftime("%Y%m%d"),
                                          end_date=datetime.now().strftime("%Y%m%d"))
            if daily_df is None or daily_df.empty:
                daily_df = self.ts_pro.daily(ts_code=ts_code,
                                              start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
                                              end_date=datetime.now().strftime("%Y%m%d"))

            basic_df = self.ts_pro.stock_basic(ts_code=ts_code, fields='ts_code,name,area,industry,market,list_date')
            stock_name = ""
            if basic_df is not None and not basic_df.empty:
                stock_name = basic_df.iloc[0].get("name", "")

            if daily_df is not None and not daily_df.empty:
                latest = daily_df.iloc[0]
                result["basic"] = {
                    "name": stock_name, "ts_code": ts_code,
                    "price": float(latest.get("close", 0)),
                    "change": float(latest.get("change", 0)),
                    "change_pct": float(latest.get("pct_chg", 0)),
                    "volume": float(latest.get("vol", 0)),
                    "turnover": float(latest.get("amount", 0)),
                    "high": float(latest.get("high", 0)),
                    "low": float(latest.get("low", 0)),
                    "open": float(latest.get("open", 0)),
                    "pre_close": float(latest.get("pre_close", 0)),
                    "trade_date": str(latest.get("trade_date", "")),
                }
                result["success"] = True
            else:
                result["error"] = f"No trading data found for {ts_code}"
                return result

        except Exception as e:
            result["error"] = str(e)
            return result

        if include_history:
            try:
                hist_df = self.ts_pro.daily(
                    ts_code=ts_code,
                    start_date=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d")
                )
                if hist_df is not None and not hist_df.empty:
                    hist_df = hist_df.sort_values("trade_date")
                    result["history"] = {
                        "data": _dataframe_to_json_records(hist_df.tail(30)),
                        "count": len(hist_df)
                    }
            except Exception as e:
                result["history"] = {"error": str(e)}

        if include_financial:
            try:
                fina_df = self.ts_pro.fina_indicator(ts_code=ts_code, limit=4)
                if fina_df is not None and not fina_df.empty:
                    latest_fina = fina_df.iloc[0]
                    result["financial"] = {
                        "date": str(latest_fina.get("end_date", "")),
                        "roe": float(latest_fina["roe"]) if pd.notna(latest_fina.get("roe")) else None,
                        "roa": float(latest_fina["roa"]) if pd.notna(latest_fina.get("roa")) else None,
                        "gross_margin": float(latest_fina["grossprofit_margin"]) if pd.notna(latest_fina.get("grossprofit_margin")) else None,
                        "net_margin": float(latest_fina["netprofit_margin"]) if pd.notna(latest_fina.get("netprofit_margin")) else None,
                        "pe": float(latest_fina["pe"]) if pd.notna(latest_fina.get("pe")) else None,
                        "data": _dataframe_to_json_records(fina_df)
                    }
            except Exception as e:
                result["financial"] = {"error": str(e)}

        return result

    def _get_cn_stock_baostock(
        self,
        symbol: str,
        include_history: bool,
        include_financial: bool
    ) -> Dict[str, Any]:
        """A-shares via BaoStock (free)."""
        result = {
            "symbol": symbol, "market": "CN", "source": "baostock",
            "timestamp": datetime.now().isoformat(), "success": False
        }

        if not self._ensure_baostock_login():
            result["error"] = "BaoStock login failed"
            return result

        bs_code = self._normalize_cn_code_baostock(symbol)

        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )

            rows = []
            while (rs.error_code == '0') and rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                result["error"] = f"BaoStock: no trade data for {bs_code}"
                return result

            df = pd.DataFrame(rows, columns=rs.fields)

            for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            latest = df.iloc[-1]

            stock_name = self._get_cn_name_baostock(bs_code)

            result["basic"] = {
                "name": stock_name,
                "bs_code": bs_code,
                "price": float(latest.get("close", 0)),
                "change": float(latest["close"] - latest["preclose"]) if pd.notna(latest.get("preclose")) else 0,
                "change_pct": float(latest.get("pctChg", 0)),
                "volume": float(latest.get("volume", 0)),
                "turnover": float(latest.get("amount", 0)),
                "high": float(latest.get("high", 0)),
                "low": float(latest.get("low", 0)),
                "open": float(latest.get("open", 0)),
                "pre_close": float(latest.get("preclose", 0)),
                "trade_date": str(latest.get("date", "")),
                "turnover_rate": float(latest.get("turn", 0)),
            }
            result["success"] = True

        except Exception as e:
            logger.error(f"BaoStock failed for {bs_code}: {e}")
            result["error"] = str(e)
            return result

        if include_history:
            try:
                hist_start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
                hist_end = datetime.now().strftime("%Y-%m-%d")

                rs_hist = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,preclose,volume,amount,turn,pctChg",
                    start_date=hist_start,
                    end_date=hist_end,
                    frequency="d",
                    adjustflag="3"
                )

                hist_rows = []
                while (rs_hist.error_code == '0') and rs_hist.next():
                    hist_rows.append(rs_hist.get_row_data())

                if hist_rows:
                    hist_df = pd.DataFrame(hist_rows, columns=rs_hist.fields)
                    for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']:
                        if col in hist_df.columns:
                            hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')

                    result["history"] = {
                        "data": _dataframe_to_json_records(hist_df.tail(30)),
                        "count": len(hist_df)
                    }
            except Exception as e:
                result["history"] = {"error": str(e)}

        if include_financial:
            try:
                year = datetime.now().year
                quarter = (datetime.now().month - 1) // 3
                if quarter == 0:
                    year -= 1
                    quarter = 4

                rs_profit = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                profit_rows = []
                while (rs_profit.error_code == '0') and rs_profit.next():
                    profit_rows.append(rs_profit.get_row_data())

                if profit_rows:
                    profit_df = pd.DataFrame(profit_rows, columns=rs_profit.fields)
                    latest_profit = profit_df.iloc[0]
                    result["financial"] = {
                        "date": f"{year}Q{quarter}",
                        "roe": float(latest_profit.get("roeAvg", 0)) if latest_profit.get("roeAvg") else None,
                        "gross_margin": float(latest_profit.get("gpMargin", 0)) if latest_profit.get("gpMargin") else None,
                        "net_margin": float(latest_profit.get("npMargin", 0)) if latest_profit.get("npMargin") else None,
                        "source": "baostock"
                    }
                else:
                    if quarter > 1:
                        quarter -= 1
                    else:
                        year -= 1
                        quarter = 4
                    rs_profit2 = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                    profit_rows2 = []
                    while (rs_profit2.error_code == '0') and rs_profit2.next():
                        profit_rows2.append(rs_profit2.get_row_data())
                    if profit_rows2:
                        profit_df2 = pd.DataFrame(profit_rows2, columns=rs_profit2.fields)
                        lp2 = profit_df2.iloc[0]
                        result["financial"] = {
                            "date": f"{year}Q{quarter}",
                            "roe": float(lp2.get("roeAvg", 0)) if lp2.get("roeAvg") else None,
                            "gross_margin": float(lp2.get("gpMargin", 0)) if lp2.get("gpMargin") else None,
                            "net_margin": float(lp2.get("npMargin", 0)) if lp2.get("npMargin") else None,
                            "source": "baostock"
                        }
            except Exception as e:
                result["financial"] = {"error": str(e)}

        return result

    def _get_cn_name_baostock(self, bs_code: str) -> str:
        """Resolve display name from BaoStock stock_basic."""
        try:
            rs = bs.query_stock_basic(code=bs_code)
            rows = []
            while (rs.error_code == '0') and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=rs.fields)
                if 'code_name' in df.columns and not df.empty:
                    return str(df.iloc[0]['code_name'])
        except Exception:
            pass
        return ""

    def _get_hk_stock(
        self,
        symbol: str,
        include_history: bool,
        include_financial: bool
    ) -> Dict[str, Any]:
        """HK stocks via Tushare."""
        if not self.use_tushare:
            return {
                "success": False,
                "error": "HK data requires Tushare (token + points).",
                "symbol": symbol, "market": "HK"
            }

        result = {
            "symbol": symbol, "market": "HK", "source": "tushare",
            "timestamp": datetime.now().isoformat(), "success": False
        }

        ts_code = symbol if ".HK" in symbol.upper() else f"{symbol}.HK"

        try:
            hk_df = self.ts_pro.hk_daily(ts_code=ts_code,
                                          start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
                                          end_date=datetime.now().strftime("%Y%m%d"))

            if hk_df is not None and not hk_df.empty:
                latest = hk_df.iloc[0]
                result["basic"] = {
                    "ts_code": ts_code,
                    "price": float(latest.get("close", 0)),
                    "change": float(latest.get("change", 0)),
                    "change_pct": float(latest.get("pct_chg", 0)),
                    "volume": float(latest.get("vol", 0)),
                    "high": float(latest.get("high", 0)),
                    "low": float(latest.get("low", 0)),
                    "open": float(latest.get("open", 0)),
                    "pre_close": float(latest.get("pre_close", 0)),
                    "trade_date": str(latest.get("trade_date", "")),
                }
                result["success"] = True
            else:
                result["error"] = f"No HK trading data found for {ts_code}"
                return result

        except Exception as e:
            result["error"] = str(e)
            return result

        if include_history:
            try:
                hist_df = self.ts_pro.hk_daily(
                    ts_code=ts_code,
                    start_date=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d")
                )
                if hist_df is not None and not hist_df.empty:
                    hist_df = hist_df.sort_values("trade_date")
                    result["history"] = {
                        "data": _dataframe_to_json_records(hist_df.tail(30)),
                        "count": len(hist_df)
                    }
            except Exception as e:
                result["history"] = {"error": str(e)}

        return result

    def _get_us_stock(
        self,
        symbol: str,
        include_history: bool
    ) -> Dict[str, Any]:
        """US stocks via Tushare us_daily."""
        if not self.use_tushare:
            return {
                "success": False,
                "error": "US data requires Tushare (token + points).",
                "symbol": symbol, "market": "US"
            }

        result = {
            "symbol": symbol, "market": "US", "source": "tushare",
            "timestamp": datetime.now().isoformat(), "success": False
        }

        ts_code = symbol.upper()

        try:
            us_df = self.ts_pro.us_daily(
                ts_code=ts_code,
                start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d")
            )

            if us_df is not None and not us_df.empty:
                latest = us_df.iloc[0]
                result["basic"] = {
                    "ts_code": ts_code,
                    "price": float(latest.get("close", 0)),
                    "change": float(latest.get("change", 0)),
                    "change_pct": float(latest.get("pct_chg", 0)),
                    "volume": float(latest.get("vol", 0)),
                    "high": float(latest.get("high", 0)),
                    "low": float(latest.get("low", 0)),
                    "open": float(latest.get("open", 0)),
                    "pre_close": float(latest.get("pre_close", 0)),
                    "trade_date": str(latest.get("trade_date", "")),
                }
                result["success"] = True
            else:
                result["error"] = f"No US trading data found for {ts_code}"
                return result

        except Exception as e:
            result["error"] = str(e)
            return result

        if include_history:
            try:
                hist_df = self.ts_pro.us_daily(
                    ts_code=ts_code,
                    start_date=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d")
                )
                if hist_df is not None and not hist_df.empty:
                    hist_df = hist_df.sort_values("trade_date")
                    result["history"] = {
                        "data": _dataframe_to_json_records(hist_df.tail(30)),
                        "count": len(hist_df)
                    }
            except Exception as e:
                result["history"] = {"error": str(e)}

        return result

    def search_stocks(self, keyword: str, market: str = "CN", limit: int = 10) -> List[Dict]:
        """
        Search tickers by name or code.

        A-shares: Tushare first, then BaoStock.
        """
        logger.info(f"Searching stocks with keyword: {keyword}")
        results = []

        if market == "CN":
            if self.use_tushare:
                try:
                    all_stocks = self.ts_pro.stock_basic(exchange='', list_status='L',
                                                          fields='ts_code,symbol,name,area,industry')
                    if all_stocks is not None and not all_stocks.empty:
                        matched = all_stocks[
                            all_stocks['name'].str.contains(keyword, na=False) |
                            all_stocks['symbol'].str.contains(keyword, na=False) |
                            all_stocks['ts_code'].str.contains(keyword, na=False)
                        ].head(limit)

                        for _, row in matched.iterrows():
                            results.append({
                                "symbol": row['symbol'],
                                "ts_code": row['ts_code'],
                                "name": row['name'],
                                "industry": row.get('industry', ''),
                                "market": "CN",
                                "source": "tushare"
                            })
                        if results:
                            return results
                except Exception as e:
                    logger.warning(f"Tushare search failed: {e}, trying BaoStock...")

            if self.use_baostock and self._ensure_baostock_login():
                try:
                    rs = bs.query_stock_basic()
                    rows = []
                    while (rs.error_code == '0') and rs.next():
                        rows.append(rs.get_row_data())

                    if rows:
                        df = pd.DataFrame(rows, columns=rs.fields)
                        kw = keyword.lower()
                        matched = df[
                            df['code_name'].str.lower().str.contains(kw, na=False) |
                            df['code'].str.contains(keyword, na=False)
                        ]
                        status_col = 'status' if 'status' in matched.columns else None
                        if status_col:
                            matched = matched[matched[status_col] == '1']

                        for _, row in matched.head(limit).iterrows():
                            code = row.get('code', '')
                            results.append({
                                "symbol": code.split('.')[1] if '.' in code else code,
                                "bs_code": code,
                                "name": row.get('code_name', ''),
                                "market": "CN",
                                "source": "baostock"
                            })
                except Exception as e:
                    logger.error(f"BaoStock search failed: {e}")

            if not results:
                return [{"error": "Search failed: Tushare unavailable and BaoStock returned no matches."}]

        elif market == "US":
            if not self.use_tushare:
                return [{"error": "US search requires Tushare (token + points)."}]
            try:
                us_stocks = self.ts_pro.us_basic()
                if us_stocks is not None and not us_stocks.empty:
                    keyword_upper = keyword.upper()
                    matched = us_stocks[
                        us_stocks['name'].str.contains(keyword, case=False, na=False) |
                        us_stocks['ts_code'].str.contains(keyword_upper, na=False)
                    ].head(limit)
                    for _, row in matched.iterrows():
                        results.append({
                            "ts_code": row['ts_code'],
                            "name": row['name'],
                            "market": "US"
                        })
            except Exception as e:
                return [{"error": f"US search failed: {str(e)}"}]

        elif market == "HK":
            if not self.use_tushare:
                return [{"error": "HK search requires Tushare (token + points)."}]
            try:
                hk_stocks = self.ts_pro.hk_basic(list_status='L')
                if hk_stocks is not None and not hk_stocks.empty:
                    matched = hk_stocks[
                        hk_stocks['name'].str.contains(keyword, na=False) |
                        hk_stocks['ts_code'].str.contains(keyword, na=False)
                    ].head(limit)
                    for _, row in matched.iterrows():
                        results.append({
                            "ts_code": row['ts_code'],
                            "name": row['name'],
                            "market": "HK"
                        })
            except Exception as e:
                return [{"error": f"HK search failed: {str(e)}"}]

        return results

    def get_available_sources(self) -> Dict[str, bool]:
        """Which upstream feeds are active."""
        return {
            "tushare": self.use_tushare,
            "baostock": self.use_baostock,
        }

    @staticmethod
    def _normalize_cn_code(symbol: str) -> str:
        """Normalize digits-only codes to Tushare form (e.g. 000001 -> 000001.SZ)."""
        symbol = symbol.strip()
        if "." in symbol:
            return symbol.upper()
        if symbol.startswith("6"):
            return f"{symbol}.SH"
        elif symbol.startswith("0") or symbol.startswith("3"):
            return f"{symbol}.SZ"
        elif symbol.startswith("8") or symbol.startswith("4"):
            return f"{symbol}.BJ"
        else:
            return f"{symbol}.SZ"

    @staticmethod
    def _normalize_cn_code_baostock(symbol: str) -> str:
        """Normalize to BaoStock form (e.g. 000001 -> sz.000001)."""
        symbol = symbol.strip()
        if symbol.startswith("sh.") or symbol.startswith("sz.") or symbol.startswith("bj."):
            return symbol
        if "." in symbol:
            parts = symbol.split(".")
            code = parts[0]
            exchange = parts[1].lower() if len(parts) > 1 else ""
            if exchange in ("sh", "sz", "bj"):
                return f"{exchange}.{code}"
            elif exchange == "ss":
                return f"sh.{code}"
        else:
            code = symbol
        if code.startswith("6"):
            return f"sh.{code}"
        elif code.startswith("0") or code.startswith("3"):
            return f"sz.{code}"
        elif code.startswith("8") or code.startswith("4"):
            return f"bj.{code}"
        else:
            return f"sz.{code}"


def get_stock_data_tool():
    """Singleton accessor."""
    if not hasattr(get_stock_data_tool, "_instance"):
        get_stock_data_tool._instance = StockDataTool()
    return get_stock_data_tool._instance


def get_tool_definitions() -> List[Dict]:
    """OpenAI-style tool definitions for stock lookup and search."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_stock_info",
                "description": """Full quote context for research (not investment advice).

Markets:
- CN: A-shares (e.g. 000001, 600519) — BaoStock fallback
- US: e.g. AAPL (Tushare points)
- HK: e.g. 00700 (Tushare points)

Returns latest print, optional ~30d history, optional financials (A-share).""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Ticker (e.g. 000001, AAPL, 00700)"
                        },
                        "market": {
                            "type": "string",
                            "enum": ["CN", "US", "HK"],
                            "description": "CN / US / HK",
                            "default": "CN"
                        },
                        "include_history": {
                            "type": "boolean",
                            "description": "Include recent daily history (~30 rows)",
                            "default": True
                        },
                        "include_financial": {
                            "type": "boolean",
                            "description": "Include financial metrics when available",
                            "default": True
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_stocks",
                "description": """Search by name or partial code (works for Chinese or English names on CN free tier).

US/HK search needs Tushare access.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Substring of the security name or ticker (including common Chinese UI labels for name/code fields)."
                        },
                        "market": {
                            "type": "string",
                            "enum": ["CN", "US", "HK"],
                            "description": "CN / US / HK",
                            "default": "CN"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results",
                            "default": 10
                        }
                    },
                    "required": ["keyword"]
                }
            }
        }
    ]
