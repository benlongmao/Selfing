"""
Resolve calendar-day strings for ``chat_turns`` queries and compute UTC windows.

- Default ``memory.chat_turns_calendar_timezone`` is UTC (matches historical ``created_at`` behavior).
- Set e.g. ``America/New_York`` or ``Asia/Shanghai`` so "yesterday" / local calendar days align with user locale.
- Accepts ISO, common English month forms, slash forms, and Chinese ``M月D日`` (for bilingual inputs).
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Any, Dict, Optional, Tuple, Union

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

TzType = Union[tzinfo, Any]

_EN_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_EN_ALT = "|".join(sorted(_EN_MONTHS.keys(), key=len, reverse=True))


def resolve_calendar_timezone_name(raw: Optional[str]) -> str:
    name = (raw or "UTC").strip()
    return name or "UTC"


def get_calendar_tzinfo(tz_name: str) -> Tuple[TzType, str]:
    """Return (tzinfo, effective_name). Invalid names fall back to UTC."""
    n = resolve_calendar_timezone_name(tz_name)
    if n.upper() == "UTC":
        return timezone.utc, "UTC"
    if ZoneInfo is None:
        return timezone.utc, "UTC"
    try:
        return ZoneInfo(n), n
    except Exception:
        return timezone.utc, "UTC"


def local_date_from_utc(ref_utc: datetime, tz: TzType) -> date:
    if tz is timezone.utc:
        return ref_utc.astimezone(timezone.utc).date()
    return ref_utc.astimezone(tz).date()


def utc_window_for_local_calendar_day(local_day: date, tz: TzType) -> Tuple[datetime, datetime]:
    """Half-open UTC interval [start_utc, end_utc) for that local calendar day."""
    if tz is timezone.utc:
        start = datetime.combine(local_day, time.min, tzinfo=timezone.utc)
        return start, start + timedelta(days=1)
    start_local = datetime.combine(local_day, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def infer_year_for_month_day(month: int, day: int, ref_local: date) -> int:
    """Month-day only: if that calendar day is still in the future vs ref_local, use previous year."""
    y = ref_local.year
    try:
        candidate = date(y, month, day)
    except ValueError:
        return y
    if candidate > ref_local:
        return y - 1
    return y


def _month_from_word(w: str) -> Optional[int]:
    return _EN_MONTHS.get((w or "").lower())


def try_resolve_calendar_date_string(
    raw: str,
    *,
    ref_utc: datetime,
    tz: TzType,
) -> Dict[str, Any]:
    """
    Parse ``calendar_date`` into a local ``date`` for the configured calendar timezone.
    """
    s = (raw or "").strip()
    if not s:
        return {"success": False, "error": "Empty date string"}

    ref_local = local_date_from_utc(ref_utc, tz)

    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dd = date(y, mo, d)
            return {"success": True, "local_date": dd, "resolved_from": "iso", "original": raw}
        except ValueError:
            return {"success": False, "error": "Invalid ISO calendar_date", "original": raw}

    # English: May 1, 2026 / May 1st, 2026
    m = re.search(
        rf"\b({_EN_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
        s,
        re.I,
    )
    if m:
        mo = _month_from_word(m.group(1))
        d, y = int(m.group(2)), int(m.group(3))
        if mo:
            try:
                dd = date(y, mo, d)
                return {
                    "success": True,
                    "local_date": dd,
                    "resolved_from": "english_mdy",
                    "original": raw,
                }
            except ValueError:
                return {"success": False, "error": "Invalid date", "original": raw}

    # English: 1 May 2026 / 1st May 2026
    m = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_EN_ALT})\s+(\d{{4}})\b",
        s,
        re.I,
    )
    if m:
        d, mo_w, y = int(m.group(1)), m.group(2), int(m.group(3))
        mo = _month_from_word(mo_w)
        if mo:
            try:
                dd = date(y, mo, d)
                return {
                    "success": True,
                    "local_date": dd,
                    "resolved_from": "english_dmy",
                    "original": raw,
                }
            except ValueError:
                return {"success": False, "error": "Invalid date", "original": raw}

    # English: May 1 / May 1st (no year)
    m = re.search(rf"\b({_EN_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", s, re.I)
    if m:
        mo = _month_from_word(m.group(1))
        d = int(m.group(2))
        if mo:
            y = infer_year_for_month_day(mo, d, ref_local)
            try:
                dd = date(y, mo, d)
                return {
                    "success": True,
                    "local_date": dd,
                    "resolved_from": "english_md",
                    "original": raw,
                    "inferred_year": y,
                }
            except ValueError:
                return {"success": False, "error": "Invalid date", "original": raw}

    # Chinese: 2025年5月1日
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dd = date(y, mo, d)
            return {
                "success": True,
                "local_date": dd,
                "resolved_from": "chinese_ymd",
                "original": raw,
            }
        except ValueError:
            return {"success": False, "error": "Invalid date", "original": raw}

    # Chinese: 5月1日 (in sentence)
    m = re.search(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日?(?!\d)", s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        y = infer_year_for_month_day(mo, d, ref_local)
        try:
            dd = date(y, mo, d)
            return {
                "success": True,
                "local_date": dd,
                "resolved_from": "chinese_md",
                "original": raw,
                "inferred_year": y,
            }
        except ValueError:
            return {"success": False, "error": "Invalid date", "original": raw}

    # 2025/5/1
    m = re.fullmatch(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dd = date(y, mo, d)
            return {
                "success": True,
                "local_date": dd,
                "resolved_from": "slash_ymd",
                "original": raw,
            }
        except ValueError:
            return {"success": False, "error": "Invalid date", "original": raw}

    # 5/1 or 5-1 (no year)
    m = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})", s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        y = infer_year_for_month_day(mo, d, ref_local)
        try:
            dd = date(y, mo, d)
            return {
                "success": True,
                "local_date": dd,
                "resolved_from": "slash_md",
                "original": raw,
                "inferred_year": y,
            }
        except ValueError:
            return {"success": False, "error": "Invalid date", "original": raw}

    return {
        "success": False,
        "error": "Unparseable date; use YYYY-MM-DD, May 1 / May 1 2026, or M/D",
        "original": raw,
    }


def effective_max_snippets(requested: int, turn_count: int, *, default_cap: int = 30) -> int:
    """
    Cap at ``default_cap`` (max 30). If the caller asks for fewer than 15 snippets, do not auto-expand
    (saves tokens). Otherwise bump toward min(turn_count, cap) on busy days.
    """
    cap = max(1, min(30, int(default_cap)))
    req = max(1, min(30, int(requested)))
    if req < 15:
        return min(req, turn_count) if turn_count else req
    return min(cap, max(req, min(turn_count, cap)))
