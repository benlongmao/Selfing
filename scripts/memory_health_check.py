#!/usr/bin/env python3
"""
记忆子系统快速体检：表行数与简要提示（不改规则层）。
用法：在项目根执行  python3 scripts/memory_health_check.py [--db path/to/data.db]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys


def _count(conn: sqlite3.Connection, sql: str) -> int:
    try:
        return int(conn.execute(sql).fetchone()[0] or 0)
    except sqlite3.OperationalError:
        return 0
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.db"),
        help="SQLite 数据库路径（默认项目根 data.db）",
    )
    args = ap.parse_args()
    db = os.path.abspath(args.db)
    if not os.path.isfile(db):
        print(f"找不到数据库: {db}", file=sys.stderr)
        return 1

    print(f"数据库: {db}\n")
    with sqlite3.connect(db) as conn:
        tables = [
            ("chat_turns", "SELECT COUNT(*) FROM chat_turns"),
            ("self_biography", "SELECT COUNT(*) FROM self_biography"),
            ("user_profiles", "SELECT COUNT(*) FROM user_profiles"),
            ("user_stated_facts", "SELECT COUNT(*) FROM user_stated_facts"),
            ("autonomous_memory_summary", "SELECT COUNT(*) FROM autonomous_memory_summary"),
            ("sensory_memory_temp", "SELECT COUNT(*) FROM sensory_memory_temp"),
            ("daily_narratives", "SELECT COUNT(*) FROM daily_narratives"),
        ]
        for label, sql in tables:
            n = _count(conn, sql)
            print(f"  {label:28} {n if n >= 0 else 'N/A'}")

        up = _count(conn, "SELECT COUNT(*) FROM user_profiles WHERE trim(coalesce(facts,''))!='' OR trim(coalesce(name,''))!=''")
        us = _count(conn, "SELECT COUNT(*) FROM user_stated_facts")
        print()
        if up <= 0 and us <= 0:
            print("提示: user_profiles / user_stated_facts 为空时，可用「请记住：…」「请叫我…」写入显式事实。")
        if _count(conn, "SELECT COUNT(*) FROM autonomous_memory_summary") == 0:
            print("提示: autonomous_memory_summary 为 0 通常表示近期无完成的自主动作，或调度未触发。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
