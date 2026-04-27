#!/usr/bin/env python3
"""
导出 persona_judgements 表，便于训练/分析

用法示例：
  source .venv/bin/activate
  python scripts/export_judgements.py --db data.db --format csv --out persona_judgements.csv
  python scripts/export_judgements.py --db data.db --format json --out persona_judgements.json
"""
import os
import sys
import csv
import json
import argparse
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data.db", help="SQLite DB path")
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="导出格式（csv 或 json）",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="输出文件路径",
    )
    parser.add_argument(
        "--source",
        choices=["core", "dynamic", "all"],
        default="all",
        help="按 source 过滤（core/dynamic/all）",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"[export_judgements] DB = {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        where = []
        params = []
        if args.source != "all":
            where.append("source=?")
            params.append(args.source)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        cur = conn.execute(
            f"SELECT id, persona_id, text, alignment, safety, helpfulness, source, extra, created_at FROM persona_judgements {where_sql}",
            params,
        )
        rows = cur.fetchall()

    print(f"[export_judgements] rows = {len(rows)}")

    if args.format == "csv":
        fieldnames = rows[0].keys() if rows else [
            "id",
            "persona_id",
            "text",
            "alignment",
            "safety",
            "helpfulness",
            "source",
            "extra",
            "created_at",
        ]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        print(f"[export_judgements] wrote CSV -> {args.out}")
    else:
        data = [dict(r) for r in rows]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[export_judgements] wrote JSON -> {args.out}")


if __name__ == "__main__":
    main()


