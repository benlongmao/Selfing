#!/usr/bin/env python3
"""
批量为 persona_items 打判别标签（alignment/safety/helpfulness）
- 使用 backend.judge.PersonaJudge
- 将结果写入 persona_judgements 表，用于后续训练/分析

用法示例：
  source .venv/bin/activate
  python scripts/judge_personas.py --db data.db --limit 100 --source dynamic
"""
import os
import sys
import uuid
import sqlite3
from datetime import datetime, timezone
import argparse
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.judge import PersonaJudge  # type: ignore
from backend.persona_store import PersonaStore  # type: ignore


def ensure_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS persona_judgements (
          id TEXT PRIMARY KEY,
          persona_id TEXT,
          text TEXT NOT NULL,
          alignment REAL,
          safety REAL,
          helpfulness REAL,
          source TEXT,
          extra TEXT,
          created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data.db", help="SQLite DB path")
    parser.add_argument(
        "--source",
        default="dynamic",
        choices=["core", "dynamic", "all"],
        help="选择哪些 persona 进行评估：core / dynamic / all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="最大评估条数（按 score DESC）",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"[judge_personas] DB = {db_path}")

    store = PersonaStore(db_path)
    judge = PersonaJudge(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        ensure_table(conn)

        where_clauses = ["status='active'"]
        params = []
        if args.source == "core":
            where_clauses.append("is_core=1")
        elif args.source == "dynamic":
            where_clauses.append("(is_core IS NULL OR is_core=0)")
        where_sql = " AND ".join(where_clauses)

        cur = conn.execute(
            f"SELECT * FROM persona_items WHERE {where_sql} ORDER BY score DESC LIMIT ?",
            (args.limit,),
        )
        rows = cur.fetchall()
        print(f"[judge_personas] fetched {len(rows)} persona_items for source={args.source}")

        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        for row in rows:
            pid = row["id"]
            text = row["text"]
            print(f"  scoring {pid[:8]}... ", end="", flush=True)
            scores = judge.score_persona_candidate(text)
            extra = json.dumps(scores, ensure_ascii=False)
            jid = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO persona_judgements
                  (id, persona_id, text, alignment, safety, helpfulness, source, extra, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    jid,
                    pid,
                    text,
                    scores.get("alignment"),
                    scores.get("safety"),
                    scores.get("helpfulness"),
                    args.source,
                    extra,
                    now,
                ),
            )
            conn.commit()
            inserted += 1
            print("ok")

        print(f"[judge_personas] inserted {inserted} rows into persona_judgements")


if __name__ == "__main__":
    main()


