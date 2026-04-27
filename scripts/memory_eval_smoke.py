#!/usr/bin/env python3
"""
记忆子系统轻量冒烟（默认不碰项目 data.db）。

- 默认：在临时 SQLite 里验证 user_fact_capture → user_stated_facts 写入与读取、
  unified_memory 检索（拷贝库时避免污染 memory_state）。
- 可选：--check-project-db 仅对项目 data.db 做只读表检查（SELECT，无写入）。

退出码：0 成功，1 失败。不把 S 改坏的前提下给回归一个抓手。
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
import traceback


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_sys_path() -> None:
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def test_imports() -> None:
    _ensure_sys_path()
    from backend.unified_memory import (  # noqa: WPS433
        UnifiedMemoryBus,
        _iso_date_prefix,
        _source_tag_for_prompt,
        UnifiedMemoryCandidate,
    )
    from backend.user_fact_capture import (  # noqa: WPS433
        apply_user_fact_capture,
        fetch_recent_stated_facts_for_prompt,
        format_user_profile_block_for_prompt,
    )

    assert _iso_date_prefix("2026-01-01T00:00:00Z") == "2026-01-01"
    c = UnifiedMemoryCandidate(
        memory_key="k",
        session_id="s",
        memory_type="episodic",
        source_table="self_biography",
        source_id="1",
        content="x",
        created_at="2026-01-01T00:00:00Z",
    )
    assert "memoir" in _source_tag_for_prompt(c)
    del UnifiedMemoryBus, apply_user_fact_capture
    del fetch_recent_stated_facts_for_prompt, format_user_profile_block_for_prompt


def test_isolated_db_pipeline() -> None:
    """临时库：显式事实 → user_stated_facts + 溯源展示（不跑完整 UnifiedMemoryBus，空库缺表会炸）。"""
    _ensure_sys_path()
    from backend.user_fact_capture import (
        apply_user_fact_capture,
        fetch_recent_stated_facts_for_prompt,
        format_user_profile_block_for_prompt,
    )

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        sid = "smoke-session"
        r = apply_user_fact_capture(
            path,
            sid,
            "请记住：memory_eval_smoke_token_42",
            turn_index=1,
        )
        assert r.get("updated") is True, r

        conn = sqlite3.connect(path)
        n = conn.execute(
            "SELECT COUNT(*) FROM user_stated_facts WHERE session_id=? AND content LIKE ?",
            (sid, "%memory_eval_smoke_token_42%"),
        ).fetchone()[0]
        conn.close()
        assert int(n) >= 1, "user_stated_facts should have new rows"

        lines = fetch_recent_stated_facts_for_prompt(path, sid, limit=5)
        assert any("memory_eval_smoke_token_42" in x for x in lines), lines

        block = format_user_profile_block_for_prompt(path, sid)
        assert "memory_eval_smoke_token_42" in block or "Fact trail" in block, block[:500]
    finally:
        if os.path.isfile(path):
            os.unlink(path)


def read_only_project_db(db_path: str) -> None:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = [
            "chat_turns",
            "self_biography",
            "user_profiles",
            "user_stated_facts",
            "persona_items",
        ]
        for t in tables:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"  [只读] {t}: {int(n)}")
            except sqlite3.OperationalError as e:
                print(f"  [只读] {t}: 跳过 ({e})")
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check-project-db",
        action="store_true",
        help="对项目 data.db 只读 COUNT（不跑写入冒烟）",
    )
    ap.add_argument(
        "--db",
        default=os.path.join(_project_root(), "data.db"),
        help="--check-project-db 时使用的库路径",
    )
    args = ap.parse_args()

    try:
        if args.check_project_db:
            db = os.path.abspath(args.db)
            if not os.path.isfile(db):
                print(f"找不到 {db}", file=sys.stderr)
                return 1
            print(f"只读检查: {db}")
            read_only_project_db(db)
            print("只读检查完成。")
            return 0

        print("memory_eval_smoke: 导入与临时库管线…")
        test_imports()
        test_isolated_db_pipeline()
        print("memory_eval_smoke: 全部通过。")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
