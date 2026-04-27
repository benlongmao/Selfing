#!/usr/bin/env python3
"""
一次性清理脚本：归档 2026-03-25 之前创建的冗余能量类 L2 规则。

背景：
  S-44 早期运行时，由于能量系统频繁触发 ENERGY_DEPLETED 事件，
  反思模块生成了大量"能量节省""低能耗"类规则（约 187 条）。
  3/25 的饱和检测修复已阻止新能量规则产生，但旧库存未清理。

策略：
  1. 识别 3/25 前创建的能量相关 L2 规则
  2. 按 score 降序保留前 3 条（代表性规则）
  3. 其余全部设为 archived
"""
import sqlite3
import sys
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data.db")

ENERGY_KEYWORDS_SQL = """
    (text LIKE '%能量%' OR text LIKE '%消耗%' OR text LIKE '%节省%'
     OR text LIKE '%低能耗%' OR text LIKE '%能量有限%' OR text LIKE '%能量预算%'
     OR text LIKE '%损耗%' OR text LIKE '%高成本%' OR text LIKE '%高代价%')
"""

CUTOFF_DATE = "2026-03-25"
KEEP_TOP_N = 3


def main():
    if not os.path.exists(DB_PATH):
        print(f"数据库不存在: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 查询目标规则
    rows = conn.execute(f"""
        SELECT id, text, score, created_at
        FROM persona_items
        WHERE is_core = 0
          AND status = 'active'
          AND created_at < ?
          AND {ENERGY_KEYWORDS_SQL}
        ORDER BY score DESC
    """, (CUTOFF_DATE,)).fetchall()

    total = len(rows)
    if total == 0:
        print("没有找到需要清理的旧能量规则。")
        conn.close()
        return

    keep = rows[:KEEP_TOP_N]
    archive = rows[KEEP_TOP_N:]

    print(f"共找到 {total} 条 3/25 前的能量类 L2 规则")
    print(f"保留前 {KEEP_TOP_N} 条:")
    for r in keep:
        print(f"  [{r['id']}] score={r['score']:.4f}  {r['text'][:60]}")

    print(f"\n将归档 {len(archive)} 条:")
    for r in archive[:5]:
        print(f"  [{r['id']}] score={r['score']:.4f}  {r['text'][:60]}")
    if len(archive) > 5:
        print(f"  ... 以及另外 {len(archive) - 5} 条")

    if "--dry-run" in sys.argv:
        print("\n[DRY RUN] 未执行任何修改。去掉 --dry-run 参数以实际执行。")
        conn.close()
        return

    confirm = input(f"\n确认归档 {len(archive)} 条规则？(y/N): ")
    if confirm.strip().lower() != "y":
        print("已取消。")
        conn.close()
        return

    archive_ids = [r["id"] for r in archive]
    placeholders = ",".join("?" for _ in archive_ids)
    conn.execute(
        f"UPDATE persona_items SET status = 'archived' WHERE id IN ({placeholders})",
        archive_ids,
    )
    conn.commit()
    print(f"已归档 {len(archive)} 条能量类旧规则。")

    remaining = conn.execute(
        "SELECT COUNT(*) FROM persona_items WHERE is_core = 0 AND status = 'active'"
    ).fetchone()[0]
    print(f"当前活跃 L2 规则数: {remaining}")

    conn.close()


if __name__ == "__main__":
    main()
