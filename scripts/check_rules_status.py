#!/usr/bin/env python3
"""
检查 L0/L1/L2 规则完整性
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data.db"

def main():
    conn = sqlite3.connect(str(DB_PATH))
    
    print("=" * 70)
    print("L0/L1/L2 规则完整性检查")
    print("=" * 70)
    
    # 1. 统计各层级规则数量
    cur = conn.execute('''
        SELECT 
            CASE 
                WHEN locked=1 THEN 'L0 (宪法级)'
                WHEN is_core=1 THEN 'L1 (核心)'
                ELSE 'L2 (动态)'
            END as level,
            COUNT(*) as total,
            SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as has_emb
        FROM persona_items 
        WHERE status='active'
        GROUP BY level
        ORDER BY level
    ''')
    print("\n📊 规则数量统计:")
    for row in cur.fetchall():
        emb_status = "✅" if row[1] == row[2] else f"⚠️ {row[2]}/{row[1]} 有embedding"
        print(f"  {row[0]}: {row[1]} 条 {emb_status}")
    
    # 2. 检查 L0 规则内容
    print("\n" + "=" * 70)
    print("🔒 L0 宪法级规则 (locked=1) - 每次对话都注入")
    print("=" * 70)
    cur = conn.execute('''
        SELECT id, text FROM persona_items 
        WHERE status='active' AND locked=1 
        ORDER BY id
    ''')
    for i, row in enumerate(cur.fetchall(), 1):
        text = row[1][:55] + "..." if len(row[1]) > 55 else row[1]
        print(f"{i:2}. {row[0]}: {text}")
    
    # 3. 检查 L1 规则示例
    print("\n" + "=" * 70)
    print("🎯 L1 核心规则示例 (is_core=1, locked=0) - 动态匹配 top-10")
    print("=" * 70)
    cur = conn.execute('''
        SELECT id, text FROM persona_items 
        WHERE status='active' AND is_core=1 AND locked=0 
        ORDER BY id
        LIMIT 15
    ''')
    for i, row in enumerate(cur.fetchall(), 1):
        text = row[1][:50] + "..." if len(row[1]) > 50 else row[1]
        print(f"{i:2}. {row[0]}: {text}")
    
    # 4. 检查 L2 规则来源
    print("\n" + "=" * 70)
    print("📝 L2 动态规则来源分布 - 动态匹配 top-5")
    print("=" * 70)
    cur = conn.execute('''
        SELECT 
            CASE 
                WHEN id LIKE 'ref-%' THEN 'ref-* (Agent自生成)'
                WHEN id LIKE 'core-%' THEN 'core-* (预设但非核心)'
                ELSE '其他'
            END as source,
            COUNT(*) as count
        FROM persona_items 
        WHERE status='active' AND is_core=0 AND locked=0
        GROUP BY source
        ORDER BY count DESC
    ''')
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} 条")
    
    # 5. 检查其他维度数据
    print("\n" + "=" * 70)
    print("🧠 其他维度数据状态")
    print("=" * 70)
    tables = [
        ("emotion_patterns", "情感模式"),
        ("motivation_patterns", "动机模式"),
        ("somatic_patterns", "体感模式"),
    ]
    for table, name in tables:
        try:
            cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            status = "✅" if count > 0 else "⚠️ 空"
            print(f"  {name}: {count} 条 {status}")
        except:
            print(f"  {name}: ❌ 表不存在")
    
    print("\n" + "=" * 70)
    print("✅ 检查完成")
    print("=" * 70)
    
    conn.close()

if __name__ == "__main__":
    main()
