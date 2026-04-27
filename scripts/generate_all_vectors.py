#!/usr/bin/env python3
"""
为所有 Persona Core 生成向量嵌入
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.persona_store import PersonaStore, PersonaItem
from datetime import datetime, timezone
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_all_vectors(db_path: str = "data.db", batch_size: int = 10):
    """为所有 Persona Core 生成向量"""
    store = PersonaStore(db_path)
    
    # 获取所有未向量化的记录
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT id, text, score, importance, novelty, reliability, evidence_count, created_at, last_seen_at, status FROM persona_items WHERE embedding IS NULL")
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        logger.info("所有记录已向量化")
        return
    
    logger.info(f"找到 {len(rows)} 条未向量化的记录，开始生成向量...")
    
    now = datetime.now(timezone.utc).isoformat()
    
    for i, row in enumerate(rows, 1):
        try:
            item = PersonaItem(
                id=row[0],
                text=row[1],
                embedding=None,  # 会自动生成
                score=row[2] if len(row) > 2 else 1.5,
                importance=row[3] if len(row) > 3 else 1.0,
                novelty=row[4] if len(row) > 4 else 0.0,
                reliability=row[5] if len(row) > 5 else 1.0,
                evidence_count=row[6] if len(row) > 6 else 1,
                created_at=row[7] if len(row) > 7 else now,
                last_seen_at=row[8] if len(row) > 8 else now,
                status=row[9] if len(row) > 9 else "active"
            )
            store.add_or_update(item, update_embedding=True)
            
            if i % batch_size == 0:
                logger.info(f"  已处理 {i}/{len(rows)} 条")
        except Exception as e:
            logger.error(f"  错误处理 {row[0]}: {e}")
    
    logger.info(f"✅ 向量生成完成: {len(rows)} 条")
    
    # 验证
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM persona_items WHERE embedding IS NOT NULL")
    count = cur.fetchone()[0]
    conn.close()
    
    logger.info(f"验证: {count} 条记录已有向量")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data.db", help="Database path")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for progress reporting")
    args = parser.parse_args()
    
    generate_all_vectors(args.db, args.batch_size)

