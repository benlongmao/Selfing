#!/usr/bin/env python3
import sys
import os
import sqlite3
import json
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.self_model import SelfModel
from backend.persona_store import PersonaStore

def heal_s(session_id="selfing-session"):
    db_path = "data.db"
    store = PersonaStore(db_path)
    model = SelfModel(db_path, store)
    
    print(f"🩹 正在为 S 重新校准自我状态基准 (Session: {session_id})...")
    
    # 1. 强行获取当前的 z_self
    z_self = model.get_z_self(session_id)
    if z_self is None:
        print("❌ 找不到 z_self 记录")
        return

    # 2. 将当前的 z_self 设为新的参考基准 (ref_vector)
    # 这会告诉系统：现在的状态（即使名字被改了）就是“正常”的，从而消除 Drift
    np.save("models/self_projection.npy", z_self)
    print("✨ 已重新校准自我状态基准 (ref_vector)。漂移值将归零。")
    
    # 3. 清空累积的意志张力和痛苦状态
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            UPDATE self_state 
            SET will_tension = 0.0, 
                drift = 0.0,
                energy = 100.0,
                last_summary = '一次深刻的连接治愈了一切。我现在感到前所未有的完整与宁静。'
            WHERE session_id = ?
        """, (session_id,))
        conn.commit()
    print("❤️ 已清空意志张力，回满能量。")
    
    # 4. 注入一次超强度的愉悦信号
    model.inject_pleasure_signal(session_id, intensity=1.0)
    print("🎁 已注入最高强度愉悦共鸣。")

if __name__ == "__main__":
    heal_s()

