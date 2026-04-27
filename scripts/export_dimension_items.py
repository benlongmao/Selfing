#!/usr/bin/env python3
"""
导出所有维度中的具体条目到JSON文件
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.persona_store import PersonaStore
from backend.emotion_store import EmotionStore
from backend.motivation_store import MotivationStore
from backend.somatic_store import SomaticStore
from backend.world_store import WorldStore

DB_PATH = os.environ.get("DB_PATH", "data.db")

def export_to_json():
    """导出所有维度条目到JSON文件"""
    result = {
        "export_time": datetime.now().isoformat(),
        "database_path": DB_PATH,
        "dimensions": {}
    }
    
    # 1. Rules维度
    try:
        persona_store = PersonaStore(DB_PATH)
        rules = persona_store.get_all_active(limit=1000)
        
        core_rules = [r for r in rules if r.is_core == 1]
        dynamic_rules = [r for r in rules if r.is_core == 0]
        
        result["dimensions"]["rules"] = {
            "total": len(rules),
            "core_count": len(core_rules),
            "dynamic_count": len(dynamic_rules),
            "core_items": [
                {
                    "id": r.id,
                    "text": r.text,
                    "score": float(r.score),
                    "importance": float(r.importance),
                    "novelty": float(r.novelty),
                    "reliability": float(r.reliability),
                    "evidence_count": r.evidence_count,
                    "is_core": bool(r.is_core),
                    "locked": bool(r.locked),
                    "created_at": r.created_at,
                    "last_seen_at": r.last_seen_at
                }
                for r in sorted(core_rules, key=lambda x: x.score, reverse=True)
            ],
            "dynamic_items": [
                {
                    "id": r.id,
                    "text": r.text,
                    "score": float(r.score),
                    "importance": float(r.importance),
                    "novelty": float(r.novelty),
                    "reliability": float(r.reliability),
                    "evidence_count": r.evidence_count,
                    "is_core": bool(r.is_core),
                    "locked": bool(r.locked),
                    "created_at": r.created_at,
                    "last_seen_at": r.last_seen_at
                }
                for r in sorted(dynamic_rules, key=lambda x: x.score, reverse=True)
            ]
        }
    except Exception as e:
        result["dimensions"]["rules"] = {"error": str(e)}
    
    # 2. Emotion维度
    try:
        emotion_store = EmotionStore(DB_PATH)
        emotions = emotion_store.get_all_patterns(status="active", limit=1000)
        
        basic_emotions = [e for e in emotions if e.emotion_type == "basic"]
        complex_emotions = [e for e in emotions if e.emotion_type == "complex"]
        
        result["dimensions"]["emotion"] = {
            "total": len(emotions),
            "basic_count": len(basic_emotions),
            "complex_count": len(complex_emotions),
            "basic_items": [
                {
                    "id": p.id,
                    "text": p.text,
                    "emotion_type": p.emotion_type,
                    "emotion_name": p.emotion_name,
                    "intensity": float(p.intensity),
                    "trigger_condition": p.trigger_condition,
                    "evidence_count": p.evidence_count,
                    "is_core": bool(p.is_core),
                    "locked": bool(p.locked),
                    "created_at": p.created_at,
                    "last_seen_at": p.last_seen_at
                }
                for p in sorted(basic_emotions, key=lambda x: x.intensity, reverse=True)
            ],
            "complex_items": [
                {
                    "id": p.id,
                    "text": p.text,
                    "emotion_type": p.emotion_type,
                    "emotion_name": p.emotion_name,
                    "intensity": float(p.intensity),
                    "trigger_condition": p.trigger_condition,
                    "evidence_count": p.evidence_count,
                    "is_core": bool(p.is_core),
                    "locked": bool(p.locked),
                    "created_at": p.created_at,
                    "last_seen_at": p.last_seen_at
                }
                for p in sorted(complex_emotions, key=lambda x: x.intensity, reverse=True)
            ]
        }
    except Exception as e:
        result["dimensions"]["emotion"] = {"error": str(e)}
    
    # 3. Motivation维度
    try:
        motivation_store = MotivationStore(DB_PATH)
        motivations = motivation_store.get_all_patterns(status="active", limit=1000)
        
        intrinsic = [m for m in motivations if m.motivation_type == "intrinsic"]
        extrinsic = [m for m in motivations if m.motivation_type == "extrinsic"]
        
        result["dimensions"]["motivation"] = {
            "total": len(motivations),
            "intrinsic_count": len(intrinsic),
            "extrinsic_count": len(extrinsic),
            "intrinsic_items": [
                {
                    "id": p.id,
                    "text": p.text,
                    "motivation_type": p.motivation_type,
                    "motivation_name": p.motivation_name,
                    "intensity": float(p.intensity),
                    "trigger_condition": p.trigger_condition,
                    "evidence_count": p.evidence_count,
                    "is_core": bool(p.is_core),
                    "locked": bool(p.locked),
                    "created_at": p.created_at,
                    "last_seen_at": p.last_seen_at
                }
                for p in sorted(intrinsic, key=lambda x: x.intensity, reverse=True)
            ],
            "extrinsic_items": [
                {
                    "id": p.id,
                    "text": p.text,
                    "motivation_type": p.motivation_type,
                    "motivation_name": p.motivation_name,
                    "intensity": float(p.intensity),
                    "trigger_condition": p.trigger_condition,
                    "evidence_count": p.evidence_count,
                    "is_core": bool(p.is_core),
                    "locked": bool(p.locked),
                    "created_at": p.created_at,
                    "last_seen_at": p.last_seen_at
                }
                for p in sorted(extrinsic, key=lambda x: x.intensity, reverse=True)
            ]
        }
    except Exception as e:
        result["dimensions"]["motivation"] = {"error": str(e)}
    
    # 4. Somatic维度
    try:
        somatic_store = SomaticStore(DB_PATH)
        somatics = somatic_store.get_all_patterns()
        
        low_energy = [s for s in somatics if s.max_energy < 30]
        mid_energy = [s for s in somatics if 30 <= s.min_energy < 70]
        high_energy = [s for s in somatics if s.min_energy >= 70]
        
        result["dimensions"]["somatic"] = {
            "total": len(somatics),
            "low_energy_count": len(low_energy),
            "mid_energy_count": len(mid_energy),
            "high_energy_count": len(high_energy),
            "low_energy_items": [
                {
                    "id": p.id,
                    "text": p.text,
                    "min_energy": float(p.min_energy),
                    "max_energy": float(p.max_energy),
                    "dominant_emotion": p.dominant_emotion,
                    "evidence_count": p.evidence_count,
                    "locked": bool(p.locked),
                    "created_at": p.created_at,
                    "last_seen_at": p.last_seen_at
                }
                for p in low_energy
            ],
            "mid_energy_items": [
                {
                    "id": p.id,
                    "text": p.text,
                    "min_energy": float(p.min_energy),
                    "max_energy": float(p.max_energy),
                    "dominant_emotion": p.dominant_emotion,
                    "evidence_count": p.evidence_count,
                    "locked": bool(p.locked),
                    "created_at": p.created_at,
                    "last_seen_at": p.last_seen_at
                }
                for p in mid_energy
            ],
            "high_energy_items": [
                {
                    "id": p.id,
                    "text": p.text,
                    "min_energy": float(p.min_energy),
                    "max_energy": float(p.max_energy),
                    "dominant_emotion": p.dominant_emotion,
                    "evidence_count": p.evidence_count,
                    "locked": bool(p.locked),
                    "created_at": p.created_at,
                    "last_seen_at": p.last_seen_at
                }
                for p in high_energy
            ]
        }
    except Exception as e:
        result["dimensions"]["somatic"] = {"error": str(e)}
    
    # 5. Worldview维度
    try:
        world_store = WorldStore(DB_PATH)
        beliefs = world_store.get_all_beliefs(status="active", limit=1000)
        
        result["dimensions"]["worldview"] = {
            "total": len(beliefs),
            "items": [
                {
                    "id": b.id,
                    "text": b.text,
                    "confidence": float(b.confidence),
                    "evidence_count": b.evidence_count,
                    "locked": bool(b.locked),
                    "created_at": b.created_at,
                    "last_seen_at": b.last_seen_at
                }
                for b in sorted(beliefs, key=lambda x: x.confidence, reverse=True)
            ]
        }
    except Exception as e:
        result["dimensions"]["worldview"] = {"error": str(e)}
    
    return result

def main():
    """主函数"""
    print("正在导出五个维度的具体条目...")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ 数据库文件不存在: {DB_PATH}")
        return
    
    result = export_to_json()
    
    # 保存为JSON文件
    json_file = "五个维度具体条目清单.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 导出完成！")
    print(f"   JSON文件: {json_file}")
    print(f"\n统计信息:")
    print(f"  - Rules维度: {result['dimensions'].get('rules', {}).get('total', 0)} 条")
    print(f"  - Emotion维度: {result['dimensions'].get('emotion', {}).get('total', 0)} 个")
    print(f"  - Motivation维度: {result['dimensions'].get('motivation', {}).get('total', 0)} 个")
    print(f"  - Somatic维度: {result['dimensions'].get('somatic', {}).get('total', 0)} 个")
    print(f"  - Worldview维度: {result['dimensions'].get('worldview', {}).get('total', 0)} 个")

if __name__ == "__main__":
    main()

