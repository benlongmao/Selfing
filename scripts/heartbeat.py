#!/usr/bin/env python3
"""
心跳脚本：模拟"持续存在" (Continuous Existence)
每隔一定时间调用 /tick 接口，触发 AI 的后台思考与自我更新。
建议在 screen 或 nohup 中运行。
"""
import time
import requests
import logging
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("heartbeat.log")
    ]
)
logger = logging.getLogger("heartbeat")

# SelfTick existence heartbeat.
# Prefer SELF_TICK_HEARTBEAT_INTERVAL so it does not collide with the in-app
# HEARTBEAT.md task heartbeat interval.
API_BASE = os.environ.get("API_BASE", "http://localhost:8080")
TICK_INTERVAL = int(
    os.environ.get("SELF_TICK_HEARTBEAT_INTERVAL")
    or os.environ.get("HEARTBEAT_INTERVAL")
    or "1800"
)
SESSION_ID = os.environ.get("HEARTBEAT_SESSION_ID", "selfing-session")

def trigger_tick():
    """调用 tick 接口"""
    try:
        # 使用正确的 Endpoint: /self/tick (定义在 backend/routers/self.py)
        url = f"{API_BASE}/self/tick"
        payload = {"sessionId": SESSION_ID} # 注意：pydantic模型使用 camelCase sessionId
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            # 检查是否有神游等触发（MetaCognition 已删除 2026-03）
            extras = []
            
            # 由于 self_tick 可能会异步触发 mind_wandering，结果可能不在返回值里
            # 但我们可以通过日志看到。这里只记录基本的
            
            logger.info(f"Tick success ({latency:.2f}s). Drift: {data.get('drift', 0.0):.4f}, Extras: {extras}")
        else:
            logger.warning(f"Tick failed ({response.status_code}): {response.text}")
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection refused. Is the backend server running at {API_BASE}?")
    except Exception as e:
        logger.error(f"Tick error: {e}")

def main():
    logger.info(f"Starting heartbeat script. Interval: {TICK_INTERVAL}s")
    logger.info("The AI is now 'living' in the background...")
    
    while True:
        trigger_tick()
        time.sleep(TICK_INTERVAL)

if __name__ == "__main__":
    main()

