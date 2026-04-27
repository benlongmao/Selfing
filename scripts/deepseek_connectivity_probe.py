#!/usr/bin/env python3
"""
DeepSeek 连通性/稳定性探测（从项目根 .env 读 DEEPSEEK_API_KEY，不打印密钥）。
用法（在仓库根）: PYTHONPATH=. python scripts/deepseek_connectivity_probe.py
可选: DEEPSEEK_FORCE_IPV4=1 PYTHONPATH=. python scripts/deepseek_connectivity_probe.py

实测结论（部分网络路径，如解析到国内 CDN eo.dnse1.com 时）：
- 短请求（GET /models、小 POST）可 200。
- 整段 JSON body 超过约 1.4KB～1.45KB 时，requests(HTTP/1.1) 常在 ~16s 出现
  RemoteDisconnected（与 S 里 DeepSeek 非流式/流式失败一致）。
- DEEPSEEK_FORCE_IPV4 不能消除该阈值。
- 这不是 S 业务逻辑 bug，而是当前出口到 api.deepseek.com 链路上的限制/异常。
  可尝试：换出口（代理/VPN/热点）、换 DNS、或改用不受该路径影响的 API 接入方式。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# 可选：与 S 启动一致，强制 IPv4
if (os.environ.get("DEEPSEEK_FORCE_IPV4") or "").strip().lower() in ("1", "true", "yes", "on"):
    try:
        import socket
        import urllib3.util.connection as u3c

        u3c.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[method-assign]
        print("[probe] DEEPSEEK_FORCE_IPV4 active")
    except Exception as e:
        print("[probe] DEEPSEEK_FORCE_IPV4 failed:", e)

import requests

BASE = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
KEY = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
if not KEY:
    print("ERROR: DEEPSEEK_API_KEY missing after load_dotenv")
    sys.exit(2)


def headers() -> dict:
    return {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "Connection": "close",
        "Accept-Encoding": "identity",
    }


def post_non_stream(payload: dict, timeout: tuple[float, float], label: str) -> tuple[bool, str]:
    url = f"{BASE}/chat/completions"
    t0 = time.perf_counter()
    try:
        r = requests.post(url, headers=headers(), json=payload, timeout=timeout)
        dt = time.perf_counter() - t0
        if not r.ok:
            return False, f"{label} HTTP {r.status_code} in {dt:.2f}s body={r.text[:120]!r}"
        data = r.json()
        ch0 = (data.get("choices") or [{}])[0]
        msg = ch0.get("message") or {}
        content = msg.get("content") or ""
        rc = msg.get("reasoning_content") or ""
        return True, f"{label} OK in {dt:.2f}s content_len={len(content)} reasoning_len={len(rc)}"
    except Exception as e:
        dt = time.perf_counter() - t0
        return False, f"{label} EX {type(e).__name__} in {dt:.2f}s {str(e)[:160]}"


def post_stream(payload: dict, timeout: tuple[float, float], label: str) -> tuple[bool, str]:
    url = f"{BASE}/chat/completions"
    body = dict(payload)
    body["stream"] = True
    t0 = time.perf_counter()
    n_chunks = 0
    nbytes = 0
    try:
        with requests.post(
            url, headers=headers(), json=body, timeout=timeout, stream=True
        ) as r:
            if r.status_code != 200:
                txt = r.text[:200] if r.text else ""
                return False, f"{label} HTTP {r.status_code} {txt!r}"
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    n_chunks += 1
                    # 粗算体积
                    nbytes += len(data)
        dt = time.perf_counter() - t0
        return True, f"{label} OK in {dt:.2f}s sse_events~={n_chunks} raw~={nbytes}b"
    except Exception as e:
        dt = time.perf_counter() - t0
        return False, f"{label} EX {type(e).__name__} in {dt:.2f}s {str(e)[:160]}"


def main() -> None:
    ok = 0
    fail = 0

    def run(one: tuple[bool, str]) -> None:
        nonlocal ok, fail
        good, msg = one
        print(msg)
        ok += int(good)
        fail += int(not good)

    # 0) models
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{BASE}/models", headers=headers(), timeout=(45, 60))
        dt = time.perf_counter() - t0
        run((r.status_code == 200, f"GET /models -> {r.status_code} in {dt:.2f}s"))
    except Exception as e:
        run((False, f"GET /models EX {type(e).__name__} {e!s}"[:200]))

    small = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Reply exactly: PING"}],
        "max_tokens": 16,
    }

    print("--- non-stream deepseek-chat x8 ---")
    for i in range(8):
        run(post_non_stream(small, (45, 120), f"nonstream_chat[{i}]"))

    print("--- stream deepseek-chat x5 ---")
    for i in range(5):
        run(post_stream(small, (45, 180), f"stream_chat[{i}]"))

    # 较大正文（拉长 POST body，模拟长上下文压力；控制 max_tokens 省费用）
    filler = ("段落测试。" * 400)  # ~2400 汉字量级，体积约数 KB～十余 KB 视编码
    large_user = filler + "\n最后只回答: LARGE_OK"
    large = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a probe. Follow the last line only."},
            {"role": "user", "content": large_user},
        ],
        "max_tokens": 32,
    }
    print("--- non-stream large body x3 ---")
    for i in range(3):
        run(post_non_stream(large, (45, 180), f"nonstream_large[{i}]"))

    print("--- stream large body x2 ---")
    for i in range(2):
        run(post_stream(large, (45, 300), f"stream_large[{i}]"))

    reasoner = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "2+2=? one digit only"}],
        "max_tokens": 64,
    }
    print("--- non-stream reasoner x2 ---")
    for i in range(2):
        run(post_non_stream(reasoner, (45, 240), f"nonstream_reasoner[{i}]"))

    print("--- stream reasoner x1 ---")
    run(post_stream(reasoner, (45, 300), "stream_reasoner[0]"))

    print(f"=== SUMMARY ok={ok} fail={fail} ===")


if __name__ == "__main__":
    main()
