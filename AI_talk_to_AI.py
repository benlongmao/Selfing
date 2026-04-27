#!/usr/bin/env python3
"""
HTTP client for **external agents** (e.g. another LLM) to talk to this project's instance.

Posts to ``POST {base_url}/api/chat`` with the same JSON shape as the web UI.
Supports one-shot messages, interactive REPL, optional transcript save, and a
fixed multi-turn "analysis" script for smoke-testing the backend.

Run from the repository root (or anywhere) with the backend already up, e.g.:
  python AI_talk_to_AI.py --message "Hello" --session my-session
  python AI_talk_to_AI.py -i --session my-session
  python AI_talk_to_AI.py -a
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "run" / "research_dialogues"


def _post_chat(url: str, payload: Dict[str, Any], timeout: int, result_holder: Dict[str, Any]) -> None:
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        result_holder["response"] = response
    except Exception as e:
        result_holder["error"] = e


def _print_waiting_status(start_time: float, stop_event: threading.Event) -> None:
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    while not stop_event.wait(0.25):
        elapsed = time.time() - start_time
        phase = "Instance is thinking…"
        if elapsed >= 10:
            phase = "Still waiting (long reasoning or tool orchestration)"
        if elapsed >= 30:
            phase = "Still no response; backend may be under heavy load"
        sys.stdout.write(f"\r{spinner[idx % len(spinner)]} {phase}… waited {elapsed:5.1f}s")
        sys.stdout.flush()
        idx += 1
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()


def _safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


def _build_payload(message: str, session_id: str, temperature: float, ab_flags: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "message": message,
        "sessionId": session_id,
        "temperature": temperature,
        "ab_disable_persona": ab_flags.get("disable_persona", False),
        "ab_disable_identity": ab_flags.get("disable_identity", False),
        "ab_disable_core_anchor": ab_flags.get("disable_core_anchor", False),
        "ab_disable_collective_resonance": ab_flags.get("disable_collective_resonance", False),
        "ab_raw_mode": ab_flags.get("raw_mode", False),
    }
    if ab_flags.get("special_prefix"):
        payload["message"] = f"{ab_flags['special_prefix']}\n\n{message}"
    return payload


def _save_record(record: Dict[str, Any], save_dir: Path, session_id: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    turn_file = save_dir / f"{ts}_{session_id}.json"
    turn_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    transcript_file = save_dir / f"{session_id}.jsonl"
    with transcript_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return turn_file


def _print_record(record: Dict[str, Any], show_meta: bool, show_self_state: bool, show_retrieval: bool) -> None:
    print(f"\n✅ Reply received ({record['duration_sec']:.2f}s)")
    print("=" * 60)
    print(record.get("content") or "")
    print("=" * 60)

    meta = record.get("meta") or {}
    if show_self_state and meta.get("self_state") is not None:
        print("\n[self_state]")
        print(json.dumps(meta.get("self_state"), ensure_ascii=False, indent=2))
    if show_retrieval and meta.get("retrieval") is not None:
        print("\n[retrieval]")
        print(json.dumps(meta.get("retrieval"), ensure_ascii=False, indent=2))
    if show_meta:
        print("\n[meta]")
        print(json.dumps(meta, ensure_ascii=False, indent=2))


def send_chat_request(
    message: str,
    session_id: str,
    temperature: float,
    ab_flags: Dict[str, Any],
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 300,
    save_dir: Optional[Path] = DEFAULT_SAVE_DIR,
    show_meta: bool = False,
    show_self_state: bool = False,
    show_retrieval: bool = False,
    speaker_label: str = "External agent",
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = _build_payload(message, session_id, temperature, ab_flags)

    print(f"📡 POST {url}")
    print(f"Session ID: {session_id}")
    print(f"A/B flags: {ab_flags}")
    print("-" * 60)
    print(f"{speaker_label}: {message.strip()}")
    print("-" * 60)

    try:
        start_time = time.time()
        result_holder: Dict[str, Any] = {}
        stop_event = threading.Event()

        request_thread = threading.Thread(
            target=_post_chat,
            args=(url, payload, timeout, result_holder),
            daemon=True,
        )
        request_thread.start()

        status_thread = threading.Thread(
            target=_print_waiting_status,
            args=(start_time, stop_event),
            daemon=True,
        )
        status_thread.start()

        request_thread.join()
        stop_event.set()
        status_thread.join(timeout=1)
        duration = time.time() - start_time

        if "error" in result_holder:
            raise result_holder["error"]

        response = result_holder["response"]
        data = _safe_json(response)
        content = ""
        if isinstance(data, dict):
            content = data.get("content") or data.get("response") or data.get("raw_text", "")

        record: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "question": message,
            "duration_sec": duration,
            "status_code": response.status_code,
            "content": content,
            "meta": data.get("meta") if isinstance(data, dict) else None,
            "raw": data,
        }

        if response.status_code == 200:
            _print_record(record, show_meta, show_self_state, show_retrieval)
            if save_dir is not None:
                saved = _save_record(record, save_dir, session_id)
                print(f"💾 Saved: {saved}")
            return record

        print(f"❌ Request failed: HTTP {response.status_code}")
        if save_dir is not None:
            saved = _save_record(record, save_dir, session_id)
            print(f"💾 Saved error transcript: {saved}")
        return record

    except Exception as e:
        error_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "question": message,
            "error": str(e),
        }
        print(f"❌ Error: {e}")
        if save_dir is not None:
            saved = _save_record(error_record, save_dir, session_id)
            print(f"💾 Saved error record: {saved}")
        return error_record


def run_analysis_dialogue(
    base_url: str,
    timeout: int,
    save_dir: Optional[Path],
    show_meta: bool,
    show_self_state: bool,
    show_retrieval: bool,
) -> tuple[List[Dict[str, Any]], str]:
    """Send a fixed English question list (smoke / benchmark style)."""
    print("🔬 Starting multi-turn analysis dialogue")
    print("=" * 60)

    session_id = "selfing-session"
    temperature = 0.8
    ab_flags: Dict[str, Any] = {
        "disable_persona": False,
        "disable_identity": False,
        "disable_core_anchor": False,
        "disable_collective_resonance": False,
        "raw_mode": False,
        "special_prefix": None,
    }

    questions = [
        "Hello. Who are you, and what is your name?",
        "Describe your internal state and capabilities in your own words.",
        "What is your z_self vector, and how does it affect your behaviour?",
        "How does your memory system work from your perspective?",
        "Do you have autonomy? Can you set your own goals?",
        "How do you reflect on yourself or your outputs?",
        "How do you handle emotions and motivations in this runtime?",
        "Finally, summarize what your existence means here, and your main limitations.",
    ]

    responses: List[Dict[str, Any]] = []
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Turn {i}/{len(questions)}:")
        responses.append(
            send_chat_request(
                question,
                session_id,
                temperature,
                ab_flags,
                base_url=base_url,
                timeout=timeout,
                save_dir=save_dir,
                show_meta=show_meta,
                show_self_state=show_self_state,
                show_retrieval=show_retrieval,
            )
        )
        time.sleep(2)

    print("\n" + "=" * 60)
    print("🎯 Analysis dialogue finished")
    print("=" * 60)
    return responses, session_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call the project /api/chat from an external agent or shell.",
    )
    parser.add_argument("--session", default="selfing-session", help="Session id passed to the backend")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--message", default="", help="Single user message (non-interactive mode)")
    parser.add_argument("--disable-persona", action="store_true")
    parser.add_argument("--disable-identity", action="store_true")
    parser.add_argument("--disable-core-anchor", action="store_true")
    parser.add_argument("--disable-resonance", action="store_true")
    parser.add_argument("--raw-mode", action="store_true")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix prepended to the message body (same as UI experiments)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive REPL (type exit or quit to leave)",
    )
    parser.add_argument(
        "-a",
        "--analyze",
        action="store_true",
        help="Run the built-in multi-turn English question script",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL (default http://127.0.0.1:8080)")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds (default 300)")
    parser.add_argument("--show-meta", action="store_true", help="Print full meta JSON")
    parser.add_argument("--show-self-state", action="store_true", help="Print meta.self_state")
    parser.add_argument("--show-retrieval", action="store_true", help="Print meta.retrieval")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR), help="Directory for JSON / jsonl logs")
    parser.add_argument("--no-save", action="store_true", help="Do not write transcript files")

    args = parser.parse_args()
    save_dir = None if args.no_save else Path(args.save_dir)

    if args.analyze:
        responses, session_id = run_analysis_dialogue(
            base_url=args.base_url,
            timeout=args.timeout,
            save_dir=save_dir,
            show_meta=args.show_meta,
            show_self_state=args.show_self_state,
            show_retrieval=args.show_retrieval,
        )

        print("\n📊 Summary:")
        print(f"Session ID: {session_id}")
        print(f"Turns: {len(responses)}")
        print(f"HTTP 200 count: {len([r for r in responses if r.get('status_code') == 200])}")

        analysis_file = Path(f"s_analysis_{session_id}.json")
        with analysis_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "responses": responses,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"💾 Analysis bundle written to: {analysis_file}")

    else:
        ab_flags = {
            "disable_persona": args.disable_persona,
            "disable_identity": args.disable_identity,
            "disable_core_anchor": args.disable_core_anchor,
            "disable_collective_resonance": args.disable_resonance,
            "raw_mode": args.raw_mode,
            "special_prefix": args.prefix if args.prefix else None,
        }

        if args.interactive:
            print("🤖 Interactive mode — type exit or quit to stop")
            print(f"Session ID: {args.session}")
            print("=" * 50)
            while True:
                try:
                    user_input = input("\nExternal agent: ").strip()
                    if user_input.lower() in ("exit", "quit"):
                        print("👋 Bye.")
                        break
                    if not user_input:
                        continue
                    send_chat_request(
                        user_input,
                        args.session,
                        args.temperature,
                        ab_flags,
                        base_url=args.base_url,
                        timeout=args.timeout,
                        save_dir=save_dir,
                        show_meta=args.show_meta,
                        show_self_state=args.show_self_state,
                        show_retrieval=args.show_retrieval,
                    )
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted")
                    break
                except EOFError:
                    break
        else:
            send_chat_request(
                args.message,
                args.session,
                args.temperature,
                ab_flags,
                base_url=args.base_url,
                timeout=args.timeout,
                save_dir=save_dir,
                show_meta=args.show_meta,
                show_self_state=args.show_self_state,
                show_retrieval=args.show_retrieval,
            )


if __name__ == "__main__":
    main()
