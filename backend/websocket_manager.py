"""
WebSocket connection manager for pushing agent updates to the browser in real time.

Design:
1. Group connections by ``session_id``.
2. Sync threads enqueue via ``queue_message()`` (thread-safe).
3. The FastAPI event loop drains the queue and sends over each socket.
"""

from typing import Dict, List, Optional, Any
from fastapi import WebSocket
import asyncio
import json
import logging
import queue
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Per-process WebSocket hub (one manager, many sessions)."""

    def __init__(self):
        # session_id -> list of websocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Cross-thread queue: sync producers -> async consumer on the main loop
        self._message_queue: queue.Queue = queue.Queue()
        # Main asyncio loop registered at startup (or lazily on first connect)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Bind the running FastAPI / Starlette event loop (call from ``startup``)."""
        self._loop = loop

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a client socket and register it under ``session_id``."""
        await websocket.accept()
        # Fallback: if startup did not register the loop, bind on first connect
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
                logger.info("[WS] Event loop bound on first WebSocket connect")
            except RuntimeError:
                pass
        with self._lock:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = []
            self.active_connections[session_id].append(websocket)
        logger.info(
            "[WS] Connected: session=%s, total=%s",
            session_id,
            len(self.active_connections.get(session_id, [])),
        )

    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a socket from its session bucket."""
        with self._lock:
            if session_id in self.active_connections:
                if websocket in self.active_connections[session_id]:
                    self.active_connections[session_id].remove(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
        logger.info("[WS] Disconnected: session=%s", session_id)

    def get_connection_count(self, session_id: str = None) -> int:
        """Return live socket count for one session or the whole process."""
        with self._lock:
            if session_id:
                return len(self.active_connections.get(session_id, []))
            return sum(len(conns) for conns in self.active_connections.values())

    async def send_to_session(self, session_id: str, message: dict):
        """JSON-encode ``message`` and send to every socket in ``session_id``."""
        with self._lock:
            connections = self.active_connections.get(session_id, []).copy()

        if not connections:
            logger.debug("[WS] No connections for session=%s", session_id)
            return

        message_json = json.dumps(message, ensure_ascii=False)
        disconnected = []

        for connection in connections:
            try:
                await connection.send_text(message_json)
                logger.debug("[WS] Sent to session=%s type=%s", session_id, message.get("type"))
            except Exception as e:
                logger.warning("[WS] Failed to send: %s", e)
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn, session_id)

    async def broadcast(self, message: dict):
        """Fan-out one payload to every tracked connection."""
        message_json = json.dumps(message, ensure_ascii=False)
        with self._lock:
            all_connections = [
                (sid, conn)
                for sid, conns in self.active_connections.items()
                for conn in conns
            ]

        for session_id, connection in all_connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                pass

    def queue_message(self, session_id: str, message: dict):
        """
        Enqueue a message from any thread; the asyncio loop will send it.

        Used by resting pulse / scheduler hooks that are not ``async``.
        """
        self._message_queue.put((session_id, message))
        logger.debug("[WS] Queued message for session=%s type=%s", session_id, message.get("type"))

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._process_queue(), self._loop)

    async def _process_queue(self):
        """Drain the cross-thread queue on the asyncio loop."""
        while not self._message_queue.empty():
            try:
                session_id, message = self._message_queue.get_nowait()
                await self.send_to_session(session_id, message)
            except queue.Empty:
                break
            except Exception as e:
                logger.error("[WS] Error processing queue: %s", e)


class WSMessageType:
    """Stable ``type`` string constants for the frontend."""

    NEW_MESSAGE = "new_message"  # assistant-initiated bubble
    STATE_UPDATE = "state_update"  # energy / affect / counters
    NOTIFICATION = "notification"  # task done, reminder, etc.
    HEARTBEAT = "heartbeat"
    PRESENCE_PULSE = "presence_pulse"
    THINKING = "thinking"  # 2026-02-27: show “thinking…” before first token


def create_ws_message(
    msg_type: str,
    content: Any = None,
    session_id: str = None,
    **extra
) -> dict:
    """Build the canonical JSON envelope consumed by ``index.html``."""
    return {
        "type": msg_type,
        "content": content,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        **extra,
    }


_manager: Optional[ConnectionManager] = None
_manager_lock = threading.Lock()


def get_websocket_manager() -> ConnectionManager:
    """Process-wide singleton (lazy)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ConnectionManager()
            logger.info("[WS] ConnectionManager initialized")
    return _manager
