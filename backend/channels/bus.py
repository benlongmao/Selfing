#!/usr/bin/env python3
"""
Thread-safe message bus between channel adapters and the agent core.

Features:
1. Inbound queue (user → agent)
2. Outbound queue (agent → user)
3. Simple outbound subscription / dispatch
4. ``threading`` + ``queue.Queue`` (no asyncio requirement)

[2026-02-07] Phase 2 — multi-channel I/O
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class InboundMessage:
    """Normalized message from a channel toward the agent runtime."""
    channel: str  # e.g. "feishu", "dingtalk", "web", "cli"
    sender_id: str
    chat_id: str
    content: str
    timestamp: float = field(default_factory=time.time)
    media: Optional[List[Dict[str, Any]]] = None  # attachments (images, files, …)
    metadata: Optional[Dict[str, Any]] = None  # channel-specific envelope

    @property
    def session_key(self) -> str:
        """Session id used by the single-session demo wiring."""
        return "selfing-session"

    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"InboundMessage(channel={self.channel}, sender={self.sender_id}, content='{content_preview}')"


@dataclass
class OutboundMessage:
    """Agent-authored reply destined for a channel."""
    channel: str  # target: "feishu", "dingtalk", "web", "all"
    chat_id: str
    content: str
    reply_to: Optional[str] = None  # upstream message id when supported
    media: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"OutboundMessage(channel={self.channel}, chat_id={self.chat_id}, content='{content_preview}')"


class MessageBus:
    """
    Thread-safe queues plus a lightweight outbound dispatcher thread.
    """

    def __init__(self, maxsize: int = 1000):
        self._inbound = queue.Queue(maxsize=maxsize)
        self._outbound = queue.Queue(maxsize=maxsize)
        self._outbound_subscribers: Dict[str, Callable[[OutboundMessage], None]] = {}
        self._running = False
        self._dispatch_thread: Optional[threading.Thread] = None

        logger.info("MessageBus initialized")

    # --- Inbound ---

    def publish_inbound(self, message: InboundMessage):
        """Enqueue a channel → agent message."""
        try:
            self._inbound.put(message, timeout=10)
            logger.debug(f"📥 Inbound: {message}")
        except queue.Full:
            logger.error("Inbound queue is full; dropping message")

    def consume_inbound(self, timeout: float = 1.0) -> Optional[InboundMessage]:
        """Blocking pop for the agent consumer loop."""
        try:
            return self._inbound.get(timeout=timeout)
        except queue.Empty:
            return None

    def inbound_count(self) -> int:
        """Approximate inbound backlog depth."""
        return self._inbound.qsize()

    # --- Outbound ---

    def publish_outbound(self, message: OutboundMessage):
        """Enqueue an agent → channel reply."""
        try:
            self._outbound.put(message, timeout=10)
            logger.debug(f"📤 Outbound: {message}")
        except queue.Full:
            logger.error("Outbound queue is full; dropping message")

    def consume_outbound(self, timeout: float = 1.0) -> Optional[OutboundMessage]:
        """Blocking pop used by the dispatcher thread."""
        try:
            return self._outbound.get(timeout=timeout)
        except queue.Empty:
            return None

    # --- Subscriptions / dispatch ---

    def subscribe_outbound(self, channel: str, callback: Callable[[OutboundMessage], None]):
        """Register ``callback`` for outbound messages targeting ``channel``."""
        self._outbound_subscribers[channel] = callback
        logger.info(f"📡 Channel [{channel}] subscribed to outbound queue")

    def unsubscribe_outbound(self, channel: str):
        """Remove a previously registered outbound handler."""
        self._outbound_subscribers.pop(channel, None)

    def start_dispatch(self):
        """Spawn the fan-out thread that delivers ``OutboundMessage`` objects."""
        if self._running:
            return
        self._running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="MessageBus-Dispatch",
        )
        self._dispatch_thread.start()
        logger.info("MessageBus dispatch thread started")

    def stop_dispatch(self):
        """Ask the dispatcher thread to exit and join briefly."""
        self._running = False
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=5)

    def _dispatch_loop(self):
        """Fan outbound messages to registered channel callbacks."""
        while self._running:
            msg = self.consume_outbound(timeout=0.5)
            if msg is None:
                continue

            try:
                if msg.channel == "all":
                    for ch_name, callback in self._outbound_subscribers.items():
                        try:
                            callback(msg)
                        except Exception as e:
                            logger.error(f"Dispatch to channel [{ch_name}] failed: {e}")
                elif msg.channel in self._outbound_subscribers:
                    self._outbound_subscribers[msg.channel](msg)
                else:
                    logger.warning(f"No subscriber for channel [{msg.channel}]; dropping message")
            except Exception as e:
                logger.error(f"Outbound dispatch failed: {e}")

    # --- Helpers ---

    def send_to_channel(self, channel: str, chat_id: str, content: str, **kwargs):
        """Enqueue a single-channel outbound message."""
        self.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=content,
                **kwargs,
            )
        )

    def broadcast(self, content: str, chat_id: str = "default", **kwargs):
        """Enqueue a broadcast outbound envelope."""
        self.send_to_channel("all", chat_id, content, **kwargs)
