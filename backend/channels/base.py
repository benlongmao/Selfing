#!/usr/bin/env python3
"""
Abstract base type for outbound chat channels (Feishu, DingTalk, web, …).

[2026-02-07] Phase 2 — multi-channel I/O
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Set

from backend.channels.bus import MessageBus, InboundMessage

logger = logging.getLogger(__name__)


class BaseChannel(ABC):
    """
    Subclasses must implement ``start``, ``stop``, and ``send``.
    """

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        allowed_users: Optional[Set[str]] = None,
    ):
        self.name = name
        self.bus = bus
        self.allowed_users = allowed_users
        self._running = False

        logger.info(f"Channel [{name}] initialized")

    @abstractmethod
    def start(self):
        """Connect / subscribe / begin receiving."""
        pass

    @abstractmethod
    def stop(self):
        """Tear down sockets and background work."""
        pass

    @abstractmethod
    def send(self, chat_id: str, content: str, **kwargs):
        """Deliver ``content`` to the remote chat/thread identified by ``chat_id``."""
        pass

    def is_allowed(self, sender_id: str) -> bool:
        """Return False when ``allowed_users`` is set and ``sender_id`` is not listed."""
        if self.allowed_users is None:
            return True  # no allowlist → accept all senders
        return sender_id in self.allowed_users

    def _handle_incoming(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: Optional[List] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Validate allowlist, wrap as ``InboundMessage``, publish to the bus.

        Intended for subclass protocol adapters after they normalize payloads.
        """
        if not self.is_allowed(sender_id):
            logger.warning(
                f"Channel [{self.name}]: rejected message from {sender_id} (not in allowlist)"
            )
            return

        message = InboundMessage(
            channel=self.name,
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media,
            metadata=metadata,
        )

        self.bus.publish_inbound(message)
        logger.info(f"Channel [{self.name}]: inbound from {sender_id}: {content[:50]}...")
