"""
Multi-channel messaging for s-main.

Channel implementations plus a thread-safe ``MessageBus``, following a small
Channel + bus split (nanobot-style), adapted to this codebase.

[2026-02-07] Phase 2 — multi-channel I/O
"""

from backend.channels.bus import MessageBus, InboundMessage, OutboundMessage
from backend.channels.base import BaseChannel
from backend.channels.manager import ChannelManager

__all__ = [
    "MessageBus",
    "InboundMessage",
    "OutboundMessage",
    "BaseChannel",
    "ChannelManager",
]
