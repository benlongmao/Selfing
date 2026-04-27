#!/usr/bin/env python3
"""
Registers channel implementations, wires outbound callbacks, and coordinates lifecycle.

[2026-02-07] Phase 2 — multi-channel I/O
"""

import logging
import threading
from typing import Dict, Optional, List

from backend.channels.bus import MessageBus, OutboundMessage
from backend.channels.base import BaseChannel
from backend.config import config

logger = logging.getLogger(__name__)


class ChannelManager:
    """
    Responsibilities:
    1. Instantiate channels from configuration
    2. Start/stop every registered adapter
    3. Subscribe each channel to outbound ``MessageBus`` traffic
    4. Expose lightweight status snapshots
    """

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.channels: Dict[str, BaseChannel] = {}
        self._started = False

        logger.info("ChannelManager initialized")

    def register(self, channel: BaseChannel):
        """Store ``channel`` and bind its ``send`` to outbound events."""
        self.channels[channel.name] = channel
        self.bus.subscribe_outbound(channel.name, self._make_send_callback(channel))
        logger.info(f"Channel [{channel.name}] registered")

    def _make_send_callback(self, channel: BaseChannel):
        """Return a closure that forwards ``OutboundMessage`` payloads."""

        def callback(msg: OutboundMessage):
            try:
                channel.send(msg.chat_id, msg.content, metadata=msg.metadata)
            except Exception as e:
                logger.error(f"Channel [{channel.name}] send failed: {e}")

        return callback

    def start_all(self):
        """Spawn per-channel threads and start bus dispatch."""
        if self._started:
            logger.warning("ChannelManager already started")
            return

        self._started = True

        for name, channel in self.channels.items():
            try:
                thread = threading.Thread(
                    target=self._start_channel,
                    args=(channel,),
                    daemon=True,
                    name=f"Channel-{name}",
                )
                thread.start()
                logger.info(f"Channel [{name}] start initiated")
            except Exception as e:
                logger.error(f"Failed to start channel [{name}]: {e}")

        self.bus.start_dispatch()

    def _start_channel(self, channel: BaseChannel):
        """Invoke ``channel.start()`` inside its dedicated thread."""
        try:
            channel.start()
        except Exception as e:
            logger.error(f"Channel [{channel.name}] error: {e}")

    def stop_all(self):
        """Stop adapters then join the dispatcher."""
        logger.info("Stopping all channels...")

        for name, channel in self.channels.items():
            try:
                channel.stop()
                logger.info(f"Channel [{name}] stopped")
            except Exception as e:
                logger.error(f"Failed to stop channel [{name}]: {e}")

        self.bus.stop_dispatch()
        self._started = False

    def get_status(self) -> Dict[str, dict]:
        """Return a shallow status dict per registered channel."""
        status = {}
        for name, channel in self.channels.items():
            status[name] = {
                "name": name,
                "running": channel._running,
                "type": type(channel).__name__,
            }
        return status

    def list_channels(self) -> List[str]:
        """Names of registered channels."""
        return list(self.channels.keys())

    @staticmethod
    def create_from_config(bus: MessageBus) -> "ChannelManager":
        """
        Build a manager from ``config`` keys under ``channels.*``.

        Reads ``config/settings.yaml`` (or merged runtime config) for Feishu/DingTalk blocks.
        """
        manager = ChannelManager(bus)

        feishu_config = config.get("channels.feishu")
        if feishu_config and feishu_config.get("enabled", False):
            try:
                from backend.channels.feishu import FeishuChannel

                channel = FeishuChannel(
                    bus=bus,
                    app_id=feishu_config.get("app_id", ""),
                    app_secret=feishu_config.get("app_secret", ""),
                    verification_token=feishu_config.get("verification_token", ""),
                    encrypt_key=feishu_config.get("encrypt_key", ""),
                    allowed_users=set(feishu_config.get("allowed_users", [])) or None,
                )
                manager.register(channel)
                logger.info("✨ Feishu channel registered")
            except ImportError as e:
                logger.warning(
                    f"Feishu channel dependency missing: {e}. Install with: pip install lark-oapi"
                )
            except Exception as e:
                logger.error(f"Feishu channel initialization failed: {e}")

        dingtalk_config = config.get("channels.dingtalk")
        if dingtalk_config and dingtalk_config.get("enabled", False):
            try:
                from backend.channels.dingtalk import DingTalkChannel

                channel = DingTalkChannel(
                    bus=bus,
                    app_key=dingtalk_config.get("app_key", ""),
                    app_secret=dingtalk_config.get("app_secret", ""),
                    robot_code=dingtalk_config.get("robot_code", ""),
                    allowed_users=set(dingtalk_config.get("allowed_users", [])) or None,
                )
                manager.register(channel)
                logger.info("✨ DingTalk channel registered")
            except ImportError as e:
                logger.warning(
                    f"DingTalk channel dependency missing: {e}. Install with: pip install dingtalk-stream"
                )
            except Exception as e:
                logger.error(f"DingTalk channel initialization failed: {e}")

        return manager
