#!/usr/bin/env python3
"""
DingTalk channel using Stream mode (WebSocket long connection).

Features:
1. Stream/WebSocket ingestion without a public inbound HTTP URL
2. Group + direct conversations
3. Markdown-capable replies (via API payload shapes)
4. Optional allowlist enforcement

Configure under ``config/settings.yaml``::

    channels:
      dingtalk:
        enabled: true
        app_key: "your_app_key"
        app_secret: "your_app_secret"
        robot_code: "your_robot_code"
        allowed_users: []

Prerequisites:
1. Create an app in the DingTalk developer console
2. Enable the bot capability
3. Enable Stream mode
4. Publish the app

[2026-02-07] Phase 2 — multi-channel I/O
"""

import json
import logging
import threading
from typing import Optional, Set, Any

from backend.channels.base import BaseChannel
from backend.channels.bus import MessageBus

logger = logging.getLogger(__name__)

try:
    import dingtalk_stream
    from dingtalk_stream import AckMessage

    DINGTALK_SDK_AVAILABLE = True
except ImportError:
    DINGTALK_SDK_AVAILABLE = False
    dingtalk_stream = None
    AckMessage = None

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class DingTalkCallbackHandler:
    """Stream callback that normalizes chatbot payloads into ``InboundMessage``."""

    def __init__(self, channel: "DingTalkChannel"):
        self.channel = channel

    def process(self, callback_data: dict) -> Any:
        """Parse ``callback_data`` and forward text into the bus."""
        try:
            msg_type = callback_data.get("msgtype", "text")
            sender_id = callback_data.get("senderStaffId", "")
            sender_nick = callback_data.get("senderNick", "")
            conversation_id = callback_data.get("conversationId", "")
            chat_type = callback_data.get("conversationType", "1")  # 1=direct, 2=group

            if msg_type == "text":
                text_content = callback_data.get("text", {})
                content = text_content.get("content", "").strip()
            elif msg_type == "richText":
                content = "[Rich text message]"
            elif msg_type == "picture":
                content = "[Image]"
            elif msg_type == "file":
                content = "[File]"
            else:
                content = f"[{msg_type}]"

            if not content:
                return AckMessage.STATUS_OK if AckMessage else None

            if chat_type == "2":
                content = content.strip()

            self.channel._handle_incoming(
                sender_id=sender_id,
                chat_id=conversation_id,
                content=content,
                metadata={
                    "sender_nick": sender_nick,
                    "chat_type": "group" if chat_type == "2" else "p2p",
                    "msg_type": msg_type,
                    "conversation_id": conversation_id,
                    "webhook_url": callback_data.get("sessionWebhook", ""),
                },
            )

            return AckMessage.STATUS_OK if AckMessage else None

        except Exception as e:
            logger.error(f"DingTalk message handling failed: {e}")
            return AckMessage.STATUS_SYSTEM_EXCEPTION if AckMessage else None


class DingTalkChannel(BaseChannel):
    """
    Primary path: ``dingtalk-stream`` WebSocket client.

    Fallback: HTTP helpers when Stream SDK is missing (limited; webhook-driven).
    """

    def __init__(
        self,
        bus: MessageBus,
        app_key: str,
        app_secret: str,
        robot_code: str = "",
        allowed_users: Optional[Set[str]] = None,
    ):
        super().__init__(name="dingtalk", bus=bus, allowed_users=allowed_users)
        self.app_key = app_key
        self.app_secret = app_secret
        self.robot_code = robot_code

        self._stream_client: Any = None
        self._access_token: Optional[str] = None
        self._token_expires: float = 0
        self._webhook_urls: dict = {}  # conversation_id -> session webhook

    def start(self):
        """Start Stream ingestion or enter degraded standby."""
        if DINGTALK_SDK_AVAILABLE:
            self._start_stream_mode()
        else:
            logger.warning(
                "DingTalk Stream SDK is not installed. Run: pip install dingtalk-stream\n"
                "Falling back to HTTP-only standby (no active polling in this build)."
            )
            self._running = True

    def _start_stream_mode(self):
        """Spin up ``DingTalkStreamClient`` (blocking inside the channel thread)."""
        if not self.app_key or not self.app_secret:
            logger.error("DingTalk app_key / app_secret are not configured")
            return

        self._running = True

        credential = dingtalk_stream.Credential(self.app_key, self.app_secret)

        self._stream_client = dingtalk_stream.DingTalkStreamClient(credential)

        handler = DingTalkCallbackHandler(self)
        self._stream_client.register_callback_handler(
            dingtalk_stream.ChatbotMessage.TOPIC,
            handler,
        )

        logger.info("🚀 DingTalk channel starting (Stream mode)...")
        try:
            self._stream_client.start_forever()
        except Exception as e:
            logger.error(f"DingTalk Stream connection failed: {e}")
            self._running = False

    def stop(self):
        """Stop Stream client if running."""
        self._running = False
        if self._stream_client:
            try:
                self._stream_client.stop()
            except Exception as e:
                logger.warning(f"DingTalk Stream stop raised: {e}")
        logger.info("DingTalk channel stopped")

    def send(self, chat_id: str, content: str, **kwargs):
        """
        Prefer cached session webhooks for group reliability, else OpenAPI batch send.
        """
        metadata = kwargs.get("metadata", {})

        webhook_url = self._webhook_urls.get(chat_id) or metadata.get("webhook_url")
        if webhook_url:
            self._send_via_webhook(webhook_url, content)
            return

        self._send_via_openapi(chat_id, content)

    def _send_via_webhook(self, webhook_url: str, content: str):
        """POST JSON to the conversation-scoped session webhook."""
        if not REQUESTS_AVAILABLE:
            logger.error("requests is not installed; cannot call DingTalk webhooks")
            return

        try:
            payload = {
                "msgtype": "text",
                "text": {
                    "content": content,
                },
            }

            resp = requests.post(webhook_url, json=payload, timeout=10)
            resp.raise_for_status()

            result = resp.json()
            if result.get("errcode", 0) != 0:
                logger.error(f"DingTalk webhook send failed: {result}")
            else:
                logger.debug("DingTalk message sent via webhook")

        except Exception as e:
            logger.error(f"DingTalk webhook send error: {e}")

    def _send_via_openapi(self, chat_id: str, content: str):
        """Robot OpenAPI path when webhook URL is unknown."""
        if not REQUESTS_AVAILABLE:
            logger.error("requests is not installed; cannot call DingTalk OpenAPI")
            return

        try:
            token = self._get_access_token()
            if not token:
                logger.error("Unable to obtain DingTalk access_token")
                return

            url = "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend"
            headers = {
                "x-acs-dingtalk-access-token": token,
                "Content-Type": "application/json",
            }
            payload = {
                "robotCode": self.robot_code,
                "userIds": [chat_id],
                "msgKey": "sampleText",
                "msgParam": json.dumps({"content": content}),
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=10)

            if resp.status_code == 200:
                logger.debug(f"DingTalk message sent via OpenAPI to {chat_id}")
            else:
                logger.error(f"DingTalk OpenAPI send failed: {resp.status_code} {resp.text}")

        except Exception as e:
            logger.error(f"DingTalk OpenAPI send error: {e}")

    def _get_access_token(self) -> Optional[str]:
        """Fetch and cache tenant access token."""
        import time

        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        if not REQUESTS_AVAILABLE:
            return None

        try:
            url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
            payload = {
                "appKey": self.app_key,
                "appSecret": self.app_secret,
            }

            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()

            data = resp.json()
            self._access_token = data.get("accessToken")
            expire_in = data.get("expireIn", 7200)
            self._token_expires = time.time() + expire_in - 300  # refresh 5 minutes early

            logger.debug(f"DingTalk access_token refreshed (ttl ~{expire_in}s)")
            return self._access_token

        except Exception as e:
            logger.error(f"DingTalk access_token fetch failed: {e}")
            return None

    def _handle_incoming(self, sender_id, chat_id, content, media=None, metadata=None):
        """Cache webhook URLs then delegate to ``BaseChannel``."""
        if metadata and metadata.get("webhook_url"):
            self._webhook_urls[chat_id] = metadata["webhook_url"]

        super()._handle_incoming(sender_id, chat_id, content, media, metadata)
