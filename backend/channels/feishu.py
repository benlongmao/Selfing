#!/usr/bin/env python3
"""
Feishu (Lark) channel using ``lark-oapi`` WebSocket long connection.

Features:
1. Event subscription without inbound public HTTP
2. Message de-duplication
3. Optional thumbs-up reaction as an ACK
4. Group + direct chats

Configure under ``config/settings.yaml``::

    channels:
      feishu:
        enabled: true
        app_id: "your_app_id"
        app_secret: "your_app_secret"
        verification_token: ""
        encrypt_key: ""
        allowed_users: []

Prerequisites:
1. Create an app on the Feishu developer portal
2. Enable bot capabilities
3. Subscribe to ``im.message.receive_v1``
4. Publish the app

[2026-02-07] Phase 2 — multi-channel I/O
"""

import json
import logging
from collections import OrderedDict
from typing import Optional, Set, Any

from backend.channels.base import BaseChannel
from backend.channels.bus import MessageBus

logger = logging.getLogger(__name__)

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        Emoji,
        P2ImMessageReceiveV1,
    )

    FEISHU_SDK_AVAILABLE = True
except ImportError:
    FEISHU_SDK_AVAILABLE = False
    lark = None
    Emoji = None

# Non-text payloads → compact English placeholders for the agent
MSG_TYPE_MAP = {
    "image": "[Image]",
    "audio": "[Voice message]",
    "file": "[File]",
    "sticker": "[Sticker]",
    "post": "[Rich text]",
    "interactive": "[Interactive card]",
}


class FeishuChannel(BaseChannel):
    """
    WebSocket client driven by ``lark.ws.Client``.

    Outbound sends use the REST client built from ``app_id`` / ``app_secret``.
    """

    def __init__(
        self,
        bus: MessageBus,
        app_id: str,
        app_secret: str,
        verification_token: str = "",
        encrypt_key: str = "",
        allowed_users: Optional[Set[str]] = None,
    ):
        super().__init__(name="feishu", bus=bus, allowed_users=allowed_users)
        self.app_id = app_id
        self.app_secret = app_secret
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key

        self._client: Any = None
        self._ws_client: Any = None
        self._processed_ids: OrderedDict = OrderedDict()  # recent message_id → None

    def start(self):
        """Build REST + WS clients and block inside the worker thread."""
        if not FEISHU_SDK_AVAILABLE:
            logger.error("Feishu SDK missing. Install with: pip install lark-oapi")
            return

        if not self.app_id or not self.app_secret:
            logger.error("Feishu app_id / app_secret are not configured")
            return

        self._running = True

        self._client = (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        event_handler = (
            lark.EventDispatcherHandler.builder(
                self.encrypt_key,
                self.verification_token,
            )
            .register_p2_im_message_receive_v1(self._on_message)
            .build()
        )

        self._ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        logger.info("🚀 Feishu channel starting (WebSocket long connection)...")
        try:
            self._ws_client.start()
        except Exception as e:
            logger.error(f"Feishu WebSocket connection failed: {e}")
            self._running = False

    def stop(self):
        """Stop WS client."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Feishu WebSocket stop raised: {e}")
        logger.info("Feishu channel stopped")

    def send(self, chat_id: str, content: str, **kwargs):
        """Send a plain-text card via im.message.create."""
        if not self._client:
            logger.warning("Feishu REST client is not initialized")
            return

        try:
            if chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"

            msg_content = json.dumps({"text": content})

            request = (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("text")
                    .content(msg_content)
                    .build()
                )
                .build()
            )

            response = self._client.im.v1.message.create(request)

            if not response.success():
                logger.error(
                    f"Feishu send failed: code={response.code}, "
                    f"msg={response.msg}, log_id={response.get_log_id()}"
                )
            else:
                logger.debug(f"Feishu message delivered to {chat_id}")

        except Exception as e:
            logger.error(f"Feishu send exception: {e}")

    def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP"):
        """Lightweight ACK reaction on the source message."""
        if not self._client or not Emoji:
            return

        try:
            request = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                )
                .build()
            )

            response = self._client.im.v1.message_reaction.create(request)

            if not response.success():
                logger.debug(f"Feishu reaction failed: {response.msg}")
        except Exception as e:
            logger.debug(f"Feishu reaction exception: {e}")

    def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """SDK callback (runs on the Feishu websocket thread)."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            message_id = message.message_id
            if message_id in self._processed_ids:
                return
            self._processed_ids[message_id] = None

            while len(self._processed_ids) > 1000:
                self._processed_ids.popitem(last=False)

            if sender.sender_type == "bot":
                return

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type

            self._add_reaction(message_id, "THUMBSUP")

            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except (json.JSONDecodeError, TypeError):
                    content = message.content or ""
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")

            if not content.strip():
                return

            reply_to = chat_id if chat_type == "group" else sender_id

            self._handle_incoming(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content.strip(),
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                },
            )

        except Exception as e:
            logger.error(f"Feishu inbound handler failed: {e}")
