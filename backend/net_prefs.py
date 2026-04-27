"""
Optional network behavior patch (call before the first urllib3/requests connection).

On WSL2 / some dual-stack setups, IPv6 or CDN edges may occasionally RST large HTTP/1.1 POSTs,
surfacing as ``RemoteDisconnected('Remote end closed connection without response')``.

When ``DEEPSEEK_FORCE_IPV4=1`` is set, urllib3 is forced to resolve and connect over IPv4 only
(useful as a contrast to curl's default happy-eyeballs path).
"""
from __future__ import annotations

import logging
import os
import socket

logger = logging.getLogger(__name__)
_applied = False


def apply_net_prefs() -> None:
    global _applied
    if _applied:
        return
    v = (os.environ.get("DEEPSEEK_FORCE_IPV4") or "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return
    try:
        import urllib3.util.connection as u3c

        u3c.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[method-assign]
        _applied = True
        logger.warning(
            "[net_prefs] DEEPSEEK_FORCE_IPV4 is on: urllib3 will use IPv4 only "
            "(mitigates some WSL/dual-stack disconnects)"
        )
    except Exception as e:
        logger.warning("[net_prefs] DEEPSEEK_FORCE_IPV4 could not be applied: %s", e)
