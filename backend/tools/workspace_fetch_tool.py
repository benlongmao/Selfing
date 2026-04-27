#!/usr/bin/env python3
"""
Download allowed https URLs into workspace/sandbox (server-side, not execute_python).

Bridges online PDFs / assets for read_pdf / read_file style tools that need a local path.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from backend.config import config

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sandbox_base() -> Path:
    base = _project_root() / "workspace" / "sandbox"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _host_allowed(hostname: str, suffixes: List[str]) -> bool:
    h = hostname.lower().strip().strip(".")
    if not h:
        return False
    for raw in suffixes:
        s = str(raw).lower().strip().strip(".")
        if not s:
            continue
        if h == s or h.endswith("." + s):
            return True
    return False


def _safe_filename(name: str, fallback: str) -> str:
    base = (name or "").strip() or fallback
    base = Path(base).name
    base = re.sub(r"[^a-zA-Z0-9._\-]", "_", base)
    if len(base) > 180:
        base = base[:180]
    return base or fallback


def _literal_host_dangerous(hostname: str) -> Optional[str]:
    """Block literal private/loopback IPs in the URL host to reduce SSRF; hostnames are not pre-resolved."""
    h = (hostname or "").strip().lower()
    if h in ("localhost",):
        return "localhost is not allowed"
    try:
        ip = ipaddress.ip_address(h)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return f"IP address type not allowed: {ip}"
    except ValueError:
        pass
    return None


class WorkspaceFetchTool:
    """Server-side download: URL -> sandbox/{session}/downloads/"""

    def fetch_url_to_workspace(
        self,
        url: str,
        session_id: str = "default",
        filename_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        enabled = bool(config.get("parameters.workspace_fetch.enabled", True))
        if not enabled:
            return {
                "success": False,
                "error": "workspace_fetch disabled (parameters.workspace_fetch.enabled=false)",
            }

        url = (url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            return {
                "success": False,
                "error": "Only https URLs are allowed (http/file/... rejected)",
            }
        host = parsed.hostname or ""
        if not host:
            return {"success": False, "error": "URL is missing a hostname"}

        danger = _literal_host_dangerous(host)
        if danger:
            return {"success": False, "error": danger}

        restrict = bool(
            config.get("parameters.workspace_fetch.restrict_to_allowed_hosts", False)
        )
        if restrict:
            suffixes = config.get("parameters.workspace_fetch.allowed_host_suffixes", []) or []
            if not isinstance(suffixes, list):
                suffixes = []
            if not suffixes:
                return {
                    "success": False,
                    "error": (
                        "restrict_to_allowed_hosts is true but allowed_host_suffixes is empty; "
                        "configure host suffixes"
                    ),
                }
            if not _host_allowed(host, suffixes):
                return {
                    "success": False,
                    "error": (
                        f"Host not on allowlist: {host!r}. "
                        "See parameters.workspace_fetch.allowed_host_suffixes"
                    ),
                }

        max_bytes = int(config.get("parameters.workspace_fetch.max_bytes", 26_214_400) or 26_214_400)
        max_bytes = max(1_048_576, min(max_bytes, 100_000_000))
        timeout = float(config.get("parameters.workspace_fetch.timeout_seconds", 60) or 60)
        timeout = max(10.0, min(timeout, 300.0))

        sid = re.sub(r"[^\w\-.]", "_", (session_id or "default").strip() or "default")[:80]
        dest_dir = _sandbox_base() / sid / "downloads"
        dest_dir.mkdir(parents=True, exist_ok=True)

        path_last = Path(parsed.path or "").name
        hint = filename_hint or path_last
        if not hint or hint == "/":
            hint = f"download_{uuid.uuid4().hex[:10]}.pdf"
        fname = _safe_filename(hint, f"fetch_{uuid.uuid4().hex[:10]}.pdf")
        if "." not in fname:
            fname += ".bin"
        out_path = dest_dir / fname
        if out_path.exists():
            stem, suf = out_path.stem, out_path.suffix
            out_path = dest_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suf}"

        req = Request(
            url,
            headers={
                "User-Agent": "S-WorkspaceFetch/1.0 (research; +https://arxiv.org/help)",
                "Accept": "application/pdf,*/*",
            },
            method="GET",
        )

        try:
            with urlopen(req, timeout=timeout) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                total = 0
                chunk_size = 256 * 1024
                with open(out_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            try:
                                f.close()
                                out_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                            return {
                                "success": False,
                                "error": f"Exceeded max size {max_bytes} bytes; download aborted",
                            }
                        f.write(chunk)
        except Exception as e:
            logger.warning("[WorkspaceFetch] failed: %s", e)
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            return {"success": False, "error": f"Download failed: {e}"}

        rel = f"{sid}/downloads/{out_path.name}"
        if "pdf" not in out_path.suffix.lower() and "pdf" in ctype:
            new_path = out_path.with_suffix(".pdf")
            try:
                out_path.rename(new_path)
                out_path = new_path
                rel = f"{sid}/downloads/{out_path.name}"
            except Exception:
                pass

        return {
            "success": True,
            "saved_relative_path": rel,
            "bytes_written": out_path.stat().st_size,
            "local_hint": (
                f"Saved under workspace/sandbox; use read_pdf(file_path=\"{rel}\") "
                "or get_pdf_info to read"
            ),
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "fetch_url_to_workspace",
                    "description": (
                        "Download an https resource into the workspace (server-side, not execute_python). "
                        "Use for online PDFs: fetch first, then read_pdf(saved_relative_path). "
                        "Public https hosts allowed by default with per-file size and timeout caps. "
                        "Literal private/loopback IPs and localhost are rejected. "
                        "To restrict hosts, set parameters.workspace_fetch.restrict_to_allowed_hosts: true "
                        "and populate allowed_host_suffixes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Full https URL (e.g. arXiv PDF link)",
                            },
                            "filename_hint": {
                                "type": "string",
                                "description": "Optional suggested filename (extension helps, e.g. paper.pdf)",
                            },
                        },
                        "required": ["url"],
                    },
                },
            }
        ]


_instance: Optional[WorkspaceFetchTool] = None


def get_workspace_fetch_tool() -> WorkspaceFetchTool:
    global _instance
    if _instance is None:
        _instance = WorkspaceFetchTool()
    return _instance
