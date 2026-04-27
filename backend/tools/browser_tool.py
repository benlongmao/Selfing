#!/usr/bin/env python3
"""
Browser control via Chrome DevTools Protocol (CDP).

Connects to Chrome on Windows from WSL, without moving the project tree.
Supports navigation, screenshots, page content, and JavaScript execution,
with WSL2-friendly CDP URL discovery.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if Playwright is available
PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright, Browser, Page, Error as PlaywrightError
    PLAYWRIGHT_AVAILABLE = True
    logger.info("✅ Playwright available for browser control")
except ImportError:
    logger.warning("⚠️ Playwright not installed. Browser control will be disabled.")
    logger.warning("   Install: pip install playwright && playwright install chromium")


class BrowserTool:
    """
    Agent-facing browser automation over CDP (Chrome on Windows from WSL).
    """
    
    def __init__(
        self,
        cdp_url: Optional[str] = None,
        default_timeout: int = 30000,
        screenshot_dir: str = "workspace/screenshots",
    ):
        """
        Args:
            cdp_url: CDP WebSocket/HTTP URL; if omitted, uses env and WSL2 heuristics.
            default_timeout: Default navigation/action timeout in milliseconds.
            screenshot_dir: Directory for screenshot files.
        """
        self.enabled = PLAYWRIGHT_AVAILABLE
        self.default_timeout = default_timeout
        self.screenshot_dir = Path(screenshot_dir)
        
        # Make sure the screenshot directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # CDP URL configuration
        self.cdp_url = cdp_url or self._get_cdp_url()
        
        # Statistics
        self.stats = {
            "total_navigations": 0,
            "total_screenshots": 0,
            "total_searches": 0,
            "errors": 0,
        }
    
    def _get_cdp_url(self) -> str:
        """
        Resolve the CDP base URL (e.g. http://<windows-host>:9222).

        WSL2 notes:
        - ``localhost`` inside WSL is not Windows' loopback.
        - The Windows vEthernet (WSL) gateway IP is usually reachable from WSL.
        - Chrome should listen with ``--remote-debugging-address=0.0.0.0`` on port 9222.

        Precedence:
        1. ``CHROME_CDP_URL`` environment variable
        2. Default gateway from ``ip route`` (Windows vEthernet IP)
        3. First ``nameserver`` in ``/etc/resolv.conf`` (fallback)
        4. Default ``172.26.80.1`` (common WSL2 gateway)
        """
        import subprocess
        import socket
        
        # 1. Environment variables
        env_url = os.environ.get("CHROME_CDP_URL")
        if env_url:
            logger.info(f"Using CDP URL from env: {env_url}")
            return env_url
        
        # 2. Get the WSL2 default gateway from ip route (this is the Windows vEthernet IP)
        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                match = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", result.stdout)
                if match:
                    gateway_ip = match.group(1)
                    # Virtual gateway that skips VPN
                    if not gateway_ip.startswith("10.255."):
                        # test connection
                        try:
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(3)
                            conn_result = sock.connect_ex((gateway_ip, 9222))
                            sock.close()
                            if conn_result == 0:
                                cdp_url = f"http://{gateway_ip}:9222"
                                logger.info(f"Using WSL2 gateway (Win11 vEthernet): {gateway_ip}")
                                return cdp_url
                            else:
                                logger.debug(f"Gateway {gateway_ip}:9222 not reachable (code: {conn_result})")
                        except Exception as e:
                            logger.debug(f"Failed to test gateway {gateway_ip}: {e}")
        except Exception as e:
            logger.warning(f"Failed to get default gateway: {e}")
        
        # 3. Get nameserver from resolv.conf (alternative solution)
        try:
            with open("/etc/resolv.conf", "r") as f:
                content = f.read()
                match = re.search(r"nameserver\s+(\d+\.\d+\.\d+\.\d+)", content)
                if match:
                    nameserver_ip = match.group(1)
                    # Virtual DNS that skips VPN
                    if not nameserver_ip.startswith("10.255."):
                        try:
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(3)
                            conn_result = sock.connect_ex((nameserver_ip, 9222))
                            sock.close()
                            if conn_result == 0:
                                cdp_url = f"http://{nameserver_ip}:9222"
                                logger.info(f"Using nameserver as Win11 IP: {nameserver_ip}")
                                return cdp_url
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Failed to read resolv.conf: {e}")
        
        # 4. Use common WSL2 gateway IP by default
        default_ip = "172.26.80.1"
        default_url = f"http://{default_ip}:9222"
        logger.info(f"Using default WSL2 gateway: {default_url}")
        return default_url
    
    def _connect_browser(self) -> Optional[Browser]:
        """Connect to Chrome over CDP."""
        if not self.enabled:
            return None
        
        try:
            playwright = sync_playwright().start()
            browser = playwright.chromium.connect_over_cdp(self.cdp_url)
            return browser
        except Exception as e:
            logger.error(f"Failed to connect to Chrome: {e}")
            self.stats["errors"] += 1
            return None
    
    def navigate(self, url: str, wait_until: str = "networkidle") -> Dict[str, Any]:
        """
        Open ``url`` in the attached browser and return a short text preview.

        Args:
            url: Target URL.
            wait_until: One of ``load``, ``domcontentloaded``, ``networkidle``.

        Returns:
            ``success``, ``url``, ``title``, ``content_preview`` (first ~1000 chars of body text), or ``error``.
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Playwright not installed. Run: pip install playwright"
            }
        
        self.stats["total_navigations"] += 1
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                # Get or create a page
                if len(browser.contexts) > 0 and len(browser.contexts[0].pages) > 0:
                    page = browser.contexts[0].pages[0]
                else:
                    context = browser.contexts[0] if browser.contexts else browser.new_context()
                    page = context.new_page()
                
                # [2026-02-22] Viewport: only set when invalid; do not force a fixed size or layout may clip.
                # navigation
                page.goto(url, wait_until=wait_until, timeout=self.default_timeout)
                
                # Get page information
                title = page.title()
                content = page.content()
                current_url = page.url
                
                # Content preview (first 1000 characters)
                text_content = page.inner_text("body")
                content_preview = text_content[:1000] if text_content else ""
                
                logger.info(f"✅ Navigated to: {current_url}")
                
                return {
                    "success": True,
                    "url": current_url,
                    "title": title,
                    "content_preview": content_preview,
                    "content_length": len(text_content) if text_content else 0,
                }
        
        except PlaywrightError as e:
            logger.error(f"❌ Navigation failed: {e}")
            self.stats["errors"] += 1
            
            # Friendly error message
            error_msg = str(e)
            if "net::ERR_CONNECTION_REFUSED" in error_msg:
                return {
                    "success": False,
                    "error": (
                        f"Cannot connect to Chrome. Check:\n"
                        f"1. Chrome is running with remote debugging (--remote-debugging-port=9222)\n"
                        f"2. Windows firewall allows port 9222\n"
                        f"3. CDP URL is correct: {self.cdp_url}"
                    ),
                }
            elif "Timeout" in error_msg:
                return {
                    "success": False,
                    "error": (
                        f"Page load timed out (>{self.default_timeout / 1000}s). Try:\n"
                        f"1. Check network connectivity\n"
                        f"2. Use wait_until='load' for a weaker wait\n"
                        f"3. Increase timeout"
                    ),
                }
            else:
                return {
                    "success": False,
                    "error": f"Navigation failed: {error_msg}",
                }
        
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }
    
    def screenshot(
        self,
        filename: Optional[str] = None,
        full_page: bool = False,
    ) -> Dict[str, Any]:
        """
        Capture a screenshot of the current page.

        Args:
            filename: Base filename (no directory); auto-generated if omitted.
            full_page: If True, capture the full scrollable page.

        Returns:
            ``success``, ``filepath``, ``filename``, ``full_page``, or ``error``.
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Playwright not installed"
            }
        
        self.stats["total_screenshots"] += 1
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {
                        "success": False,
                        "error": "No open page. Call browse_web with a URL first.",
                    }
                
                page = browser.contexts[0].pages[0]
                
                # Screenshot: ensure a usable viewport (or set a default)
                try:
                    viewport = page.viewport_size
                    if not viewport:
                        # Try to get the actual size of the window, if that fails, use the default value
                        logger.info("Viewport is None, setting default for screenshot")
                        page.set_viewport_size({"width": 1280, "height": 800})
                except Exception as e:
                    logger.warning(f"Could not check/set viewport: {e}")
                
                # Wait for the page to stabilize
                page.wait_for_load_state("domcontentloaded")
                time.sleep(0.5)  # brief settle for rendering
                
                # Generate file name
                if not filename:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                
                # Make sure the filename has a .png suffix
                if not filename.endswith(".png"):
                    filename += ".png"
                
                filepath = self.screenshot_dir / filename
                
                # screenshot
                page.screenshot(path=str(filepath), full_page=full_page)
                
                logger.info(f"✅ Screenshot saved: {filepath}")
                
                return {
                    "success": True,
                    "filepath": str(filepath),
                    "filename": filename,
                    "full_page": full_page,
                }
        
        except Exception as e:
            logger.error(f"❌ Screenshot failed: {e}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Screenshot failed: {str(e)}",
            }
    
    def search_baidu(self, query: str) -> Dict[str, Any]:
        """
        Run a Baidu web search (legacy helper; prefer tavily_search for new flows).

        Args:
            query: Search query string.

        Returns:
            ``success``, ``query``, ``results_count``, ``top_results`` (up to 5), or ``error``.
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Playwright not installed"
            }
        
        self.stats["total_searches"] += 1
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                # Get or create a page
                if len(browser.contexts) > 0 and len(browser.contexts[0].pages) > 0:
                    page = browser.contexts[0].pages[0]
                else:
                    context = browser.contexts[0] if browser.contexts else browser.new_context()
                    page = context.new_page()
                
                # Visit Baidu
                page.goto("https://www.baidu.com", timeout=self.default_timeout)

                # Enter search term
                page.fill("#kw", query)

                # Click search
                page.click("#su")

                # Wait for results
                page.wait_for_selector(".result", timeout=self.default_timeout)
                
                # Extract search results
                results = page.query_selector_all(".result")
                
                top_results = []
                for idx, result in enumerate(results[:5], 1):
                    try:
                        title_elem = result.query_selector("h3 a")
                        title = title_elem.inner_text() if title_elem else "(no title)"
                        
                        url_elem = result.query_selector("h3 a")
                        url = url_elem.get_attribute("href") if url_elem else ""
                        
                        abstract_elem = result.query_selector(".c-abstract")
                        abstract = abstract_elem.inner_text() if abstract_elem else ""
                        
                        top_results.append({
                            "rank": idx,
                            "title": title,
                            "url": url,
                            "abstract": abstract[:200]
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse result {idx}: {e}")
                
                logger.info(f"✅ Baidu search completed: {query}")
                
                return {
                    "success": True,
                    "query": query,
                    "results_count": len(results),
                    "top_results": top_results,
                }
        
        except Exception as e:
            logger.error(f"❌ Baidu search failed: {e}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Baidu search failed: {str(e)}",
            }
    
    def execute_script(self, script: str) -> Dict[str, Any]:
        """
        Evaluate JavaScript in the current page context.

        Args:
            script: JavaScript source to run in the page.

        Returns:
            ``success``, ``result``, or ``error``.
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Playwright not installed"
            }
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {
                        "success": False,
                        "error": "No open page",
                    }
                
                page = browser.contexts[0].pages[0]
                result = page.evaluate(script)
                
                return {
                    "success": True,
                    "result": result,
                }
        
        except Exception as e:
            logger.error(f"❌ Script execution failed: {e}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Script execution failed: {str(e)}",
            }
    
    def get_current_page_info(self) -> Dict[str, Any]:
        """Return URL/title for the current tab, or ``has_page: false`` if none."""
        if not self.enabled:
            return {
                "success": False,
                "error": "Playwright not installed"
            }
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {
                        "success": True,
                        "has_page": False,
                        "message": "No open page",
                    }
                
                page = browser.contexts[0].pages[0]
                
                return {
                    "success": True,
                    "has_page": True,
                    "url": page.url,
                    "title": page.title(),
                }
        
        except Exception as e:
            logger.error(f"❌ Failed to get page info: {e}")
            return {
                "success": False,
                "error": f"Failed to read page info: {str(e)}",
            }
    
    # ========== [2026-01-31] ​​New: Interactive function ==========    
    def fill_input(self, selector: str, value: str, clear_first: bool = True) -> Dict[str, Any]:
        """
        Fill a form field identified by CSS or XPath.

        Args:
            selector: CSS selector or XPath (e.g. ``#username``, ``input[name='email']``).
            value: Text to type into the field.
            clear_first: Clear the field before filling (default True).
        """
        if not self.enabled:
            return {"success": False, "error": "Playwright not installed"}
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {
                        "success": False,
                        "error": "No open page; call browse_web with a URL first.",
                    }
                
                page = browser.contexts[0].pages[0]
                
                # [2026-02-22] Wait for the element to exist (visibility is not required because CDP visibility detection is inaccurate)
                page.wait_for_selector(selector, state="attached", timeout=self.default_timeout)
                
                # Empty and fill (use force=True to skip visibility checks)
                if clear_first:
                    page.locator(selector).fill("", force=True)
                page.locator(selector).fill(value, force=True)
                
                logger.info(f"✅ Filled input '{selector}' with value (len={len(value)})")
                
                return {
                    "success": True,
                    "selector": selector,
                    "filled_value": value if len(value) < 50 else f"{value[:50]}...",
                }
        
        except Exception as e:
            logger.error(f"❌ Fill input failed: {e}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Fill failed: {str(e)}",
            }

    def click_element(self, selector: str, wait_after: int = 1000) -> Dict[str, Any]:
        """
        Click an element; optionally wait after click for navigation/JS.

        Args:
            selector: CSS selector or XPath.
            wait_after: Milliseconds to wait after the click.
        """
        if not self.enabled:
            return {"success": False, "error": "Playwright not installed"}
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {"success": False, "error": "No open page"}

                page = browser.contexts[0].pages[0]
                old_url = page.url

                # [2026-02-22] Wait for the element to exist (visibility is not required because CDP visibility detection is inaccurate)
                page.wait_for_selector(selector, state="attached", timeout=self.default_timeout)
                
                # [2026-02-22] Click: Use force first, if failed, use JS event
                try:
                    page.locator(selector).click(force=True, timeout=5000)
                except Exception:
                    # JS click fallback
                    page.locator(selector).dispatch_event("click")
                
                # Wait for page response
                if wait_after > 0:
                    page.wait_for_timeout(wait_after)
                
                new_url = page.url
                navigated = old_url != new_url
                
                logger.info(f"✅ Clicked element '{selector}'" + (f" → navigated to {new_url}" if navigated else ""))
                
                return {
                    "success": True,
                    "selector": selector,
                    "old_url": old_url,
                    "new_url": new_url,
                    "navigated": navigated,
                }
        
        except Exception as e:
            logger.error(f"❌ Click failed: {e}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Click failed: {str(e)}",
            }

    def wait_for_element(
        self,
        selector: str,
        state: str = "visible",
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Wait until ``selector`` reaches the given Playwright state.

        Args:
            selector: CSS selector.
            state: ``attached`` | ``detached`` | ``visible`` | ``hidden``.
            timeout: Timeout in ms (defaults to ``default_timeout``).
        """
        if not self.enabled:
            return {"success": False, "error": "Playwright not installed"}
        
        timeout = timeout or self.default_timeout
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {"success": False, "error": "No open page"}

                page = browser.contexts[0].pages[0]

                # await element
                page.wait_for_selector(selector, state=state, timeout=timeout)
                
                logger.info(f"✅ Element '{selector}' is now {state}")
                
                return {
                    "success": True,
                    "selector": selector,
                    "state": state,
                    "found": True,
                }
        
        except Exception as e:
            logger.error(f"❌ Wait for element failed: {e}")
            return {
                "success": False,
                "selector": selector,
                "found": False,
                "error": f"Wait timed out: {str(e)}",
            }

    def get_element_text(self, selector: str) -> Dict[str, Any]:
        """Return ``inner_text`` for the first element matching ``selector``."""
        if not self.enabled:
            return {"success": False, "error": "Playwright not installed"}
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {"success": False, "error": "No open page"}

                page = browser.contexts[0].pages[0]

                # [2026-02-22] Wait for element to exist
                page.wait_for_selector(selector, state="attached", timeout=self.default_timeout)

                # Get text
                text = page.locator(selector).inner_text()
                
                logger.info(f"✅ Got text from '{selector}' (len={len(text)})")
                
                return {
                    "success": True,
                    "selector": selector,
                    "text": text,
                }
        
        except Exception as e:
            logger.error(f"❌ Get text failed: {e}")
            return {
                "success": False,
                "error": f"Failed to read text: {str(e)}",
            }

    def get_page_elements(self, selector: str, limit: int = 10) -> Dict[str, Any]:
        """
        Summarize up to ``limit`` elements matching ``selector`` (tag, short text, attrs).
        """
        if not self.enabled:
            return {"success": False, "error": "Playwright not installed"}
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {"success": False, "error": "No open page"}

                page = browser.contexts[0].pages[0]

                elements = page.query_selector_all(selector)
                
                results = []
                for i, elem in enumerate(elements[:limit]):
                    try:
                        results.append({
                            "index": i,
                            "tag": elem.evaluate("e => e.tagName.toLowerCase()"),
                            "text": (elem.inner_text() or "")[:200],  # cap text length
                            "href": elem.get_attribute("href"),
                            "id": elem.get_attribute("id"),
                            "class": elem.get_attribute("class"),
                            "name": elem.get_attribute("name"),
                            "type": elem.get_attribute("type"),
                        })
                    except Exception:
                        continue
                
                logger.info(f"✅ Found {len(elements)} elements matching '{selector}'")
                
                return {
                    "success": True,
                    "selector": selector,
                    "count": len(elements),
                    "elements": results,
                }
        
        except Exception as e:
            logger.error(f"❌ Get elements failed: {e}")
            return {
                "success": False,
                "error": f"Failed to list elements: {str(e)}",
            }

    def login_to_site(
        self,
        url: str,
        username_selector: str,
        password_selector: str,
        submit_selector: str,
        username: str,
        password: str,
        success_indicator: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Open ``url``, fill credentials, submit, and optionally wait for ``success_indicator``.
        """
        if not self.enabled:
            return {"success": False, "error": "Playwright not installed"}
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                # Get or create a page
                if len(browser.contexts) > 0 and len(browser.contexts[0].pages) > 0:
                    page = browser.contexts[0].pages[0]
                else:
                    context = browser.contexts[0] if browser.contexts else browser.new_context()
                    page = context.new_page()
                
                # 1. Navigate to the login page
                logger.info(f"🔐 Step 1: Navigating to {url}")
                page.goto(url, wait_until="networkidle", timeout=self.default_timeout)
                
                # 2. Fill in the user name
                logger.info(f"🔐 Step 2: Filling username")
                page.wait_for_selector(username_selector, state="visible", timeout=10000)
                page.fill(username_selector, username)
                
                # 3. Fill in the password
                logger.info(f"🔐 Step 3: Filling password")
                page.wait_for_selector(password_selector, state="visible", timeout=10000)
                page.fill(password_selector, password)
                
                # 4. Click to log in
                logger.info(f"🔐 Step 4: Clicking submit")
                page.click(submit_selector)
                
                # 5. Wait for page response
                page.wait_for_load_state("networkidle", timeout=self.default_timeout)
                
                # 6. Verify successful login
                logged_in = True
                if success_indicator:
                    try:
                        page.wait_for_selector(success_indicator, state="visible", timeout=5000)
                        logger.info(f"✅ Login successful! Found success indicator: {success_indicator}")
                    except Exception:
                        logged_in = False
                        logger.warning(f"⚠️ Login might have failed: success indicator not found")
                
                final_url = page.url
                
                return {
                    "success": True,
                    "logged_in": logged_in,
                    "final_url": final_url,
                    "message": (
                        "Login flow finished"
                        + ("; success indicator seen" if logged_in else "; verify success manually")
                    ),
                }
        
        except Exception as e:
            logger.error(f"❌ Login failed: {e}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "logged_in": False,
                "error": f"Login failed: {str(e)}",
            }

    def type_text(self, selector: str, text: str, delay: int = 50) -> Dict[str, Any]:
        """
        Type text character-by-character (useful when sites validate keystrokes).

        Args:
            selector: Field selector.
            text: Full string to type.
            delay: Delay between keys in milliseconds.
        """
        if not self.enabled:
            return {"success": False, "error": "Playwright not installed"}
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_url)
                
                if not browser.contexts or not browser.contexts[0].pages:
                    return {"success": False, "error": "No open page"}

                page = browser.contexts[0].pages[0]

                # [2026-02-22] Wait for element to exist and focus
                page.wait_for_selector(selector, state="attached", timeout=self.default_timeout)
                page.locator(selector).click(force=True)  # focus field first
                
                # Enter word by word
                page.locator(selector).type(text, delay=delay)
                
                logger.info(f"✅ Typed {len(text)} characters into '{selector}'")
                
                return {
                    "success": True,
                    "selector": selector,
                    "typed_length": len(text),
                }
        
        except Exception as e:
            logger.error(f"❌ Type text failed: {e}")
            return {
                "success": False,
                "error": f"Typing failed: {str(e)}",
            }

    def get_stats(self) -> Dict[str, Any]:
        """Return usage counters and CDP config for this tool."""
        return {
            **self.stats,
            "enabled": self.enabled,
            "cdp_url": self.cdp_url,
        }
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        OpenAI-style tool definitions for the tool router.

        [2026-02-22] Reduced surface: browse_web, screenshot_page, browser_interact, browser_login.
        Legacy helpers (e.g. Baidu search) are not exposed; use tavily_search for web search.
        """
        if not self.enabled:
            return []
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "browse_web",
                    "description": """Open a URL in a real Chrome tab (CDP) and return a short text preview.

Use when you need rendered DOM or a specific site; for broad web search prefer tavily_search.

Example: browse_web("https://example.com")""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to open",
                            },
                            "wait_until": {
                                "type": "string",
                                "enum": ["load", "domcontentloaded", "networkidle"],
                                "description": "Playwright wait_until; default networkidle",
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "screenshot_page",
                    "description": """Save a PNG screenshot of the current tab.

Requires browse_web first. Example: screenshot_page("result.png", full_page=True)""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Optional filename (basename only)",
                            },
                            "full_page": {
                                "type": "boolean",
                                "description": "If true, capture the full scrollable page",
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_interact",
                    "description": """DOM interaction on the active tab: click, fill, type, read text, list elements, or wait.

Actions: click | fill | type | get_text | get_elements | wait.

Examples:
- browser_interact(action="click", selector="#submit-btn")
- browser_interact(action="fill", selector="#username", value="myuser")
- browser_interact(action="get_text", selector=".result")""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["click", "fill", "type", "get_text", "get_elements", "wait"],
                                "description": "Action to perform",
                            },
                            "selector": {
                                "type": "string",
                                "description": "CSS selector (or XPath where supported)",
                            },
                            "value": {
                                "type": "string",
                                "description": "Value for fill/type actions",
                            },
                            "wait_after": {
                                "type": "integer",
                                "description": "Milliseconds to wait after click (click only)",
                            }
                        },
                        "required": ["action", "selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_login",
                    "description": """Fill username/password on a login page and submit.

Example: browser_login(url="...", username_selector="#user", password_selector="#pass", submit_selector="#login", username="...", password="...")""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Login page URL"},
                            "username_selector": {"type": "string", "description": "Selector for username field"},
                            "password_selector": {"type": "string", "description": "Selector for password field"},
                            "submit_selector": {"type": "string", "description": "Selector for submit/login control"},
                            "username": {"type": "string", "description": "Username value"},
                            "password": {"type": "string", "description": "Password value"},
                            "success_indicator": {
                                "type": "string",
                                "description": "Optional selector that should appear after a successful login",
                            },
                        },
                        "required": ["url", "username_selector", "password_selector", "submit_selector", "username", "password"]
                    }
                }
            }
        ]
    
    def browser_interact(self, action: str, selector: str, value: str = "", wait_after: int = 1000) -> Dict[str, Any]:
        """
        Unified page interaction (2026-02-22): dispatches click, fill, type, get_text, get_elements, wait.
        """
        if action == "click":
            return self.click_element(selector, wait_after)
        elif action == "fill":
            return self.fill_input(selector, value, clear_first=True)
        elif action == "type":
            return self.type_text(selector, value, delay=50)
        elif action == "get_text":
            return self.get_element_text(selector)
        elif action == "get_elements":
            return self.get_page_elements(selector, limit=10)
        elif action == "wait":
            return self.wait_for_element(selector, state="visible")
        else:
            return {"success": False, "error": f"Unknown action: {action}"}


_browser_tool_instance: Optional[BrowserTool] = None


def get_browser_tool() -> BrowserTool:
    """Return the process-wide BrowserTool singleton."""
    global _browser_tool_instance
    if _browser_tool_instance is None:
        _browser_tool_instance = BrowserTool()
    return _browser_tool_instance
