#!/usr/bin/env python3
"""
PDF reader: extract text, metadata, search, and list PDFs under the workspace.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def _reject_if_url(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Only local/workspace paths are allowed; direct http(s) URLs are rejected
    to avoid the model misusing this tool. Use ``fetch_url_to_workspace`` first.
    """
    fp = (file_path or "").strip()
    low = fp.lower()
    if low.startswith("http://") or low.startswith("https://"):
        return {
            "success": False,
            "error": (
                "read_pdf does not accept a raw URL. Call fetch_url_to_workspace(url=...) to save a file, "
                "then read_pdf(file_path=<saved relative path>). "
                "Do not use execute_python to download; it has no network access."
            ),
        }
    return None


PYPDF_AVAILABLE = False
PDFPLUMBER_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
    logger.info("pypdf available for PDF reading")
except ImportError:
    logger.warning("pypdf not installed")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logger.info("pdfplumber available for PDF reading")
except ImportError:
    logger.warning("pdfplumber not installed")

PDF_AVAILABLE = PYPDF_AVAILABLE or PDFPLUMBER_AVAILABLE
if not PDF_AVAILABLE:
    logger.warning("No PDF library. Install: pip install pypdf pdfplumber")


class PDFReaderTool:
    """Read and search PDF files in the agent workspace."""

    def __init__(self, workspace_dir: str = "workspace/sandbox"):
        """*workspace_dir*: base directory (relative to cwd or absolute)."""
        self.workspace_dir = Path(workspace_dir)
        self.enabled = PDF_AVAILABLE
        # Prefer pdfplumber for table detection
        self.use_pdfplumber = PDFPLUMBER_AVAILABLE

    def read_pdf(
        self,
        file_path: str,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_chars: int = 10000
    ) -> Dict[str, Any]:
        """Extract text from a PDF, optionally by page range and max character cap."""
        if not self.enabled:
            return {
                "success": False,
                "error": "No PDF library. Run: pip install pypdf pdfplumber"
            }

        url_err = _reject_if_url(file_path)
        if url_err:
            return url_err

        path = self._resolve_path(file_path)
        if not path:
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        if path.suffix.lower() != '.pdf':
            return {
                "success": False,
                "error": f"Not a PDF: {file_path}"
            }

        try:
            if self.use_pdfplumber:
                return self._read_with_pdfplumber(path, start_page, end_page, max_chars)
            return self._read_with_pypdf(path, start_page, end_page, max_chars)

        except Exception as e:
            logger.error(f"[PDF] Read failed: {e}")
            return {
                "success": False,
                "error": f"Read failed: {e}"
            }

    def _read_with_pdfplumber(
        self,
        path: Path,
        start_page: int,
        end_page: Optional[int],
        max_chars: int
    ) -> Dict[str, Any]:
        """Read using pdfplumber."""
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)

            start_idx = max(0, start_page - 1)
            end_idx = min(total_pages, end_page) if end_page else total_pages

            texts: List[str] = []
            tables_count = 0
            char_count = 0

            for i in range(start_idx, end_idx):
                page = pdf.pages[i]
                page_text = page.extract_text() or ""

                tables = page.extract_tables()
                if tables:
                    tables_count += len(tables)

                if char_count + len(page_text) > max_chars:
                    remaining = max_chars - char_count
                    texts.append(
                        f"--- Page {i+1} ---\n{page_text[:remaining]}...\n[truncated]"
                    )
                    break
                texts.append(f"--- Page {i+1} ---\n{page_text}")
                char_count += len(page_text)

            metadata = pdf.metadata or {}

            return {
                "success": True,
                "file": str(path.name),
                "total_pages": total_pages,
                "pages_read": f"{start_page}-{end_idx}",
                "content": "\n\n".join(texts),
                "char_count": char_count,
                "tables_detected": tables_count,
                "metadata": {
                    "title": metadata.get("Title", ""),
                    "author": metadata.get("Author", ""),
                    "creator": metadata.get("Creator", ""),
                    "creation_date": str(metadata.get("CreationDate", "")),
                }
            }

    def _read_with_pypdf(
        self,
        path: Path,
        start_page: int,
        end_page: Optional[int],
        max_chars: int
    ) -> Dict[str, Any]:
        """Read using pypdf."""
        import pypdf

        with open(path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)

            start_idx = max(0, start_page - 1)
            end_idx = min(total_pages, end_page) if end_page else total_pages

            texts: List[str] = []
            char_count = 0

            for i in range(start_idx, end_idx):
                page = reader.pages[i]
                page_text = page.extract_text() or ""

                if char_count + len(page_text) > max_chars:
                    remaining = max_chars - char_count
                    texts.append(
                        f"--- Page {i+1} ---\n{page_text[:remaining]}...\n[truncated]"
                    )
                    break
                texts.append(f"--- Page {i+1} ---\n{page_text}")
                char_count += len(page_text)

            metadata = reader.metadata or {}

            return {
                "success": True,
                "file": str(path.name),
                "total_pages": total_pages,
                "pages_read": f"{start_page}-{end_idx}",
                "content": "\n\n".join(texts),
                "char_count": char_count,
                "metadata": {
                    "title": metadata.get("/Title", ""),
                    "author": metadata.get("/Author", ""),
                    "creator": metadata.get("/Creator", ""),
                }
            }

    def get_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """Metadata and page count only (faster than full read)."""
        if not self.enabled:
            return {"success": False, "error": "No PDF library installed"}

        url_err = _reject_if_url(file_path)
        if url_err:
            return url_err

        path = self._resolve_path(file_path)
        if not path:
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            if self.use_pdfplumber:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    metadata = pdf.metadata or {}
                    return {
                        "success": True,
                        "file": str(path.name),
                        "size_kb": round(path.stat().st_size / 1024, 1),
                        "total_pages": len(pdf.pages),
                        "title": metadata.get("Title", ""),
                        "author": metadata.get("Author", ""),
                        "creator": metadata.get("Creator", ""),
                    }
            import pypdf
            with open(path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                metadata = reader.metadata or {}
                return {
                    "success": True,
                    "file": str(path.name),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "total_pages": len(reader.pages),
                    "title": metadata.get("/Title", ""),
                    "author": metadata.get("/Author", ""),
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_pdf(
        self,
        file_path: str,
        keyword: str,
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Search for *keyword* across pages, returning snippets and counts."""
        if not self.enabled:
            return {"success": False, "error": "No PDF library installed"}

        url_err = _reject_if_url(file_path)
        if url_err:
            return url_err

        path = self._resolve_path(file_path)
        if not path:
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            results = []
            search_key = keyword if case_sensitive else keyword.lower()

            if self.use_pdfplumber:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        search_text = text if case_sensitive else text.lower()

                        if search_key in search_text:
                            idx = search_text.find(search_key)
                            start = max(0, idx - 50)
                            end = min(len(text), idx + len(keyword) + 50)
                            context = text[start:end]

                            results.append({
                                "page": i + 1,
                                "context": f"...{context}...",
                                "count": search_text.count(search_key)
                            })
            else:
                import pypdf
                with open(path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        search_text = text if case_sensitive else text.lower()

                        if search_key in search_text:
                            idx = search_text.find(search_key)
                            start = max(0, idx - 50)
                            end = min(len(text), idx + len(keyword) + 50)
                            context = text[start:end]

                            results.append({
                                "page": i + 1,
                                "context": f"...{context}...",
                                "count": search_text.count(search_key)
                            })

            total_matches = sum(r["count"] for r in results)

            return {
                "success": True,
                "file": str(path.name),
                "keyword": keyword,
                "total_matches": total_matches,
                "pages_with_matches": len(results),
                "results": results[:20]  # cap
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_pdfs(self, directory: str = "") -> Dict[str, Any]:
        """List ``*.pdf`` under *directory* (relative to the workspace) or the whole workspace."""
        try:
            if directory:
                search_dir = self.workspace_dir / directory
            else:
                search_dir = self.workspace_dir

            if not search_dir.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}

            pdfs: List[Dict] = []
            for pdf_file in search_dir.rglob("*.pdf"):
                try:
                    stat = pdf_file.stat()
                    pdfs.append({
                        "name": pdf_file.name,
                        "path": str(pdf_file.relative_to(self.workspace_dir)),
                        "size_kb": round(stat.st_size / 1024, 1),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception:
                    continue

            pdfs.sort(key=lambda x: x["modified"], reverse=True)

            return {
                "success": True,
                "directory": str(search_dir),
                "count": len(pdfs),
                "files": pdfs[:50]
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _resolve_path(self, file_path: str) -> Optional[Path]:
        """Resolve a user path against absolute, workspace, or CWD."""
        path = Path(file_path)
        if path.is_absolute() and path.exists():
            return path

        path = self.workspace_dir / file_path
        if path.exists():
            return path

        path = Path(file_path)
        if path.exists():
            return path

        return None

    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_pdf",
                    "description": (
                        "Read PDF text from a local file under the workspace. "
                        "Supports page ranges, optional max length, and table count when using pdfplumber. "
                        "Scanned (image) PDFs may have no text. Use fetch_url_to_workspace for URLs, not raw https here."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Workspace-relative or existing local file path; not https://",
                            },
                            "start_page": {
                                "type": "integer",
                                "description": "First 1-based page (default 1).",
                            },
                            "end_page": {
                                "type": "integer",
                                "description": "Last 1-based page, optional (default: last page).",
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pdf_info",
                    "description": (
                        "Get PDF metadata and page count without full extraction; no raw URLs."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Local path only; not https?://",
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_pdf",
                    "description": "Search a PDF for a keyword: page, snippet, hit counts (max 20 hits in response).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the PDF in the workspace.",
                            },
                            "keyword": {
                                "type": "string",
                                "description": "Search term (supports mixed-language text).",
                            }
                        },
                        "required": ["file_path", "keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_pdfs",
                    "description": "List PDFs under a subdirectory of the workspace (or entire workspace if omitted).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Optional subfolder; default = workspace root.",
                            }
                        }
                    }
                }
            }
        ]


_instance: Optional[PDFReaderTool] = None


def get_pdf_reader_tool() -> PDFReaderTool:
    """Singleton accessor for this process."""
    global _instance
    if _instance is None:
        _instance = PDFReaderTool()
    return _instance
