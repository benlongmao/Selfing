#!/usr/bin/env python3
"""
Embedding fallback when ``sentence-transformers`` is unavailable.

Uses deterministic hashing to fabricate pseudo-vectors for **tests only** — not semantic.
"""
import hashlib
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class SimpleEmbedder:
    """Hash-based pseudo-embedder (non-semantic) for offline / CI environments."""

    def __init__(self, dim: int = 384):  # matches BAAI/bge-small-en-v1.5
        self.dim = dim
        logger.warning("Using SimpleEmbedder (fallback) - not semantic, only for testing")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Map each input string to a fixed-length float vector via MD5 expansion.

        Args:
            texts: a single string or a list of strings
            normalize: L2-normalize each row when ``True``

        Returns:
            ``np.ndarray`` shaped ``(n, dim)`` or ``(dim,)`` when a single string is passed.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()

            vec = np.frombuffer(hash_bytes * (self.dim // len(hash_bytes) + 1), dtype=np.uint8)[:self.dim]
            vec = vec.astype(np.float32) / 255.0
            vec = vec * 2.0 - 1.0

            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

            embeddings.append(vec)

        result = np.array(embeddings)
        return result[0] if len(result) == 1 else result

    @property
    def dimension(self) -> int:
        return self.dim

def get_embedder_fallback():
    """Prefer the real embedder; fall back to ``SimpleEmbedder`` on import/runtime errors."""
    try:
        from backend.embedder import get_embedder as get_real_embedder
        return get_real_embedder()
    except Exception as e:
        logger.warning(f"Failed to load real embedder: {e}, using fallback")
        return SimpleEmbedder(dim=384)
