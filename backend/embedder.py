#!/usr/bin/env python3
"""
Sentence embeddings via SentenceTransformer.

Default model: **BAAI/bge-small-en-v1.5** (384-d, English retrieval).

Load order (same as main `s` repo):
1. `EMBEDDER_MODEL` if it points to an **existing local directory** (offline SentenceTransformer folder).
2. Cached tree under `MODELSCOPE_CACHE` / `EMBEDDER_MODEL_SCOPE` if that path exists on disk.
3. `modelscope.snapshot_download(EMBEDDER_MODEL_SCOPE)` when `modelscope` is installed.
4. HuggingFace `SentenceTransformer(MODEL_NAME)` with `HF_ENDPOINT` (e.g. hf-mirror).

Override with env: `EMBEDDER_MODEL`, `EMBEDDER_MODEL_SCOPE`, `MODELSCOPE_CACHE`, `HF_ENDPOINT`.
"""
import os
from pathlib import Path
import threading
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# HuggingFace model id when falling back to HF / when EMBEDDER_MODEL is not a path
_DEFAULT_HF_ID = "BAAI/bge-small-en-v1.5"
MODEL_NAME = os.environ.get("EMBEDDER_MODEL", _DEFAULT_HF_ID)

# Default ModelScope / cache layout id (often same card name as on HF)
_DEFAULT_SCOPE_ID = "BAAI/bge-small-en-v1.5"

_DEFAULT_MODELS_DIR = (Path(__file__).resolve().parents[1] / "models")


class Embedder:
    """Text embedder (singleton)."""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_init_done", False):
            return
        self._load_lock = threading.Lock()
        self._init_done = True

    def _load_model(self) -> None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            env_model = os.environ.get("EMBEDDER_MODEL", "").strip()
            if env_model:
                p = Path(env_model)
                if p.exists():
                    logger.info("Loading embedder from local path (EMBEDDER_MODEL): %s", p)
                    self._model = SentenceTransformer(str(p), device=device)
                    logger.info("Loaded embedder from local path (EMBEDDER_MODEL).")
                    return

            modelscope_id = os.environ.get("EMBEDDER_MODEL_SCOPE", _DEFAULT_SCOPE_ID).strip()
            cache_dir = os.environ.get("MODELSCOPE_CACHE", str(_DEFAULT_MODELS_DIR)).strip() or str(_DEFAULT_MODELS_DIR)
            cache_path = Path(cache_dir)
            if not cache_path.is_absolute():
                cache_path = (Path(__file__).resolve().parents[1] / cache_path).resolve()

            local_candidate = cache_path / modelscope_id
            if local_candidate.exists():
                logger.info("Loading embedder from cache path: %s", local_candidate)
                self._model = SentenceTransformer(str(local_candidate), device=device)
                logger.info("Loaded embedder from cache path.")
                return

            try:
                from modelscope import snapshot_download

                logger.info(
                    "Downloading embedder from ModelScope: %s (cache_dir=%s)",
                    modelscope_id,
                    cache_path,
                )
                model_path = snapshot_download(modelscope_id, cache_dir=str(cache_path))
                self._model = SentenceTransformer(model_path, device=device)
                logger.info("Loaded embedder from ModelScope: %s", model_path)
            except ImportError:
                logger.warning("ModelScope not available, falling back to HuggingFace")
                hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
                os.environ["HF_ENDPOINT"] = hf_endpoint
                hf_id = MODEL_NAME if "/" in MODEL_NAME else _DEFAULT_HF_ID
                self._model = SentenceTransformer(hf_id, device=device)
                logger.info("Loaded embedder from HF (via %s): %s", hf_endpoint, hf_id)
            except Exception as e:
                logger.warning("ModelScope download failed: %s, falling back to HuggingFace", e)
                hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
                os.environ["HF_ENDPOINT"] = hf_endpoint
                hf_id = MODEL_NAME if "/" in MODEL_NAME else _DEFAULT_HF_ID
                logger.info("Initializing SentenceTransformer on device: %s", device)
                self._model = SentenceTransformer(hf_id, device=device)
                logger.info("Loaded embedder from HF (via %s): %s", hf_endpoint, hf_id)
        except Exception as e:
            logger.error("Failed to load embedder: %s", e)
            raise

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        return embeddings[0] if len(embeddings) == 1 else embeddings

    def encode_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._load_model()

        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )

    @property
    def dimension(self) -> int:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._load_model()
        return int(self._model.get_sentence_embedding_dimension())


_embedder = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
