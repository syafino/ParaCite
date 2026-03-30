from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Embedder

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(Embedder):
    """BERT-family embedder backed by ``sentence-transformers``."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        log.info("Loading SentenceTransformer model '%s'", model_name)
        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32, copy=False)
