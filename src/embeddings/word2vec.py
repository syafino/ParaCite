from __future__ import annotations

import logging
import re

import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api

from .base import Embedder

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DEFAULT_PRETRAINED = "glove-wiki-gigaword-100"


class Word2VecEmbedder(Embedder):
    """Average-pooled word vectors via gensim KeyedVectors.

    Parameters
    ----------
    pretrained : str
        Name recognised by ``gensim.downloader`` (downloaded on first use).
    keyed_vectors_path : str | None
        Path to a saved ``KeyedVectors`` file.  When given, *pretrained* is
        ignored and no download happens.
    """

    def __init__(
        self,
        pretrained: str = _DEFAULT_PRETRAINED,
        keyed_vectors_path: str | None = None,
    ) -> None:
        if keyed_vectors_path is not None:
            log.info("Loading KeyedVectors from %s", keyed_vectors_path)
            self._kv = KeyedVectors.load(keyed_vectors_path)
        else:
            log.info("Loading pretrained model '%s' via gensim downloader", pretrained)
            self._kv = api.load(pretrained)

    @property
    def dim(self) -> int:
        return self._kv.vector_size

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            tokens = _TOKEN_RE.findall(text.lower())
            word_vecs = [self._kv[t] for t in tokens if t in self._kv]
            if word_vecs:
                vec = np.mean(word_vecs, axis=0)
            else:
                vec = np.zeros(self.dim, dtype=np.float32)
            rows.append(vec)
        return np.asarray(rows, dtype=np.float32)
