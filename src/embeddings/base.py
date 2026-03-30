from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Uniform interface for text embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 matrix of embeddings for *texts*."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...
