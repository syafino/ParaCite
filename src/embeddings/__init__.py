from .base import Embedder

__all__ = ["Embedder", "Word2VecEmbedder", "SentenceTransformerEmbedder"]


def __getattr__(name: str):
    """Lazy-load concrete embedders so heavy deps are only imported on use."""
    if name == "Word2VecEmbedder":
        from .word2vec import Word2VecEmbedder
        return Word2VecEmbedder
    if name == "SentenceTransformerEmbedder":
        from .sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
