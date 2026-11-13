"""Shared BERT embedding model singleton — used by faithfulness and context management."""

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True)


def semantic_similarity(text_a: str, text_b: str) -> float:
    vecs = embed([text_a, text_b])
    cos = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-9)
    return float(cos)
