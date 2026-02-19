from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

try:
    from config import MODEL_NAME as _CONFIG_MODEL
except ImportError:
    _CONFIG_MODEL = None

if TYPE_CHECKING:
    import numpy as np

# Default Hugging Face model: M3E base, 768 dim, Chinese + English
M3E_MODEL_ID = "moka-ai/m3e-base"

_lock = threading.Lock()
_model: SentenceTransformer | None = None


def get_encoder(model_name: str | None = None) -> SentenceTransformer:
    """Return the global M3E encoder singleton. Thread-safe lazy init."""
    global _model
    if model_name is None:
        model_name = _CONFIG_MODEL if _CONFIG_MODEL else M3E_MODEL_ID
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(model_name)
    return _model


def encode(texts: list[str], *, normalize: bool = False) -> np.ndarray:
    """
    Encode texts with the shared M3E model. Thread-safe via lock.
    Returns shape (len(texts), 768). If normalize=True, L2-normalize each row.
    """
    import numpy as np

    encoder = get_encoder()
    with _lock:
        emb = encoder.encode(texts, convert_to_numpy=True)
    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms
    return emb
