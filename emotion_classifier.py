from __future__ import annotations

import threading
from typing import NamedTuple

from embedding import encode

try:
    from config import EMOTION_LABELS
except ImportError:
    EMOTION_LABELS = ["开心", "悲伤", "愤怒", "惊讶", "恐惧", "厌恶", "中性"]

_lock = threading.Lock()
_label_embeddings: list[list[float]] | None = None


class EmotionResult(NamedTuple):
    emotion: str
    score: float


def _ensure_label_embeddings() -> list[list[float]]:
    """Compute and cache L2-normalized label embeddings once. Thread-safe."""
    global _label_embeddings
    if _label_embeddings is None:
        with _lock:
            if _label_embeddings is None:
                import numpy as np

                emb = encode(EMOTION_LABELS, normalize=True)
                _label_embeddings = [emb[i].tolist() for i in range(len(EMOTION_LABELS))]
    return _label_embeddings


def classify(text: str) -> EmotionResult:
    """
    Classify text into one of the predefined emotions by cosine similarity.
    Returns the top-1 emotion label and score.
    """
    import numpy as np

    if not text or not text.strip():
        return EmotionResult(emotion=EMOTION_LABELS[-1], score=0.0)  # neutral

    label_emb = _ensure_label_embeddings()
    q = encode([text.strip()], normalize=True)  # (1, 768)
    C = np.array(label_emb, dtype=np.float64)  # (n_classes, 768)
    scores = np.dot(q, C.T).ravel()  # (n_classes,) cosine already (normalized)
    idx = int(np.argmax(scores))
    return EmotionResult(emotion=EMOTION_LABELS[idx], score=float(scores[idx]))


def is_ready() -> bool:
    """Return True if label embeddings are cached (model and categories ready)."""
    return _label_embeddings is not None
