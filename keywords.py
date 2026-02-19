
from __future__ import annotations

import re
from embedding import encode


def _segment_candidates(text: str, min_len: int = 2, max_len: int = 6) -> list[str]:
    """Extract word/phrase candidates via jieba. Filter by length and dedupe."""
    import jieba

    text = text.strip()
    if not text:
        return []
    # Words
    words = [w for w in jieba.lcut(text) if min_len <= len(w) <= max_len and re.match(r"[\u4e00-\u9fff\w]+", w)]
    # Optional: add bigrams
    seen = set(words)
    for i in range(len(words) - 1):
        bigram = words[i] + words[i + 1]
        if min_len <= len(bigram) <= max_len and bigram not in seen:
            seen.add(bigram)
            words.append(bigram)
    return list(dict.fromkeys(words))[:50]  # cap candidates to avoid huge encode


def extract_keywords(text: str, top_k: int = 5) -> list[str]:
    """
    Extract top_k keywords from text using m3e embedding similarity.
    Document and candidates are encoded together; candidates are ranked by
    cosine similarity to the document vector.
    """
    text = text.strip()
    if not text:
        return []
    candidates = _segment_candidates(text)
    if not candidates:
        return []

    # Encode doc and all candidates in one batch
    batch = [text] + candidates
    emb = encode(batch, normalize=True)
    doc_vec = emb[0:1]  # (1, 768)
    cand_vec = emb[1:]   # (n_cand, 768)
    scores = (doc_vec @ cand_vec.T).ravel()

    # Top-k indices (skip if score too low)
    top_indices = scores.argsort()[::-1][:top_k]
    result: list[str] = []
    for i in top_indices:
        if scores[i] > 0.3:  # minimal relevance
            result.append(candidates[i])
    return result[:top_k] if result else candidates[:top_k]
