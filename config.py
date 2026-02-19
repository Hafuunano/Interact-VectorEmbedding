"""
App config: emotion labels, model name, keyword options. Override via env if needed.
"""

from __future__ import annotations

import os

# Model
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "moka-ai/m3e-base")

# Basic emotion categories (psychology standard)
EMOTION_LABELS = [
    "开心",
    "悲伤",
    "愤怒",
    "惊讶",
    "恐惧",
    "厌恶",
    "中性",
]

# Keyword extraction
KEYWORDS_TOP_K_DEFAULT = int(os.environ.get("KEYWORDS_TOP_K", "5"))
KEYWORDS_TOP_K_MAX = 20

# API
CLASSIFY_TIMEOUT_SECONDS = float(os.environ.get("CLASSIFY_TIMEOUT", "30"))
EXECUTOR_WORKERS = int(os.environ.get("EXECUTOR_WORKERS", "2"))
