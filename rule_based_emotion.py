"""

Reference From : https://github.com/NickCharlie/astrbot_plugin_self_learning/

"""

from __future__ import annotations

import re
from typing import NamedTuple

try:
    from config import EMOTION_LABELS
except ImportError:
    EMOTION_LABELS = ["开心", "悲伤", "愤怒", "惊讶", "恐惧", "厌恶", "中性"]


class RuleBasedEmotionResult(NamedTuple):
    """Same shape as EmotionResult for compatibility."""
    emotion: str
    score: float
    scores: dict[str, float] | None = None  # optional: all category scores


# Keyword lists per emotion (Chinese + common emoji). Order matches EMOTION_LABELS usage.
EMOTION_KEYWORDS: dict[str, list[str]] = {
    "开心": [
        "开心", "高兴", "兴奋", "满意", "喜欢", "爱", "好棒", "太好了", "哈哈",
        "快乐", "愉快", "幸福", "赞", "棒", "谢谢", "😄", "😊", "👍", "❤️",
    ],
    "悲伤": [
        "难过", "伤心", "悲哀", "沮丧", "郁闷", "哭", "痛苦", "失落", "失望",
        "😭", "😢", "💔",
    ],
    "愤怒": [
        "生气", "愤怒", "烦", "讨厌", "火大", "气", "恼火", "暴躁",
        "😡",
    ],
    "惊讶": [
        "哇", "天哪", "真的", "不会吧", "竟然", "居然", "惊", "吓",
        "😱", "😯", "🤔",
    ],
    "恐惧": [
        "害怕", "恐惧", "慌", "吓", "恐怖", "担心", "不安", "紧张",
    ],
    "厌恶": [
        "恶心", "嫌弃", "反感", "讨厌", "烦", "糟糕", "差", "不行",
        "坏", "烂", "糟", "屎", "滚", "呸",
    ],
    "中性": [
        "知道", "明白", "可以", "好的", "嗯", "哦", "这样", "然后",
        "吗", "呢", "什么", "怎么", "为什么", "哪里",
    ],
}


def _tokenize(text: str) -> list[str]:
    """Split text by spaces and common Chinese punctuation; filter empty. No embedding."""
    if not text or not text.strip():
        return []
    # Same pattern as astrbot: spaces + ，。！？；：
    parts = re.split(r"\s+|[，。！？；：、]", text.strip())
    return [p for p in parts if p]


def classify_rule_based(text: str) -> RuleBasedEmotionResult:
    """
    Classify text into one emotion by keyword matching only (no LLM, no embedding).
    Returns the emotion with highest score and that score; optionally all scores in result.
    """
    words = _tokenize(text)
    total = max(len(words), 1)

    # Build score per label (only for labels we have keywords for)
    labels = [lab for lab in EMOTION_LABELS if lab in EMOTION_KEYWORDS]
    if not labels:
        labels = list(EMOTION_KEYWORDS.keys())

    scores_dict: dict[str, float] = {}
    for label in labels:
        keywords = EMOTION_KEYWORDS.get(label, [])
        count = sum(1 for w in words if w in keywords)
        scores_dict[label] = count / total

    if not scores_dict:
        default = EMOTION_LABELS[-1] if EMOTION_LABELS else "中性"
        return RuleBasedEmotionResult(emotion=default, score=0.0, scores=None)

    best_label = max(scores_dict, key=scores_dict.get)
    best_score = scores_dict[best_label]

    # When no keyword matched (all zeros), return neutral instead of first label (e.g. 开心)
    if best_score <= 0.0:
        neutral = "中性"
        if neutral in EMOTION_LABELS:
            best_label = neutral
        else:
            best_label = EMOTION_LABELS[-1] if EMOTION_LABELS else "中性"

    return RuleBasedEmotionResult(
        emotion=best_label,
        score=best_score,
        scores=scores_dict,
    )


def is_available() -> bool:
    """Rule-based path is always available (no model load)."""
    return True
