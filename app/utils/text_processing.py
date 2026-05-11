"""
Text processing utilities — shared helpers used across all services.
"""

import re
import string
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer after cleaning."""
    return clean_text(text).split()


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using TF-IDF vectors.
    Returns a float between 0.0 and 1.0.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    except Exception:
        return 0.0


def count_words(text: str) -> int:
    """Return word count of text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Return approximate sentence count."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def avg_sentence_length(text: str) -> float:
    """Return average words per sentence."""
    words = count_words(text)
    sentences = count_sentences(text)
    if sentences == 0:
        return 0.0
    return words / sentences


def has_structured_elements(text: str) -> bool:
    """Check if text contains lists, headers, or numbered points."""
    patterns = [
        r"^\s*[-*•]\s",       # bullet points
        r"^\s*\d+\.\s",       # numbered list
        r"^#+\s",              # markdown headers
        r"^\s*[A-Z][^.!?]*:$" # label-style headers
    ]
    for line in text.splitlines():
        for pattern in patterns:
            if re.match(pattern, line):
                return True
    return False


def scale_to_ten(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Scale a value from [min_val, max_val] to [0, 10]."""
    if max_val == min_val:
        return 5.0
    scaled = (value - min_val) / (max_val - min_val) * 10
    return round(min(10.0, max(0.0, scaled)), 2)
