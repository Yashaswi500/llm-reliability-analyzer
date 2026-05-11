"""
Structure & Clarity Service
Evaluates the structural quality, readability, and completeness of an AI response.

Checks:
- Response length (too short = vague, too long = bloated)
- Sentence complexity and average length
- Presence of organized structure (lists, headers, paragraphs)
- Vague / filler language patterns
- Completeness signals
"""

import re
from app.utils.text_processing import (
    count_words,
    count_sentences,
    avg_sentence_length,
    has_structured_elements,
)


# Vague filler phrases that reduce clarity
VAGUE_PATTERNS = [
    r"\b(it depends|various factors|many things|some people|in some cases)\b",
    r"\b(etc\.|and so on|and so forth|blah blah)\b",
    r"\b(basically|kind of|sort of|you know|like,)\b",
    r"\b(i think|i believe|i guess|maybe|perhaps|possibly)\b",
]

# Incomplete answer signals
INCOMPLETE_PATTERNS = [
    r"\b(i (don't|do not|can't|cannot) (know|answer|help|say))\b",
    r"\b(i'm not sure|i am not sure|unclear|not certain)\b",
    r"\b(more research (is needed|needed)|consult a professional)\b",
]


def _count_vague_phrases(text: str) -> int:
    """Count number of vague filler phrase matches."""
    count = 0
    text_lower = text.lower()
    for pattern in VAGUE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        count += len(matches)
    return count


def _count_incomplete_signals(text: str) -> int:
    """Count number of incomplete/evasive answer signals."""
    count = 0
    text_lower = text.lower()
    for pattern in INCOMPLETE_PATTERNS:
        if re.search(pattern, text_lower):
            count += 1
    return count


def _readability_penalty(avg_len: float) -> float:
    """
    Penalize responses with very long average sentence length (hard to read)
    or very short (choppy / underdeveloped).
    Ideal: 12–22 words per sentence.
    """
    if 12 <= avg_len <= 22:
        return 0.0
    elif avg_len < 6:
        return 1.5
    elif avg_len > 35:
        return 1.5
    elif avg_len > 28:
        return 0.8
    else:
        return 0.3


def compute_structure(response: str) -> dict:
    """
    Evaluate the structural quality and clarity of the response.

    Returns:
        dict with score (0-10), note, issues, suggestions.
    """
    issues = []
    suggestions = []
    score = 10.0

    word_count = count_words(response)
    sentence_count = count_sentences(response)
    avg_len = avg_sentence_length(response)
    is_structured = has_structured_elements(response)
    vague_count = _count_vague_phrases(response)
    incomplete_count = _count_incomplete_signals(response)

    # --- Length check ---
    if word_count < 20:
        score -= 4.5
        issues.append("Response is too short — lacks sufficient detail or explanation.")
        suggestions.append("Expand the answer with relevant details, examples, or steps.")
    elif word_count < 50:
        score -= 1.5
        issues.append("Response is brief — may be incomplete for complex topics.")
    elif word_count > 600:
        score -= 1.0
        issues.append("Response is very long — consider trimming for readability.")
        suggestions.append("Break the response into clearly labeled sections or bullet points.")

    # --- Readability ---
    penalty = _readability_penalty(avg_len)
    score -= penalty
    if avg_len > 30:
        issues.append(f"Average sentence length is {avg_len:.1f} words — sentences are too complex.")
        suggestions.append("Split long sentences into shorter, clearer statements.")
    elif avg_len < 6 and sentence_count > 3:
        issues.append("Sentences are very short and choppy — may reduce comprehension.")

    # --- Vague language ---
    if vague_count >= 3:
        score -= 1.5
        issues.append(f"Response contains {vague_count} vague or filler phrases.")
        suggestions.append("Replace vague phrases with specific, factual statements.")
    elif vague_count > 0:
        score -= 0.5

    # --- Incomplete signals ---
    if incomplete_count >= 2:
        score -= 2.0
        issues.append("Response appears evasive or incomplete.")
        suggestions.append("Provide a direct and complete answer instead of deflecting.")
    elif incomplete_count == 1:
        score -= 0.8

    # --- Bonus for well-structured responses ---
    if is_structured and word_count >= 60:
        score = min(10.0, score + 0.5)
    elif not is_structured and word_count >= 150:
        issues.append("Long response lacks headers or lists — consider adding structure.")
        suggestions.append("Use bullet points, numbered steps, or section headers to organize information.")

    score = round(min(10.0, max(0.0, score)), 2)

    # --- Note ---
    if score >= 8:
        note = "Well-structured, clear, and appropriately detailed response."
    elif score >= 6:
        note = "Adequate structure but has room for improvement in clarity or depth."
    elif score >= 4:
        note = "Response lacks structure or clarity — notable quality issues present."
    else:
        note = "Poor structure — response is vague, incomplete, or incoherent."

    return {
        "score": score,
        "note": note,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_len, 1),
        "has_structure": is_structured,
        "vague_phrase_count": vague_count,
        "issues": issues,
        "suggestions": suggestions,
    }
