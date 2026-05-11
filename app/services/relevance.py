"""
Relevance Service
Checks if the AI response actually answers the given prompt.
Uses TF-IDF cosine similarity between prompt and response.
"""

from app.utils.text_processing import (
    tfidf_cosine_similarity,
    count_words,
    scale_to_ten,
    clean_text,
)


def compute_relevance(prompt: str, response: str) -> dict:
    """
    Evaluate how relevant the response is to the prompt.

    Returns:
        dict with score (0-10), label, note, and issues list.
    """
    issues = []
    suggestions = []

    # --- Core similarity score ---
    # TF-IDF cosine for prompt-response pairs realistically peaks around 0.3–0.4
    # for good answers (different vocabulary = lower overlap). Scale [0, 0.35] → [0, 10].
    similarity = tfidf_cosine_similarity(prompt, response)
    score = scale_to_ten(similarity, min_val=0.0, max_val=0.35)

    # --- Word count check: too short is likely off-topic or incomplete ---
    word_count = count_words(response)
    if word_count < 15:
        score = min(score, 4.0)
        issues.append("Response is very short — likely incomplete or vague.")
        suggestions.append("Expand the response to fully address the prompt.")

    # --- Check for keyword overlap between prompt and response ---
    prompt_keywords = set(clean_text(prompt).split()) - {"what", "is", "the", "a", "an", "of", "in", "to", "how", "why", "when", "who", "does", "do", "are", "was", "were", "be", "been"}
    response_words = set(clean_text(response).split())
    overlap = prompt_keywords & response_words
    keyword_coverage = len(overlap) / max(len(prompt_keywords), 1)

    if keyword_coverage < 0.1 and score > 5:
        score = min(score, 5.5)
        issues.append("Response shares few keywords with the prompt — may be off-topic.")

    # --- Final score clamp ---
    score = round(min(10.0, max(0.0, score)), 2)

    # --- Interpret score ---
    if score >= 8:
        label = "Highly relevant"
        note = "Response directly and comprehensively addresses the prompt."
    elif score >= 6:
        label = "Mostly relevant"
        note = "Response is generally on-topic but may miss some aspects of the prompt."
    elif score >= 4:
        label = "Partially relevant"
        note = "Response touches the topic but drifts or lacks focus."
        if not issues:
            issues.append("Response only partially answers the prompt.")
    else:
        label = "Low relevance"
        note = "Response does not adequately address the prompt."
        if not issues:
            issues.append("Response appears off-topic or fails to answer the question.")
        suggestions.append("Rewrite the response to directly target the prompt's question.")

    return {
        "score": score,
        "label": label,
        "note": note,
        "similarity_raw": round(similarity, 4),
        "keyword_coverage": round(keyword_coverage, 4),
        "issues": issues,
        "suggestions": suggestions,
    }
