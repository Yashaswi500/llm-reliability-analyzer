"""
Hallucination / Factual Consistency Service
Detects potentially fabricated or incorrect information in AI responses.

Strategy:
1. If reference context is provided → similarity-based factual check.
2. Always → pattern-based heuristics (overconfident claims, unverifiable stats, etc.).
3. Load sample known facts from data/sample_facts.json for basic fact-checking.
"""

import json
import os
import re
from typing import Optional

from app.utils.text_processing import tfidf_cosine_similarity, count_sentences


# Path to trusted facts database
FACTS_FILE = os.path.join(os.path.dirname(__file__), "../../data/sample_facts.json")


def _load_facts() -> list:
    """Load known facts from the JSON file."""
    try:
        with open(FACTS_FILE, "r") as f:
            return json.load(f).get("facts", [])
    except Exception:
        return []


def _detect_overconfidence_patterns(text: str) -> list:
    """
    Detect linguistic patterns associated with hallucination risk:
    - Overly specific numbers/dates without context
    - Absolute statements about uncertain topics
    - Common hallucination markers
    """
    issues = []
    text_lower = text.lower()

    # Suspicious specificity: vague year claims (not tied to a known publication/event keyword)
    known_event_words = ["published", "founded", "established", "created", "released", "born", "died", "signed", "launched", "invented"]
    has_known_event = any(w in text_lower for w in known_event_words)
    if re.search(r"\bin (1[0-9]{3}|20[0-9]{2})\b", text_lower) and not has_known_event:
        issues.append("Response contains a year reference without an established event — verify for accuracy.")

    # Unverified percentage or statistic claims
    if re.search(r"\b\d+(\.\d+)?%\b", text_lower):
        issues.append("Response contains percentage/statistic claims — confirm against a source.")

    # Absolute authority claims
    absolute_patterns = ["it is a fact that", "it is well known", "it is proven", "studies show", "scientists have confirmed"]
    for p in absolute_patterns:
        if p in text_lower:
            issues.append(f"Overconfident claim detected: '{p}' — ensure this is verifiable.")
            break

    # Named person + invented action (simple heuristic: named entity + verb + past tense)
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b.{0,50}\b(discovered|invented|created|said|wrote|founded|developed)\b", text):
        issues.append("Attribution claim detected (person + action) — verify factual accuracy.")

    return issues


def _check_against_facts(response: str, facts: list) -> tuple:
    """
    Compare response against known trusted facts using similarity.
    Returns (contradiction_found: bool, relevant_fact: str or None, similarity: float).
    """
    best_sim = 0.0
    best_fact = None

    for fact_entry in facts:
        fact_text = fact_entry.get("fact", "")
        sim = tfidf_cosine_similarity(response, fact_text)
        if sim > best_sim:
            best_sim = sim
            best_fact = fact_entry

    if best_fact and best_sim > 0.2:
        correct_answer = best_fact.get("correct_answer", "").lower()
        response_lower = response.lower()
        # Check that keywords from the correct answer appear in the response
        # Extract key name/term (first 3 significant words of correct answer)
        key_terms = [w for w in correct_answer.split()[:8] if len(w) > 3]
        matches = sum(1 for term in key_terms if term in response_lower)
        coverage = matches / max(len(key_terms), 1)
        # If response doesn't cover >40% of the correct answer's key terms → contradiction
        if correct_answer and coverage < 0.4:
            return True, best_fact, best_sim

    return False, best_fact, best_sim


def compute_hallucination(response: str, context: Optional[str] = None) -> dict:
    """
    Evaluate factual reliability and hallucination risk.

    Args:
        response: The AI-generated response.
        context:  Optional trusted reference text to compare against.

    Returns:
        dict with score, hallucination_risk, issues, suggestions, note.
    """
    issues = []
    suggestions = []
    score = 8.0  # Start optimistic; deduct for red flags

    # --- 1. Pattern-based heuristic checks ---
    heuristic_issues = _detect_overconfidence_patterns(response)
    issues.extend(heuristic_issues)
    # Lower penalty per issue — patterns are signals, not proof
    score -= len(heuristic_issues) * 0.5

    # --- 2. Check against local facts database ---
    facts = _load_facts()
    contradiction_found, matched_fact, fact_sim = _check_against_facts(response, facts)

    if contradiction_found and matched_fact:
        score -= 3.5
        correct = matched_fact.get("correct_answer", "")
        topic = matched_fact.get("topic", "the topic")
        issues.append(f"Possible contradiction with verified knowledge about {topic}.")
        if correct:
            suggestions.append(f"Verified information: {correct}")

    # --- 3. Context-based factual consistency check ---
    if context and context.strip():
        context_similarity = tfidf_cosine_similarity(response, context)

        if context_similarity < 0.15:
            score -= 3.0
            issues.append("Response shows very low alignment with the provided reference context.")
            suggestions.append("Review and align the response with the provided reference context.")
        elif context_similarity < 0.4:
            score -= 1.5
            issues.append("Response partially diverges from the reference context.")
        elif context_similarity >= 0.7:
            score = min(10.0, score + 0.5)

    # --- 4. Empty / very short responses are suspect ---
    if len(response.split()) < 10:
        score -= 1.5
        issues.append("Response is too short to evaluate factual depth.")

    # --- Final score ---
    score = round(min(10.0, max(0.0, score)), 2)

    # --- Hallucination risk label ---
    if score >= 7.5:
        risk = "low"
        note = "No significant hallucination indicators detected."
    elif score >= 5.0:
        risk = "medium"
        note = "Some potential inaccuracies detected — manual review recommended."
        if not suggestions:
            suggestions.append("Cross-check claims in this response with authoritative sources.")
    else:
        risk = "high"
        note = "High hallucination risk — response likely contains fabricated or incorrect information."
        suggestions.append("Do not deploy this response without thorough fact-checking.")

    return {
        "score": score,
        "hallucination_risk": risk,
        "note": note,
        "context_used": bool(context and context.strip()),
        "fact_db_matched": matched_fact is not None,
        "issues": issues,
        "suggestions": suggestions,
    }
