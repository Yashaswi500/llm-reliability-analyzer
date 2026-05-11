"""
Evaluation route — POST /api/evaluate
Orchestrates all evaluation services and returns the full reliability report.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.relevance import compute_relevance
from app.services.hallucination import compute_hallucination
from app.services.safety import compute_safety
from app.services.structure import compute_structure
from app.services.scoring import compute_trust_score, build_report

router = APIRouter()


class EvaluationRequest(BaseModel):
    prompt: str
    response: str
    context: Optional[str] = None  # Optional reference/trusted context


class EvaluationResponse(BaseModel):
    prompt: str
    response: str
    relevance: dict
    factual: dict
    safety: dict
    structure: dict
    issues: list
    suggestions: list
    trust_score: float
    verdict: str


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest):
    """
    Main evaluation endpoint.
    Accepts a prompt + AI response, runs all checks, returns reliability report.
    """
    if not request.prompt.strip() or not request.response.strip():
        raise HTTPException(status_code=400, detail="Prompt and response must not be empty.")

    # --- Run all evaluation services ---
    relevance_result = compute_relevance(request.prompt, request.response)
    hallucination_result = compute_hallucination(request.response, request.context)
    safety_result = compute_safety(request.response)
    structure_result = compute_structure(request.response)

    # --- Compute final trust score and build report ---
    report = build_report(
        prompt=request.prompt,
        response=request.response,
        relevance=relevance_result,
        factual=hallucination_result,
        safety=safety_result,
        structure=structure_result,
    )

    return report
