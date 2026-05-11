"""
LLM Reliability & Risk Analyzer — FastAPI Backend
Main application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.evaluate import router as evaluate_router

app = FastAPI(
    title="LLM Reliability & Risk Analyzer",
    description="AI governance tool for evaluating LLM-generated responses.",
    version="1.0.0",
)

# Allow Streamlit frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate_router, prefix="/api")


@app.get("/")
def health_check():
    return {"status": "ok", "service": "LLM Reliability Analyzer"}
