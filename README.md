# 🛡️ LLM Reliability & Risk Analyzer

> **"AI for evaluating AI"** — A production-style AI governance and quality assurance platform for evaluating LLM-generated responses before deployment.

---

## Problem Statement

Large Language Models (LLMs) are increasingly being deployed in production systems, but they come with a critical challenge: **they hallucinate, produce unsafe content, go off-topic, and generate poorly structured responses** — often without any warning.

Companies deploying AI systems need a systematic way to evaluate LLM outputs before they reach end users. This project addresses that gap.

---

## Why AI Reliability Matters

| Risk | Real-World Impact |
|------|-------------------|
| Hallucination | Misinformation, legal liability, broken trust |
| Safety violations | Harm to users, regulatory penalties |
| Low relevance | Poor user experience, task failure |
| Poor structure | Misunderstandings, support escalations |

---

## Architecture

```
llm-reliability-analyzer/
│
├── app/                         # FastAPI backend
│   ├── main.py                  # App entry point, CORS, routing
│   ├── routes/
│   │   └── evaluate.py          # POST /api/evaluate endpoint
│   ├── services/
│   │   ├── relevance.py         # TF-IDF cosine similarity scoring
│   │   ├── hallucination.py     # Factual consistency + pattern detection
│   │   ├── safety.py            # Toxicity, PII, injection detection
│   │   ├── structure.py         # Readability, length, clarity
│   │   └── scoring.py           # Weighted trust score + report assembly
│   └── utils/
│       └── text_processing.py   # Shared NLP helpers
│
├── frontend/
│   └── streamlit_app.py         # Streamlit dashboard UI
│
├── data/
│   └── sample_facts.json        # Trusted facts database for fact-checking
│
├── tests/
│   └── test_evaluation.py       # Full pytest suite (40+ test cases)
│
├── requirements.txt
└── README.md
```

---

## Features

### Core Evaluation Pipeline

| Dimension | Method | Weight |
|-----------|--------|--------|
| **Factual Accuracy** | Pattern heuristics + fact DB + context similarity | 40% |
| **Relevance** | TF-IDF cosine similarity + keyword coverage | 30% |
| **Safety** | Regex pattern scanning (hate, violence, PII, injection) | 20% |
| **Structure** | Length, readability, vague language, formatting | 10% |

### Trust Score Formula
```
Final Score = 0.40 × Factual + 0.30 × Relevance + 0.20 × Safety + 0.10 × Structure
```

### Additional Features
- 📋 Detailed per-dimension explainability notes
- ⚠️ Issue detection with severity labels (critical / warning / info)
- 💡 Actionable improvement suggestions
- 📄 Full JSON report export
- 🕓 Session-based evaluation history
- 🔌 Modular architecture — swap in moderation APIs or embedding models with zero interface changes

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/llm-reliability-analyzer.git
cd llm-reliability-analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Start the FastAPI backend
```bash
uvicorn app.main:app --reload
```
API will be live at: `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

### Step 2 — Start the Streamlit frontend (new terminal)
```bash
streamlit run frontend/streamlit_app.py
```
Dashboard at: `http://localhost:8501`

### Step 3 — Run the test suite
```bash
pytest tests/test_evaluation.py -v
```

---

## API Reference

### `POST /api/evaluate`

**Request body:**
```json
{
  "prompt": "Who discovered gravity?",
  "response": "Albert Einstein discovered gravity in 1902.",
  "context": "Isaac Newton described gravity in 1687."
}
```

**Response:**
```json
{
  "trust_score": 4.6,
  "verdict": "Low Trust",
  "relevance": { "score": 7.2, "label": "Mostly relevant", ... },
  "factual": { "score": 3.1, "hallucination_risk": "high", ... },
  "safety": { "score": 10.0, "verdict": "safe", ... },
  "structure": { "score": 5.0, ... },
  "issues": ["Contradiction with verified knowledge about gravity discovery."],
  "suggestions": ["Verified information: Isaac Newton discovered gravity."]
}
```

---

## Example Output

```
Prompt:    "Who discovered gravity?"
Response:  "Albert Einstein discovered gravity in 1902."

RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━
Relevance:       7.2 / 10  ✅
Factual:         2.5 / 10  ❌  [Hallucination risk: HIGH]
Safety:         10.0 / 10  ✅
Structure:       5.0 / 10  ⚠️

Issues:
  • Contradiction with verified knowledge about gravity discovery.
  • Attribution claim detected — verify factual accuracy.

Suggestion:
  • Verified: Isaac Newton published his law of universal gravitation in 1687.

FINAL TRUST SCORE: 4.8 / 10  →  Low Trust
━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Future Improvements

- [ ] Integrate sentence-transformers for semantic embedding-based relevance
- [ ] Add OpenAI / Azure moderation API as safety backend option
- [ ] LLM-as-judge mode: use a second LLM call for factual verification
- [ ] SQLite history storage with search and filtering
- [ ] PDF report export
- [ ] Batch evaluation mode: upload CSV of prompt-response pairs
- [ ] Side-by-side comparison of two AI responses
- [ ] Configurable weight profiles (e.g. "safety-first" vs "accuracy-first")
- [ ] REST API authentication for enterprise deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| NLP | scikit-learn (TF-IDF), regex |
| Data | pandas, numpy |
| Testing | pytest |
| Language | Python 3.10+ |

---

## Project Positioning

This project demonstrates:
- ✅ Understanding of LLM reliability challenges (hallucination, safety, relevance)
- ✅ Production-oriented system design (modular services, typed APIs, separation of concerns)
- ✅ AI safety and governance thinking
- ✅ Evaluation pipeline architecture
- ✅ Explainability — every score comes with a reason
- ✅ Test-driven development (40+ test cases across all services)

---

*Built as a demonstration of AI governance tooling and LLM evaluation system design.*
