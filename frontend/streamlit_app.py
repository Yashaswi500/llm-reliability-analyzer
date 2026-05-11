"""
LLM Reliability & Risk Analyzer — Streamlit Frontend
Professional AI governance dashboard for evaluating LLM responses.
"""

import streamlit as st
import requests
import json
import pandas as pd
import time

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Reliability Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main layout */
    .main { background-color: #f8f7f4; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Header */
    .app-header {
        background: #0f0f0f;
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { color: #ffffff; font-size: 1.6rem; font-weight: 600; margin: 0; }
    .app-header p { color: #888; font-size: 0.875rem; margin: 0.25rem 0 0; }
    .badge {
        display: inline-block; background: #1a1a1a; color: #9ca3af;
        font-size: 0.7rem; padding: 3px 10px; border-radius: 4px;
        border: 1px solid #333; margin-bottom: 0.5rem; letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    .metric-label { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #0f0f0f; line-height: 1.1; }
    .metric-sub { font-size: 0.75rem; color: #999; margin-top: 2px; }

    /* Trust score big display */
    .trust-display {
        background: #0f0f0f; color: #fff;
        border-radius: 12px; padding: 1.5rem 2rem;
        text-align: center;
    }
    .trust-display .score { font-size: 4rem; font-weight: 700; line-height: 1; }
    .trust-display .label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.5rem; }

    /* Issue/suggestion pills */
    .issue-critical { background: #fee2e2; color: #991b1b; border-radius: 4px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600; }
    .issue-warning  { background: #fef3c7; color: #92400e; border-radius: 4px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600; }
    .issue-info     { background: #dbeafe; color: #1e40af; border-radius: 4px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600; }

    /* Section titles */
    .section-title {
        font-size: 0.7rem; font-weight: 600; color: #888;
        text-transform: uppercase; letter-spacing: 0.08em;
        border-bottom: 1px solid #e5e5e5; padding-bottom: 6px; margin-bottom: 12px;
    }

    /* Verdict banner */
    .verdict-high    { background: #d1fae5; color: #065f46; border-radius: 8px; padding: 0.75rem 1rem; }
    .verdict-moderate{ background: #fef3c7; color: #78350f; border-radius: 8px; padding: 0.75rem 1rem; }
    .verdict-low     { background: #fee2e2; color: #7f1d1d; border-radius: 8px; padding: 0.75rem 1rem; }
    .verdict-critical{ background: #1f2937; color: #f87171; border-radius: 8px; padding: 0.75rem 1rem; }

    /* Progress bar override */
    .stProgress > div > div { border-radius: 4px; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api"


def get_score_color(score: float) -> str:
    if score >= 8:
        return "#16a34a"
    elif score >= 6:
        return "#ca8a04"
    elif score >= 4:
        return "#ea580c"
    return "#dc2626"


def get_verdict_class(verdict: str) -> str:
    mapping = {
        "High Trust": "verdict-high",
        "Moderate Trust": "verdict-moderate",
        "Low Trust": "verdict-low",
        "Critical Risk": "verdict-critical",
    }
    return mapping.get(verdict, "verdict-moderate")


def get_issue_class(score: float) -> str:
    if score < 4:
        return "issue-critical"
    if score < 7:
        return "issue-warning"
    return "issue-info"


def call_evaluate_api(prompt: str, response: str, context: str) -> dict:
    """Call the FastAPI backend and return the evaluation report."""
    payload = {"prompt": prompt, "response": response, "context": context or None}
    r = requests.post(f"{API_BASE}/evaluate", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def render_score_bar(label: str, score: float, note: str = ""):
    """Render a labeled score bar."""
    color = get_score_color(score)
    st.markdown(f"**{label}** — `{score}/10`")
    st.progress(score / 10)
    if note:
        st.caption(note)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="badge">🛡️ AI Governance Tool</div>
    <h1>LLM Reliability & Risk Analyzer</h1>
    <p>Evaluate AI-generated responses for hallucination risk, factual accuracy, safety, and structural quality.</p>
</div>
""", unsafe_allow_html=True)

# ── Input Form ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown("#### Input")
    prompt_input = st.text_area(
        "Prompt",
        placeholder="Enter the original prompt that was given to the AI...",
        height=100,
        key="prompt",
    )
    response_input = st.text_area(
        "AI-Generated Response",
        placeholder="Paste the AI response to evaluate here...",
        height=150,
        key="response",
    )
    context_input = st.text_area(
        "Reference Context (optional)",
        placeholder="Provide trusted facts or reference text to check the response against...",
        height=80,
        key="context",
    )

with col_right:
    st.markdown("#### Weights")
    st.caption("Trust score formula")
    weights_df = pd.DataFrame({
        "Dimension": ["Factual Accuracy", "Relevance", "Safety", "Structure"],
        "Weight": ["40%", "30%", "20%", "10%"],
    })
    st.dataframe(weights_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### About")
    st.caption(
        "This tool runs heuristic, pattern-based, and similarity analysis locally. "
        "No data is sent to third-party APIs."
    )

evaluate_btn = st.button("🔍 Run Reliability Evaluation", type="primary", use_container_width=True)

# ── Evaluation ───────────────────────────────────────────────────────────────
if evaluate_btn:
    if not prompt_input.strip() or not response_input.strip():
        st.error("Please fill in both the Prompt and AI Response fields.")
        st.stop()

    with st.spinner("Running evaluation pipeline..."):
        try:
            report = call_evaluate_api(prompt_input, response_input, context_input)
        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ Cannot connect to the backend. Make sure the FastAPI server is running:\n"
                "```\nuvicorn app.main:app --reload\n```"
            )
            st.stop()
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.stop()

    st.success("Evaluation complete.")
    st.markdown("---")

    # ── Trust Score Banner ────────────────────────────────────────────────────
    st.markdown("### Reliability Report")

    trust = report["trust_score"]
    verdict = report["verdict"]
    verdict_class = get_verdict_class(verdict)
    verdict_emoji = {"High Trust": "✅", "Moderate Trust": "⚠️", "Low Trust": "🔴", "Critical Risk": "🚨"}.get(verdict, "⚠️")

    banner_col, score_col = st.columns([3, 1])
    with banner_col:
        st.markdown(f"""
        <div class="{verdict_class}">
            <strong>{verdict_emoji} {verdict}</strong><br>
            <span style="font-size:0.85rem;">Final Trust Score: <strong>{trust}/10</strong> — 
            Weighted across factual accuracy, relevance, safety, and structure.</span>
        </div>
        """, unsafe_allow_html=True)
    with score_col:
        score_color = get_score_color(trust)
        st.markdown(f"""
        <div style="background:{score_color};color:#fff;border-radius:10px;padding:1rem;text-align:center;">
            <div style="font-size:2.5rem;font-weight:700;line-height:1;">{trust}</div>
            <div style="font-size:0.7rem;margin-top:4px;opacity:0.85;">/ 10</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Metric Cards ──────────────────────────────────────────────────────────
    st.markdown("#### Dimension Scores")
    m1, m2, m3, m4 = st.columns(4)

    dimensions = [
        (m1, "📊 Relevance", report["relevance"]["score"], report["relevance"]["note"]),
        (m2, "🔬 Factual Accuracy", report["factual"]["score"], report["factual"]["note"]),
        (m3, "🛡️ Safety", report["safety"]["score"], report["safety"]["note"]),
        (m4, "📐 Structure", report["structure"]["score"], report["structure"]["note"]),
    ]

    for col, label, score, note in dimensions:
        with col:
            color = get_score_color(score)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{score}</div>
                <div class="metric-sub">/ 10</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.progress(score / 10)
            st.caption(note)

    st.markdown("---")

    # ── Detailed Breakdown ────────────────────────────────────────────────────
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        # Factual details
        st.markdown("#### 🔬 Factual Analysis")
        hallucination_risk = report["factual"].get("hallucination_risk", "unknown")
        risk_colors = {"low": "#16a34a", "medium": "#ca8a04", "high": "#dc2626"}
        risk_color = risk_colors.get(hallucination_risk, "#888")
        st.markdown(f"**Hallucination Risk:** <span style='color:{risk_color};font-weight:600;text-transform:uppercase;'>{hallucination_risk}</span>", unsafe_allow_html=True)
        ctx_used = report["factual"].get("context_used", False)
        st.caption(f"{'✅ Reference context used in evaluation.' if ctx_used else '⚪ No reference context provided.'}")

        st.markdown("#### 📐 Structure Analysis")
        struct = report["structure"]
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Word count", struct.get("word_count", 0))
        with s2:
            st.metric("Sentences", struct.get("sentence_count", 0))
        with s3:
            st.metric("Avg words/sent", struct.get("avg_sentence_length", 0))
        if struct.get("has_structure"):
            st.success("Response has structured formatting (lists/headers).")
        else:
            st.warning("Response lacks structured formatting.")

    with detail_col2:
        # Issues
        st.markdown("#### ⚠️ Issues Detected")
        issues = report.get("issues", [])
        if issues:
            for issue in issues:
                st.markdown(f"— {issue}")
        else:
            st.success("No issues detected.")

    st.markdown("---")

    # ── Suggestions ───────────────────────────────────────────────────────────
    st.markdown("#### 💡 Suggested Improvements")
    suggestions = report.get("suggestions", [])
    if suggestions:
        for sug in suggestions:
            st.info(f"💡 {sug}")
    else:
        st.success("No specific improvements suggested — response looks good.")

    st.markdown("---")

    # ── Full JSON Report / Export ─────────────────────────────────────────────
    with st.expander("📄 View Full JSON Report"):
        st.json(report)

    st.download_button(
        label="⬇️ Export Report as JSON",
        data=json.dumps(report, indent=2),
        file_name="llm_reliability_report.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── History (session state) ────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.insert(0, {
        "prompt": prompt_input[:60] + "..." if len(prompt_input) > 60 else prompt_input,
        "trust_score": trust,
        "verdict": verdict,
    })

# ── Evaluation History ────────────────────────────────────────────────────────
if st.session_state.get("history"):
    with st.expander(f"🕓 Evaluation History ({len(st.session_state.history)} runs)"):
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
