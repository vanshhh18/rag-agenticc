
import streamlit as st
import os
import time
import traceback

from langchain_groq import ChatGroq
from retrieval import hybrid_search


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="RAG vs LLM — Agentic Insight",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D0F14;
    color: #E2E4EA;
}
.stApp { background-color: #0D0F14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }
.hero { display: flex; flex-direction: column; align-items: flex-start; padding: 2.5rem 0 1.5rem; border-bottom: 1px solid #1E2130; margin-bottom: 2rem; }
.hero-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.18em; color: #5B63F5; text-transform: uppercase; margin-bottom: 0.6rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; line-height: 1.1; color: #F0F2FF; margin: 0; }
.hero-title span { background: linear-gradient(90deg, #5B63F5, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { margin-top: 0.8rem; font-size: 0.95rem; color: #6B7280; font-weight: 300; max-width: 560px; line-height: 1.6; }
div[data-testid="stTextInput"] > div > div > input { background: #13151E !important; border: 1px solid #252838 !important; border-radius: 10px !important; color: #E2E4EA !important; font-size: 1rem !important; padding: 0.75rem 1rem !important; }
div[data-testid="stTextInput"] > div > div > input:focus { border-color: #5B63F5 !important; box-shadow: 0 0 0 3px rgba(91,99,245,0.15) !important; }
div[data-testid="stButton"] > button { background: linear-gradient(135deg, #5B63F5, #7C3AED) !important; color: #fff !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 2rem !important; font-weight: 500 !important; font-size: 0.95rem !important; }
div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.panel { background: #13151E; border: 1px solid #1E2130; border-radius: 14px; padding: 1.5rem 1.75rem; min-height: 220px; }
.panel-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.1rem; }
.panel-badge { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; padding: 0.2rem 0.55rem; border-radius: 4px; font-weight: 500; }
.badge-rag { background: rgba(91,99,245,0.15); color: #818CF8; border: 1px solid rgba(91,99,245,0.3); }
.badge-llm { background: rgba(167,139,250,0.12); color: #A78BFA; border: 1px solid rgba(167,139,250,0.3); }
.panel-title { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: #C7CAE0; }
.panel-body { font-size: 0.93rem; line-height: 1.75; color: #9CA3AF; }
.ctx-header { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: #4B5563; margin: 1.4rem 0 0.7rem; }
.ctx-chip { background: #0D0F14; border: 1px solid #1E2130; border-left: 3px solid #5B63F5; border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin-bottom: 0.6rem; font-size: 0.84rem; color: #6B7280; line-height: 1.6; }
.fancy-divider { display: flex; align-items: center; gap: 1rem; margin: 2.5rem 0; }
.fancy-divider-line { flex: 1; height: 1px; background: #1E2130; }
.fancy-divider-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.18em; color: #374151; text-transform: uppercase; }
.insight-card { background: linear-gradient(135deg, #0F1120 0%, #131626 100%); border: 1px solid #252838; border-radius: 14px; padding: 1.75rem 2rem; }
.insight-card-header { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1rem; }
.insight-icon { font-size: 1.3rem; }
.insight-title { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: #C7CAE0; }
.insight-body { font-size: 0.93rem; line-height: 1.8; color: #9CA3AF; }
.metric-row { display: flex; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap; }
.metric-chip { background: #0D0F14; border: 1px solid #1E2130; border-radius: 8px; padding: 0.45rem 0.9rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #6B7280; }
.metric-chip strong { color: #818CF8; }
</style>
""", unsafe_allow_html=True)


# =========================
# HERO HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Agentic AI System</div>
    <h1 class="hero-title">RAG vs <span>Generic LLM</span></h1>
    <p class="hero-sub">
        Ask a question. See how grounded retrieval compares to open-ended generation — then let the agent judge the difference.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# INPUT ROW
# =========================
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="query",
        placeholder="e.g. What are the key findings in the Q3 report?",
        label_visibility="collapsed",
    )
with col_btn:
    run = st.button("⚡ Run", use_container_width=True)


# =========================
# LLM SETUP
# =========================
@st.cache_resource
def get_llms():
    rag_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    generic_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
    )
    return rag_llm, generic_llm


# =========================
# RUN PIPELINE
# =========================
if run and query:
    try:
        rag_llm, generic_llm = get_llms()

        # --- RAG ---
        with st.spinner("Retrieving from your data…"):
            t0 = time.time()
            docs = hybrid_search(query)
            context = "\n".join(d.page_content[:200] for d in docs)
            rag_prompt = f"""You are a precise assistant. Use ONLY this context:
{context}

Question: {query}"""
            rag_response = rag_llm.invoke(rag_prompt).content
            rag_time = round(time.time() - t0, 2)

        # --- Generic ---
        with st.spinner("Querying generic LLM…"):
            t1 = time.time()
            generic_prompt = f"Answer the question:\n{query}"
            generic_response = generic_llm.invoke(generic_prompt).content
            generic_time = round(time.time() - t1, 2)

        # --- Panels ---
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-badge badge-rag">RAG · Your Data</span>
                    <span class="panel-title">🔍 Retrieval-Augmented</span>
                </div>
                <div class="panel-body">{rag_response}</div>
                <div class="metric-row">
                    <div class="metric-chip">⏱ <strong>{rag_time}s</strong></div>
                    <div class="metric-chip">📄 <strong>{len(docs)}</strong> chunks retrieved</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if docs:
                st.markdown('<div class="ctx-header">Retrieved context</div>', unsafe_allow_html=True)
                for d in docs:
                    st.markdown(f'<div class="ctx-chip">{d.page_content[:300]}</div>', unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-badge badge-llm">Generic · No context</span>
                    <span class="panel-title">🤖 Open-ended LLM</span>
                </div>
                <div class="panel-body">{generic_response}</div>
                <div class="metric-row">
                    <div class="metric-chip">⏱ <strong>{generic_time}s</strong></div>
                    <div class="metric-chip">🌐 Trained knowledge only</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Agent Insight ---
        st.markdown("""
        <div class="fancy-divider">
            <div class="fancy-divider-line"></div>
            <div class="fancy-divider-label">Agent Verdict</div>
            <div class="fancy-divider-line"></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Agent comparing both answers…"):
            judge_prompt = f"""Compare these two answers and decide which is more accurate and why. Be concise.

Question: {query}

RAG Answer: {rag_response}

Generic Answer: {generic_response}"""
            judge = rag_llm.invoke(judge_prompt).content

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-card-header">
                <span class="insight-icon">🧠</span>
                <span class="insight-title">Agent Insight</span>
            </div>
            <div class="insight-body">{judge}</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"**Error:** {e}")
        st.code(traceback.format_exc())

