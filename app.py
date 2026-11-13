"""
Academic Document Assistant.
A clean, minimal, ChatGPT-like interface for research papers and documents.
"""

import importlib
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Ensure rag_engine is reloaded fresh on any changes
import rag_engine
importlib.reload(rag_engine)

from rag_engine import (
    IDONTKNOW,
    SIMILARITY_CUTOFF,
    build_bm25_index,
    chunk_academic_pages,
    embed_single_query,
    embed_texts_batched,
    evaluate_rag_triad,
    extract_text_from_pdf,
    generate_grounded_answer,
    generate_grounded_answer_stream,
    hybrid_search,
)

load_dotenv()


# ---------------------------------------------------------------------------
# API Key Helper
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key.strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


def save_api_key_to_env(key_val: str):
    """Saves the key to .env file for permanent local use."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(f"GEMINI_API_KEY={key_val.strip()}\n")
    load_dotenv(override=True)


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "chunks": [],
        "metas": [],
        "embeddings": None,
        "bm25_index": None,
        "messages": [],
        "active_docs": [],
        "pending_prompt": None,
        "is_generating": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ---------------------------------------------------------------------------
# Page Configuration & Clean Dark/Light Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Academic Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        max-width: 48rem;
    }
    h1 {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .text-muted {
        color: #888888;
        font-size: 0.9rem;
        margin-bottom: 1.25rem;
    }
    .stButton button {
        border-radius: 6px;
    }
    #chat-bottom-anchor {
        height: 1px;
        width: 100%;
        visibility: hidden;
    }

    /* 1. Hide Main Menu (three dots) */
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }

    /* 2. Hide Top Header clutter (Fork, Deploy, GitHub icon) while preserving sidebar toggle (>>) */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    header [data-testid="stToolbar"],
    header [data-testid="stHeaderActionElements"],
    header a,
    .stDeployButton,
    header a[href*="github.com"],
    [data-testid="stDecoration"] {
        visibility: hidden !important;
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        pointer-events: none !important;
    }
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
    }

    /* 3. Hide Footer ("Made with Streamlit") */
    footer, [data-testid="stFooter"] {
        visibility: hidden !important;
        display: none !important;
    }

    /* 4. Hide Streamlit Cloud viewer toolbar (avatar, red crown button, manage app) */
    [class*="viewerBadge"],
    [class*="manageApp"],
    [data-testid="manage-app-button"],
    #manage-app-button,
    div[class*="ProfileButton"],
    div[class*="floatingButton"],
    div[class*="ViewerBadge"],
    div[class*="HostBadge"],
    div[class*="ConnectionStatus"] {
        visibility: hidden !important;
        display: none !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = get_api_key()


# ---------------------------------------------------------------------------
# Ingestion Helper
# ---------------------------------------------------------------------------
def index_documents(files, key: str):
    all_chunks = []
    all_metas = []
    doc_names = []

    with st.spinner("Processing documents..."):
        for i, f in enumerate(files):
            name = getattr(f, "name", f"document_{i+1}.pdf")
            doc_names.append(name)
            try:
                pages = extract_text_from_pdf(f)
            except Exception as e:
                st.error(f"Failed to read {name}: {e}")
                continue

            if not pages:
                st.warning(f"No readable text in {name}. It might be a scanned image.")
                continue

            chunks, metas = chunk_academic_pages(name, pages, chunk_size=800, overlap=120)
            all_chunks.extend(chunks)
            all_metas.extend(metas)

        if not all_chunks:
            st.error("No extractable text found in uploaded files.")
            return False

        try:
            embeddings = embed_texts_batched(all_chunks, api_key=key, batch_size=50)
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            return False

        bm25_idx = build_bm25_index(all_chunks)

        st.session_state.chunks = all_chunks
        st.session_state.metas = all_metas
        st.session_state.embeddings = embeddings
        st.session_state.bm25_index = bm25_idx
        st.session_state.active_docs = doc_names
        st.session_state.messages = []
        return True


# ---------------------------------------------------------------------------
# Sidebar (ChatGPT-style navigation & documents)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Academic Assistant")
    st.caption("Document-grounded research tool")

    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.divider()

    st.markdown("**Documents**")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        process_btn = st.button(
            "Process",
            type="primary",
            use_container_width=True,
            disabled=not uploaded,
        )
    with col2:
        load_sample = st.button("Load sample", use_container_width=True)

    # Active documents status
    if st.session_state.active_docs:
        st.caption(
            f"Active: {len(st.session_state.active_docs)} file(s), "
            f"{len(st.session_state.chunks)} passages indexed."
        )
        for d in st.session_state.active_docs:
            st.text(f"• {d}")

    # Settings
    top_k: int = 4
    dense_ratio: float = 0.6
    with st.expander("Settings"):
        top_k = st.slider("Passages retrieved", 2, 8, 4)
        dense_ratio = st.slider("Dense / Sparse balance", 0.0, 1.0, 0.6, 0.1)

    # Viva Benchmark Tool
    with st.expander("Benchmark Evaluation"):
        st.caption("Quantitative RAG Triad audit (Faithfulness & Relevance)")
        bench_query = st.text_input(
            "Audit query",
            value="What is the mathematical formulation of attention?",
        )
        if st.button("Run Audit", use_container_width=True):
            if not api_key:
                st.error("API key missing in .env.")
            elif not st.session_state.chunks:
                st.error("Upload a document first.")
            else:
                with st.spinner("Auditing..."):
                    q_vec = embed_single_query(bench_query, api_key)
                    hits = hybrid_search(
                        bench_query,
                        q_vec,
                        st.session_state.embeddings,
                        st.session_state.chunks,
                        st.session_state.bm25_index,
                        k=top_k,
                    )
                    top_cos = hits[0][2] if hits else 0.0
                    if not hits or top_cos < SIMILARITY_CUTOFF:
                        ans = IDONTKNOW
                        ctx_list = []
                    else:
                        retrieved = [
                            (st.session_state.chunks[i], st.session_state.metas[i], c)
                            for i, _, c, _ in hits
                        ]
                        ans = generate_grounded_answer(bench_query, retrieved, api_key)
                        ctx_list = [st.session_state.chunks[i] for i, _, _, _ in hits]
                    metrics = evaluate_rag_triad(bench_query, ans, ctx_list, api_key)
                    st.write(f"**Faithfulness:** {metrics.get('faithfulness', 0)}%")
                    st.write(f"**Context Relevance:** {metrics.get('context_relevance', 0)}%")
                    st.write(f"**Answer Relevance:** {metrics.get('answer_relevance', 0)}%")
                    st.write(f"**Verdict:** {metrics.get('verdict', 'N/A')}")

# Handle file uploads
if process_btn and uploaded:
    if not api_key:
        st.error("API key is not configured. Set GEMINI_API_KEY in .env.")
    else:
        if index_documents(uploaded, api_key):
            st.success("Indexed successfully.")
            st.rerun()

# Handle sample document loading
if load_sample:
    if not api_key:
        st.error("API key is not configured. Set GEMINI_API_KEY in .env.")
    else:
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs", "attention_mechanism_study.pdf")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                import io
                buf = io.BytesIO(f.read())
                buf.name = "attention_mechanism_study.pdf"
                if index_documents([buf], api_key):
                    st.success("Sample paper loaded.")
                    st.rerun()
        else:
            st.error("Sample document not found in sample_docs/.")


# ---------------------------------------------------------------------------
# Main Chat Interface (Clean ChatGPT Style)
# ---------------------------------------------------------------------------
if not api_key:
    st.markdown("<h1>Academic Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='text-muted'>To enable document analysis and question answering, "
        "connect your free Gemini API key once below. It will be saved locally to your <code>.env</code> file.</p>",
        unsafe_allow_html=True,
    )
    with st.form("api_key_form"):
        key_input = st.text_input("Gemini API Key", type="password", placeholder="Paste your API key here (AIzaSy...)")
        submitted = st.form_submit_button("Save and Start", type="primary")
        if submitted and key_input.strip():
            save_api_key_to_env(key_input)
            st.rerun()

    st.markdown(
        "<span style='font-size: 0.85rem; color: #888;'>"
        "Need a key? Get one 100% free at <a href='https://aistudio.google.com' target='_blank'>aistudio.google.com</a> (takes 1 minute, no credit card required)."
        "</span>",
        unsafe_allow_html=True,
    )
    st.stop()

# Header with Clear Chat option
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown("<h1>Academic Assistant</h1>", unsafe_allow_html=True)
with col_h2:
    if st.session_state.messages:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

# Empty State: Clean welcome screen
if not st.session_state.messages:
    st.markdown(
        "<p class='text-muted'>Ask questions grounded in your research papers, lecture notes, or textbooks.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.chunks:
        st.info("To begin, upload a PDF in the sidebar or click 'Load sample'.")
    else:
        st.markdown("**Suggested prompts:**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("What is the mathematical formulation of attention?", use_container_width=True):
                st.session_state.pending_prompt = "What is the mathematical formulation of attention?"
                st.rerun()
            if st.button("What datasets and benchmark results were reported?", use_container_width=True):
                st.session_state.pending_prompt = "What datasets and benchmark results were reported?"
                st.rerun()
        with col_b:
            if st.button("What are the main limitations discussed?", use_container_width=True):
                st.session_state.pending_prompt = "What are the main limitations discussed?"
                st.rerun()
            if st.button("Who won the 1994 FIFA World Cup? (Honesty test)", use_container_width=True):
                st.session_state.pending_prompt = "Who won the 1994 FIFA World Cup?"
                st.rerun()

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for text, meta, rrf_score, cos_score in msg["sources"]:
                    st.markdown(
                        f"**{meta['doc']}** (Page {meta['page']}) — *Relevance: {cos_score:.2f}*"
                    )
                    st.caption(text[:400] + ("..." if len(text) > 400 else ""))
                    st.divider()

# Input Bar: Captures input from either clicked suggestion button or chat input
is_busy = st.session_state.get("is_generating", False)
user_input = None
if st.session_state.get("pending_prompt") and not is_busy:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif typed_query := st.chat_input(
    "Assistant is responding..." if is_busy else "Ask a question about your documents...",
    disabled=is_busy,
):
    user_input = typed_query

if user_input and not is_busy:
    st.session_state.is_generating = True

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        t0 = time.time()
        sources = []
        retrieved_triples = []

        # Run retrieval if documents are loaded
        if st.session_state.chunks:
            with st.spinner("Searching documents..."):
                # Fast embedding of the query
                q_vec = embed_single_query(user_input, api_key)

                # Hybrid search (Dense + BM25 via RRF)
                hits = hybrid_search(
                    query=user_input,
                    query_vec=q_vec,
                    matrix=st.session_state.embeddings,
                    chunks=st.session_state.chunks,
                    bm25_index=st.session_state.bm25_index,
                    k=top_k,
                    dense_weight=dense_ratio,
                    sparse_weight=round(1.0 - dense_ratio, 2),
                )

            top_cosine = hits[0][2] if hits else 0.0
            if hits and top_cosine >= SIMILARITY_CUTOFF:
                retrieved_triples = [
                    (st.session_state.chunks[idx], st.session_state.metas[idx], cos_score)
                    for idx, rrf_score, cos_score, bm25_score in hits
                ]
                sources = [
                    (st.session_state.chunks[idx], st.session_state.metas[idx], rrf_score, cos_score)
                    for idx, rrf_score, cos_score, bm25_score in hits
                ]

        # Unified ChatGPT-style streaming for ALL questions
        try:
            stream = generate_grounded_answer_stream(
                query=user_input,
                retrieved_chunks=retrieved_triples,
                api_key=api_key,
                chat_history=st.session_state.messages,
                active_docs=st.session_state.active_docs,
            )
            answer = st.write_stream(stream)
        except Exception:
            answer = generate_grounded_answer(
                query=user_input,
                retrieved_chunks=retrieved_triples,
                api_key=api_key,
                chat_history=st.session_state.messages,
                active_docs=st.session_state.active_docs,
            )
            st.markdown(answer)

        elapsed = time.time() - t0

        if not answer or not isinstance(answer, str):
            answer = IDONTKNOW

        is_valid_answer = not answer.startswith(IDONTKNOW)
        if sources and is_valid_answer:
            with st.expander(f"Sources ({len(sources)})"):
                for text, meta, rrf_score, cos_score in sources:
                    st.markdown(
                        f"**{meta['doc']}** (Page {meta['page']}) — *Relevance: {cos_score:.2f}*"
                    )
                    st.caption(text[:400] + ("..." if len(text) > 400 else ""))
                    st.divider()

        st.caption(f"Answered in {elapsed:.1f}s.")

        # Save assistant message to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources if is_valid_answer else [],
        })

    st.session_state.is_generating = False
    st.rerun()

# Auto-scroll directly inside the page and strip Streamlit Cloud branding badges
st.html(
    """
    <div id="chat-bottom-anchor"></div>
    <script>
    (function() {
        function scrollDown() {
            var el = document.getElementById('chat-bottom-anchor');
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'end' });
                return;
            }
            var container = document.querySelector('[data-testid="stAppViewContainer"]') ||
                            document.querySelector('[data-testid="stMain"]') ||
                            document.querySelector('section.main') ||
                            document.documentElement;
            if (container) {
                container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
            }
        }
        scrollDown();
        setTimeout(scrollDown, 50);
        setTimeout(scrollDown, 150);
        setTimeout(scrollDown, 300);
        setTimeout(scrollDown, 600);

        function removeCloudBadges() {
            var selectors = [
                '#MainMenu',
                'footer',
                '[data-testid="stFooter"]',
                '[data-testid="stDecoration"]',
                '[data-testid="stToolbar"]',
                '[data-testid="stHeaderActionElements"]',
                '.stDeployButton',
                'header a',
                '[data-testid="manage-app-button"]',
                '#manage-app-button',
                '[class*="viewerBadge"]',
                '[class*="ViewerBadge"]',
                '[class*="manageApp"]',
                '[class*="ProfileButton"]',
                '[class*="floatingButton"]',
                '[class*="HostBadge"]'
            ];
            var docs = [document];
            try {
                if (window.parent && window.parent.document && window.parent.document !== document) {
                    docs.push(window.parent.document);
                }
            } catch(e) {}
            try {
                if (window.top && window.top.document && !docs.includes(window.top.document)) {
                    docs.push(window.top.document);
                }
            } catch(e) {}

            docs.forEach(function(doc) {
                selectors.forEach(function(sel) {
                    try {
                        var elements = doc.querySelectorAll(sel);
                        elements.forEach(function(el) {
                            el.style.setProperty('display', 'none', 'important');
                            el.style.setProperty('visibility', 'hidden', 'important');
                            el.style.setProperty('opacity', '0', 'important');
                            el.style.setProperty('pointer-events', 'none', 'important');
                        });
                    } catch(e) {}
                });
            });
        }
        removeCloudBadges();
        setInterval(removeCloudBadges, 800);
    })();
    </script>
    """
)
