"""
Academic Document Assistant — a RAG chatbot that answers questions from uploaded PDFs.

How to run:
    pip install -r requirements.txt
    streamlit run app.py

You need a free Gemini API key from https://aistudio.google.com
Enter it in the sidebar (or set it as GEMINI_API_KEY env var / Streamlit secret).
"""

import os
import time

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai

load_dotenv()

# ---------------------------------------------------------------------------
# Settings you may want to change
# ---------------------------------------------------------------------------
EMBED_MODEL = "models/gemini-embedding-001"   # free embedding model from Google
GEMINI_MODEL = "gemini-3.5-flash"           # free chat model — swap for any current flash model
SIMILARITY_CUTOFF = 0.45                    # below this, the question is "not in the documents"


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------
def extract_text(pdf_file):
    """Read every page of one PDF, return a list of (page_number, text)."""
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_pages(doc_name, pages, size):
    """Split page text into overlapping pieces so meaning is not cut mid-sentence."""
    chunks, metas = [], []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append(piece)
                metas.append({"doc": doc_name, "page": page_num})
            start += size - 100  # 100-char overlap keeps context at the edges
    return chunks, metas


def embed_texts(texts):
    """Convert texts to embedding vectors using Google's free embedding API."""
    result = genai.embed_content(model=EMBED_MODEL, content=texts)
    return np.array(result["embedding"], dtype=float)


def cosine_top_k(query_vec, matrix, k):
    """Return indices and scores of the k most similar rows (cosine similarity)."""
    q = np.array(query_vec, dtype=float)
    sims = (matrix @ q) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-10)
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


def answer_question(question, api_key, k):
    """Find the most relevant chunks, then ask Gemini to answer only from them."""
    chunks = st.session_state.chunks
    metas = st.session_state.metas
    matrix = st.session_state.embeddings

    if not chunks:
        return "I don't know — no documents have been processed yet. Upload a PDF and click **Process documents** first.", []

    genai.configure(api_key=api_key, transport="rest")
    q_vec = embed_texts([question])[0]
    hits = cosine_top_k(q_vec, matrix, k)

    if not hits or hits[0][1] < SIMILARITY_CUTOFF:
        # Nothing in the documents is close enough to the question.
        return "I don't know — that answer is not in the uploaded documents.", []

    context = "\n\n".join(
        "[Document: {}, Page: {}]\n{}".format(metas[i]["doc"], metas[i]["page"], chunks[i])
        for i, _ in hits
    )
    prompt = (
        "Answer the question using ONLY the context below. Do not use any outside knowledge.\n"
        "If the answer is not in the context, reply exactly: I don't know.\n"
        "After each fact, cite where it came from like this: [Document: name, Page: N].\n\n"
        "Context:\n" + context + "\n\nQuestion: " + question
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    sources = [(chunks[i], metas[i], score) for i, score in hits]
    return response.text, sources


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Academic Document Assistant", page_icon="📚")
st.title("📚 Academic Document Assistant")
st.caption("Ask questions about your uploaded PDFs — answers come with the document and page number.")

# ---------------------------------------------------------------------------
# Sidebar: API key + document upload + settings
# ---------------------------------------------------------------------------
default_key = os.getenv("GEMINI_API_KEY", "")
try:
    if not default_key:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    pass

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Gemini API key",
        value=default_key,
        type="password",
        help="Free key from https://aistudio.google.com — used for embeddings and answers.",
    )

    st.divider()
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    process_clicked = st.button("Process documents", type="primary", disabled=not uploaded)

    st.divider()
    st.header("Settings")
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, 50)
    top_k = st.slider("Chunks to read per answer", 2, 8, 4)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key_name, default in [("chunks", []), ("metas", []),
                          ("embeddings", None), ("messages", [])]:
    if key_name not in st.session_state:
        st.session_state[key_name] = default

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
if process_clicked and uploaded:
    if not api_key:
        st.error("Please enter your free Gemini API key first (https://aistudio.google.com).")
        st.stop()
    with st.spinner("Reading PDFs, chunking and embedding..."):
        genai.configure(api_key=api_key, transport="rest")
        all_chunks, all_metas = [], []
        for f in uploaded:
            pages = extract_text(f)
            if not pages:
                st.warning("Could not read any text from **{}** — it may be a scanned PDF (needs OCR).".format(f.name))
                continue
            chunks, metas = chunk_pages(f.name, pages, chunk_size)
            all_chunks.extend(chunks)
            all_metas.extend(metas)
        if not all_chunks:
            st.error("No readable text found in the uploaded PDFs.")
            st.stop()
        st.session_state.embeddings = embed_texts(all_chunks)
        st.session_state.chunks = all_chunks
        st.session_state.metas = all_metas
        st.session_state.messages = []
    st.success("Done! {} text chunks saved from {} file(s). Ask your questions below.".format(
        len(all_chunks), len(uploaded)))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for d, m, s in msg.get("sources", []):
            with st.expander("Source: {} — page {}".format(m["doc"], m["page"])):
                st.caption(d[:800] + ("..." if len(d) > 800 else ""))

if prompt := st.chat_input("Ask a question about your documents..."):
    if not api_key:
        st.error("Please enter your free Gemini API key in the sidebar first (https://aistudio.google.com).")
        st.stop()
    if not st.session_state.chunks:
        st.error("Upload a PDF and click **Process documents** first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            t0 = time.time()
            answer, sources = answer_question(prompt, api_key, top_k)
            elapsed = time.time() - t0
        st.markdown(answer)
        for d, m, s in sources:
            with st.expander("Source: {} — page {}".format(m["doc"], m["page"])):
                st.caption(d[:800] + ("..." if len(d) > 800 else ""))
        st.caption("Answered in {:.1f}s from {} chunk(s).".format(elapsed, len(sources)))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
