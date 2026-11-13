# 🎓 Academic Document Assistant (Hybrid RAG)

> **Next-Generation Academic RAG Chatbot:** Ground conversational AI strictly in research papers, lecture notes, and textbooks with **Hybrid Search (Dense + BM25)**, **Recursive Semantic Chunking**, **Contextual Conversational Memory**, and **Exact Page Citations**.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Gemini API](https://img.shields.io/badge/AI-Google%20Gemini%20(Free%20Tier)-green.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Subscription](https://img.shields.io/badge/Cost-%240.00%20(100%25%20Free)-success.svg)](#)

---

## 🌟 Why This Stands Out (Beyond Ordinary College RAG Bots)

Most student RAG projects use naive character splitting and basic single-prompt retrieval. This project implements state-of-the-art information retrieval techniques with zero paid subscriptions:

1. **Hybrid Retrieval (Dense + Sparse with RRF):**
   * Combines **Dense Semantic Embeddings** (`gemini-text-embedding-004`) with **Sparse Keyword Matching** (`BM25 Okapi`).
   * Fuses candidate rankings using **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF}(d) = \frac{0.6}{60 + \text{rank}_{\text{dense}}(d)} + \frac{0.4}{60 + \text{rank}_{\text{sparse}}(d)}$$
   * Accurately catches both abstract concepts and exact technical tokens, formulas, and acronyms.

2. **Layout-Aware Academic PDF Parsing:**
   * Utilizes layout-preserving extraction to parse **two-column academic papers** (IEEE, ACM, arXiv) sequentially without intermingling column text.

3. **Recursive Boundary Chunking:**
   * Respects natural grammatical boundaries (`\n\n` $\rightarrow$ `\n` $\rightarrow$ `.` $\rightarrow$ `;` $\rightarrow$ ` `).
   * Never slices formulas, words, or sentences mid-character.

4. **Conversational Query Contextualization:**
   * Automatically resolves pronouns and follow-up queries (e.g., *"What are its main limitations?"* $\rightarrow$ *"What are the limitations of the Scaled Cosine Attention mechanism?"*).

5. **Built-in RAG Triad Benchmarking (Viva Ready):**
   * Features an interactive evaluation panel measuring **Faithfulness**, **Context Relevance**, and **Answer Relevance** to quantitatively prove zero hallucination.

6. **Academic Utility Suite:**
   * **LaTeX Math Rendering:** Displays formulas cleanly ($E = mc^2$ and block equations).
   * **Executive Research Briefs:** 1-click synthesis of Problem, Methodology, Results, and Limitations.
   * **BibTeX Exporter:** Instant `@article` reference formatting.
   * **Persistent Knowledge Base:** Export and import pre-computed embeddings as `.npz` files.

---

## 📁 Repository Structure

```
academic-doc-assistant/
├── app.py                  # Streamlit Web Application (UI, Tabs, Visualizer)
├── rag_engine.py           # Core RAG Algorithms (Chunking, Hybrid Search, RRF, LLM)
├── ingest.py               # Standalone Headless CLI Ingestion Script
├── requirements.txt        # Lightweight, zero-cost dependencies
├── .env.example            # Environment template for Gemini API key
├── HOW_IT_WORKS.md         # Comprehensive viva defense & technical cheat-sheet
├── documents/              # Directory for PDFs to ingest via CLI
└── sample_docs/            # Pre-packaged sample paper for instant live testing
    └── attention_mechanism_study.pdf
```

---

## 🚀 Quickstart (Local Run in 3 Minutes)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/academic-doc-assistant.git
cd academic-doc-assistant
pip install -r requirements.txt
```

### 2. Configure Free Gemini API Key
Get a 100% free Gemini API key from [Google AI Studio](https://aistudio.google.com) (no credit card required).

Create a `.env` file from the example:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
GEMINI_API_KEY=AIzaSy...your_key_here
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Click **"Load Sample Paper"** in the sidebar to try it immediately!

---

## 🌐 How to Deploy 100% Free on Streamlit Cloud

You **do not need any paid hosting, VPS, or cloud subscription**. Streamlit Community Cloud hosts this app with a free public link and automated SSL:

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "feat: complete academic doc assistant with hybrid rag"
   git push origin main
   ```
   *(Ensure `.env` is ignored by `.gitignore` so your API key is never pushed!)*

2. **Sign in to Streamlit Community Cloud:**
   * Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.

3. **Deploy New App:**
   * Click **"Create app"** $\rightarrow$ **"Deploy a public app from GitHub"**.
   * Select your repository: `your-username/academic-doc-assistant`.
   * Branch: `main`.
   * Main file path: `app.py`.

4. **Set Up the Secret API Key:**
   * Under **Advanced settings** $\rightarrow$ **Secrets**, paste:
     ```toml
     GEMINI_API_KEY = "AIzaSy...your_actual_key"
     ```
   * Click **Save** and **Deploy**.

5. **Share Your App:**
   * Your app will be live at `https://<your-custom-name>.streamlit.app` in under 2 minutes. Share this link on your resume, LinkedIn, and during your project viva!

---

## 🖥️ CLI Batch Ingestion (Headless)

To pre-index a large collection of textbooks or papers without opening the browser:
```bash
# Place PDFs into documents/ folder
python ingest.py --folder documents --chunk-size 800 --out knowledge_base.npz
```
You can then upload `knowledge_base.npz` directly into the web UI using the sidebar.

---

## 🧪 Testing the Anti-Hallucination Guardrail

Professors test RAG systems by asking questions that are intentionally **not** present in the document.

1. Load `attention_mechanism_study.pdf`.
2. Ask: *"What is the capital of Australia?"* or *"Who won the 1994 FIFA World Cup?"*
3. **Observed Output:**
   > *"I don't know — that information is not in the uploaded documents."*
4. **Why it happens:** The system evaluates cosine similarity and BM25 alignment. If the top candidate score falls below `SIMILARITY_CUTOFF` (0.40), the query is blocked before reaching the generator, eliminating hallucinations.

---

## 📖 Viva Preparation

Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for in-depth mathematical formulations of Reciprocal Rank Fusion, cosine distance vs. Euclidean metrics, the RAG Triad, and model viva answers for external examiners.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
