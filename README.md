# Academic Document Assistant

A RAG (Retrieval Augmented Generation) chatbot that answers questions from
academic PDFs you upload — textbooks, research papers, lecture notes.
Every answer cites the **document and page number**, and it says
**"I don't know"** when the answer is not in your documents (no made-up answers).

## What is inside

| File | What it does |
|---|---|
| `app.py` | The Streamlit web app: upload PDFs, ask questions, chat history, source citations |
| `ingest.py` | Standalone script that builds the knowledge base from PDFs in `documents/` |
| `requirements.txt` | All Python packages needed (lightweight, no heavy ML downloads) |
| `.env.example` | Template for your Gemini API key |
| `HOW_IT_WORKS.md` | Simple explanation of the code — read this before the viva |
| `documents/` | Put PDFs here for the standalone ingest script |
| `sample_docs/` | A small sample PDF to try the app instantly |

## Run it (5 minutes)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then:
1. Get a **free** Gemini API key at https://aistudio.google.com (2 minutes, no card)
2. Paste the key in the sidebar of the app
3. Upload 1–2 PDFs, click **Process documents**
4. Ask questions in the chat box

No heavy downloads: embeddings come from Google's free embedding API,
so the app starts instantly on any machine.

## How it works (one paragraph)

Each PDF is read page by page and split into small text chunks. Every chunk is
converted into an **embedding** (a list of numbers that captures its meaning)
using Google's free `gemini-embedding-001` model, and the vectors are kept in
memory. When you ask a question, the question is embedded the same way, the app
finds the chunks with the most similar meaning using **cosine similarity**,
and **Gemini** writes the final answer using only those chunks —
with a citation `[Document: name, Page: N]` under each fact.

## Deploy free on Streamlit Cloud

1. Push this folder to GitHub
2. Go to https://share.streamlit.io → **New app** → pick your repo → `app.py`
3. In the app settings, add a secret: `GEMINI_API_KEY = your_key_here`
   (or paste the key in the sidebar each time — never commit the real key)

## Honesty test (great for the demo)

Ask something that is definitely NOT in your PDFs. The app must reply
"I don't know" — professors love this because it proves the bot does not
hallucinate answers.

## Semester plan this was built for

- Weeks 1–2: learn RAG basics, pick the stack
- Weeks 3–5: PDF upload, chunking, vector search ✅ (done here)
- Weeks 6–8: AI model + chat UI with citations ✅ (done here)
- Weeks 9–10: test with 20 real questions, measure accuracy
- Weeks 11–12: polish, report, demo
