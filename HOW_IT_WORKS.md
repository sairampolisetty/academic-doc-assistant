# How the code works — simple explanation (read before the viva)

Written in plain words so you can explain it confidently when a professor asks.

## The big idea in one line

Instead of the AI guessing answers from its training, we **first find the right
pages** in your PDFs, then make the AI answer **only from those pages**.

This is called **RAG** = Retrieval Augmented Generation.
- **Retrieval** = finding the right pages from your documents
- **Augmented Generation** = the AI writes the answer using those pages, not its memory

## The pipeline, step by step (matches `app.py` exactly)

1. **Upload** — you pick PDF files in the app (`st.file_uploader`).
2. **Read text** — `extract_text()` uses `pypdf` to pull text out of every page,
   and remembers which page each piece of text came from.
3. **Chunking** — `chunk_pages()` splits long text into pieces of ~800 characters
   with a small overlap. Why? The AI reads more precisely when it gets small,
   focused pieces instead of whole chapters.
4. **Embeddings** — each chunk is converted into a list of numbers (a **vector**)
   by Google's free `gemini-embedding-001` model. Texts with similar *meaning*
   get similar numbers. Because it is an API, there is no heavy model to
   download — the app stays light.
5. **Store** — chunks + their vectors are kept in memory for the session, so
   search is instant. (The standalone `ingest.py` can also save them to a
   `knowledge_base.npz` file on disk.)
6. **Search** — when you ask a question, the question is converted into numbers
   the same way, and **cosine similarity** picks the 4 chunks with the closest
   meaning. Cosine similarity is just a score from -1 to 1 for "how alike are
   these two meanings".
7. **Answer** — those chunks + your question are sent to **Gemini** with a strict
   instruction: "Answer ONLY from this text. If the answer is not here, say
   I don't know." The answer comes back with citations like
   `[Document: notes.pdf, Page: 12]`.
8. **Honesty guard** — if even the best-matching chunk is too low in similarity
   (score below `SIMILARITY_CUTOFF`), the app answers "I don't know" without
   even asking Gemini. This stops hallucination.

## Likely viva questions and short answers

**What is an embedding?**
A list of numbers that represents the *meaning* of text. Similar meanings →
similar numbers, so we can search by meaning instead of by exact words.

**Why chunk the text instead of sending the whole PDF?**
Whole PDFs are too long, cost more, and the model misses details. Small chunks
give precise search results — only the relevant pages reach the model.

**Why not just paste the PDF into Gemini directly?**
Three reasons: the model has a limit on how much text it can read at once,
long documents cost more, and retrieval lets the answer cite exact pages.

**What is hallucination, and how does your project reduce it?**
Hallucination is when the AI invents answers that sound right but are false.
We reduce it two ways: Gemini is instructed to answer only from the retrieved
chunks, and the distance cutoff makes the app say "I don't know" when nothing
in the documents is close enough.

**Why API embeddings instead of a local model?**
Local models need heavy downloads (torch is hundreds of MB) and a strong
machine. The free Google embedding API needs no download, works on any laptop
and on free Streamlit Cloud hosting — simpler to demo and easier to explain.

**Where does the API key go, and why not in the code?**
In the `.env` file (or the Streamlit Cloud secrets). A key in the code would be
visible to anyone who sees the repo — and `.gitignore` keeps `.env` out of
GitHub.

**What if the PDF is a scanned image (photo of pages)?**
`pypdf` finds no text, and the app warns you. Those PDFs need **OCR**
(e.g. `pytesseract`) to turn images into text first — that is a known limit,
listed in the README.

## Demo flow (under 5 minutes)

1. Upload 2 PDFs live, click **Process documents**
2. Ask one easy question (answer in one paragraph) → show the page citation
3. Ask one hard question (needs two pages) → show it combined both
4. Ask something NOT in the documents → show the **"I don't know"** reply
