"""
Standalone ingest script: builds the knowledge base from PDFs in documents/
without opening the web app. Useful for testing and for the report.

Needs a Gemini API key: set GEMINI_API_KEY env var or put it in a .env file.

Usage:
    python ingest.py
    python ingest.py --folder my_pdfs --chunk-size 800

Output: knowledge_base.npz with chunks, metadata and embedding vectors.
"""

import argparse
import glob
import os

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai

load_dotenv()

EMBED_MODEL = "models/gemini-embedding-001"


def extract_text(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_pages(doc_name, pages, size):
    chunks, metas = [], []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append(piece)
                metas.append({"doc": doc_name, "page": page_num})
            start += size - 100
    return chunks, metas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="documents")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--out", default="knowledge_base.npz")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Set GEMINI_API_KEY first (free key: https://aistudio.google.com).")
        return
    genai.configure(api_key=api_key, transport="rest")

    pdfs = sorted(glob.glob(os.path.join(args.folder, "*.pdf")))
    if not pdfs:
        print("No PDFs found in {}/ — drop some files there and re-run.".format(args.folder))
        return

    all_chunks, all_metas = [], []
    for path in pdfs:
        name = os.path.basename(path)
        pages = extract_text(path)
        if not pages:
            print("  SKIP {}: no readable text (scanned PDF? needs OCR)".format(name))
            continue
        chunks, metas = chunk_pages(name, pages, args.chunk_size)
        all_chunks.extend(chunks)
        all_metas.extend(metas)
        print("  OK   {}: {} pages -> {} chunks".format(name, len(pages), len(chunks)))

    print("Embedding {} chunks...".format(len(all_chunks)))
    result = genai.embed_content(model=EMBED_MODEL, content=all_chunks)
    vectors = np.array(result["embedding"], dtype=np.float32)

    docs = np.array([m["doc"] for m in all_metas])
    pages = np.array([m["page"] for m in all_metas])
    np.savez(args.out, chunks=np.array(all_chunks), docs=docs,
             pages=pages, vectors=vectors)
    print("Done. Saved {} chunks to {}.".format(len(all_chunks), args.out))


if __name__ == "__main__":
    main()
