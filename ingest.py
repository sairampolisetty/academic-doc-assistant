"""
ingest.py — Standalone Batch Ingestion Script for Academic Document Assistant.

Builds a persistent knowledge base (.npz) from PDF files in documents/
without opening the web application.

Usage:
    python ingest.py
    python ingest.py --folder documents --chunk-size 800 --out knowledge_base.npz

Output:
    knowledge_base.npz containing chunks, metadata, and embedding vectors.
"""

import argparse
import glob
import os
import sys

import numpy as np
from dotenv import load_dotenv

# Ensure local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import (
    DEFAULT_EMBED_MODEL,
    chunk_academic_pages,
    embed_texts_batched,
    extract_text_from_pdf,
)

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Batch Ingest PDFs for Academic Document Assistant")
    parser.add_argument("--folder", default="documents", help="Folder containing PDF files")
    parser.add_argument("--chunk-size", type=int, default=800, help="Max chunk size in characters")
    parser.add_argument("--overlap", type=int, default=120, help="Overlap in characters")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    parser.add_argument("--out", default="knowledge_base.npz", help="Output .npz file path")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        print("Get your free API key at: https://aistudio.google.com and set it in .env")
        sys.exit(1)

    pdfs = sorted(glob.glob(os.path.join(args.folder, "*.pdf")))
    if not pdfs:
        print(f"⚠️ No PDF files found in '{args.folder}/'.")
        print("Place your academic PDFs into that folder and re-run this script.")
        sys.exit(0)

    print(f"📚 Found {len(pdfs)} PDF(s) in '{args.folder}/'...")

    all_chunks = []
    all_metas = []

    for path in pdfs:
        name = os.path.basename(path)
        print(f"  📖 Reading: {name}...")
        try:
            pages = extract_text_from_pdf(path)
        except Exception as e:
            print(f"  ❌ Error reading {name}: {e}")
            continue

        if not pages:
            print(f"  ⚠️ Skipping {name}: No extractable text found (scanned or image-only PDF).")
            continue

        chunks, metas = chunk_academic_pages(name, pages, chunk_size=args.chunk_size, overlap=args.overlap)
        all_chunks.extend(chunks)
        all_metas.extend(metas)
        print(f"     -> {len(pages)} pages split into {len(chunks)} recursive semantic chunks.")

    if not all_chunks:
        print("❌ No readable chunks were extracted from the documents. Aborting.")
        sys.exit(1)

    print(f"\n⚡ Generating embeddings for {len(all_chunks)} chunks using '{args.embed_model}'...")
    try:
        vectors = embed_texts_batched(all_chunks, api_key=api_key, embed_model=args.embed_model, batch_size=40)
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        sys.exit(1)

    docs = np.array([m["doc"] for m in all_metas])
    pages = np.array([m["page"] for m in all_metas])
    chunk_ids = np.array([m["chunk_id"] for m in all_metas])

    np.savez_compressed(
        args.out,
        chunks=np.array(all_chunks, dtype=object),
        docs=docs,
        pages=pages,
        chunk_ids=chunk_ids,
        vectors=vectors,
    )

    print(f"\n✅ Knowledge base successfully saved to: {args.out}")
    print(f"   Indexed {len(all_chunks)} chunks across {len(pdfs)} document(s).")
    print(f"   Embeddings matrix shape: {vectors.shape}")
    print("\nYou can now launch the web app with: streamlit run app.py")


if __name__ == "__main__":
    main()
