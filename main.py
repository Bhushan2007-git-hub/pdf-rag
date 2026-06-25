import os
import math
import json
from collections import defaultdict
from typing import AsyncGenerator

import fitz  # pymupdf
from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory document store: { filename: [chunk, chunk, ...] }
doc_store: dict[str, list[str]] = {}


# ── PDF ingestion ──────────────────────────────────────────────────────────────

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk) > 20:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── BM25 retrieval ─────────────────────────────────────────────────────────────

def bm25_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Score every chunk across all docs with BM25 and return top_k hits."""
    all_chunks = [
        {"text": chunk, "doc": fname}
        for fname, chunks in doc_store.items()
        for chunk in chunks
    ]
    if not all_chunks:
        return []

    q_words = [w.lower() for w in query.split() if len(w) > 2]
    if not q_words:
        return []

    # Document frequency
    df: dict[str, int] = defaultdict(int)
    for item in all_chunks:
        seen = set(item["text"].lower().split())
        for w in seen:
            df[w] += 1

    N = len(all_chunks)
    avg_len = sum(len(c["text"].split()) for c in all_chunks) / N
    k1, b = 1.5, 0.75

    scored = []
    for item in all_chunks:
        words = item["text"].lower().split()
        freq: dict[str, int] = defaultdict(int)
        for w in words:
            freq[w] += 1
        doc_len = len(words)
        score = 0.0
        for qw in q_words:
            f = freq[qw]
            if f == 0:
                continue
            idf = math.log((N - df[qw] + 0.5) / (df[qw] + 0.5) + 1)
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avg_len))
        scored.append({**item, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [c for c in scored[:top_k] if c["score"] > 0]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    data = await file.read()
    text = extract_text(data)
    chunks = chunk_text(text)
    doc_store[file.filename] = chunks
    return {"filename": file.filename, "chunks": len(chunks)}


@app.delete("/doc/{filename}")
async def delete_doc(filename: str):
    if filename not in doc_store:
        raise HTTPException(404, "Document not found.")
    del doc_store[filename]
    return {"removed": filename}


@app.get("/docs")
async def list_docs():
    return [{"filename": f, "chunks": len(c)} for f, c in doc_store.items()]


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    if not doc_store:
        raise HTTPException(400, "No documents uploaded yet.")

    hits = bm25_retrieve(req.question)

    if hits:
        context = "\n\n---\n\n".join(
            f"[Excerpt {i+1} from: {h['doc']}]\n{h['text']}"
            for i, h in enumerate(hits)
        )
        system = (
            "You are a PDF analysis assistant. Your ONLY job is to answer questions "
            "based strictly on the document excerpts provided below. "
            "Rules you must never break:\n"
            "1. Never use any knowledge outside of the provided excerpts.\n"
            "2. If the answer is not found in the excerpts, respond only with: "
            "'This information is not present in the uploaded documents.'\n"
            "3. Always mention which document the answer came from.\n"
            "4. Never engage in general conversation, small talk, or answer anything unrelated to the documents.\n\n"
            f"DOCUMENT EXCERPTS:\n{context}"
        )
    else:
        async def no_match():
            yield f"event: sources\ndata: {json.dumps([])}\n\n"
            yield f"data: {json.dumps({'text': 'I could not find anything related to that in your uploaded documents.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_match(), media_type="text/event-stream")

    sources = [{"doc": h["doc"], "score": round(h["score"], 3)} for h in hits]

    async def stream() -> AsyncGenerator[str, None]:
        # First send the source metadata as a special SSE event
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": req.question},
            ],
        )
        for chunk in completion:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# Serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
