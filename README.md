# PDF RAG - Q&A

A Retrieval-Augmented Generation (RAG) app that answers questions strictly based on uploaded PDF documents. Built with FastAPI, BM25, and Groq (Llama 3.3).

## How it works
1. Upload a PDF
2. Text is extracted and split into chunks
3. BM25 retrieves the most relevant chunks for your question
4. Llama 3.3 (via Groq) answers using only those chunks

## Tech Stack
- **Backend** — FastAPI, PyMuPDF, BM25
- **LLM** — Llama 3.3 70B via Groq API
- **Frontend** — Vanilla HTML/CSS/JS

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Groq API key (free at console.groq.com)
4. Run: `py -3.13 -m uvicorn main:app --reload`
5. Open `http://localhost:8000`

## Notes
- Only answers from uploaded PDFs — no general knowledge
- Scanned PDFs (image-based) may not extract properly
- Documents are stored in memory — cleared on server restart
