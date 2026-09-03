# nerveLLM — Local RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) application.
PDFs are ingested, chunked, embedded, and stored in **PostgreSQL + pgvector** (running in Docker). Questions are answered by a **local Ollama LLM**. Everything is orchestrated by **Inngest** for durable, observable step-functions and exposed through a **Streamlit** UI.

---

## Project Structure

```
nerveLLM/
├── main.py              # FastAPI app + Inngest function definitions (ingest & query)
├── streamlit_app.py     # Streamlit UI — upload PDFs, ask questions, view answers
├── data_loader.py       # PDF loading, text chunking, embedding generation (Ollama)
├── vector_db.py         # PgVectorStorage — upsert & cosine-similarity search
├── setup_db.py          # One-time DB initialisation script (pgvector extension + table + index)
├── custom_types.py      # Pydantic models for Inngest step I/O
├── docker-compose.yml   # pgvector/pgvector:pg16 container definition
├── pyproject.toml       # Python dependencies (managed with uv)
├── .env.example         # Template — copy to .env and fill in values
```

---

## RAG Pipeline

### Ingestion Pipeline

```
  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐    ┌─────────────┐
  │  Upload  │───▶│  PDF Loader │───▶│   Sentence Chunker   │───▶│  Embedding  │
  │  PDF     │    │ (LlamaIndex │    │  (1 000 tok chunks / │    │  Model      │
  │          │    │  PDFReader) │    │   200 tok overlap)   │    │  (Ollama or │
  └──────────┘    └─────────────┘    └──────────────────────┘    │   OpenAI)   │
                                                                  └──────┬──────┘
                                                                         │
                                                                  ┌──────▼──────┐
                                                                  │  pgvector   │
                                                                  │  (Docker)   │
                                                                  │  HNSW Index │
                                                                  └─────────────┘
```

### Query Pipeline

```
  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐    ┌─────────────┐
  │  User    │───▶│  Embed      │───▶│  Cosine Search       │───▶│  Top-K      │
  │ Question │    │  Question   │    │  (pgvector <=>)      │    │  Chunks     │
  └──────────┘    └─────────────┘    └──────────────────────┘    └──────┬──────┘
                                                                         │
                                                                  ┌──────▼──────┐
                                                                  │   Prompt    │
                                                                  │  Assembly   │
                                                                  │ (context +  │
                                                                  │  question)  │
                                                                  └──────┬──────┘
                                                                         │
                                                                  ┌──────▼──────┐
                                                                  │  Ollama LLM │
                                                                  │ llama3.1:8b │
                                                                  └──────┬──────┘
                                                                         │
                                                                  ┌──────▼──────┐
                                                                  │  Answer +   │
                                                                  │   Sources   │
                                                                  └─────────────┘
```

### Orchestration Layer (Inngest)

```
  Streamlit UI ──send event──▶ Inngest Dev Server ──trigger──▶ FastAPI Backend

  rag/ingest_pdf    →  [load-and-chunk]   →  [embed-and-upsert]
  rag/query_pdf_ai  →  [embed-and-search] →  [llm-answer]

  • Throttle  : max 2 ingestion runs / minute
  • Rate limit: max 1 ingest per source file per 4 hours (deduplication)
  • Step-level retries and full run history in the Inngest dashboard (:8288)
```

### Pipeline Steps in Detail

| # | Step | Module / Function | What happens |
|---|---|---|---|
| 1 | **Load PDF** | `data_loader.load_and_chunk_pdf()` | `PDFReader` extracts raw text page-by-page |
| 2 | **Chunk** | `data_loader` → `SentenceSplitter` | Text split into 1 000-token chunks with 200-token overlap |
| 3 | **Embed** | `data_loader.embed_texts()` | Chunks sent to Ollama (`nomic-embed-text`, 768 dims) or OpenAI (`text-embedding-3-large`, 3 072 dims) |
| 4 | **Upsert** | `vector_db.PgVectorStorage.upsert()` | Text + vector + metadata stored in PostgreSQL with deterministic UUID5 IDs (idempotent) |
| 5 | **Query Embed** | `data_loader.embed_texts()` | User question embedded with the same model |
| 6 | **Search** | `vector_db.PgVectorStorage.search()` | Cosine distance (`<=>`) with HNSW index returns top-K chunks |
| 7 | **Generate** | `main._generate_answer()` | `llama3.1:8b` via Ollama answers using only the retrieved context |

---

## Requirements

### System Requirements

| Requirement | Version / Notes |
|---|---|
| Python | ≥ 3.13 |
| Docker & Docker Compose | Any recent version |
| [Ollama](https://ollama.com) | Latest — for local LLM & embeddings |
| [uv](https://docs.astral.sh/uv/) | Latest (recommended package manager) |
| Node.js / npx | For running the Inngest dev server |
---
### Ollama Models

```bash
ollama pull llama3.1:8b        # LLM for answer generation
ollama pull nomic-embed-text   # Embedding model (768 dims)
```

---

## How to Run
the Python application (Streamlit, FastAPI, Inngest) runs directly on your machine. Only the database runs in Docker.

#### 1. Clone & Install

```bash
git clone <repo-url>
cd nerveLLM

# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

#### 2. Configure Environment

```bash
cp .env.example .env
# Defaults work out-of-the-box for local development — no edits required
```

#### 3. Start pgvector in Docker

```bash
docker compose up -d
```

Starts PostgreSQL 16 with the pgvector extension pre-installed, exposed on `localhost:5432`.
Data is persisted in the `pgvector_data` Docker volume across restarts.

#### 4. Initialise the Database

```bash
uv run python setup_db.py
```

This script:
- Enables the `vector` extension
- Creates the `documents` table with the correct vector dimension
- Builds an HNSW cosine similarity index
- Runs a probe insert/search to verify everything works

#### 5. Start Ollama

```bash
# Terminal 1
ollama serve

# Pull models if not already done (one-time)
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

#### 6. Start the FastAPI Backend

```bash
# Terminal 2
uv run uvicorn main:app --reload --port 8000
```

#### 7. Start the Inngest Dev Server

```bash
# Terminal 3
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

Inngest dashboard → [http://127.0.0.1:8288](http://127.0.0.1:8288)
Use it to inspect function runs, step outputs, retries, and event history.

#### 8. Start the Streamlit UI

```bash
# Terminal 4
uv run streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### Quick-Start (all steps combined)

```bash
# 1. Start database
docker compose up -d
uv run python setup_db.py

# 2. Start Ollama (separate terminal)
ollama serve

# 3. Start FastAPI backend (separate terminal)
uv run uvicorn main:app --reload --port 8000

# 4. Start Inngest dev server (separate terminal)
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

# 5. Start Streamlit UI (separate terminal)
uv run streamlit run streamlit_app.py
```

Then open **http://localhost:8501**, upload a PDF, and ask questions.

---