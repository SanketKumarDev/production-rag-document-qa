# AskMyTechDocs — Production RAG Document Q&A

> A production-oriented Retrieval-Augmented Generation (RAG) application that lets users ask questions about their own documents using hybrid retrieval, Cross-Encoder reranking, local LLM inference, and citation-backed answers.

## Overview

**AskMyTechDocs** is a domain-specific **"Ask My Docs"** system designed to demonstrate a practical production RAG pipeline.

Instead of relying only on an LLM's pretrained knowledge, the application retrieves relevant information from a user's documents and uses that evidence to generate grounded answers.

The retrieval pipeline combines:

- **BM25** for keyword-based retrieval
- **Vector search** for semantic retrieval
- **Reciprocal Rank Fusion (RRF)** for combining retrieval results
- **Cross-Encoder reranking** for improving relevance
- **Ollama + Llama 3.2** for local answer generation
- **Citation validation** for traceable responses

The project is built with **Python, FastAPI, Streamlit, FAISS, Sentence Transformers, Pytest, Docker, and GitHub Actions**.

---

## Key Features

- PDF and text document ingestion
- Document chunking and metadata tracking
- Semantic vector search
- BM25 keyword search
- Hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Local LLM inference with Ollama
- Citation-backed answers
- Citation validation
- FastAPI backend
- Streamlit frontend
- Automated tests
- CI workflow
- Docker support
- Reproducible Python dependencies with `requirements.txt`

---

## How It Works

```text
Documents
    |
    v
Ingestion & Chunking
    |
    +--------------------+
    |                    |
    v                    v
BM25 Index          Vector Index
    |                    |
    +---------+----------+
              |
              v
       Hybrid Retrieval
              |
              v
          RRF Fusion
              |
              v
    Cross-Encoder Reranking
              |
              v
      Relevant Context
              |
              v
        Ollama / LLM
              |
              v
    Citation Validation
              |
              v
     Answer + Citations
```

### 1. Document Ingestion

Documents are loaded from the document collection and converted into text.

### 2. Chunking

Long documents are divided into smaller chunks while retaining useful metadata such as:

- Source filename
- Page number, when available
- Chunk ID

### 3. Indexing

Each chunk is prepared for two retrieval methods:

- A vector representation for semantic search
- A BM25 representation for keyword search

### 4. Hybrid Retrieval

The user's question is searched using both retrieval methods.

This allows the system to handle both:

- Exact technical terms and keywords
- Meaning-based or semantic questions

### 5. RRF Fusion

Results from BM25 and vector search are combined using **Reciprocal Rank Fusion** to produce a stronger candidate set.

### 6. Cross-Encoder Reranking

The retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The most relevant chunks are selected for the final context.

### 7. Answer Generation

The selected context is sent to a local LLM through Ollama.

The default model is:

```text
llama3.2
```

The LLM generates an answer based on the retrieved evidence.

### 8. Citation Validation

The application validates the citations included in the generated answer against the retrieved source documents.

The final response includes the answer and supporting evidence.

---

## Example

A user can ask:

```text
What programming languages does Sanket Kumar know?
```

The system retrieves relevant evidence from the indexed documents and may return an answer such as:

```text
According to [S1] and [S3], Sanket Kumar knows:

- JavaScript
- Java
- Python
- HTML5
- CSS3
```

The UI also displays the corresponding sources and retrieved evidence.

If the documents do not contain sufficient information, the system can respond that it could not find enough evidence instead of inventing an answer.

---

## Architecture

```text
                    +----------------+
                    |     User       |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Streamlit UI   |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | FastAPI API    |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | User Question  |
                    +-------+--------+
                            |
                +-----------+-----------+
                |                       |
                v                       v
          +-----------+           +-----------+
          |    BM25   |           |  Vector   |
          |  Search   |           |  Search   |
          +-----+-----+           +-----+-----+
                |                       |
                +-----------+-----------+
                            |
                            v
                    +----------------+
                    |   RRF Fusion   |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Cross-Encoder  |
                    |   Reranking    |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Context Builder|
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Ollama / LLM   |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Citation Check |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Answer + Source|
                    +----------------+
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Embeddings | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Retrieval Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Reranker Model | `ms-marco-MiniLM-L-6-v2` |
| LLM Runtime | Ollama |
| LLM | Llama 3.2 |
| Testing | Pytest |
| CI | GitHub Actions |
| Containerization | Docker |

---

## Project Structure

```text
ask-my-tech-docs/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── config/
│   ├── core/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── retrieval/
│   └── main.py
│
├── data/
│   └── documents/
│
├── evaluation/
├── frontend/
│   └── streamlit_app.py
├── indexes/
├── scripts/
│   └── ingest.py
├── tests/
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

# Setup

## Prerequisites

Install:

- Python 3.11+
- Git
- Ollama
- Docker (optional)

The core application uses a local LLM through Ollama, so a paid LLM API key is not required.

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ask-my-tech-docs.git
cd ask-my-tech-docs
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

---

## 4. Set Up Ollama

Install Ollama and verify the installation:

```bash
ollama --version
```

Download the model:

```bash
ollama pull llama3.2
```

Verify:

```bash
ollama list
```

> `ollama pull` downloads the selected LLM model. It does not reinstall Ollama itself.

---

## 5. Configure Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```env
APP_NAME=AskMyTechDocs
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

Do not commit `.env` to GitHub.

---

## 6. Add Documents

Place your PDF and text documents in:

```text
data/documents/
```

Example:

```text
data/documents/
├── resume.pdf
├── documentation.pdf
└── notes.txt
```

---

## 7. Build the Search Index

Run the ingestion pipeline from the project root:

```bash
python -m scripts.ingest
```

Using `python -m` is recommended because it correctly resolves project-level imports such as:

```python
from app.config.settings import get_settings
```

The first run may download the embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The Cross-Encoder model is also downloaded automatically when first required.

---

## 8. Start the Backend

Run:

```bash
uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

> If `/` returns `404 Not Found`, that is expected when no root endpoint is defined. Use `/docs` to access the API documentation.

---

## 9. Start the Frontend

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run frontend/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The application is now ready to answer questions about your indexed documents.

---

# Running the Application

You normally need:

### Terminal 1 — FastAPI

```bash
uvicorn app.main:app --reload
```

### Terminal 2 — Streamlit

```bash
streamlit run frontend/streamlit_app.py
```

Ollama must be installed and available locally with the required model.

---

# Testing

Run the test suite:

```bash
pytest
```

Verbose output:

```bash
pytest -v
```

The tests cover core application behavior such as retrieval, citation validation, and API functionality.

---

# Evaluation and CI

The project includes an evaluation and CI workflow intended to improve reliability as the application evolves.

The CI pipeline follows the general flow:

```text
Push / Pull Request
        |
        v
Install Dependencies
        |
        v
Run Tests
        |
        v
Validation
        |
        v
Pass / Fail
```

The evaluation process can be extended to measure:

- Retrieval quality
- Context relevance
- Answer grounding
- Citation correctness

This helps identify whether an issue originates in retrieval, reranking, context construction, or generation.

---

# Docker

Build the image:

```bash
docker build -t ask-my-tech-docs .
```

Run:

```bash
docker run -p 8000:8000 ask-my-tech-docs
```

Or use Docker Compose:

```bash
docker compose up --build
```

The exact Docker and Ollama networking configuration may vary depending on whether Ollama runs on the host or inside a container.

---

# Common Issues

### `ModuleNotFoundError: No module named 'app'`

Run the ingestion script from the project root:

```bash
python -m scripts.ingest
```

instead of:

```bash
python scripts/ingest.py
```

### `ollama: command not found`

Restart VS Code or the terminal after installing Ollama, then check:

```bash
ollama --version
```

### Git Bash virtual environment activation

Use:

```bash
source .venv/Scripts/activate
```

### Streamlit file not found

Run the command from the project root:

```bash
streamlit run frontend/streamlit_app.py
```

### FastAPI returns `404 Not Found`

Open:

```text
http://127.0.0.1:8000/docs
```

The root `/` endpoint may not be implemented.

### First response is slow

The first request may take longer while the embedding model, Cross-Encoder, and Ollama LLM are loaded into memory. Local CPU-based LLM inference can also be slower than cloud inference.

---

# Future Improvements

Possible extensions include:

- Streaming responses
- Multi-user document collections
- Conversation history
- Query rewriting
- Metadata filtering
- PostgreSQL + pgvector
- Redis caching
- Advanced RAG evaluation
- Observability and metrics
- Authentication and authorization
- Cloud deployment

---

# Learning Outcomes

This project provides practical experience with:

- Retrieval-Augmented Generation
- Document ingestion and chunking
- Embeddings and vector search
- BM25 keyword retrieval
- Hybrid search
- Reciprocal Rank Fusion
- Cross-Encoder reranking
- Local LLM inference with Ollama
- Citation validation
- FastAPI
- Streamlit
- Automated testing
- CI workflows
- Docker
- Production-oriented AI application architecture

---

## Author

**Sanket Kumar**

Built as a hands-on AI engineering project focused on production-oriented RAG systems, hybrid information retrieval, LLM applications, and reliable document-grounded question answering.

---

## Keywords

`RAG` `Retrieval-Augmented Generation` `Production RAG` `LLM` `Generative AI` `AI Engineering` `Hybrid Search` `BM25` `Vector Search` `FAISS` `Cross-Encoder` `Reranking` `RRF` `Ollama` `Llama 3.2` `FastAPI` `Streamlit` `Python` `Sentence Transformers` `RAG Evaluation` `Docker` `GitHub Actions`
