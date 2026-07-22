import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer

from app.api.schemas import AskRequest, AskResponse
from app.config.settings import get_settings
from app.core.models import DocumentChunk
from app.generation.citation_validator import enforce_grounding
from app.generation.context_builder import build_context
from app.generation.ollama_client import OllamaClient
from app.generation.prompts import build_prompt
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore

router = APIRouter()
settings = get_settings()
INDEX_DIR = Path("indexes")


def _load_components():
    chunks_data = json.loads(
        (INDEX_DIR / "chunks.json").read_text(encoding="utf-8")
    )
    chunks = [
        DocumentChunk(
            chunk_id=x["chunk_id"],
            document_id=x["document_id"],
            text=x["text"],
            metadata=x["metadata"],
        )
        for x in chunks_data
    ]

    vector = VectorStore(SentenceTransformer(settings.embedding_model))
    vector.load(INDEX_DIR, chunks)

    bm25 = BM25Index()
    bm25.load(INDEX_DIR)

    return (
        HybridRetriever(vector, bm25, settings.rrf_k),
        CrossEncoderReranker(settings.reranker_model),
        OllamaClient(settings.ollama_base_url, settings.ollama_model),
    )


@router.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        retriever, reranker, llm = _load_components()

        candidates = retriever.search(
            request.question,
            settings.vector_top_k,
            settings.bm25_top_k,
            settings.hybrid_top_k,
        )
        results = reranker.rerank(request.question, candidates, request.top_k)
        context, citations = build_context(results)

        raw_answer = llm.generate(build_prompt(request.question, context))
        answer, grounded, used_ids = enforce_grounding(
            raw_answer,
            {c["id"] for c in citations},
        )

        used_citations = [c for c in citations if c["id"] in used_ids]

        return AskResponse(
            answer=answer,
            grounded=grounded,
            citations=used_citations,
            retrieved_sources=citations,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Indexes are not built. Run: python scripts/ingest.py",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
