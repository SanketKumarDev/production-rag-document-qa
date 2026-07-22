import time

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import AskRequest, AskResponse
from app.generation.citation_validator import enforce_grounding
from app.generation.context_builder import build_context
from app.generation.prompts import build_prompt

router = APIRouter()


@router.get("/health")
def health(request: Request):
    rag = request.app.state.rag

    return {
        "status": "ok" if rag.is_ready() else "starting",
        "app": rag.settings.app_name,
    }


@router.post("/ask", response_model=AskResponse)
def ask(
    request: Request,
    payload: AskRequest,
):
    rag = request.app.state.rag

    if not rag.is_ready():
        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing.",
        )

    try:
        total_start = time.perf_counter()

        # ---------------------------------
        # 1. Hybrid Retrieval
        # ---------------------------------
        retrieval_start = time.perf_counter()

        candidates = rag.retriever.search(
            payload.question,
            rag.settings.vector_top_k,
            rag.settings.bm25_top_k,
            rag.settings.hybrid_top_k,
        )

        retrieval_time = time.perf_counter() - retrieval_start

        # ---------------------------------
        # 2. Cross-Encoder Reranking
        # ---------------------------------
        rerank_start = time.perf_counter()

        results = rag.reranker.rerank(
            payload.question,
            candidates,
            payload.top_k,
        )

        rerank_time = time.perf_counter() - rerank_start

        # ---------------------------------
        # 3. Build Context
        # ---------------------------------
        context_start = time.perf_counter()

        context, citations = build_context(results)

        context_time = time.perf_counter() - context_start

        # ---------------------------------
        # 4. LLM Generation
        # ---------------------------------
        llm_start = time.perf_counter()

        prompt = build_prompt(
            payload.question,
            context,
        )

        raw_answer = rag.llm.generate(prompt)

        llm_time = time.perf_counter() - llm_start

        # ---------------------------------
        # 5. Citation Validation
        # ---------------------------------
        validation_start = time.perf_counter()

        answer, grounded, used_ids = enforce_grounding(
            raw_answer,
            {citation["id"] for citation in citations},
        )

        validation_time = time.perf_counter() - validation_start

        # ---------------------------------
        # Total
        # ---------------------------------
        total_time = time.perf_counter() - total_start

        print("\n" + "=" * 50)
        print("RAG PERFORMANCE")
        print("=" * 50)
        print(f"Retrieval:       {retrieval_time:.3f}s")
        print(f"Reranking:       {rerank_time:.3f}s")
        print(f"Context:         {context_time:.3f}s")
        print(f"LLM Generation:  {llm_time:.3f}s")
        print(f"Citation Check:  {validation_time:.3f}s")
        print("-" * 50)
        print(f"TOTAL:           {total_time:.3f}s")
        print("=" * 50 + "\n")

        used_citations = [
            citation
            for citation in citations
            if citation["id"] in used_ids
        ]

        return AskResponse(
            answer=answer,
            grounded=grounded,
            citations=used_citations,
            retrieved_sources=citations,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Indexes are not built. "
                "Run: python -m scripts.ingest"
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc