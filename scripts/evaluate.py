import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.core.models import DocumentChunk
from app.evaluation.evaluator import RetrievalEvaluator
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore


def main():
    settings = get_settings()
    index_dir = Path("indexes")

    chunks_data = json.loads(
        (index_dir / "chunks.json").read_text(encoding="utf-8")
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
    vector.load(index_dir, chunks)

    bm25 = BM25Index()
    bm25.load(index_dir)

    retriever = HybridRetriever(vector, bm25, settings.rrf_k)
    reranker = CrossEncoderReranker(settings.reranker_model)

    metrics = RetrievalEvaluator(retriever, reranker).evaluate(
        Path("evaluation/dataset.json"),
        settings.rerank_top_k,
    )

    print(json.dumps(metrics, indent=2))

    if metrics["recall_at_k"] < 0.80:
        raise SystemExit("Quality gate failed: Recall@K < 0.80")


if __name__ == "__main__":
    main()
