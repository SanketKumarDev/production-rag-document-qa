import json
from pathlib import Path

from app.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


class RetrievalEvaluator:
    def __init__(self, retriever: HybridRetriever, reranker: CrossEncoderReranker):
        self.retriever = retriever
        self.reranker = reranker

    def evaluate(self, dataset_path: Path, top_k: int = 5) -> dict:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        rows = []

        for item in dataset:
            candidates = self.retriever.search(item["question"], top_k=20)
            reranked = self.reranker.rerank(item["question"], candidates, top_k)
            ids = [r.chunk.document_id for r in reranked]
            relevant = set(item["relevant_documents"])

            rows.append({
                "question": item["question"],
                "recall_at_k": recall_at_k(ids, relevant, top_k),
                "precision_at_k": precision_at_k(ids, relevant, top_k),
                "mrr": reciprocal_rank(ids, relevant),
            })

        count = max(len(rows), 1)
        return {
            "samples": len(rows),
            "recall_at_k": sum(x["recall_at_k"] for x in rows) / count,
            "precision_at_k": sum(x["precision_at_k"] for x in rows) / count,
            "mrr": sum(x["mrr"] for x in rows) / count,
            "details": rows,
        }
