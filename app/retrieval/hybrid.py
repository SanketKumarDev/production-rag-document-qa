from app.core.models import SearchResult
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25: BM25Index, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        top_k: int = 20,
    ) -> list[SearchResult]:
        return reciprocal_rank_fusion(
            [
                self.vector_store.search(query, vector_top_k),
                self.bm25.search(query, bm25_top_k),
            ],
            k=self.rrf_k,
            top_k=top_k,
        )
