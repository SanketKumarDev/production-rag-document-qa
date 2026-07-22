from sentence_transformers import CrossEncoder
from app.core.models import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        scores = self.model.predict(
            [(query, result.chunk.text) for result in candidates]
        )

        reranked = [
            SearchResult(result.chunk, float(score), "cross_encoder")
            for result, score in zip(candidates, scores)
        ]

        return sorted(reranked, key=lambda x: x.score, reverse=True)[:top_k]
