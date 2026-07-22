from collections import defaultdict
from app.core.models import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 20,
) -> list[SearchResult]:
    scores = defaultdict(float)
    chunks = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] += 1.0 / (k + rank)
            chunks[chunk_id] = result.chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        SearchResult(chunks[chunk_id], score, "hybrid")
        for chunk_id, score in ranked
    ]
