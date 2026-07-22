from app.core.models import DocumentChunk, SearchResult
from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_merges_duplicate_chunks():
    chunk = DocumentChunk("1", "doc", "text")
    result = SearchResult(chunk, 1.0)
    merged = reciprocal_rank_fusion([[result], [result]], top_k=5)

    assert len(merged) == 1
    assert merged[0].chunk.chunk_id == "1"
