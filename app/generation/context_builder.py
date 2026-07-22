from app.core.models import SearchResult


def build_context(results: list[SearchResult]) -> tuple[str, list[dict]]:
    citations = []
    blocks = []

    for index, result in enumerate(results, start=1):
        citation_id = f"S{index}"
        source = result.chunk.metadata.get("file_name", result.chunk.document_id)
        page = result.chunk.metadata.get("page")

        citations.append({
            "id": citation_id,
            "source": source,
            "page": page,
            "chunk_id": result.chunk.chunk_id,
            "score": result.score,
        })

        page_text = f", page {page}" if page else ""
        blocks.append(
            f"[{citation_id}] Source: {source}{page_text}\n"
            f"Chunk ID: {result.chunk.chunk_id}\n"
            f"{result.chunk.text}"
        )

    return "\n\n".join(blocks), citations
