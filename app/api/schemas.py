from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    id: str
    source: str
    page: int | None = None
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[Citation]
    retrieved_sources: list[Citation]
