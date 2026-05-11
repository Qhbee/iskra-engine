from __future__ import annotations

from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    top_k: int | None = Field(None, ge=1, le=100, description="覆盖默认 RETRIEVAL_TOP_K")
    path_prefix: str | None = Field(
        None,
        description="仅检索 rel_path 此前缀（会自动加 % LIKE）",
    )


class RagSourceItem(BaseModel):
    rel_path: str | None = None
    title: str | None = None
    book: str | None = None
    chunk_index: int | None = None
    chunk_id: int | None = None
    score: float | None = None
    snippet: str = ""


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[RagSourceItem]
