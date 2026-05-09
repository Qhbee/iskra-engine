"""
LlamaIndex ``BaseEmbedding`` 适配：`GgufEmbedding`（可与其它实现如 ``openai_llama_embedding`` 并存）。
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field

from iskra_engine.embeddings.gguf_embed import embed_document, embed_query


class GgufEmbedding(BaseEmbedding):
    """向量维数必须与 ``chunk.embedding``、``ISKRA_EMBED_DIM`` 一致。"""

    embed_dim: int = Field(ge=1, description="向量维数")

    def __init__(self, *, embed_dim: int | None = None, **kwargs: Any) -> None:
        load_dotenv()
        dim = (
            embed_dim
            if embed_dim is not None
            else int(os.environ.get("ISKRA_EMBED_DIM", "1024"))
        )
        super().__init__(embed_dim=dim, model_name="iskra-jina-embeddings-v5-retrieval-gguf", **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "GgufEmbedding"

    def _get_query_embedding(self, query: str) -> Embedding:
        return embed_query(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await asyncio.to_thread(embed_query, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return embed_document(text)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await asyncio.to_thread(embed_document, text)
