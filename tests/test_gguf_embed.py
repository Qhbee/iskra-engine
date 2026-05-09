"""GGUF/embeddings：不加载 .gguf，mock 下层调用。"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from iskra_engine.embeddings.gguf_embed import embed_document, embed_query
from iskra_engine.embeddings.gguf_llama_embedding import GgufEmbedding


def test_gguf_embedding_class_name() -> None:
    assert GgufEmbedding.class_name() == "GgufEmbedding"


def test_gguf_embedding_get_query_embedding_delegates() -> None:
    vec = [0.0] * 4
    with patch(
        "iskra_engine.embeddings.gguf_llama_embedding.embed_query",
        return_value=vec,
    ) as m_query, patch(
        "iskra_engine.embeddings.gguf_llama_embedding.embed_document",
        return_value=[],
    ):
        emb = GgufEmbedding(embed_dim=4)
        assert emb.get_query_embedding(" hello ") == vec
        m_query.assert_called_once_with(" hello ")


def test_gguf_embedding_get_text_embedding_delegates() -> None:
    vec = [0.0] * 4
    with patch(
        "iskra_engine.embeddings.gguf_llama_embedding.embed_document",
        return_value=vec,
    ) as m_doc, patch(
        "iskra_engine.embeddings.gguf_llama_embedding.embed_query",
        return_value=[],
    ):
        emb = GgufEmbedding(embed_dim=4)
        assert emb.get_text_embedding("chunk") == vec
        m_doc.assert_called_once_with("chunk")


def test_gguf_embedding_aget_methods_delegates() -> None:
    qv = [1.0, 0.0]
    dv = [0.0, 1.0]
    with patch(
        "iskra_engine.embeddings.gguf_llama_embedding.embed_query",
        return_value=qv,
    ) as m_query, patch(
        "iskra_engine.embeddings.gguf_llama_embedding.embed_document",
        return_value=dv,
    ) as m_doc:

        emb = GgufEmbedding(embed_dim=2)

        async def run() -> None:
            rq = await emb.aget_query_embedding("q")
            rt = await emb.aget_text_embedding("t")
            assert rq == qv
            assert rt == dv

        asyncio.run(run())
        m_query.assert_called_once_with("q")
        m_doc.assert_called_once_with("t")


def test_embed_query_prefix_and_normalize(monkeypatch) -> None:
    monkeypatch.setenv("ISKRA_EMBED_DIM", "2")
    mock_llm = MagicMock()
    mock_llm.embed.return_value = [[3.0, 4.0]]
    with patch(
        "iskra_engine.embeddings.gguf_embed._get_llama_embedder",
        return_value=mock_llm,
    ):
        out = embed_query("hi")
    mock_llm.embed.assert_called_once_with("Query: hi")
    assert len(out) == 2
    assert abs(out[0] - 0.6) < 1e-9 and abs(out[1] - 0.8) < 1e-9


def test_embed_document_prefix_and_normalize(monkeypatch) -> None:
    monkeypatch.setenv("ISKRA_EMBED_DIM", "2")
    mock_llm = MagicMock()
    mock_llm.embed.return_value = [[0.0, 5.0]]
    with patch(
        "iskra_engine.embeddings.gguf_embed._get_llama_embedder",
        return_value=mock_llm,
    ):
        out = embed_document("body")
    mock_llm.embed.assert_called_once_with("Document: body")
    assert len(out) == 2
    assert abs(out[0] - 0.0) < 1e-9 and abs(out[1] - 1.0) < 1e-9
