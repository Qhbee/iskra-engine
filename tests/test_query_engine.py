"""``query_engine.build_rag_query_engine``：用 mock 验证装配契约，不接 PG / GGUF / 真实 LLM。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from llama_index.core.response_synthesizers import ResponseMode

from iskra_engine.query import query_engine as qe


def test_build_rag_query_engine_passes_through_top_k_and_path_prefix() -> None:
    mock_llm_factory = MagicMock()
    llm_inst = MagicMock()
    mock_llm_factory.return_value = llm_inst

    embed_inst = MagicMock()
    mock_embed_cls = MagicMock(return_value=embed_inst)

    ret_inst = MagicMock()
    mock_ret_cls = MagicMock(return_value=ret_inst)

    engine_inst = MagicMock()
    mock_from_args = MagicMock(return_value=engine_inst)

    with (
        patch.object(qe, "build_openai_compatible_llm", mock_llm_factory),
        patch.object(qe, "GgufEmbedding", mock_embed_cls),
        patch.object(qe, "PgVectorRetriever", mock_ret_cls),
        patch.object(qe.RetrieverQueryEngine, "from_args", mock_from_args),
    ):
        out = qe.build_rag_query_engine(
            top_k=7,
            path_prefix="docs/",
            load_dotenv_=False,
        )

    assert out is engine_inst
    mock_llm_factory.assert_called_once_with()
    mock_embed_cls.assert_called_once_with()
    mock_ret_cls.assert_called_once_with(
        embed_inst,
        top_k=7,
        path_prefix="docs/",
    )
    mock_from_args.assert_called_once_with(
        retriever=ret_inst,
        llm=llm_inst,
        response_mode=ResponseMode.COMPACT,
    )


def test_build_rag_query_engine_uses_injected_llm_and_embedding() -> None:
    custom_llm = MagicMock()
    custom_emb = MagicMock()
    mock_ret_cls = MagicMock(return_value=MagicMock())
    engine_inst = MagicMock()
    mock_from_args = MagicMock(return_value=engine_inst)

    with (
        patch.object(qe, "build_openai_compatible_llm") as mock_llm_factory,
        patch.object(qe, "GgufEmbedding") as mock_embed_cls,
        patch.object(qe, "PgVectorRetriever", mock_ret_cls),
        patch.object(qe.RetrieverQueryEngine, "from_args", mock_from_args),
    ):
        out = qe.build_rag_query_engine(
            llm=custom_llm,
            embedding_model=custom_emb,
            load_dotenv_=False,
        )

    assert out is engine_inst
    mock_llm_factory.assert_not_called()
    mock_embed_cls.assert_not_called()
    mock_ret_cls.assert_called_once_with(
        custom_emb,
        top_k=None,
        path_prefix=None,
    )
    kwargs = mock_from_args.call_args.kwargs
    assert kwargs["retriever"] is mock_ret_cls.return_value
    assert kwargs["llm"] is custom_llm


def test_build_rag_query_engine_load_dotenv_flag() -> None:
    with (
        patch.object(qe, "load_dotenv") as mock_load,
        patch.object(qe, "build_openai_compatible_llm", return_value=MagicMock()),
        patch.object(qe, "GgufEmbedding", return_value=MagicMock()),
        patch.object(qe, "PgVectorRetriever", return_value=MagicMock()),
        patch.object(
            qe.RetrieverQueryEngine,
            "from_args",
            return_value=MagicMock(),
        ),
    ):
        qe.build_rag_query_engine(load_dotenv_=True)
        mock_load.assert_called_once()
        mock_load.reset_mock()
        qe.build_rag_query_engine(load_dotenv_=False)
        mock_load.assert_not_called()
