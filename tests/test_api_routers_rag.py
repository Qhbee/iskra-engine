"""``api/routers/rag.py``：LlamaIndex ``Response`` → ``RagQueryResponse`` 映射."""

from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode

from iskra_engine.api.routers.rag import _to_rag_query_response


def test_llama_index_response_to_rag_query_response() -> None:
    n = TextNode(
        text="chunk body",
        metadata={"rel_path": "a/b.md", "chunk_index": 0, "chunk_id": 9},
        id_="chunk:9:0",
    )
    llama_response = Response(
        response="合成答案",
        source_nodes=[NodeWithScore(node=n, score=0.88)],
    )
    out = _to_rag_query_response(llama_response)
    assert out.answer == "合成答案"
    assert len(out.sources) == 1
    assert out.sources[0].rel_path == "a/b.md"
    assert out.sources[0].chunk_index == 0
    assert out.sources[0].chunk_id == 9
    assert out.sources[0].score == 0.88
    assert "chunk" in out.sources[0].snippet
