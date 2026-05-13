"""RAG 编排层：把「根据用户提问检索文档」与「组装发给大模型生成答案」接成 LlamaIndex 的 QueryEngine。

只做**组装**（工厂）：不负责 SQL、不负责 GGUF 细节。具体分工：

- ``embeddings/``：问句（及文档风格）向量；
- ``retrieval/``：仅从 PG/pgvector 取 Top-K chunks；
- 本模块 ``query/``：组装 RetrieverQueryEngine = Retriever + Response Synthesizer + LLM。
  对 ``engine.query()``，返回的 ``Response.response`` 即为答案正文，并可从 ``source_nodes`` 追到引用片段。
"""
from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

from iskra_engine.embeddings.gguf_llama_embedding import GgufEmbedding
from iskra_engine.llm.client import build_openai_compatible_llm
from iskra_engine.retrieval.pg_vector_retriever import PgVectorRetriever


def build_rag_query_engine(
    *,
    top_k: Optional[int] = None,
    path_prefix: Optional[str] = None,
    model: Optional[str] = None,
    llm: Optional[LLM] = None,
    embedding_model: Optional[BaseEmbedding] = None,
    load_dotenv_: bool = True,
) -> RetrieverQueryEngine:
    """组装端到端「用户问句 → 文档检索 → 上下文合成 → 发给 LLM 回答」流水线（LlamaIndex RAG）。
    组装 LlamaIndex RAG 的 **QueryEngine**：**会调用大模型生成对用户问题的最终文字回答**。

    流水线：用户问句 → 嵌入 → **文档检索**（Top-K）→ **compact 合成**（把检索块塞进提示）→ **LLM 生成答复**。
    ``engine.query(问题)`` 会发起一次（或多次）聊天补全；``Response.response`` 即模型给出的答案，``source_nodes`` 为引用依据。

    不负责回答、只查库时勿用本工厂；请用 ``PgVectorRetriever.retrieve()`` 或 probe 脚本。
    """
    if load_dotenv_:
        load_dotenv()
    llm_eff = llm or build_openai_compatible_llm(model=model)
    embedding_eff = embedding_model or GgufEmbedding()

    retriever = PgVectorRetriever(
        embedding_eff,
        top_k=top_k,
        path_prefix=path_prefix,
    )

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm_eff,
        response_mode=ResponseMode.COMPACT,
    )
