"""从 PostgreSQL / pgvector 拉 Top-K chunks：LlamaIndex BaseRetriever。

HNSW ``ef_search``（环境变量 ``PG_HNSW_EF_SEARCH``）与 ``top_k``（环境变量 ``RETRIEVAL_TOP_K``）的工程搭配（经验区间，须按延迟与抽检召回率微调）：

+---------------------------+------------------+-------------------+-------------------------------------------------+
| 业务场景                    |  top_k（召回条数） |  ef_search 推荐值  | 适用说明                                          |
+---------------------------+------------------+-------------------+-------------------------------------------------+
| 极速响应型（在线客服、端侧检索）|      3 ~ 5       |      16 ~ 32      | 追求极致响应时间（毫秒级），允许牺牲极小概率的边缘召回。   |
| 标准生产型（大多数企业默认值）  |      5 ~ 10      |      64 ~ 128     | 性价比最高的区间。在召回率和耗时之间达到最佳平衡；       |
| 学术法律型（高精度，严禁漏检）  |     10 ~ 20      |     256 ~ 512     | 更重 CPU、查询更慢；宁可多花几毫秒，也必须保证高召回。   |
+---------------------------+------------------+-------------------+-------------------------------------------------+

``ef_search`` 与数据集规模 ``n`` 一般弱相关，**无必经比例**：调大主要是为了 ANN 少漏近邻，
不是 ``n`` 翻倍就要 ``ef_search`` 翻倍。未设 ``PG_HNSW_EF_SEARCH`` 时本模块会用内置
默认值（参见 ``_session_hnsw_ef_search``）。
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, cast

import psycopg
from dotenv import load_dotenv
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pgvector.psycopg import Vector
from psycopg import Connection

from iskra_engine.db.conn import pg_connect_kwargs

ConnectionFactory = Callable[[], Connection]


def _session_hnsw_ef_search() -> Optional[int]:
    """pgvector HNSW：每条检索连接 ``SET LOCAL hnsw.ef_search``。

    环境变量 ``PG_HNSW_EF_SEARCH``：

    - **未设置**：内置 ``128``（比 pgvector 默认的 40 更宽，介于典型生产与高召回之间，减少漏召回）。
    - ``0`` / ``off`` / ``false`` / ``no``：**不 SET**（沿用服务器默认）。
    - 其它正整数：使用该值。

    搭配 ``top_k`` / 场景的推荐见模块 docstring 表格；``ef_search`` 与文档条数 ``n``
    **一般无必经解析关系**，以抽检召回与延迟为准。
    """
    raw = os.environ.get("PG_HNSW_EF_SEARCH")
    if raw is None:
        return 128
    s = raw.strip().lower()
    if not s or s in ("0", "off", "false", "no"):
        return None
    return max(1, int(s))


def _apply_hnsw_ef_search(cur: psycopg.Cursor, ef: Optional[int]) -> None:
    """``SET LOCAL`` 不接受 ``$n`` 占位符，须在 SQL 字面量里写整数。"""
    if ef is None:
        return
    n = int(ef)
    if n < 1:
        return
    cur.execute(f"SET LOCAL hnsw.ef_search = {n}")


def _like_prefix(rel_prefix: Optional[str]) -> Optional[str]:
    if rel_prefix is None or str(rel_prefix).strip() == "":
        return None
    s = rel_prefix.strip()
    return s if s.endswith("%") else f"{s}%"


def _register_pgvector(conn: Connection) -> None:
    from pgvector.psycopg import register_vector  # noqa: PLC0415

    register_vector(conn)


class PgVectorRetriever(BaseRetriever):
    """基于 pgvector：``embedding <=> q``（余弦距离）越小越靠前；``score`` 用 ``1 - distance`` 裁剪到 [0,1]。
    返回 chunk 正文，``metadata`` 中含文档侧字段（``rel_path`` / ``title`` / ``book`` 等）。
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        *,
        top_k: Optional[int] = None,
        path_prefix: Optional[str] = None,
        conn_factory: Optional[ConnectionFactory] = None,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: object,
    ) -> None:
        if kwargs:
            msg = f"未知构造函数参数: {set(kwargs)!r}"
            raise TypeError(msg)
        super().__init__(callback_manager=callback_manager, verbose=verbose)
        load_dotenv()
        self._embedding_model = embedding_model
        self._top_k = (
            top_k
            if top_k is not None
            else int(os.environ.get("RETRIEVAL_TOP_K", "8"))
        )
        self._prefix = path_prefix
        self._conn_factory = conn_factory or (
            lambda: psycopg.connect(**pg_connect_kwargs())
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        qstr = query_bundle.query_str.strip()
        if not qstr:
            return []

        embedding = self._embedding_model.get_query_embedding(qstr)
        pattern = _like_prefix(self._prefix)
        vec = Vector(embedding)
        nodes: list[NodeWithScore] = []

        ef_search = _session_hnsw_ef_search()

        with self._conn_factory() as conn:
            _register_pgvector(conn)
            with conn.cursor() as cur:
                _apply_hnsw_ef_search(cur, ef_search)

                if pattern is None:
                    cur.execute(
                        """
                        SELECT chunk.id AS chunk_id, chunk.chunk_index, chunk.text,
                            d.rel_path AS rel_path, d.title AS doc_title, d.book AS doc_book,
                            chunk.embedding <=> %s::vector AS distance
                        FROM chunk
                        INNER JOIN document AS d ON d.id = chunk.document_id
                        ORDER BY distance ASC NULLS LAST
                        LIMIT %s
                        """,
                        (vec, self._top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT chunk.id AS chunk_id, chunk.chunk_index, chunk.text,
                            d.rel_path AS rel_path, d.title AS doc_title, d.book AS doc_book,
                            chunk.embedding <=> %s::vector AS distance
                        FROM chunk
                        INNER JOIN document AS d ON d.id = chunk.document_id
                        WHERE d.rel_path LIKE %s ESCAPE '\\'
                        ORDER BY distance ASC NULLS LAST
                        LIMIT %s
                        """,
                        (vec, pattern, self._top_k),
                    )

                rows = cast(Sequence[tuple], cur.fetchall())

        for chunk_id, chunk_index, chunk_txt, rel_path, doc_title, doc_book, distance in rows:
            distance = float(distance) if distance is not None else 1.0
            similarity = max(0.0, min(1.0, 1.0 - distance))

            metadata = {
                "rel_path": rel_path,
                "title": doc_title,
                "book": doc_book,
                "chunk_id": int(chunk_id),
                "chunk_index": int(chunk_index),
                "embedding_distance": distance,
            }
            node_id = f"chunk:{chunk_id}:{chunk_index}"
            node = TextNode(text=str(chunk_txt), metadata=metadata, id_=node_id)
            nodes.append(NodeWithScore(node=node, score=similarity))

        return nodes
