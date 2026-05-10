"""``retrieval/pg_vector_retriever``：纯函数、HNSW 会话参数与带 mock 库的 ``PgVectorRetriever``。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import QueryBundle

from iskra_engine.retrieval import pg_vector_retriever as pv


def test_like_prefix_none() -> None:
    assert pv._like_prefix(None) is None
    assert pv._like_prefix("") is None
    assert pv._like_prefix("  ") is None


def test_like_suffix_percent() -> None:
    assert pv._like_prefix("books/vol1") == "books/vol1%"
    assert pv._like_prefix("p%") == "p%"


def test_session_hnsw_ef_search_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PG_HNSW_EF_SEARCH", raising=False)
    assert pv._session_hnsw_ef_search() == 128


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("0", None),
        ("off", None),
        ("FALSE", None),
        ("no", None),
        ("  ", None),
    ],
)
def test_session_hnsw_ef_search_disabled(
    monkeypatch: pytest.MonkeyPatch, raw: str, want: int | None
) -> None:
    monkeypatch.setenv("PG_HNSW_EF_SEARCH", raw)
    assert pv._session_hnsw_ef_search() is want


def test_session_hnsw_ef_search_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG_HNSW_EF_SEARCH", "64")
    assert pv._session_hnsw_ef_search() == 64
    monkeypatch.setenv("PG_HNSW_EF_SEARCH", "1")
    assert pv._session_hnsw_ef_search() == 1


def test_session_hnsw_ef_search_clamped_min(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG_HNSW_EF_SEARCH", "-3")
    assert pv._session_hnsw_ef_search() == 1


def test_apply_hnsw_ef_search_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    cur = MagicMock()
    pv._apply_hnsw_ef_search(cur, None)
    cur.execute.assert_not_called()


def test_apply_hnsw_ef_search_sets_integer() -> None:
    cur = MagicMock()
    pv._apply_hnsw_ef_search(cur, 99)
    cur.execute.assert_called_once_with("SET LOCAL hnsw.ef_search = 99")


@pytest.mark.parametrize("_ef", [0, -1])
def test_apply_hnsw_ef_search_non_positive(_ef: int) -> None:
    cur = MagicMock()
    pv._apply_hnsw_ef_search(cur, _ef)
    cur.execute.assert_not_called()


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, object]] = []

    def execute(self, query: str, params: object | None = None) -> None:
        self.calls.append((query, params))

    def fetchall(self) -> list[tuple]:
        return self.rows


class _CursorCtx:
    def __init__(self, cur: _FakeCursor) -> None:
        self._cur = cur

    def __enter__(self) -> _FakeCursor:
        return self._cur

    def __exit__(self, *_a: object) -> None:
        return None


class _FakeConn:
    """``with conn_factory() as conn`` + ``with conn.cursor() as cur``。"""

    def __init__(self, rows: list[tuple]) -> None:
        self._fake_cur = _FakeCursor(rows)

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *_a: object) -> None:
        return None

    def cursor(self) -> _CursorCtx:
        return _CursorCtx(self._fake_cur)


def _embedding_stub(dim: int) -> list[float]:
    return [0.01 * float(i % 17) for i in range(dim)]


@patch.object(pv, "_register_pgvector", lambda _c: None)
@patch.object(pv, "_session_hnsw_ef_search", lambda: None)
def test_pg_vector_retriever_retrieve_maps_rows() -> None:
    rows = [
        (
            42,
            3,
            "chunk text",
            "mao/sn.md",
            "与斯诺",
            "选集",
            0.2,
        ),
    ]

    outer = _FakeConn(rows)

    def factory() -> _FakeConn:
        return outer

    emb = MagicMock()
    emb.get_query_embedding.return_value = _embedding_stub(1024)

    r = pv.PgVectorRetriever(emb, top_k=5, path_prefix=None, conn_factory=factory)
    out = r._retrieve(QueryBundle("  毛泽东与斯诺  "))

    assert len(out) == 1
    nws = out[0]
    assert nws.score == pytest.approx(0.8)
    assert nws.node.text == "chunk text"
    assert nws.node.metadata["rel_path"] == "mao/sn.md"
    assert nws.node.metadata["chunk_id"] == 42
    assert nws.node.metadata["chunk_index"] == 3
    assert nws.node.metadata["embedding_distance"] == pytest.approx(0.2)
    assert nws.node.id_ == "chunk:42:3"

    emb.get_query_embedding.assert_called_once_with("毛泽东与斯诺")
    selects = [
        c for c in outer._fake_cur.calls if "SELECT chunk.id" in c[0].replace("\n", " ")
    ]
    assert len(selects) == 1
    _sql, params = selects[0]
    assert params[1] == 5


@patch.object(pv, "_register_pgvector", lambda _c: None)
@patch.object(pv, "_session_hnsw_ef_search", lambda: None)
def test_pg_vector_retriever_path_prefix_like() -> None:
    rows: list[tuple] = []

    outer = _FakeConn(rows)

    def factory() -> _FakeConn:
        return outer

    emb = MagicMock()
    emb.get_query_embedding.return_value = _embedding_stub(512)

    r = pv.PgVectorRetriever(emb, top_k=2, path_prefix="books/v1", conn_factory=factory)
    out = r._retrieve(QueryBundle("q"))
    assert out == []

    selects = [
        c for c in outer._fake_cur.calls if "WHERE d.rel_path LIKE" in c[0]
    ]
    assert len(selects) == 1
    _sql, params = selects[0]
    assert params[1] == "books/v1%"
    assert params[2] == 2


@patch.object(pv, "_session_hnsw_ef_search", lambda: 77)
def test_pg_vector_retriever_sets_ef_search(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[tuple] = []
    outer = _FakeConn(rows)

    monkeypatch.setattr(pv, "_register_pgvector", lambda _c: None)

    emb = MagicMock()
    emb.get_query_embedding.return_value = _embedding_stub(16)

    r = pv.PgVectorRetriever(emb, top_k=1, conn_factory=lambda: outer)
    r._retrieve(QueryBundle("x"))

    set_calls = [c for c in outer._fake_cur.calls if "SET LOCAL hnsw" in c[0]]
    assert len(set_calls) == 1
    assert set_calls[0][0] == "SET LOCAL hnsw.ef_search = 77"


def test_pg_vector_retriever_empty_query() -> None:
    emb = MagicMock()
    r = pv.PgVectorRetriever(emb, top_k=3, conn_factory=lambda: None)  # not used

    assert r._retrieve(QueryBundle(" \t")) == []

    emb.get_query_embedding.assert_not_called()


def test_pg_vector_retriever_unknown_kwargs() -> None:
    emb = MagicMock()
    with pytest.raises(TypeError, match="未知构造函数参数"):
        pv.PgVectorRetriever(emb, foo=1)  # type: ignore[call-arg]


def test_pg_vector_retriever_top_k_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", "11")
    emb = MagicMock()
    r = pv.PgVectorRetriever(emb, top_k=None, conn_factory=lambda: None)
    assert r._top_k == 11
