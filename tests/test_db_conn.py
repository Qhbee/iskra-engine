"""``db/conn``：环境变量映射与 ``connect_pg`` 行为。"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from psycopg.rows import dict_row

from iskra_engine.db.conn import connect_pg, pg_connect_kwargs

# skipif 在导入阶段就会求值；且 PyCharm 常以 tests/ 为 cwd，需显式指向仓库根 .env
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")


def _env_truthy(name: str) -> bool:
    """与常见 .env 习惯一致：1 / true / yes / on（大小写不敏感）为开启。"""
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _clear_pg_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "PGHOST",
        "PGPORT",
        "PGDATABASE",
        "PGUSER",
        "PGPASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)


def test_pg_connect_kwargs_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pg_env(monkeypatch)
    assert pg_connect_kwargs() == {
        "host": "127.0.0.1",
        "port": "5432",
        "dbname": "db_name",
        "user": "db_user",
        "password": "db_password",
    }


def test_pg_connect_kwargs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGHOST", "db.example.test")
    monkeypatch.setenv("PGPORT", "5433")
    monkeypatch.setenv("PGDATABASE", "mydb")
    monkeypatch.setenv("PGUSER", "alice")
    monkeypatch.setenv("PGPASSWORD", "secret")
    assert pg_connect_kwargs() == {
        "host": "db.example.test",
        "port": "5433",
        "dbname": "mydb",
        "user": "alice",
        "password": "secret",
    }


@patch("iskra_engine.db.conn.psycopg.connect")
def test_connect_pg_calls_psycopg_with_kwargs(mock_connect: MagicMock) -> None:
    fake = MagicMock()
    mock_connect.return_value = fake

    conn = connect_pg(dict_rows=False)

    mock_connect.assert_called_once_with(**pg_connect_kwargs())
    assert conn is fake


@patch("iskra_engine.db.conn.psycopg.connect")
def test_connect_pg_dict_rows_sets_row_factory(
    mock_connect: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_pg_env(monkeypatch)
    fake = MagicMock()
    mock_connect.return_value = fake

    connect_pg(dict_rows=True)

    assert fake.row_factory is dict_row


@pytest.mark.skipif(
    not _env_truthy("ISKRA_DB_SMOKE"),
    reason=("在进程环境或仓库根 .env 中设置 ISKRA_DB_SMOKE=true（或 1、yes、on）后运行烟测：对真实库做一次 SELECT 1"),
)
def test_connect_pg_smoke_select_one() -> None:
    with connect_pg() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS n")
            row = cur.fetchone()
            assert row is not None and row[0] == 1
