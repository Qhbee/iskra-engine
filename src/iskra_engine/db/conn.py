"""PostgreSQL：`PG*` 环境变量→psycopg 连接。"""
from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row


def pg_connect_kwargs() -> dict[str, Any]:
    return {
        "host": os.environ.get("PGHOST", "127.0.0.1"),
        "port": os.environ.get("PGPORT", "5432"),
        "dbname": os.environ.get("PGDATABASE", "db_name"),
        "user": os.environ.get("PGUSER", "db_user"),
        "password": os.environ.get("PGPASSWORD", "db_password"),
    }


def connect_pg(*, dict_rows: bool = False) -> Connection:
    kw = pg_connect_kwargs()
    conn = psycopg.connect(**kw)
    if dict_rows:
        conn.row_factory = dict_row  # type: ignore[assignment]
    return conn
