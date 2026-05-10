"""只验证向量召回：``PgVectorRetriever.retrieve`` → 打印 ``rel_path`` / ``chunk_index`` / ``score``。

不调用大模型，不写 ``RetrieverQueryEngine``。

默认 stdout 输出带缩进的 JSON；``top_k`` 未在命令行指定时，优先用下方脚本常量，再读环境变量 ``RETRIEVAL_TOP_K``。
未写命令行的 ``query`` / ``--path-prefix`` 时，先用脚本内 ``DEFAULT_QUERY`` / ``DEFAULT_PATH_PREFIX``（strip 后）；
仍为空则控制台交互（前缀回车表示全库）。

用法（项目根、已配置 ``.env`` 的 ``PG*``、``ISKRA_GGUF_PATH`` 等）::

    uv run python scripts/probe_vector_retrieval.py
    uv run python scripts/probe_vector_retrieval.py "你的问句"
    uv run python scripts/probe_vector_retrieval.py "问句" --top-k 10 --path-prefix books/vol1
    uv run python scripts/probe_vector_retrieval.py --no-json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from iskra_engine.embeddings.gguf_llama_embedding import GgufEmbedding
from iskra_engine.retrieval.pg_vector_retriever import PgVectorRetriever

# ----- 按需修改：脚本内默认（CLI 可覆盖 top-k / 路径 / 输出格式）-----
DEFAULT_OUTPUT_JSON = True
# 未传 --top-k 时：先用此值；若为 None 则读环境变量 RETRIEVAL_TOP_K，再缺省 8
DEFAULT_TOP_K: int | None = 8
# 未传检索问句 positional 时：先用此字符串（strip 后）；仍为空则控制台询问
DEFAULT_QUERY = ""
# 未传 --path-prefix 时：先用此字符串（strip 后）；仍为空则控制台询问
DEFAULT_PATH_PREFIX = ""


def _resolve_top_k(cli_top_k: int | None) -> int:
    if cli_top_k is not None:
        return cli_top_k
    if DEFAULT_TOP_K is not None:
        return int(DEFAULT_TOP_K)
    return int(os.environ.get("RETRIEVAL_TOP_K", "8"))


def _resolve_path_prefix(cli_prefix: str | None) -> str | None:
    if cli_prefix is not None:
        raw = cli_prefix
    else:
        raw = DEFAULT_PATH_PREFIX
    s = (raw or "").strip()
    if s:
        return s
    line = input("路径前缀（可选，回车表示全库）: ").strip()
    return line or None


def _resolve_query(cli_query: str | None) -> str:
    if cli_query is not None:
        raw = cli_query
    else:
        raw = DEFAULT_QUERY
    s = (raw or "").strip()
    if s:
        return s
    line = input("查询: ").strip()
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description="仅向量检索 Top-K，不调 LLM")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="检索问句；省略则用脚本 DEFAULT_QUERY，strip 后仍空则交互输入",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--path-prefix",
        type=str,
        default=None,
        help="document.rel_path 前缀；省略则用脚本 DEFAULT_PATH_PREFIX，仍空则交互输入",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="人类可读多行输出（默认 JSON）",
    )
    args = parser.parse_args()

    load_dotenv()

    q = _resolve_query(args.query)
    if not q:
        print("query 不能为空", file=sys.stderr)
        return 1

    top_k = _resolve_top_k(args.top_k)
    path_prefix = _resolve_path_prefix(args.path_prefix)
    use_json = DEFAULT_OUTPUT_JSON and not args.no_json

    try:
        emb = GgufEmbedding()
        retriever = PgVectorRetriever(
            emb,
            top_k=top_k,
            path_prefix=path_prefix,
        )
        nodes = retriever.retrieve(q)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        return 1

    sources = []
    for nws in nodes:
        md = nws.node.metadata or {}
        sources.append(
            {
                "rel_path": md.get("rel_path"),
                "chunk_index": md.get("chunk_index"),
                "score": float(nws.score) if nws.score is not None else None,
                # 纯正文截取（DB chunk.text）；LLM 模式会把 metadata 拼进正文，与 JSON 其它字段重复
                "snippet": (getattr(nws.node, "text", "") or "")[:400],
            }
        )

    if use_json:
        print(
            json.dumps(
                {"query": q, "sources": sources},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"query: {q}\n--- sources ({len(sources)}) ---")
        for i, s in enumerate(sources, 1):
            print(
                f"{i}. score={s['score']!s}  "
                f"rel_path={s['rel_path']!r}  chunk_index={s['chunk_index']}",
            )
            if s["snippet"]:
                print(f"   {s['snippet'][:200]!s}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
