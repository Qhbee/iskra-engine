"""命令行跑一次 RAG（需 ``.env``：``PG*``、``API_KEY``、``ISKRA_GGUF_PATH`` 等）。"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from iskra_engine.query.query_engine import build_rag_query_engine


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 单次提问（RetrieverQueryEngine）")
    parser.add_argument("query", help="问题")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--path-prefix", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="整段 JSON 打到 stdout")
    args = parser.parse_args()

    load_dotenv()
    try:
        engine = build_rag_query_engine(
            top_k=args.top_k,
            path_prefix=args.path_prefix,
            load_dotenv_=False,
        )
        resp = engine.query(args.query.strip())
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "answer": resp.response,
            "sources": [
                {
                    "rel_path": (s.node.metadata or {}).get("rel_path"),
                    "chunk_index": (s.node.metadata or {}).get("chunk_index"),
                    "score": float(s.score) if s.score is not None else None,
                }
                for s in resp.source_nodes
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(resp.response or "")
        print("\n--- sources ---")
        for s in resp.source_nodes:
            md = s.node.metadata or {}
            print(
                f"[{s.score:.4f}] {md.get('rel_path')} "
                f"#idx{md.get('chunk_index')}\n{(s.node.text or '')[:200]}...",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
