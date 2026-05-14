from __future__ import annotations

import logging
from typing import cast

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import MetadataMode

from iskra_engine.api.schemas.rag import RagQueryRequest, RagQueryResponse, RagSourceItem
from iskra_engine.query.query_engine import build_rag_query_engine

log = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["rag"])


@router.post("/rag/query", response_model=RagQueryResponse)
def rag_query(body: RagQueryRequest) -> RagQueryResponse:
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query 不能为空")
    load_dotenv()

    def _run() -> RagQueryResponse:
        engine = build_rag_query_engine(
            top_k=body.top_k,
            path_prefix=body.path_prefix,
            model=body.model.strip() if body.model and body.model.strip() else None,
            load_dotenv_=False,
        )
        llama_index_response = cast(Response, engine.query(q))
        return _to_rag_query_response(llama_index_response)

    try:
        return _run()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        log.exception("RAG 请求失败")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _to_rag_query_response(llama_index_response: Response) -> RagQueryResponse:
    ans = (llama_index_response.response or "").strip()
    out: list[RagSourceItem] = []
    for sn in llama_index_response.source_nodes:
        md = sn.node.metadata or {}
        # ``MetadataMode.LLM`` 会把 metadata 拼进正文（rel_path、embedding_distance 等），
        # 与 DB ``chunk.text`` 重复且易被误认为「多列拼成一行」。API 的 snippet 只要纯文本。
        text = sn.node.get_content(metadata_mode=MetadataMode.NONE)
        out.append(
            RagSourceItem(
                rel_path=md.get("rel_path"),
                title=md.get("title"),
                book=md.get("book"),
                chunk_index=md.get("chunk_index"),
                chunk_id=md.get("chunk_id"),
                score=float(sn.score) if sn.score is not None else None,
                snippet=text[:800] if isinstance(text, str) else str(text)[:800],
            )
        )
    return RagQueryResponse(answer=ans, sources=out)
