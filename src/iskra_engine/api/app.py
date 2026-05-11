from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from iskra_engine.api.routers import health, rag

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_dotenv()
    try:
        from iskra_engine.embeddings.gguf_embed import warmup_gguf  # noqa: PLC0415

        await asyncio.to_thread(warmup_gguf)
    except Exception as e:
        log.warning("GGUF 预热未执行（请检查 ISKRA_GGUF_PATH）；首次 RAG 请求再加载: %s", e)
    yield


app = FastAPI(
    title="iskra-engine",
    version="0.1.0",
    description=(
        "Iskra 的智能引擎，通过对内的 HTTP API 提供 RAG 服务："
        "基于 PostgreSQL/pgvector 的文档上下文检索 + 包装提示词 LLM 对话调用 + 应答生成接口；"
        "不负责对话存储与多端加载等外层 Web 服务。"
    ),
    lifespan=lifespan,
)


app.include_router(health.router)
app.include_router(rag.router)
