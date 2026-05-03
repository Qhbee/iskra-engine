from fastapi import FastAPI

app = FastAPI(
    title="iskra-engine",
    version="0.1.0",
    description=(
        "Iskra 的智能引擎，通过对内的 HTTP API 提供 RAG 服务："
        "基于 PostgreSQL/pgvector 的文档上下文检索 + 包装提示词 LLM 对话调用 + 应答生成接口；"
        "不负责对话存储与多端加载等外层 Web 服务。"
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
