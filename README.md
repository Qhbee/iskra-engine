# iskra-engine

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Active-success)

---

`iskra-engine` 是 `iskra` 的智能引擎 🧠，采用 [iskra-data](https://github.com/Qhbee/iskra-data) 的数据 。

---

## Start

在项目根目录安装可编辑模式后启动：

```bash
uv pip install -e .
python main.py
```

或直接使用 uvicorn：

```bash
uvicorn iskra_engine.api.app:app --reload --host 127.0.0.1 --port 8000
```

健康检查：`GET http://127.0.0.1:8000/health`
API 文档（Swagger UI）：`GET http://127.0.0.1:8000/docs`
API 文档（ReDoc）：`GET http://127.0.0.1:8000/redoc`

### RAG（PostgreSQL + pgvector + LlamaIndex）

**只测向量召回（不调大模型）**：用脚本 [`scripts/probe_vector_retrieval.py`](scripts/probe_vector_retrieval.py)，输出与 `sources` 类似的 `rel_path` / `chunk_index` / `score` / 片段预览。

```bash
uv run python scripts/probe_vector_retrieval.py "你的问题" --json
# 可选：--top-k 10  --path-prefix books/vol1
```

涉及代码：`[src/iskra_engine/db/conn.py](src/iskra_engine/db/conn.py)`、`[src/iskra_engine/embeddings/gguf_embed.py](src/iskra_engine/embeddings/gguf_embed.py)`、`[src/iskra_engine/embeddings/gguf_llama_embedding.py](src/iskra_engine/embeddings/gguf_llama_embedding.py)`、`[src/iskra_engine/retrieval/pg_vector_retriever.py](src/iskra_engine/retrieval/pg_vector_retriever.py)`。

**完整 RAG（检索 + LLM 合成）** 需 `.env` 中配置 `PG*`、`ISKRA_GGUF_PATH`（查询向量，与入库 chunk 向量同空间）、以及合成用 `API_KEY` / `BASE_URL` / `MODEL` 等（见 [.env.example](.env.example) 中 RAG 段）。

起服务后调用（示例）：

```bash
curl -s -X POST http://127.0.0.1:8000/rag/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"你的问题\", \"top_k\": 6}"
```

PowerShell 下也可：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/rag/query -Method POST -ContentType 'application/json' -Body '{"query":"你的问题"}'
```

不拉起 HTTP 时可（**注意**：`uv run` 默认会 **同步虚拟环境**，可能访问 GitHub 校验 [`pyproject.toml`](pyproject.toml) 里的 **llama-cpp-python 直链轮子**，离线或防火墙会失败）：

```bash
uv run --no-sync python scripts/rag_query_once.py "你的问题" --json
```

依赖已在 `.venv` 装好、仅需避免联网校验时：**加 `--no-sync`**；或直接使用解释器（与手动跑 `scripts/smoke_gguf_embed.py` 等价）：

```powershell
.\.venv\Scripts\python.exe scripts/rag_query_once.py "你的问题" --json
```

也可设环境变量 `UV_NO_SYNC=1`，再照常 `uv run python ...`。

**说明**：报错「要下载」多半是 **UV 在为 URL 依赖取元数据/同步**，不一定是 `.venv` 里缺包——你已在该环境里跑通 GGUF 烟测就说明 **llama-cpp-python 已在本地**。

### 大模型配置（LlamaIndex，OpenAI 兼容）

1. 复制 [`.env.example`](.env.example) 为 `.env`，按厂商文档填写 `API_KEY`，按需填 `BASE_URL`、`MODEL`。若 ``MODEL`` 为 DeepSeek 等**非 OpenAI 官方名称**，在 `.env` 设置 ``CONTEXT_WINDOW``（例如厂商文档里的上下文长度），见 [.env.example](.env.example)。 
2. 仓库根目录**已可编辑安装**后，在项目根执行：

单次提问

```bash
uv run python scripts/llm_chat_once.py "你好，用一句话说明你能做什么"
```

多轮对话（REPL，带上下文；**脚本内默认自带系统提示词**）：

```bash
uv run python scripts/llm_chat_loop.py
# 覆盖：`--system "别的提示"`；不要 system：`--system ""`
```

### 开发依赖与测试：
`uv sync --extra dev`，再 `uv run pytest`（单测 mock，不访问外网）。
