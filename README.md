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
