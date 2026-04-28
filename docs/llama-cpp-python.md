# llama-cpp-python（本仓库）

## 选型

|   项目   | 选择                                                                                                                   |
|:------:|:---------------------------------------------------------------------------------------------------------------------|
|  向量化   | Hugging Face：`jinaai/jina-embeddings-v5-text-small-retrieval-GGUF`                                                   |
|   推理   | `llama-cpp-python==0.3.19`，Linux x86_64，CPU（abetlen [cpu 轮子索引](https://abetlen.github.io/llama-cpp-python/whl/cpu)）  |
| Python | **仅 3.12.x**（勿升 3.13+，避免无可靠预编译轮子）                                                                                    |

```text
requires-python = ">=3.12,<3.13"
```

## 部署步骤（Linux）

1. `uv sync`（从 `pyproject.toml` + `[tool.uv].extra-index-url` 拉 **CPU** 预编译 `llama-cpp-python`）。
2. `.env`：**`ISKRA_GGUF_PATH`** = 服务器上 GGUF 绝对路径；**`ISKRA_N_GPU_LAYERS=0`**。
3. `uv run python scripts/smoke_gguf_embed.py` → 期望 **`dim=1024`**。

## 备注

- 最初计划是开发环境 Windows + GPU 离线处理文档，生产环境 Linux + CPU 在线处理查询。
- 但 Windows 上 `llama-cpp-python` 的 CUDA 支持不稳定，且预编译轮子缺乏，导致开发效率极低，各种尝试均失败。
- 决定拆分项目，新建 `iskra-etl` 负责 Windows + CUDA 的离线处理，采用 `sentence-tranformer`；`iskra-engine` 仍采用 `gguf` + `llama-cpp-python`，但只负责 Linux + CPU 的在线处理。
- Windows 下[第三方 GPU wheel](https://github.com/dougeeai/llama-cpp-python-wheels) 安装脚本、源码编译安装脚本已移除，不纳入 `iskra-engine` 仓库。
