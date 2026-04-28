"""
加载 Jina Q8_0 GGUF，试跑一条 embedding，确认 dim=1024。

部署目标：Linux x86_64 CPU；需 .env 中 ISKRA_GGUF_PATH 与 ISKRA_N_GPU_LAYERS（生产用 0）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EXPECTED_DIM = 1024


def _drop_stale_cuda_path() -> None:
    """卸载 CUDA 后常残留 CUDA_PATH；llama_cpp 会在 import 时对 CUDA_PATH\\bin 调用 add_dll_directory，路径不存在即失败。纯 CPU 可不设这些变量。"""
    keys = [k for k in os.environ if k.upper().startswith("CUDA_PATH")]
    for k in keys:
        base = os.environ[k]
        bin_dir = os.path.join(base, "bin")
        if base and not os.path.isdir(bin_dir):
            del os.environ[k]


_drop_stale_cuda_path()


def main() -> None:
    gguf = os.environ.get("ISKRA_GGUF_PATH", "").strip()
    if not gguf:
        print("请设置环境变量 ISKRA_GGUF_PATH 为 .gguf 文件的绝对路径", file=sys.stderr)
        sys.exit(1)
    p = Path(gguf)
    if not p.is_file():
        print(f"文件不存在: {p}", file=sys.stderr)
        sys.exit(1)

    n_gpu = int(os.environ.get("ISKRA_N_GPU_LAYERS", "0"))

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(p),
        embedding=True,
        n_ctx=8192,
        n_batch=512,
        verbose=False,
        n_gpu_layers=n_gpu,
    )
    # 与 Jina retrieval 习惯一致：索引侧用 Document: 前缀（以官方文档为准）
    text = "Document: 这是一条测试文本，用于确认向量维度。"
    out = llm.embed(text)
    if not out:
        print("embed 返回空", file=sys.stderr)
        sys.exit(1)
    # 0.3.x 有的返回 list[float]（一维），有的 list[list[float]]；勿对一维再取 [0] 成标量
    first = out[0]
    vec = first if isinstance(first, (list, tuple)) else out
    if not vec:
        print("embed 向量为空", file=sys.stderr)
        sys.exit(1)
    dim = len(vec)
    print(f"OK  dim={dim}  n_gpu_layers={n_gpu}")
    if dim != EXPECTED_DIM:
        print(
            f"警告: 期望 {EXPECTED_DIM} 维（jina-v5-text-small-retrieval），当前 {dim}",
            file=sys.stderr,
        )
        sys.exit(2)
    print("前 5 维:", vec[:5])


if __name__ == "__main__":
    main()
