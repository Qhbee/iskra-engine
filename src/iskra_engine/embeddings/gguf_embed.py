"""GGUF 向量：与 [`scripts/smoke_gguf_embed.py`](../../../scripts/smoke_gguf_embed.py) 标尺一致（L2、维数）。"""
from __future__ import annotations

import math
import os
from pathlib import Path
from threading import RLock
from typing import Any, Optional

_llama: Optional[Any] = None
# RLock：可重入，`embed_prefixed` 整段与同线程内的 `_get_llama_embedder` 嵌套获取不会死锁；多线程下串行化 `Llama.embed`。
_lock = RLock()


def _l2_normalize(vec: list[float]) -> list[float]:
    sq = sum(x * x for x in vec)
    norm = math.sqrt(sq) if sq > 0.0 else 0.0
    if norm <= 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _drop_stale_cuda_path() -> None:
    keys = [k for k in os.environ if k.upper().startswith("CUDA_PATH")]
    for k in keys:
        base = os.environ[k]
        bin_dir = os.path.join(base, "bin") if base else ""
        if base and not os.path.isdir(bin_dir):
            del os.environ[k]


def _get_llama_embedder():
    """懒加载 Llama(embedding=True)。"""
    global _llama
    with _lock:
        if _llama is None:
            _drop_stale_cuda_path()
            from llama_cpp import Llama  # noqa: PLC0415

            gguf = os.environ.get("ISKRA_GGUF_PATH", "").strip()
            if not gguf:
                msg = "未设置 ISKRA_GGUF_PATH，无法 GGUF embed"
                raise ValueError(msg)
            p = Path(gguf)
            if not p.is_file():
                msg = f"GGUF 不存在: {p}"
                raise FileNotFoundError(msg)
            n_gpu = int(os.environ.get("ISKRA_N_GPU_LAYERS", "0"))
            _llama = Llama(
                model_path=str(p),
                embedding=True,
                n_ctx=8192,
                n_batch=512,
                verbose=False,
                n_gpu_layers=n_gpu,
            )
        return _llama


def embed_prefixed(prompt: str) -> list[float]:
    """单行文本（已含前缀）→ L2 单位向量列表。

    与进程内全局单例 ``Llama`` 共享；多线程（如 FastAPI sync 路由线程池）下对 ``embed`` 串行化以保证安全。
    """
    with _lock:
        llm = _get_llama_embedder()
        out = llm.embed(prompt)
        if not out:
            msg = "embed 返回空"
            raise RuntimeError(msg)
        first = out[0]
        vec = list(first if isinstance(first, (list, tuple)) else out)
        if len(vec) == 0:
            msg = "向量为空"
            raise RuntimeError(msg)
        expect = int(os.environ.get("ISKRA_EMBED_DIM", "1024"))
        if len(vec) != expect:
            msg = f"向量维数期望 {expect}，实际 {len(vec)}"
            raise ValueError(msg)
        return _l2_normalize(vec)


def embed_query(query: str) -> list[float]:
    """检索问句：前缀默认 ``Query: ``。"""
    q = query.strip()
    prefix = os.environ.get("ISKRA_QUERY_PREFIX", "Query: ").strip()
    if not prefix.endswith(" ") and prefix:
        prefix = f"{prefix} "
    prompt = f"{prefix}{q}" if q else prefix.strip()
    return embed_prefixed(prompt)


def embed_document(text: str) -> list[float]:
    """索引/文档侧文本（通常即 chunk 正文）：前缀默认 ``Document: ``，与离线索引入库习惯一致。"""
    t = text.strip()
    prefix = os.environ.get("ISKRA_DOCUMENT_PREFIX", "Document: ").strip()
    if not prefix.endswith(" ") and prefix:
        prefix = f"{prefix} "
    prompt = f"{prefix}{t}" if t else prefix.strip()
    return embed_prefixed(prompt)


def warmup_gguf() -> None:
    """启动时预热：装载权重。"""
    _get_llama_embedder()
