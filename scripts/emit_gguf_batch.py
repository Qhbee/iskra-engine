"""从 stdin 或 --input JSON 读取多段文本，用同一 GGUF 会话逐个 embed；stdout 打出 JSON。

供 iskra-etl/scripts/compare_st_gguf_cos.py 单次子进程批量取向量，避免每个样本重复加载权重。

payload: {"texts": ["...", ...]}
stdout: {"dim":1024,"l2_before_norm":[[float],...],"unit_vectors":[[float],...]}

环境：ISKRA_GGUF_PATH（必填）、ISKRA_N_GPU_LAYERS（默认 0）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _l2_normalize(vec: list[float]) -> tuple[list[float], float]:
    sq = sum(x * x for x in vec)
    norm = math.sqrt(sq) if sq > 0.0 else 0.0
    if norm <= 0.0:
        return (list(vec), norm)
    return ([x / norm for x in vec], norm)


def _drop_stale_cuda_path() -> None:
    keys = [k for k in os.environ if k.upper().startswith("CUDA_PATH")]
    for k in keys:
        base = os.environ[k]
        bin_dir = os.path.join(base, "bin")
        if base and not os.path.isdir(bin_dir):
            del os.environ[k]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="UTF-8 JSON，形如 {\"texts\": [\"...\"]}")
    args = parser.parse_args()

    p_in = Path(args.input)
    if not p_in.is_file():
        print(f"--input 不是文件: {p_in}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(p_in.read_text(encoding="utf-8"))
    texts = data.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        print('JSON 顶层需包含 "texts": [字符串, ...]', file=sys.stderr)
        sys.exit(1)

    gguf = os.environ.get("ISKRA_GGUF_PATH", "").strip()
    if not gguf:
        print("ISKRA_GGUF_PATH 未设置", file=sys.stderr)
        sys.exit(1)
    p = Path(gguf)
    if not p.is_file():
        print(f"GGUF 不存在: {p}", file=sys.stderr)
        sys.exit(1)

    n_gpu = int(os.environ.get("ISKRA_N_GPU_LAYERS", "0"))

    _drop_stale_cuda_path()

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(p),
        embedding=True,
        n_ctx=8192,
        n_batch=512,
        verbose=False,
        n_gpu_layers=n_gpu,
    )

    l2_before: list[float] = []
    unit_vectors: list[list[float]] = []
    for text in texts:
        out = llm.embed(text)
        if not out:
            print("embed 返回空", file=sys.stderr)
            sys.exit(1)
        first = out[0]
        vec_raw = list(first if isinstance(first, (list, tuple)) else out)
        if not vec_raw:
            print("embed 向量为空", file=sys.stderr)
            sys.exit(1)
        unit, raw_norm = _l2_normalize(vec_raw)
        l2_before.append(raw_norm)
        unit_vectors.append(unit)

    dim = len(unit_vectors[0]) if unit_vectors else 0
    print(
        json.dumps(
            {
                "dim": dim,
                "l2_before_norm": l2_before,
                "unit_vectors": unit_vectors,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
