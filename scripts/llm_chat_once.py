"""单次提问：跑一次 ``ChatLlm.query``（需项目根 ``.env``：``API_KEY`` 等）。

多轮对话见同目录 ``llm_chat_loop.py``。
"""

from __future__ import annotations

import argparse
import sys

from iskra_engine.llm import ChatLlm


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenAI 兼容 API：单次 query（与 README 示例一致）",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="你好，你是谁？一句话说明你能做什么？",
        help="发给模型的文本；省略则用默认占位句（你好，你是谁？一句话说明你能做什么？）",
    )
    args = parser.parse_args()
    try:
        llm = ChatLlm()
    except ValueError as e:
        print(f"{e}", file=sys.stderr)
        return 1
    response = llm.query(args.question)
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
