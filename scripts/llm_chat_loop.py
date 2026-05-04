"""多轮对话 REPL：保留用户与助手的上下文，走 OpenAI 兼容 Chat API。

内置默认 ``system`` 提示（可按需改 ``DEFAULT_SYSTEM_PROMPT``）。

单次提问见 ``llm_chat_once.py``。

用法（项目根、已配置 ``.env``）::

    uv run python scripts/llm_chat_loop.py

命令：``/quit`` 或 ``/exit`` 退出；``/clear`` 清空本轮上下文（默认保留 ``system``）。

命令行可用 ``--system`` 覆盖默认系统提示；传空字符串 ``--system ""`` 则不带 system。
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock

from iskra_engine.llm import build_openai_compatible_llm


DEFAULT_SYSTEM_PROMPT = "你叫 Iskra，是马列主义专家，请尽量用中文回答。"

def _text_from_message(message: ChatMessage) -> str:
    return "".join(
        b.text for b in message.blocks if isinstance(b, TextBlock)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI 兼容：多轮对话（带上下文）")
    parser.add_argument(
        "--system",
        default=None,
        metavar="TEXT",
        help="覆盖内置默认系统提示；传空字符串则关闭 system 消息",
    )
    args = parser.parse_args()

    try:
        llm = build_openai_compatible_llm()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    history: List[ChatMessage] = []
    if args.system is None:
        system_text = DEFAULT_SYSTEM_PROMPT.strip()
    else:
        system_text = args.system.strip()
    if system_text:
        history.append(
            ChatMessage(role=MessageRole.SYSTEM, content=system_text)
        )

    print("多轮对话（输入 /quit 退出，/clear 清空上下文）")
    while True:
        try:
            raw = input("\n你> ")
        except EOFError:
            print()
            break

        user_line = raw.strip()
        if user_line.lower() in ("/quit", "/exit", "quit", "exit"):
            break
        if user_line == "/clear":
            history = [
                m
                for m in history
                if m.role == MessageRole.SYSTEM
            ]
            print("（已清空对话，仅保留 system 提示如有）")
            continue
        if not user_line:
            continue

        history.append(ChatMessage(role=MessageRole.USER, content=user_line))
        resp = llm.chat(history)
        assistant_msg = resp.message
        history.append(assistant_msg)

        body = _text_from_message(assistant_msg)
        print(f"\n助手> {body}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
