"""通过 LlamaIndex 的 OpenAI 兼容 LLM 做简单对话封装。"""

from iskra_engine.llm.client import ChatLlm, OpenAICompatibleAny, build_openai_compatible_llm

__all__ = ["ChatLlm", "OpenAICompatibleAny", "build_openai_compatible_llm"]
