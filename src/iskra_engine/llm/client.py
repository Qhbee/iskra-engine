"""LlamaIndex OpenAI 兼容 LLM：`ChatLlm().query("...")` 即一次补全。"""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import Tokenizer
from llama_index.llms.openai.utils import (
    O1_MODELS,
    is_chat_model,
    is_function_calling_model,
    openai_modelname_to_contextsize,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _fallback_context_window() -> int:
    raw = os.environ.get("CONTEXT_WINDOW", "8192").strip()
    if raw.isdigit():
        return int(raw)
    return 8192


class OpenAICompatibleAny(OpenAI):
    """接入 OpenAI 兼容网关（DeepSeek、硅基流动等）时 ``MODEL`` 可任取；不在 LlamaIndex 内置名单里也能工作。

    - ``metadata``：未知模型用 ``CONTEXT_WINDOW``（默认 8192）并默认定位 **Chat Completions**。
    - ``_tokenizer``：tiktoken 无法识别模型名时不做精确计数字典，避免 KeyError。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 转发到父类，避免 IDE 误判「子类不接受 model/api_key/temperature」等关键字。
        super().__init__(*args, **kwargs)

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        try:
            return super()._tokenizer
        except (KeyError, ValueError):
            return None
        except Exception:
            return None

    @property
    def metadata(self) -> LLMMetadata:
        name = self._get_model_name()
        try:
            cw = openai_modelname_to_contextsize(name)
            in_openai_registry = True
        except ValueError:
            cw = _fallback_context_window()
            in_openai_registry = False

        if in_openai_registry:
            chat = is_chat_model(model=name)
            fc = is_function_calling_model(model=name)
        else:
            chat = _env_bool("LLM_USE_CHAT_COMPLETIONS", True)
            fc = _env_bool("LLM_FUNCTION_CALLING", True)

        return LLMMetadata(
            context_window=cw,
            num_output=self.max_tokens or -1,
            is_chat_model=chat,
            is_function_calling_model=fc,
            model_name=self.model,
            system_role=MessageRole.USER
            if self.model in O1_MODELS
            else MessageRole.SYSTEM,
        )


def build_openai_compatible_llm(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.2,
) -> OpenAICompatibleAny:
    """组装 LlamaIndex 的 OpenAI 兼容 LLM（可换 ``BASE_URL``、任意 ``MODEL`` 名）。"""

    load_dotenv()
    key = api_key if api_key is not None else os.environ.get("API_KEY")
    if not key or not str(key).strip():
        msg = "未设置 API_KEY，无法直连大模型 HTTP API"
        raise ValueError(msg)
    raw_base = api_base if api_base is not None else os.environ.get("BASE_URL")
    base = raw_base.strip() if isinstance(raw_base, str) and raw_base.strip() else None
    m = model if model is not None else os.environ.get("MODEL", "gpt-4o-mini")
    return OpenAICompatibleAny(
        model=m,
        api_key=key,
        api_base=base,
        temperature=temperature,
    )


class ChatLlm:
    """薄封装：对外只暴露 ``query``，底层用 LlamaIndex 的 ``OpenAI.complete``。"""

    def __init__(self, llm: OpenAI | None = None) -> None:
        self._llm = llm if llm is not None else build_openai_compatible_llm()

    def query(self, question: str) -> str:
        text = question.strip()
        if not text:
            return ""
        resp = self._llm.complete(text)
        return resp.text or ""
