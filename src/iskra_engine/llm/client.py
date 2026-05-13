"""LlamaIndex OpenAI 兼容 LLM：`ChatLlm().query("...")` 即一次补全。"""

from __future__ import annotations

import os
from typing import Any, Optional, override

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


# 厂商前缀（与前端 ``provider/model`` 对齐）→ 仅读**该厂商**下列出的环境变量（不再回退通用 ``API_KEY`` / ``BASE_URL``）。
_VENDOR_API_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY", "GPT_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "xiaomi": ("XIAOMI_API_KEY", "MIMO_API_KEY"),
    "zhipu": ("ZHIPU_API_KEY", "BIGMODEL_API_KEY"),
    "qwen": ("ALI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"),
}

_VENDOR_API_BASES: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_BASE_URL", "GPT_BASE_URL"),
    "anthropic": ("ANTHROPIC_BASE_URL", "CLAUDE_BASE_URL"),
    "google": ("GOOGLE_BASE_URL", "GEMINI_BASE_URL"),
    "deepseek": ("DEEPSEEK_BASE_URL",),
    "moonshot": ("MOONSHOT_BASE_URL", "KIMI_BASE_URL"),
    "xiaomi": ("XIAOMI_BASE_URL", "MIMO_BASE_URL"),
    "zhipu": ("ZHIPU_BASE_URL", "BIGMODEL_BASE_URL"),
    "qwen": ("ALI_BASE_URL", "DASHSCOPE_BASE_URL", "QWEN_BASE_URL"),
}


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


def _first_nonempty_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if not _is_blank_str(raw):
            return str(raw).strip()
    return None


def _is_blank_str(s: str | None) -> bool:
    """``True`` 当且仅当 ``s`` 为 ``None`` 或 ``strip()`` 后为空。"""
    return s is None or not bool(s.strip())


def _parse_vendor_prefixed_model(model_raw: str) -> tuple[str | None, str]:
    """``厂商/模型`` → ``(厂商小写, 模型 id)``；无法解析则 ``(None, 原串)``。

    使用 ``str.partition``：返回 ``(厂商段, 分隔符, 模型名段)``；分隔符在命中时恒为 ``"/"``。
    """
    if "/" not in model_raw:
        return None, model_raw
    vendor, separator, model_id = model_raw.partition("/")
    assert separator == "/"
    vendor = vendor.strip().lower()
    model_id = model_id.strip()
    if not vendor or not model_id:
        return None, model_raw
    return vendor, model_id


def _reject_explicit_blank(name: str, value: str | None) -> None:
    if value is None:
        return
    if not str(value).strip():
        raise ValueError(f"参数 {name} 若传入则不得为空串或仅空白")


def _require_resolved_vendor_model(raw_model: str) -> tuple[str, str]:
    """校验模型字符串为「已登记厂商/模型」，返回 ``(厂商, 发给 HTTP 的 model id)``。"""
    s = (raw_model or "").strip()
    if not s:
        msg = "模型标识不能为空；须为「厂商/模型」，例如 xiaomi/mimo-v2.5"
        raise ValueError(msg)
    vendor, model_id = _parse_vendor_prefixed_model(s)
    if vendor is None or vendor not in _VENDOR_API_KEYS:
        allowed = ", ".join(sorted(_VENDOR_API_KEYS))
        msg = f"模型标识须为「支持的厂商/模型」（当前为 {s!r}）。支持厂商前缀: {allowed}"
        raise ValueError(msg)
    if not model_id.strip():
        msg = f"模型标识中「/」右侧模型名不能为空（当前为 {s!r}）"
        raise ValueError(msg)
    return vendor, model_id.strip()


class OpenAICompatibleAny(OpenAI):
    """接入 OpenAI 兼容网关（DeepSeek、硅基流动等）时 model 可任取；不在 LlamaIndex 内置名单里也能工作。

    - ``metadata``：未知模型用 ``CONTEXT_WINDOW``（默认 8192）并默认定位 **Chat Completions**。
    - ``_tokenizer``：tiktoken 无法识别模型名时不做精确计数字典，避免 KeyError。
    """

    @override
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 转发到父类，避免 IDE 误判「子类不接受 model/api_key/temperature」等关键字。
        super().__init__(*args, **kwargs)

    @property
    @override
    def _tokenizer(self) -> Optional[Tokenizer]:
        try:
            return super()._tokenizer
        except (KeyError, ValueError):
            return None
        except Exception:
            return None

    @property
    @override
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
    """组装 LlamaIndex 的 OpenAI 兼容 LLM（可换参数）。

    - **模型**：须为 ``{厂商}/{模型 id}``（如 ``openai/gpt-4o-mini``）。显式传参 ``model`` 优先；若为 None 则读环境变量 ``DEFAULT_MODEL``。发给 HTTP 的 ``model`` 仅为 / 右侧的模型 id。
    - **密钥**：未显式传入 ``api_key``（None 视为未传）时，只在该厂商对应的环境变量中解析（见 ``_VENDOR_API_KEYS``），不再读取通用 ``API_KEY``。在无环境变量且无显式传入时，视为配置错误并抛出异常。
    - **链接**：未显式传入 ``api_base``（None 视为未传）时，只在该厂商对应的环境变量中解析（见 ``_VENDOR_API_BASES``），不再读取通用 ``BASE_URL``；若仍无值，则 ``api_base`` 可为 ``None``。厂商为 openai 时允许始终为 None（交由 LlamaIndex / 官方默认端点）；其余厂商在无环境变量且无显式传入时，视为配置错误并抛出异常。
    - **显式参数**：若传入 ``model`` / ``api_key`` / ``api_base``，则 **不得** 为空串或仅空白（视为非法调用，与「未传None」区分）。
    """

    _reject_explicit_blank("model", model)
    _reject_explicit_blank("api_key", api_key)
    _reject_explicit_blank("api_base", api_base)

    load_dotenv()
    raw_model = str(model).strip() if model is not None else os.environ.get("DEFAULT_MODEL", "openai/gpt-4o-mini")
    if _is_blank_str(raw_model):
        raise ValueError("未设置模型：请配置环境变量 DEFAULT_MODEL，或传入 model=…（须为「厂商/模型」，例如 openai/gpt-4o-mini）")

    vendor, model_id = _require_resolved_vendor_model(raw_model)

    key = str(api_key).strip() if api_key is not None else _first_nonempty_env(_VENDOR_API_KEYS[vendor])
    if _is_blank_str(key):
        raise ValueError(f"未配置厂商「{vendor}」的 API Key，请设置环境变量之一: {", ".join(_VENDOR_API_KEYS[vendor])}，或者显式传参 api_key")

    base = str(api_base).strip() if api_base is not None else _first_nonempty_env(_VENDOR_API_BASES[vendor])
    if _is_blank_str(base) and vendor != "openai":
        raise ValueError(f"未配置厂商「{vendor}」的 API Base Url，请设置环境变量之一: {", ".join(_VENDOR_API_BASES[vendor])}，或者显式传参 api_base")

    return OpenAICompatibleAny(
        model=model_id,
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
