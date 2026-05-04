from unittest.mock import MagicMock

from llama_index.core.base.llms.types import CompletionResponse

from iskra_engine.llm import ChatLlm


def test_chat_llm_query_delegates_to_complete() -> None:
    inner = MagicMock()
    inner.complete.return_value = CompletionResponse(text="mocked reply")
    llm = ChatLlm(llm=inner)
    response = llm.query("Some question")
    assert response == "mocked reply"
    inner.complete.assert_called_once()


def test_chat_llm_query_strips_and_empty() -> None:
    inner = MagicMock()
    inner.complete.return_value = CompletionResponse(text="x")
    llm = ChatLlm(llm=inner)
    assert llm.query("   ") == ""
    inner.complete.assert_not_called()


def test_openai_compatible_any_unknown_model_metadata(monkeypatch: object) -> None:
    from llama_index.llms.openai.utils import openai_modelname_to_contextsize

    from iskra_engine.llm.client import OpenAICompatibleAny

    monkeypatch.setenv("CONTEXT_WINDOW", "131072")
    llm = OpenAICompatibleAny(
        model="example-unknown-model",
        api_key="sk-test",
        api_base="https://api.example.invalid",
    )
    assert llm.metadata.is_chat_model is True
    assert llm.metadata.context_window == 131072
    cw_gpt = openai_modelname_to_contextsize("gpt-4o-mini")
    reg = OpenAICompatibleAny(model="gpt-4o-mini", api_key="sk-test")
    assert reg.metadata.context_window == cw_gpt
