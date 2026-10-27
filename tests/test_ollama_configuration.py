"""Pruebas offline de configuración/fábrica (sin Ollama ni LangChain instalados)."""

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import Settings, LLMProvider
from utils.llm_factory import LLMFactory
from config.settings import settings


def test_ollama_provider_requires_no_api_keys(monkeypatch):
    monkeypatch.setattr(Settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(Settings, "OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr(Settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(Settings, "OLLAMA_NUM_CTX", 8192)
    monkeypatch.setattr(Settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(Settings, "GOOGLE_API_KEY", "")
    Settings.validate()


def test_factory_uses_chato_llama(monkeypatch):
    captured = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_ollama", types.SimpleNamespace(ChatOllama=FakeChatOllama))
    monkeypatch.setattr(settings, "LLM_PROVIDER", LLMProvider.OLLAMA.value)
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "OLLAMA_NUM_CTX", 8192)
    llm = LLMFactory.create_llm()
    assert isinstance(llm, FakeChatOllama)
    assert captured["model"] == "qwen3:4b"
    assert captured["num_ctx"] == 8192
    assert captured["reasoning"] is False


def test_invalid_provider(monkeypatch):
    import pytest

    monkeypatch.setattr(Settings, "LLM_PROVIDER", "otro")
    with pytest.raises(ValueError, match="Proveedor LLM inválido"):
        Settings.validate()
