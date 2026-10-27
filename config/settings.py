"""Configuración global de NAREL-Mapper (OpenAI, Gemini u Ollama local)."""

import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class Settings:
    """Opciones de configuración, leídas del .env al iniciar el programa."""

    # LLM: Ollama es el valor por defecto; los proveedores comerciales siguen disponibles.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", LLMProvider.OLLAMA.value).strip().lower()
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    # Google Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Ollama: servidor local; no se necesita ninguna API key.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "6144"))

    # Conversation & Memory
    MAX_MEMORY_MESSAGES: int = int(os.getenv("MAX_MEMORY_MESSAGES", "6"))
    MAX_CONTEXT_MESSAGES: int = int(os.getenv("MAX_CONTEXT_MESSAGES", "2"))

    # System
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    @classmethod
    def validate(cls) -> None:
        """Comprueba proveedor y credenciales; Ollama no requiere clave."""
        valid_providers = [p.value for p in LLMProvider]
        if cls.LLM_PROVIDER not in valid_providers:
            raise ValueError(
                f"Proveedor LLM inválido: {cls.LLM_PROVIDER!r}. "
                f"Valores válidos: {valid_providers}"
            )
        if cls.LLM_PROVIDER == LLMProvider.OPENAI and not cls.OPENAI_API_KEY:
            raise ValueError("Faltan credenciales: OPENAI_API_KEY no configurada.")
        if cls.LLM_PROVIDER == LLMProvider.GEMINI and not cls.GOOGLE_API_KEY:
            raise ValueError("Faltan credenciales: GOOGLE_API_KEY no configurada.")
        if cls.LLM_PROVIDER == LLMProvider.OLLAMA:
            if not cls.OLLAMA_MODEL.strip():
                raise ValueError("OLLAMA_MODEL no puede estar vacío.")
            if not cls.OLLAMA_BASE_URL.startswith(("http://", "https://")):
                raise ValueError("OLLAMA_BASE_URL debe empezar por http:// o https://.")
            if cls.OLLAMA_NUM_CTX <= 0:
                raise ValueError("OLLAMA_NUM_CTX debe ser un entero positivo.")


settings = Settings()
