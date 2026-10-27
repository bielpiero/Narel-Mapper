"""Fábrica de proveedores LLM para NAREL-Mapper."""

from config.settings import LLMProvider, settings


class LLMFactory:
    """Selecciona un cliente de modelo a partir de la configuración."""

    @staticmethod
    def create_llm():
        provider = settings.LLM_PROVIDER

        if provider == LLMProvider.OLLAMA:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                num_ctx=settings.OLLAMA_NUM_CTX,
                reasoning=False,  # Qwen3: deshabilita el modo de razonamiento extenso.
            )

        if provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                api_key=settings.OPENAI_API_KEY,
            )

        if provider == LLMProvider.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                google_api_key=settings.GOOGLE_API_KEY,
            )

        raise ValueError(
            f"Proveedor LLM no soportado: {provider!r}. "
            f"Valores válidos: {[p.value for p in LLMProvider]}"
        )
