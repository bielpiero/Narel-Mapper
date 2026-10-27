"""Prueba real del prompt + Pydantic usando Ollama (requiere servidor local)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from graph.map_graph import get_llm
from models.semantic_models import AgentResponse
from prompts.map_prompts import get_agent_prompt


def main():
    if settings.LLM_PROVIDER != "ollama":
        raise SystemExit("Configura LLM_PROVIDER=ollama en .env")

    print(f"Proveedor: {settings.LLM_PROVIDER}")
    print(f"Modelo:    {settings.OLLAMA_MODEL}")
    print(f"Servidor:  {settings.OLLAMA_BASE_URL}")
    chain = get_agent_prompt() | get_llm()
    result = chain.invoke(
        {
            "current_map": {},
            "chat_history": "No history.",
            "message": "Crea el espacio 'Casa' de 10 por 8 metros. No añadas habitaciones todavía.",
            "overlap_warning": "",
        }
    )
    if not isinstance(result, AgentResponse):
        raise TypeError(f"Se esperaba AgentResponse; recibido {type(result).__name__}")
    print("\nSalida estructurada Pydantic válida:")
    print(result.model_dump_json(indent=2))
    if result.updated_map.global_space is None:
        raise AssertionError("El modelo no construyó global_space. Revisar prompt/modelo.")
    print("\nSMOKE TEST: OK (esquema y global_space)")
    print("AVISO: este test NO valida las restricciones geométricas del backend.")


if __name__ == "__main__":
    main()
