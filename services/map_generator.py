# services/map_generator.py
import copy
from graph.map_graph import create_map_graph
from utils.logger import setup_logger

logger = setup_logger(__name__)

_graph = None
_graph_state = {
    "user_input": "",
    "semantic_map": None,
    "agent_response": None,
    "messages": [],
    "conflict_rooms": [],
    "overlapping_rooms": [],
}
_history: list = []  # pila undo
_redo_stack: list = []  # pila redo


def get_runtime_snapshot() -> dict:
    """Exporta el estado runtime para persistirlo en st.session_state."""
    return {
        "graph_state": copy.deepcopy(_graph_state),
        "history": copy.deepcopy(_history),
        "redo_stack": copy.deepcopy(_redo_stack),
    }


def restore_runtime_snapshot(snapshot: dict | None):
    """Restaura el estado runtime previamente exportado."""
    global _graph_state, _history, _redo_stack, _graph
    if not snapshot:
        return

    _graph_state = copy.deepcopy(snapshot.get("graph_state") or {
        "user_input": "",
        "semantic_map": None,
        "agent_response": None,
        "messages": [],
        "conflict_rooms": [],
    })
    _history = copy.deepcopy(snapshot.get("history") or [])
    _redo_stack = copy.deepcopy(snapshot.get("redo_stack") or [])

    # La restauración no cambia la estructura del grafo. Evitamos recrearlo en cada rerun.


def get_graph():
    global _graph
    if _graph is None:
        logger.info("🔧 Creando grafo de LangGraph...")
        _graph = create_map_graph()
        logger.info("✅ Grafo creado")
    return _graph


def process_message(user_message: str) -> dict:
    """
    Procesa un mensaje del usuario a través del grafo.
    Returns: Dict con 'agent_response' y 'semantic_map'
    """
    global _graph_state, _history, _redo_stack

    if not user_message or not user_message.strip():
        return {"agent_response": "Por favor, envía un mensaje válido.", "semantic_map": _graph_state["semantic_map"]}

    # Guardar snapshot antes de modificar y limpiar redo al tomar un camino nuevo
    _history.append(copy.deepcopy(_graph_state))
    _redo_stack.clear()

    _graph_state["user_input"] = user_message

    try:
        _graph_state = get_graph().invoke(_graph_state)
        logger.info("✅ Mensaje procesado")
        return {
            "agent_response": _graph_state["agent_response"],
            "semantic_map": _graph_state["semantic_map"]
        }
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        _history.pop()  # descartar snapshot si falló
        _redo_stack.clear()
        return {
            "agent_response": f"⚠️ Error: {str(e)}",
            "semantic_map": _graph_state["semantic_map"]
        }


def undo_state() -> dict | None:
    """Restaura el estado anterior. Mueve el estado actual a la pila redo."""
    global _graph_state, _history, _redo_stack
    if not _history:
        return None
    _redo_stack.append(copy.deepcopy(_graph_state))
    _graph_state = _history.pop()
    logger.info("↩️ Estado restaurado (undo)")
    return _graph_state["semantic_map"]


def redo_state() -> dict | None:
    """Avanza al estado siguiente. Mueve el estado actual a la pila undo."""
    global _graph_state, _history, _redo_stack
    if not _redo_stack:
        return None
    _history.append(copy.deepcopy(_graph_state))
    _graph_state = _redo_stack.pop()
    logger.info("↪️ Estado avanzado (redo)")
    return _graph_state["semantic_map"]


def can_undo() -> bool:
    return len(_history) > 0


def can_redo() -> bool:
    return len(_redo_stack) > 0


def reset_state():
    """Reinicia el estado del grafo."""
    global _graph_state, _graph, _history, _redo_stack
    _graph = None
    _history = []
    _redo_stack = []
    _graph_state = {
        "user_input": "",
        "semantic_map": None,
        "agent_response": None,
        "messages": [],
        "conflict_rooms": [],
        "overlapping_rooms": [],
    }
    logger.info("🔄 Estado reiniciado")


def load_local_demo_map() -> dict:
    """Load a deterministic test map for visual debugging; no API/Ollama request."""
    global _graph_state, _history, _redo_stack
    example = {
        "global_space": {"name": "Demo de anidación", "width": 12, "height": 10},
        "rooms": [
            {"name": "Dormitorio", "position_x": 1, "position_y": 1, "width": 6, "height": 5,
             "nested_rooms": [{"name": "Baño", "position_x": 3, "position_y": 1,
                                "width": 2, "height": 2}]},
            {"name": "Cocina", "position_x": 7, "position_y": 1, "width": 3, "height": 5},
        ],
        "connections": [
            {"origin_room": "Dormitorio", "destination_room": "Cocina", "width": 1.0, "wall_position": "east"},
            {"origin_room": "Baño", "destination_room": "Dormitorio", "width": 0.9, "wall_position": "north"},
        ],
    }
    from models.semantic_models import SemanticMap
    validated = SemanticMap.model_validate(example).model_dump()
    _graph_state = {"user_input": "", "semantic_map": validated, "agent_response": None,
                    "messages": [], "conflict_rooms": [], "overlapping_rooms": []}
    _history = []
    _redo_stack = []
    return validated
