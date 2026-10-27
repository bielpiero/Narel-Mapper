# ui/streamlit_ui.py
import io
import json
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
import streamlit as st
from models.semantic_models import SemanticMap
from services.map_generator import (
    process_message,
    reset_state,
    undo_state,
    redo_state,
    can_undo,
    can_redo,
    get_runtime_snapshot,
    restore_runtime_snapshot,
)
from services.map_visualizer import generate_floor_plan
from services.map_generator import load_local_demo_map


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hola! Soy tu asistente de mapas semanticos. Para empezar, dime el nombre del espacio que quieres modelar (por ejemplo, \"Casa de la abuela\") y sus dimensiones totales en metros (ancho x alto)."
        }]
    if "current_map" not in st.session_state:
        st.session_state.current_map = None
    if "messages_history" not in st.session_state:
        st.session_state.messages_history = []  # pila undo de mensajes del chat
    if "messages_redo" not in st.session_state:
        st.session_state.messages_redo = []  # pila redo de mensajes del chat
    if "graph_runtime_snapshot" not in st.session_state:
        st.session_state.graph_runtime_snapshot = None


def _restore_backend_runtime():
    """Sincroniza estado backend desde session_state en cada rerun."""
    restore_runtime_snapshot(st.session_state.graph_runtime_snapshot)


def _persist_backend_runtime():
    """Guarda estado backend en session_state tras mutaciones."""
    st.session_state.graph_runtime_snapshot = get_runtime_snapshot()


def _dict_to_xml(data, root_tag="semantic_map") -> str:
    """Converts a dict to a pretty-printed XML string."""
    def _build(parent, obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                child = ET.SubElement(parent, str(key))
                _build(child, val)
        elif isinstance(obj, list):
            for item in obj:
                child = ET.SubElement(parent, "item")
                _build(child, item)
        elif obj is None:
            parent.text = ""
        else:
            parent.text = str(obj)

    root = ET.Element(root_tag)
    _build(root, data)
    raw = ET.tostring(root, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_map_png(map_dict: dict) -> bytes:
    """Compute once per map state; invalidate automatically when the map changes."""
    semantic_map = SemanticMap.model_validate(map_dict)
    fig = generate_floor_plan(semantic_map, dpi=90, figsize=(10, 8))
    try:
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=90, bbox_inches="tight")
        return output.getvalue()
    finally:
        plt.close(fig)


def _render_map_figure(map_dict: dict):
    """Always report errors: the old bare except silently hid the missing map."""
    try:
        semantic_map = SemanticMap.model_validate(map_dict)
        if not semantic_map.rooms and not semantic_map.global_space:
            st.info("Define un espacio global para visualizar el plano.")
            return
        st.image(_cached_map_png(map_dict))
    except Exception as exc:
        st.error(f"Error al generar o dibujar el plano: {exc}")
        if st.session_state.get("show_debug_details", False):
            st.exception(exc)


def main():
    st.set_page_config(page_title="Semantic Map Builder", page_icon="", layout="wide")
    initialize_state()
    _restore_backend_runtime()

    # Sidebar: current floor plan + JSON
    with st.sidebar:
        st.header(" Mapa Semantico")

        if st.button(" Nueva sesion", use_container_width=True):
            reset_state()
            _persist_backend_runtime()
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Sesion reiniciada. Dime el nombre del espacio que quieres modelar y sus dimensiones totales en metros (ancho x alto)."
            }]
            st.session_state.current_map = None
            st.session_state.messages_history = []
            st.session_state.messages_redo = []
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if can_undo():
                if st.button("↩ Deshacer", use_container_width=True):
                    # Guardar estado actual en redo antes de deshacer
                    st.session_state.messages_redo.append(list(st.session_state.messages))
                    prev_map = undo_state()
                    _persist_backend_runtime()
                    if st.session_state.messages_history:
                        st.session_state.messages = st.session_state.messages_history.pop()
                    st.session_state.current_map = prev_map
                    st.rerun()
        with col2:
            if can_redo():
                if st.button("↪ Rehacer", use_container_width=True):
                    # Guardar estado actual en undo antes de rehacer
                    st.session_state.messages_history.append(list(st.session_state.messages))
                    next_map = redo_state()
                    _persist_backend_runtime()
                    if st.session_state.messages_redo:
                        st.session_state.messages = st.session_state.messages_redo.pop()
                    st.session_state.current_map = next_map
                    st.rerun()

        if st.button("🧪 Demo 2D sin LLM", use_container_width=True):
            demo = load_local_demo_map()
            _persist_backend_runtime()
            st.session_state.current_map = demo
            st.session_state.messages = [{"role": "assistant", "content": "Mapa de demostración cargado localmente; no se ha consultado ningún LLM."}]
            st.session_state.messages_history = []
            st.session_state.messages_redo = []
            st.rerun()

        st.checkbox("Mostrar detalles de errores", key="show_debug_details")
        st.divider()

        if st.session_state.current_map:
            st.caption("El plano aparece en el panel principal, bajo el título.")
            with st.expander("Ver / Exportar mapa", expanded=False):
                tab_json, tab_yaml, tab_xml = st.tabs(["JSON", "YAML", "XML"])

                with tab_json:
                    json_str = json.dumps(st.session_state.current_map, indent=2, ensure_ascii=False)
                    st.code(json_str, language="json")
                    st.download_button(
                        "Descargar JSON",
                        data=json_str,
                        file_name="mapa_semantico.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                with tab_yaml:
                    yaml_str = yaml.dump(
                        st.session_state.current_map,
                        allow_unicode=True,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                    st.code(yaml_str, language="yaml")
                    st.download_button(
                        "Descargar YAML",
                        data=yaml_str,
                        file_name="mapa_semantico.yaml",
                        mime="text/yaml",
                        use_container_width=True,
                    )

                with tab_xml:
                    xml_str = _dict_to_xml(st.session_state.current_map)
                    st.code(xml_str, language="xml")
                    st.download_button(
                        "Descargar XML",
                        data=xml_str,
                        file_name="mapa_semantico.xml",
                        mime="application/xml",
                        use_container_width=True,
                    )
        else:
            st.info("El mapa aparecera aqui conforme vayas describiendo el entorno.")

    # Chat
    st.title(" Semantic Map Builder")
    if st.session_state.current_map:
        st.subheader("Plano 2D actual")
        _render_map_figure(st.session_state.current_map)
        snapshot = st.session_state.graph_runtime_snapshot or {}
        timing = (snapshot.get("graph_state") or {}).get("last_llm_seconds")
        if isinstance(timing, (int, float)):
            st.caption(f"Última consulta LLM: {timing:.1f} segundos (no incluye dibujado).")


    # Render conversation history (text + inline floor plan if the message has a map snapshot)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Describe el entorno..."):
        # Guardar snapshot ANTES de añadir nada, para que undo elimine también el mensaje del usuario
        st.session_state.messages_history.append(list(st.session_state.messages))
        st.session_state.messages_redo.clear()  # nueva rama: se pierde el redo

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                try:
                    result = process_message(prompt)
                    _persist_backend_runtime()
                    response = result["agent_response"]
                    st.session_state.current_map = result["semantic_map"]
                    st.markdown(response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
