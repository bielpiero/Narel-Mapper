from typing import TypedDict, Optional
from functools import lru_cache
from time import perf_counter
from langgraph.graph import StateGraph, END
from models.semantic_models import SemanticMap, AgentResponse
from prompts.map_prompts import get_agent_prompt
from config.settings import settings
from utils.llm_factory import LLMFactory


class MapState(TypedDict):
    user_input: str
    semantic_map: Optional[SemanticMap]
    agent_response: Optional[str]
    messages: list
    move_intents: Optional[list]  # filled by LLM when user explicitly requests moving room(s)
    conflict_rooms: list          # room names already reported as conflicted (not re-reported)
    overlapping_rooms: list       # pairs [{room_a, room_b}] currently overlapping; cleared when resolved
    last_llm_seconds: float      # observability for local models


@lru_cache(maxsize=1)
def get_llm():
    settings.validate()
    llm = LLMFactory.create_llm()
    # Para Ollama, la API nativa de JSON Schema es más adecuada que depender
    # de function/tool calling para rellenar la respuesta Pydantic anidada.
    if settings.LLM_PROVIDER == "ollama":
        return llm.with_structured_output(AgentResponse, method="json_schema")
    return llm.with_structured_output(AgentResponse)


def _format_history(messages: list) -> str:
    recent = messages[-settings.MAX_CONTEXT_MESSAGES:] if messages else []
    if not recent:
        return "No history."
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"- {role}: {msg['content']}")
    return "\n".join(lines)


def _pair_key(room_a: str, room_b: str) -> tuple:
    return tuple(sorted((room_a, room_b)))


def _normalize_blocked_pairs(raw_blocked: list) -> set:
    pairs = set()
    for item in raw_blocked or []:
        a = item.get("room_a") or item.get("origin_room")
        b = item.get("room_b") or item.get("destination_room")
        if a and b and a != b:
            pairs.add(_pair_key(a, b))
    return pairs


def _pairs_to_blocked_list(pairs: set) -> list:
    return [{"room_a": a, "room_b": b} for a, b in sorted(pairs)]


def _nested_ancestor_pairs(rooms: list) -> set:
    """Pairs that belong to the same ancestry chain; internal links are explicit."""
    result = set()
    def walk(children, ancestors):
        for room in children:
            name = room['name']
            for a in ancestors:
                result.add(_pair_key(a, name))
            walk(room.get('nested_rooms', []), [*ancestors, name])
    walk(rooms, [])
    return result


def _connection_pairs(connections: list) -> set:
    return {
        _pair_key(c["origin_room"], c["destination_room"])
        for c in connections or []
        if c.get("origin_room") and c.get("destination_room") and c["origin_room"] != c["destination_room"]
    }


def _build_overlap_warning(overlapping_rooms: list) -> str:
    if not overlapping_rooms:
        return ""
    pairs_str = ", ".join(f"'{p['room_a']}' y '{p['room_b']}'" for p in overlapping_rooms)
    return (
        "### SOLAPAMIENTO ACTIVO ⚠️ (LEE ESTO PRIMERO)\n"
        f"Las siguientes habitaciones están solapadas geométricamente: {pairs_str}.\n"
        "REGLAS OBLIGATORIAS mientras exista este conflicto:\n"
        "1. NO añadas nuevas habitaciones ni modifiques posiciones en updated_map.\n"
        "2. Informa al usuario de las habitaciones solapadas y pídele que especifique cómo moverlas.\n"
        "3. Si el usuario pide mover una habitación, emite los move_intents correspondientes.\n"
        "4. Mantén updated_map.rooms idéntico al estado actual (sin cambios de posición ni nuevas habitaciones).\n"
        "5. No uses la pregunta de cierre habitual — pide al usuario que resuelva el solapamiento primero."
    )


@lru_cache(maxsize=1)
def _get_llm_chain():
    return get_agent_prompt() | get_llm()

def llm_update_map(state: MapState):
    chain = _get_llm_chain()
    messages: list = state.get("messages") or []
    overlap_warning = _build_overlap_warning(state.get("overlapping_rooms") or [])

    llm_started = perf_counter()
    resultado = chain.invoke({
        "current_map": state.get("semantic_map") or {},
        "chat_history": _format_history(messages),
        "message": state["user_input"],
        "overlap_warning": overlap_warning,
    })

    llm_seconds = perf_counter() - llm_started

    # Only append the user message here. The assistant message is appended
    # at the end of auto_place_node (after backend additions) so the LLM sees
    # the final response including conflict notices and placement annotations.
    updated_messages = (messages + [
        {"role": "user", "content": state["user_input"]},
    ])[-settings.MAX_MEMORY_MESSAGES:]

    move_intents = None
    if resultado.move_intents:
        move_intents = [m.model_dump() for m in resultado.move_intents]

    previous_map = state.get("semantic_map") or {}
    previous_connections = list(previous_map.get("connections", []))
    previous_blocked = _normalize_blocked_pairs(previous_map.get("blocked_connections", []))

    updated_map = resultado.updated_map.model_dump()
    updated_connections = list(updated_map.get("connections", []))
    updated_blocked = _normalize_blocked_pairs(updated_map.get("blocked_connections", []))

    removed_pairs = _connection_pairs(previous_connections) - _connection_pairs(updated_connections)
    added_pairs = _connection_pairs(updated_connections) - _connection_pairs(previous_connections)

    # If a pair was removed in this turn, keep it blocked to avoid auto-recreate by adjacency.
    # If user adds it again later, unblock it.
    merged_blocked = (previous_blocked | updated_blocked | removed_pairs) - added_pairs
    updated_map["blocked_connections"] = _pairs_to_blocked_list(merged_blocked)

    return {
        **state,
        "semantic_map": updated_map,
        "agent_response": resultado.agent_response,
        "messages": updated_messages,
        "move_intents": move_intents,
        "last_llm_seconds": llm_seconds,
    }



_TOUCH_TOLERANCE = 0.05  # metros
_WALL_STACK_GAP = 0.1  # small separation between siblings on the same wall
_ROTATION_EPSILON = 1e-3  # tolerance for angle comparisons


def _wall_key(reference_room: str, direction: str, gap: float) -> tuple:
    return reference_room, direction, round(float(gap), 4)


def _initialize_wall_cursors(placed: dict) -> dict:
    """Builds per-wall cursors using already placed rooms with spatial_relation."""
    cursors = {}
    for room in placed.values():
        relation = room.get("spatial_relation")
        if not relation:
            continue

        ref_name = relation.get("reference_room")
        direction = relation.get("direction")
        gap = relation.get("gap", 0.0)
        ref = placed.get(ref_name)
        if not ref:
            continue

        key = _wall_key(ref_name, direction, gap)
        ref_x, ref_y = ref["position_x"], ref["position_y"]

        if direction in ("north", "south"):
            used = (room["position_x"] - ref_x) + room["width"] + _WALL_STACK_GAP
        elif direction in ("east", "west"):
            used = (room["position_y"] - ref_y) + room["height"] + _WALL_STACK_GAP
        else:
            continue

        cursors[key] = max(cursors.get(key, 0.0), max(0.0, used))

    return cursors


def _opposite_wall(wall_position: str) -> Optional[str]:
    opposite = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }
    return opposite.get(wall_position)


def _preferred_anchor_wall(room_name: str, connections: list) -> Optional[str]:
    """Returns the wall to keep fixed during pure rotation, prioritizing widest connection."""
    candidates = []
    for connection in connections or []:
        origin = connection.get("origin_room")
        destination = connection.get("destination_room")
        if origin != room_name and destination != room_name:
            continue

        if origin == room_name:
            wall = connection.get("wall_position")
        else:
            wall = _opposite_wall(connection.get("wall_position"))

        if wall in {"north", "south", "east", "west"}:
            width = float(connection.get("width", 0.0) or 0.0)
            candidates.append((width, wall))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _quarter_turn_steps(rotation_degrees: float) -> Optional[int]:
    """Returns quarter-turn steps if angle is a supported multiple of 90, else None."""
    steps = int(round(rotation_degrees / 90.0))
    if abs(rotation_degrees - (steps * 90.0)) <= _ROTATION_EPSILON:
        return steps
    return None


def move_room_node(state: MapState) -> MapState:
    """
    Applies explicit displacements (delta_x, delta_y) and rotations to one or more rooms.
    Clears spatial_relation for changed rooms so they maintain absolute positions.
    Supports multiple room updates in a single turn.

    Rotation semantics:
    - orientation is updated by adding rotation_degrees.
    - for odd quarter-turns (90/270), width/height are swapped.
    - for pure rotations (no translation), try to keep connected wall fixed.
    - fallback keeps room center after swap when no anchor wall is available.
    """
    moves = state.get("move_intents") or []
    if not moves:
        return state

    semantic_map = dict(state.get("semantic_map") or {})
    rooms = list(semantic_map.get("rooms", []))
    existing_connections = list(semantic_map.get("connections", []))
    blocked_pairs = _normalize_blocked_pairs(semantic_map.get("blocked_connections", []))

    # Build a map of room_name -> move for quick lookup
    moves_by_name = {m["room_name"]: m for m in moves}

    updated_rooms = []
    changed_names = set()
    unsupported_rotation_rooms = []
    for room in rooms:
        if room["name"] in moves_by_name:
            room = dict(room)
            move = moves_by_name[room["name"]]
            delta_x = float(move.get("delta_x", 0.0) or 0.0)
            delta_y = float(move.get("delta_y", 0.0) or 0.0)
            rotation_degrees = float(move.get("rotation_degrees", 0.0) or 0.0)
            quarter_turn_steps = _quarter_turn_steps(rotation_degrees)
            has_supported_rotation = quarter_turn_steps is not None and quarter_turn_steps != 0
            has_unsupported_rotation = abs(rotation_degrees) > 1e-9 and quarter_turn_steps is None
            if has_supported_rotation and room.get("nested_rooms"):
                # Proper rigid-body rotation of nested frames is outside this minimal restoration.
                has_supported_rotation = False
                has_unsupported_rotation = True
            pure_rotation = has_supported_rotation and abs(delta_x) <= 1e-9 and abs(delta_y) <= 1e-9

            room["position_x"] = round(room["position_x"] + delta_x, 4)
            room["position_y"] = round(room["position_y"] + delta_y, 4)

            if has_supported_rotation:
                current_orientation = float(room.get("orientation", 0.0) or 0.0)
                room["orientation"] = round((current_orientation + (quarter_turn_steps * 90.0)) % 360, 4)

                # Axis-aligned geometry only supports exact 90/270 swaps.
                if quarter_turn_steps % 2 != 0:
                    old_x, old_y = room["position_x"], room["position_y"]
                    old_w, old_h = room["width"], room["height"]
                    old_right = old_x + old_w
                    old_top = old_y + old_h

                    room["width"], room["height"] = old_h, old_w

                    anchor_wall = _preferred_anchor_wall(room["name"], existing_connections) if pure_rotation else None

                    if anchor_wall == "east":
                        room["position_x"] = round(old_right - room["width"], 4)
                        room["position_y"] = round(old_y + (old_h - room["height"]) / 2, 4)
                    elif anchor_wall == "west":
                        room["position_x"] = round(old_x, 4)
                        room["position_y"] = round(old_y + (old_h - room["height"]) / 2, 4)
                    elif anchor_wall == "north":
                        room["position_x"] = round(old_x + (old_w - room["width"]) / 2, 4)
                        room["position_y"] = round(old_top - room["height"], 4)
                    elif anchor_wall == "south":
                        room["position_x"] = round(old_x + (old_w - room["width"]) / 2, 4)
                        room["position_y"] = round(old_y, 4)
                    else:
                        center_x = old_x + old_w / 2
                        center_y = old_y + old_h / 2
                        room["position_x"] = round(center_x - room["width"] / 2, 4)
                        room["position_y"] = round(center_y - room["height"] / 2, 4)

            if has_unsupported_rotation:
                unsupported_rotation_rooms.append(room["name"])

            room["spatial_relation"] = None  # absolute position now — must not be recalculated
            if abs(delta_x) > 1e-9 or abs(delta_y) > 1e-9 or has_supported_rotation:
                changed_names.add(room["name"])
        updated_rooms.append(room)

    # Resolve explicit move intents referring to nested rooms by name.
    def _move_nested(children):
        transformed = []
        for child in children:
            child = dict(child)
            if child["name"] in moves_by_name:
                move = moves_by_name[child["name"]]
                dx = float(move.get("delta_x", 0) or 0)
                dy = float(move.get("delta_y", 0) or 0)
                angle = float(move.get("rotation_degrees", 0) or 0)
                if abs(angle) > 1e-9:
                    unsupported_rotation_rooms.append(child["name"])
                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                    child["position_x"] = round(child["position_x"] + dx, 4)
                    child["position_y"] = round(child["position_y"] + dy, 4)
                    changed_names.add(child["name"])
            child["nested_rooms"] = _move_nested(child.get("nested_rooms", []))
            transformed.append(child)
        return transformed
    updated_rooms = [dict(r, nested_rooms=_move_nested(r.get("nested_rooms", []))) for r in updated_rooms]

    agent_response = state.get("agent_response", "")
    if unsupported_rotation_rooms:
        unsupported_names = ", ".join(sorted(set(unsupported_rotation_rooms)))
        agent_response += (
            f" [Rotacion ignorada para: {unsupported_names}. "
            "Solo se admiten rotaciones en multiplos de 90 grados.]"
        )

    if not changed_names:
        return {**state, "move_intents": None, "agent_response": agent_response}

    # Save intended positions (post-delta, pre-clamp) to compare after clamping
    intended_positions = {
        r["name"]: (r["position_x"], r["position_y"])
        for r in updated_rooms
        if r["name"] in changed_names
    }

    # Clamp moved rooms to global space before rebuilding connections
    updated_rooms, clamped_names = _clamp_rooms_to_global_space(
        updated_rooms, semantic_map.get("global_space")
    )
    if clamped_names:
        gs = semantic_map.get("global_space") or {}
        gs_name = gs.get("name", "el espacio global")
        clamped_positions = {r["name"]: (r["position_x"], r["position_y"]) for r in updated_rooms}
        details = []
        for name in clamped_names:
            ix, iy = intended_positions.get(name, (0, 0))
            ax_, ay_ = clamped_positions.get(name, (ix, iy))
            details.append(
                f"'{name}' (destino pedido: ({ix:.2f}, {iy:.2f}) → ajustado a: ({ax_:.2f}, {ay_:.2f}))"
            )
        agent_response = (
            f"No he podido mover la habitación a la posición solicitada porque quedaría fuera de '{gs_name}'. "
            f"La he dejado en el borde más cercano posible: {'; '.join(details)}. "
            f"Si quieres, puedes ampliar el espacio global o elegir un desplazamiento menor."
        )

    # Hierarchical links are explicit; don't erase them when a parent or child moves.
    internal_pairs = _nested_ancestor_pairs(updated_rooms)
    # Rebuild geometry-based connections touching changed rooms.
    connections_base = [
        c for c in existing_connections
        if (
            _pair_key(c["origin_room"], c["destination_room"]) in internal_pairs
            or (c["origin_room"] not in changed_names and c["destination_room"] not in changed_names)
        )
        and _pair_key(c["origin_room"], c["destination_room"]) not in blocked_pairs
    ]
    new_connections = _detect_connections(updated_rooms, connections_base, blocked_pairs)

    semantic_map = {
        **semantic_map,
        "rooms": updated_rooms,
        "connections": connections_base + new_connections,
    }
    return {
        **state,
        "semantic_map": semantic_map,
        "move_intents": None,
        "agent_response": agent_response,
    }


def _is_connection_geom_valid(conn: dict, rooms_by_name: dict) -> bool:
    """Return False if the connection's wall_position contradicts actual room geometry."""
    wall = conn.get("wall_position")
    if not wall:
        return True
    a = rooms_by_name.get(conn["origin_room"])
    b = rooms_by_name.get(conn["destination_room"])
    if a is None or b is None:
        return True  # can't validate, keep it
    tol = _TOUCH_TOLERANCE
    ax, ay, aw, ah = a["position_x"], a["position_y"], a["width"], a["height"]
    bx, by, bw, bh = b["position_x"], b["position_y"], b["width"], b["height"]
    if wall == "east":
        return abs((ax + aw) - bx) <= tol
    if wall == "west":
        return abs((bx + bw) - ax) <= tol
    if wall == "north":
        return abs((ay + ah) - by) <= tol
    if wall == "south":
        return abs((by + bh) - ay) <= tol
    return True


def _detect_connections(rooms: list, existing_connections: list, blocked_pairs: Optional[set] = None) -> list:
    tol = _TOUCH_TOLERANCE
    blocked_pairs = blocked_pairs or set()
    existing_pairs = {
        frozenset([c["origin_room"], c["destination_room"]])
        for c in existing_connections
    }
    new_connections = []

    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            pair = frozenset([a["name"], b["name"]])
            if _pair_key(a["name"], b["name"]) in blocked_pairs:
                continue
            if pair in existing_pairs:
                continue

            ax, ay, aw, ah = a["position_x"], a["position_y"], a["width"], a["height"]
            bx, by, bw, bh = b["position_x"], b["position_y"], b["width"], b["height"]

            a_right, a_left, a_top, a_bottom = ax + aw, ax, ay + ah, ay
            b_right, b_left, b_top, b_bottom = bx + bw, bx, by + bh, by

            DEFAULT_DOOR_WIDTH = 1.0

            if abs(a_right - b_left) <= tol:
                overlap = min(a_top, b_top) - max(a_bottom, b_bottom)
                if overlap > tol:
                    center_y = max(a_bottom, b_bottom) + overlap / 2
                    new_connections.append({
                        "origin_room": a["name"], "destination_room": b["name"],
                        "connection_type": "door",
                        "width": min(DEFAULT_DOOR_WIDTH, round(overlap, 2)),
                        "wall_position": "east",
                        "wall_offset": round(center_y - ay, 4),
                    })
                    existing_pairs.add(pair)
                    continue

            if abs(b_right - a_left) <= tol:
                overlap = min(a_top, b_top) - max(a_bottom, b_bottom)
                if overlap > tol:
                    center_y = max(a_bottom, b_bottom) + overlap / 2
                    new_connections.append({
                        "origin_room": a["name"], "destination_room": b["name"],
                        "connection_type": "door",
                        "width": min(DEFAULT_DOOR_WIDTH, round(overlap, 2)),
                        "wall_position": "west",
                        "wall_offset": round(center_y - ay, 4),
                    })
                    existing_pairs.add(pair)
                    continue

            if abs(a_top - b_bottom) <= tol:
                overlap = min(a_right, b_right) - max(a_left, b_left)
                if overlap > tol:
                    center_x = max(a_left, b_left) + overlap / 2
                    new_connections.append({
                        "origin_room": a["name"], "destination_room": b["name"],
                        "connection_type": "door",
                        "width": min(DEFAULT_DOOR_WIDTH, round(overlap, 2)),
                        "wall_position": "north",
                        "wall_offset": round(center_x - ax, 4),
                    })
                    existing_pairs.add(pair)
                    continue

            if abs(b_top - a_bottom) <= tol:
                overlap = min(a_right, b_right) - max(a_left, b_left)
                if overlap > tol:
                    center_x = max(a_left, b_left) + overlap / 2
                    new_connections.append({
                        "origin_room": a["name"], "destination_room": b["name"],
                        "connection_type": "door",
                        "width": min(DEFAULT_DOOR_WIDTH, round(overlap, 2)),
                        "wall_position": "south",
                        "wall_offset": round(center_x - ax, 4),
                    })
                    existing_pairs.add(pair)

    return new_connections


def _clamp_rooms_to_global_space(rooms: list, global_space: Optional[dict]) -> tuple[list, list]:
    """
    Clamps every room so it stays within [0, gs_w] x [0, gs_h].
    Returns (clamped_rooms, list_of_clamped_names).
    If global_space is None, rooms are returned unchanged.
    """
    if not global_space:
        return rooms, []
    gs_w = float(global_space.get("width", 0) or 0)
    gs_h = float(global_space.get("height", 0) or 0)
    clamped_names = []
    result = []
    for room in rooms:
        r = dict(room)
        rw = float(r.get("width", 0) or 0)
        rh = float(r.get("height", 0) or 0)
        orig_x, orig_y = r.get("position_x", 0.0), r.get("position_y", 0.0)
        new_x = round(max(0.0, min(float(orig_x), gs_w - rw)), 4)
        new_y = round(max(0.0, min(float(orig_y), gs_h - rh)), 4)
        if abs(new_x - float(orig_x)) > 1e-6 or abs(new_y - float(orig_y)) > 1e-6:
            r["position_x"] = new_x
            r["position_y"] = new_y
            clamped_names.append(r["name"])
        result.append(r)
    return result, clamped_names


_OVERLAP_MIN = 1e-3  # minimum overlap in each dimension (meters) to flag as overlapping


def _detect_overlaps(rooms: list) -> list:
    """Returns [{room_a, room_b}] for every pair of rooms whose interiors intersect."""
    overlaps = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            ax, ay, aw, ah = a["position_x"], a["position_y"], a["width"], a["height"]
            bx, by, bw, bh = b["position_x"], b["position_y"], b["width"], b["height"]
            overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
            overlap_y = min(ay + ah, by + bh) - max(ay, by)
            if overlap_x > _OVERLAP_MIN and overlap_y > _OVERLAP_MIN:
                overlaps.append({"room_a": a["name"], "room_b": b["name"]})
    return overlaps


def auto_place_node(state: MapState) -> MapState:
    """
    Resuelve posiciones relativas y detecta conexiones por adyacencia.
    position_x/y es la esquina inferior izquierda de cada habitación.
    """
    semantic_map: dict = state.get("semantic_map") or {}
    rooms: list = list(semantic_map.get("rooms", []))
    pending_rooms: list = list(semantic_map.get("pending_rooms", []))
    existing_connections: list = list(semantic_map.get("connections", []))
    blocked_pairs = _normalize_blocked_pairs(semantic_map.get("blocked_connections", []))

    # Snapshot de posiciones antes de cualquier cambio, para detectar qué realmente se movió
    original_positions = {r["name"]: (r["position_x"], r["position_y"]) for r in rooms}

    # Rooms con spatial_relation → recalcular posición
    promoted_to_pending = []
    clean_rooms = []
    for r in rooms:
        if r.get("spatial_relation"):
            promoted_to_pending.append({
                "name": r["name"],
                "width": r["width"],
                "height": r["height"],
                "orientation": r.get("orientation", 0.0),
                "objects": r.get("objects", []),
                "nested_rooms": r.get("nested_rooms", []),
                "spatial_relation": r["spatial_relation"],
            })
        else:
            clean_rooms.append(r)

    pending_names_existing = {pr["name"] for pr in pending_rooms}
    combined_pending = pending_rooms + [
        p for p in promoted_to_pending if p["name"] not in pending_names_existing
    ]
    rooms = clean_rooms

    placed = {r["name"]: r for r in rooms}

    newly_placed = []
    placement_conflicts = []   # rooms that can't be placed due to global_space bounds
    conflicted_names: set = set()

    gs = semantic_map.get("global_space")
    gs_w = float(gs.get("width", 0)) if gs else None
    gs_h = float(gs.get("height", 0)) if gs else None

    # Center rooms that the LLM placed at the default (0, 0) placeholder inside the global space
    if gs_w is not None and gs_h is not None:
        for r in rooms:
            if (r.get("position_x", 0.0) == 0.0
                    and r.get("position_y", 0.0) == 0.0
                    and not r.get("spatial_relation")):
                r["position_x"] = max(0.0, round((gs_w - float(r["width"])) / 2, 4))
                r["position_y"] = max(0.0, round((gs_h - float(r["height"])) / 2, 4))

    changed = True
    remaining = list(combined_pending)
    while changed:
        changed = False
        next_remaining = []
        wall_cursors = _initialize_wall_cursors(placed)
        for pr in remaining:
            # Skip rooms already flagged as conflicted in a previous iteration
            if pr["name"] in conflicted_names:
                next_remaining.append(pr)
                continue

            relation = pr.get("spatial_relation")
            width = pr.get("width")
            height = pr.get("height")

            if not relation or width is None or height is None:
                next_remaining.append(pr)
                continue

            ref_name = relation.get("reference_room")
            direction = relation.get("direction")
            gap = relation.get("gap", 0.0)
            ref = placed.get(ref_name)

            if ref is None:
                next_remaining.append(pr)
                continue

            ref_x, ref_y = ref["position_x"], ref["position_y"]
            ref_w, ref_h = ref["width"], ref["height"]
            wall_key = _wall_key(ref_name, direction, gap)
            cursor = wall_cursors.get(wall_key, 0.0)

            if direction == "east":
                new_x = ref_x + ref_w + gap
                new_y = ref_y + cursor
                wall_cursors[wall_key] = cursor + height + _WALL_STACK_GAP
            elif direction == "west":
                new_x = ref_x - width - gap
                new_y = ref_y + cursor
                wall_cursors[wall_key] = cursor + height + _WALL_STACK_GAP
            elif direction == "north":
                new_x = ref_x + cursor
                new_y = ref_y + ref_h + gap
                wall_cursors[wall_key] = cursor + width + _WALL_STACK_GAP
            elif direction == "south":
                new_x = ref_x + cursor
                new_y = ref_y - height - gap
                wall_cursors[wall_key] = cursor + width + _WALL_STACK_GAP
            else:
                next_remaining.append(pr)
                continue

            # Check global_space bounds before promoting
            if gs_w is not None and gs_h is not None:
                out = (
                    new_x < 0 or new_y < 0
                    or new_x + width > gs_w
                    or new_y + height > gs_h
                )
                if out:
                    conflicted_names.add(pr["name"])
                    placement_conflicts.append({
                        "room": pr["name"],
                        "direction": direction,
                        "reference": ref_name,
                    })
                    next_remaining.append(pr)
                    continue

            promoted = {
                "name": pr["name"],
                "position_x": round(new_x, 4),
                "position_y": round(new_y, 4),
                "width": width,
                "height": height,
                "orientation": pr.get("orientation", 0.0),
                "objects": pr.get("objects", []),
                "nested_rooms": pr.get("nested_rooms", []),
                "spatial_relation": pr.get("spatial_relation"),  # conservar info semantica
            }
            newly_placed.append(promoted)
            placed[pr["name"]] = promoted
            changed = True

        remaining = next_remaining

    still_pending = remaining

    promoted_names = {r["name"] for r in newly_placed}
    all_rooms = [r for r in rooms if r["name"] not in promoted_names] + newly_placed

    # Clamp all rooms to global space bounds (deterministic, no room can escape)
    all_rooms, clamped_names = _clamp_rooms_to_global_space(all_rooms, semantic_map.get("global_space"))

    # If a room changed position in this turn, stale connections for that room
    # must be rebuilt from geometry (including updated wall_position).
    _tol = 0.001
    moved_names = {
        r["name"] for r in newly_placed
        if r["name"] not in original_positions
        or abs(r["position_x"] - original_positions[r["name"]][0]) > _tol
        or abs(r["position_y"] - original_positions[r["name"]][1]) > _tol
    }
    rooms_by_name_final = {r["name"]: r for r in all_rooms}
    internal_pairs = _nested_ancestor_pairs(all_rooms)
    connections_base = [
        c for c in existing_connections
        if (
            _pair_key(c["origin_room"], c["destination_room"]) in internal_pairs
            or (c["origin_room"] not in moved_names and c["destination_room"] not in moved_names)
        )
        and _pair_key(c["origin_room"], c["destination_room"]) not in blocked_pairs
        and _is_connection_geom_valid(c, rooms_by_name_final)
    ]

    new_connections = _detect_connections(all_rooms, connections_base, blocked_pairs)

    updated_map = {
        **semantic_map,
        "rooms": all_rooms,
        "pending_rooms": still_pending,
        "connections": connections_base + new_connections,
    }

    agent_response = state.get("agent_response", "")

    if clamped_names:
        # This should rarely trigger now (conflicts caught above), but handle edge cases
        names_str = ", ".join(clamped_names)
        agent_response += f" [Ajustado al borde del espacio global: {names_str}.]"

    # Solo anunciar habitaciones cuya posición realmente cambió respecto al estado anterior
    actually_moved = [r for r in newly_placed if r["name"] in moved_names]
    if actually_moved:
        names = ", ".join(r["name"] for r in actually_moved)
        agent_response += f" [Posicionadas automáticamente: {names}.]"
    if new_connections:
        pairs = ", ".join(f"{c['origin_room']}↔{c['destination_room']}" for c in new_connections)
        agent_response += f" [Conexiones detectadas: {pairs}.]"

    known_conflict_rooms = set(state.get("conflict_rooms") or [])

    # Rooms successfully placed this turn are no longer conflicted
    placed_names = {r["name"] for r in newly_placed}
    resolved_conflicts = known_conflict_rooms & placed_names
    known_conflict_rooms -= resolved_conflicts

    # Only report conflicts for rooms not already reported in a previous turn
    new_conflicts = [c for c in placement_conflicts if c["room"] not in known_conflict_rooms]
    for c in placement_conflicts:
        known_conflict_rooms.add(c["room"])

    if new_conflicts:
        gs_name = gs.get("name", "el espacio global") if gs else "el espacio global"
        conflict_msgs = []
        for c in new_conflicts:
            dir_es = {"north": "norte", "south": "sur", "east": "este", "west": "oeste"}.get(c["direction"], c["direction"])
            conflict_msgs.append(
                f"He registrado '{c['room']}' como habitación pendiente porque al colocarla al {dir_es} de '{c['reference']}' "
                f"quedaría fuera de '{gs_name}'.\n"
                f"¿Qué prefieres?\n"
                f"  **A)** Indicarme otra pared de '{c['reference']}' donde colocarlo "
                f"(norte / sur / este / oeste).\n"
                f"  **B)** Mover '{c['reference']}' para hacer hueco y volver a intentarlo.\n"
                f"  **C)** Ampliar el espacio global '{gs_name}'."
            )
        # Replace LLM response entirely — it said "placed" but backend couldn't place it
        agent_response = "\n\n".join(conflict_msgs)

    # Detect geometric overlaps between placed rooms
    previous_overlapping = state.get("overlapping_rooms") or []
    overlapping_pairs = _detect_overlaps(all_rooms)

    if overlapping_pairs and not new_conflicts:
        # Override with overlap warning (only when there is no out-of-bounds conflict
        # message already replacing it — they're mutually exclusive in practice).
        pair_labels = ", ".join(f"'{p['room_a']}' y '{p['room_b']}'" for p in overlapping_pairs)
        agent_response = (
            f"⚠️ **Hay habitaciones solapadas:** {pair_labels}.\n\n"
            f"Por favor, indícame qué movimientos hacer para recolocarlas (por ejemplo: "
            f"\"mueve X 2 metros al norte\"). No se pueden añadir más habitaciones hasta "
            f"que se resuelva el solapamiento."
        )
    elif not overlapping_pairs and previous_overlapping:
        # Overlap just resolved — the LLM response was produced before the backend
        # applied the movement, so it may be stale or confused. Replace it with a
        # clear confirmation that the conflict is gone.
        agent_response = "✅ Solapamiento resuelto. Las habitaciones ya no se superponen. ¿Te gustaría añadir o modificar algo más?"

    # Append the final assistant message (with all backend annotations) to history
    messages = list(state.get("messages") or [])
    final_messages = (messages + [
        {"role": "assistant", "content": agent_response}
    ])[-settings.MAX_MEMORY_MESSAGES:]

    return {
        **state,
        "semantic_map": updated_map,
        "agent_response": agent_response,
        "messages": final_messages,
        "conflict_rooms": list(known_conflict_rooms),
        "overlapping_rooms": overlapping_pairs,
    }


def _route_after_llm(state: MapState) -> str:
    return "move_room_node" if state.get("move_intents") else "auto_place_node"


def create_map_graph():
    workflow = StateGraph(MapState)
    workflow.add_node("llm_update_map", llm_update_map)
    workflow.add_node("move_room_node", move_room_node)
    workflow.add_node("auto_place_node", auto_place_node)
    workflow.set_entry_point("llm_update_map")
    workflow.add_conditional_edges(
        "llm_update_map",
        _route_after_llm,
        {"move_room_node": "move_room_node", "auto_place_node": "auto_place_node"},
    )
    # After move_room_node, always pass through auto_place_node so pending rooms
    # that were blocked by the old position can be retried automatically.
    workflow.add_edge("move_room_node", "auto_place_node")
    workflow.add_edge("auto_place_node", END)
    return workflow.compile()