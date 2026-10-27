import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from typing import Tuple
from models.semantic_models import SemanticMap, Room, RoboticObject, Connection


# Color configuration
COLOR_ROOM_FILL = '#E8F4F8'
COLOR_ROOM_BORDER = '#2C3E50'
COLOR_OBJECT_NAVIGABLE = '#A8E6CF'
COLOR_OBJECT_NON_NAVIGABLE = '#FFB6B6'
COLOR_CONNECTION = '#FF6B6B'
COLOR_TEXT = '#2C3E50'
COLOR_COORDINATES = '#D32F2F'  # Red for coordinate guides
COLOR_GLOBAL_SPACE_BORDER = '#1A1A2E'
COLOR_GLOBAL_SPACE_FILL = '#F5F5F0'


def _iter_absolute_rooms(rooms: list[Room], offset_x: float = 0.0, offset_y: float = 0.0, depth: int = 0):
    """Walk the tree; nested room coordinates are relative to their immediate parent."""
    for room in rooms:
        abs_x = offset_x + room.position_x
        abs_y = offset_y + room.position_y
        yield room, abs_x, abs_y, depth
        yield from _iter_absolute_rooms(room.nested_rooms, abs_x, abs_y, depth + 1)


def generate_floor_plan(semantic_map: SemanticMap, dpi: int = 100, figsize: Tuple[int, int] = (12, 10)) -> Figure:
    """Draw top-level and nested rooms in global coordinates, without an LLM call."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(which='major', color='#999999', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    title = semantic_map.global_space.name if semantic_map.global_space else "Semantic Map"
    ax.set_title(f"Technical Floor Plan: {title}", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)

    if semantic_map.global_space:
        _draw_global_space(ax, semantic_map.global_space)

    absolute_rooms = list(_iter_absolute_rooms(semantic_map.rooms))
    # Lookup by name is unambiguous only when names are globally unique.
    name_to_absolute = {room.name: (room, x, y) for room, x, y, _ in absolute_rooms}
    for room, x, y, depth in absolute_rooms:
        _draw_room(ax, room, x, y, depth)
        for obj in room.objects:
            _draw_object(ax, obj, x, y)
        ax.plot(x, y, marker='+', color=COLOR_COORDINATES, markersize=8)
        ax.text(x + 0.1, y + 0.1, f"({x:g}, {y:g})",
                fontsize=7, color=COLOR_COORDINATES, ha='left', va='bottom', fontweight='bold')
    for connection in semantic_map.connections:
        _draw_connection(ax, connection, name_to_absolute)
    _adjust_limits(ax, semantic_map)
    _add_legend(ax)
    fig.tight_layout()
    return fig


def _draw_global_space(ax, gs):
    """Draws the outer bounding rectangle for the global space."""
    rect = patches.Rectangle(
        (0, 0),
        gs.width,
        gs.height,
        linewidth=3,
        edgecolor=COLOR_GLOBAL_SPACE_BORDER,
        facecolor=COLOR_GLOBAL_SPACE_FILL,
        linestyle='--',
        alpha=0.4,
        zorder=0,
        label='Global Space'
    )
    ax.add_patch(rect)
    ax.text(
        gs.width / 2,
        gs.height + 0.3,
        gs.name,
        fontsize=12,
        fontweight='bold',
        ha='center',
        va='bottom',
        color=COLOR_GLOBAL_SPACE_BORDER,
    )


def _draw_room(ax, room: Room, x: float, y: float, depth: int = 0):
    """Draw a rectangle using resolved absolute coordinates."""
    rect = patches.Rectangle(
        (x, y), room.width, room.height,
        linewidth=2 if depth == 0 else 1.7,
        edgecolor=COLOR_ROOM_BORDER if depth == 0 else '#386fa4',
        facecolor=COLOR_ROOM_FILL if depth == 0 else '#d3e9ff',
        alpha=0.65 if depth == 0 else 0.85,
        zorder=2 + depth, label='Room'
    )
    ax.add_patch(rect)
    ax.text(x + room.width / 2, y + room.height / 2, room.name,
            fontsize=10 if depth == 0 else 8, fontweight='bold',
            ha='center', va='center', color=COLOR_TEXT, zorder=5 + depth,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))


def _draw_object(ax, obj: RoboticObject, offset_x: float = 0.0, offset_y: float = 0.0):
    """Draws an object in the coordinate frame of its room."""
    color = COLOR_OBJECT_NAVIGABLE if obj.navigable else COLOR_OBJECT_NON_NAVIGABLE
    
    if obj.shape == "circle":
        circle = patches.Circle(
            (obj.position_x + offset_x, obj.position_y + offset_y),
            obj.width / 2,
            linewidth=1.5,
            edgecolor=COLOR_TEXT,
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(circle)
    else:
        x_corner = obj.position_x + offset_x - obj.width / 2
        y_corner = obj.position_y + offset_y - obj.height / 2
        rect = patches.Rectangle(
            (x_corner, y_corner),
            obj.width,
            obj.height,
            linewidth=1.5,
            edgecolor=COLOR_TEXT,
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(rect)
    
    # Object label
    ax.text(
        obj.position_x + offset_x,
        obj.position_y + offset_y,
        obj.object_class,
        fontsize=7,
        ha='center',
        va='center',
        color='black'
    )


def _draw_connection(ax, connection: Connection, absolute_rooms: dict):
    """Draw connections using absolute coordinates (also for nested rooms)."""
    found = absolute_rooms.get(connection.origin_room)
    if not found:
        return
    origin_room, rx, ry = found
    rw, rh = origin_room.width, origin_room.height
    offset = connection.wall_offset

    if connection.wall_position == "north":
        x = rx + (offset if offset is not None else rw / 2)
        y = ry + rh
        dx, dy = connection.width, 0
    elif connection.wall_position == "south":
        x = rx + (offset if offset is not None else rw / 2)
        y = ry
        dx, dy = connection.width, 0
    elif connection.wall_position == "east":
        x = rx + rw
        y = ry + (offset if offset is not None else rh / 2)
        dx, dy = 0, connection.width
    else:  # west
        x = rx
        y = ry + (offset if offset is not None else rh / 2)
        dx, dy = 0, connection.width
    
    ax.plot(
        [x - dx/2, x + dx/2], [y - dy/2, y + dy/2],
        color=COLOR_CONNECTION, linewidth=5, solid_capstyle='round'
    )


def _adjust_limits(ax, semantic_map: SemanticMap):
    """Adjusts limits with margin. Uses global_space bounds when defined."""
    margin = 2.0

    if semantic_map.global_space:
        gs = semantic_map.global_space
        ax.set_xlim(-margin, gs.width + margin)
        ax.set_ylim(-margin, gs.height + margin)
        return

    if not semantic_map.rooms:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        return

    x_min = min(r.position_x for r in semantic_map.rooms)
    x_max = max(r.position_x + r.width for r in semantic_map.rooms)
    y_min = min(r.position_y for r in semantic_map.rooms)
    y_max = max(r.position_y + r.height for r in semantic_map.rooms)

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)


def _add_legend(ax):
    from matplotlib.lines import Line2D
    legend_elements = [
        patches.Rectangle((0,0),1,1, facecolor=COLOR_GLOBAL_SPACE_FILL, edgecolor=COLOR_GLOBAL_SPACE_BORDER, linestyle='--', linewidth=2, label='Global Space'),
        patches.Rectangle((0,0),1,1, facecolor=COLOR_ROOM_FILL, edgecolor=COLOR_ROOM_BORDER, label='Room'),
        patches.Rectangle((0,0),1,1, facecolor=COLOR_OBJECT_NON_NAVIGABLE, label='Obstacle'),
        Line2D([0],[0], color=COLOR_CONNECTION, linewidth=4, label='Door'),
        Line2D([0],[0], marker='+', color=COLOR_COORDINATES, linestyle='None', label='Origin corner (X,Y)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def export_floor_plan(fig: Figure, file_path: str, format: str = 'png'):
    fig.savefig(f"{file_path}.{format}", dpi=300, bbox_inches='tight')