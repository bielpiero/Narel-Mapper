from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SpatialRelation(BaseModel):
    """Relative position of a pending room with respect to an already-placed room."""
    reference_room: str = Field(description="Name of the already-placed room used as reference.")
    direction: Literal["north", "south", "east", "west"] = Field(
        description="Direction from the reference room where the new room is located."
    )
    gap: float = Field(default=0.0, description="Distance (meters) between the two rooms.")

class RoboticObject(BaseModel):
    object_id: str = Field(description="Unique identifier (e.g., chair_01).")
    object_class: str
    navigable: bool
    position_x: float
    position_y: float
    width: float
    height: float
    shape: Literal["rectangle", "circle"] = "rectangle"
    orientation: float = 0.0

class Room(BaseModel):
    name: str
    position_x: float = 0.0
    position_y: float = 0.0
    width: float
    height: float
    orientation: float = 0.0
    objects: List[RoboticObject] = Field(default_factory=list)
    nested_rooms: List["Room"] = Field(
        default_factory=list,
        description="Child rooms; positions are relative to this room’s lower-left corner.",
    )
    spatial_relation: Optional[SpatialRelation] = None  # filled by LLM; auto_place_node overwrites position

class Connection(BaseModel):
    origin_room: str
    destination_room: str
    connection_type: Literal["door", "hallway", "arch", "opening"] = "door"
    width: float
    wall_position: Literal["north", "south", "east", "west"]
    wall_offset: Optional[float] = None  # offset (meters) from origin room bottom-left along the wall axis

class PendingRoom(BaseModel):
    """Rooms identified but lacking full geometric data."""
    name: str
    width: Optional[float] = None
    height: Optional[float] = None
    spatial_relation: Optional[SpatialRelation] = None

class PendingConnection(BaseModel):
    room_A: str
    room_B: str
    suggested_type: str = "door"
    origin_wall: Optional[str] = None


class BlockedConnection(BaseModel):
    """Pair of rooms where auto door detection must not recreate a connection."""
    room_a: str
    room_b: str

class GlobalSpace(BaseModel):
    """The outer bounding area that contains all rooms (e.g. a house, a floor, a warehouse)."""
    name: str = Field(description="Name of the global space (e.g. 'Casa de la abuela').")
    width: float = Field(description="Total width of the space in meters.")
    height: float = Field(description="Total height (depth) of the space in meters.")

class SemanticMap(BaseModel):
    """Root State of the World."""
    global_space: Optional[GlobalSpace] = Field(
        default=None,
        description="The outer bounding area that must be defined before placing any rooms."
    )
    rooms: List[Room] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    pending_rooms: List[PendingRoom] = Field(default_factory=list)
    pending_connections: List[PendingConnection] = Field(default_factory=list)
    blocked_connections: List[BlockedConnection] = Field(default_factory=list)

class MoveIntent(BaseModel):
    """Explicit displacement request for an already-placed room."""
    room_name: str = Field(description="Exact name of the room to move (must match an existing room).")
    delta_x: float = Field(default=0.0, description="Horizontal displacement in meters. Positive = east, negative = west.")
    delta_y: float = Field(default=0.0, description="Vertical displacement in meters. Positive = north, negative = south.")
    rotation_degrees: float = Field(
        default=0.0,
        description="Rotation to apply in degrees. Positive = counterclockwise. Typical values: +/-90, 180."
    )

class AgentResponse(BaseModel):
    """Structured output from the LLM."""
    agent_response: str = Field(description="Natural language reply to the user.")
    updated_map: SemanticMap = Field(description="The full updated state of the map.")
    move_intents: Optional[list[MoveIntent]] = Field(
        default=None,
        description="List of room displacement requests. Filled ONLY when the user explicitly asks to move/displace rooms by a distance. Can contain multiple moves in a single turn. Leave null for all other operations."
    )
# Rebuild recursive references for Pydantic v2 / JSON Schema.
Room.model_rebuild()
