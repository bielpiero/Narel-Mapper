"""Local tests: no Ollama, no API, no Streamlit needed."""
import os
os.environ.setdefault('MPLBACKEND','Agg')

from models.semantic_models import SemanticMap
from services.map_visualizer import generate_floor_plan, _iter_absolute_rooms
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt


def sample_map():
    return SemanticMap.model_validate({
        'global_space': {'name':'Casa', 'width':12, 'height':10},
        'rooms': [
            {'name':'Dormitorio', 'position_x':1,'position_y':1,'width':6,'height':5,
             'nested_rooms':[{'name':'Baño','position_x':3,'position_y':1,'width':2,'height':2,
                              'nested_rooms':[{'name':'Ducha','position_x':0.2,'position_y':0.2,'width':1,'height':1}]}]},
            {'name':'Cocina','position_x':7,'position_y':1,'width':3,'height':5}
        ],
        'connections': [
            {'origin_room':'Baño','destination_room':'Dormitorio','wall_position':'north','width':0.9}
        ]
    })


def test_schema_preserves_nested_rooms():
    m = sample_map()
    restored = SemanticMap.model_validate_json(m.model_dump_json())
    assert restored.rooms[0].nested_rooms[0].nested_rooms[0].name == 'Ducha'


def test_absolute_coordinate_composition():
    rooms = list(_iter_absolute_rooms(sample_map().rooms))
    assert [(r.name,round(x,1),round(y,1)) for r,x,y,_ in rooms] == [
        ('Dormitorio',1,1),('Baño',4,2),('Ducha',4.2,2.2),('Cocina',7,1)]


def test_parent_translation_moves_child_without_mutating_local_coordinates():
    m = sample_map()
    m.rooms[0].position_x += 2
    assert m.rooms[0].nested_rooms[0].position_x == 3
    child = list(_iter_absolute_rooms(m.rooms))[1]
    assert child[1:3] == (6,2)


def test_render_nested_patches_and_connections():
    fig = generate_floor_plan(sample_map())
    try:
        rects=[p for p in fig.axes[0].patches if isinstance(p,Rectangle)]
        assert any(tuple(r.get_xy()) == (4,2) and r.get_width()==2 for r in rects)
        assert any(tuple(round(v,1) for v in r.get_xy()) == (4.2,2.2) for r in rects)
        assert len(fig.axes[0].lines) >= 1
    finally:
        plt.close(fig)
