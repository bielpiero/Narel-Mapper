"""Check nested rendering locally without Streamlit, Ollama or any LLM API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matplotlib import pyplot as plt
from models.semantic_models import SemanticMap
from services.map_visualizer import generate_floor_plan


def main():
    source = {
        'global_space': {'name':'Prueba sin LLM', 'width':12, 'height':10},
        'rooms': [
            {'name':'Dormitorio', 'position_x':1, 'position_y':1, 'width':6, 'height':5,
             'nested_rooms':[{'name':'Baño', 'position_x':3, 'position_y':1, 'width':2, 'height':2}]},
            {'name':'Cocina', 'position_x':7,'position_y':1,'width':3,'height':5}
        ],
        'connections':[{'origin_room':'Baño','destination_room':'Dormitorio','width':0.9,'wall_position':'north'}]
    }
    semantic_map = SemanticMap.model_validate(source)
    out_path = Path(__file__).resolve().parents[1] / 'test_nested_map.png'
    fig = generate_floor_plan(semantic_map)
    try:
        fig.savefig(out_path, dpi=120)
    finally:
        plt.close(fig)
    print('MAP VISUALIZER: OK')
    print('Imagen creada:', out_path)
    print('Habitaciones anidadas:', len(semantic_map.rooms[0].nested_rooms))


if __name__ == '__main__':
    main()
