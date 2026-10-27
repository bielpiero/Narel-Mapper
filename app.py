import os
from dotenv import load_dotenv
from ui.streamlit_ui import main

# Cargar variables de entorno (Ollama no necesita API key).
load_dotenv()

if __name__ == "__main__":
    main()