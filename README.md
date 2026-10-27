# ROBOT-MAP-BUILDER
Chatbot capaz de generar mapa semántico para un robot junto con una representación grafica

# 🤖 AI Semantic Map Builder (LangGraph Edition)

&gt; **Asistente inteligente para la generación de mapas semánticos robóticos mediante lenguaje natural.**

Este proyecto implementa un sistema avanzado capaz de generar mapas semánticos para robots (habitaciones, conexiones, objetos) interpretando comandos de voz o texto. Utiliza **LangGraph** para la gestión del estado conversacional y **Streamlit** para la visualización en tiempo real.

---

## ✨ Características Principales

* **🗣️ Interfaz Conversacional:** Describe el entorno ("Crea una cocina de 4x3m al norte del salón") y el bot lo interpretará.
* **🧠 Grafo de Estado (LangGraph):** Gestión robusta de la memoria y el flujo de la conversación.
* **📐 Planos Técnicos 2D:** Visualización automática con `matplotlib` (cuadrícula de ingeniería, cotas y centros).
* **💾 Exportación Multi-formato:** Descarga el mapa generado en **JSON**, **XML** y **YAML**.
* **🤖 Modelos Flexibles:** Soporte para **Ollama local**, **Google Gemini** y **OpenAI GPT**.

---

## 🚀 Requisitos Previos

* **Python 3.10** o superior.
* Ollama instalado para inferencia local; alternativamente, una **API Key** de Gemini u OpenAI.
* **Git** instalado en tu sistema.

---

## 🛠️ Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu ordenador local:

### 1. Clonar el repositorio

Descarga el código fuente y entra en la carpeta:

```bash
git clone https://github.com/nataliadiaz1/map-robot-builder.git
cd map-robot-builder
```

### 2. Crear un entrono virtual

Es muy recomendable usar un entorno virtual para aislar las dependencias:

En Windows:
```bash
python -m venv <nombre_entorno>
.\venv\Scripts\activate
```

En Mac/Linux (cualqueira de las dos opciones):
```bash
python -m venv venv
source venv/bin/activate

python -m venv venv
source ./Scripts/activate
```

### 2. Instalar dependencias

Instala todas las librerías necesarias:
```bash
pip install -r requirements.txt
```

### 3. Configuracion de variables

Para la opción local, Ollama no necesita API key. Crea `.env` junto a `app.py` copiando `.env.example`. Una configuración mínima es:
```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_NUM_CTX=8192
LLM_TEMPERATURE=0
```

### 4. Ejecución
Una vez instalado y configurado, inicia la interfaz web con:
```bash
streamlit run app.py
```
Esto abrirá automáticamente una pestaña en tu navegador (generalmente en http://localhost:8501) donde podrás empezar a diseñar tus mapas.







---

## Ejecutar NAREL-Mapper localmente con Ollama (Windows)

El proyecto ahora acepta `LLM_PROVIDER=ollama` y no requiere credenciales de
OpenAI ni de Google cuando se usa el servidor Ollama local.

1. Instala Ollama para Windows: https://ollama.com/download/windows
2. En PowerShell, descarga el modelo:

   ```powershell
   ollama pull qwen3:4b
   ollama list
   ```

3. Dentro de la carpeta que contiene `app.py`, crea el entorno virtual,
   actívalo e instala las dependencias:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   Si la política de PowerShell impide activar el entorno, también puedes
   ejecutar directamente ` .\.venv\Scripts\python.exe` sin activarlo.

4. Copia `.env.example` como `.env` sin incluir claves de servicios comerciales:

   ```powershell
   Copy-Item .env.example .env
   ```

   El valor inicial de `OLLAMA_MODEL` es `qwen3:4b`. Para cambiar a un modelo
   más pequeño, descarga `qwen3:1.7b` y cambia esa línea en `.env`.

5. Comprueba el servidor y la **salida estructurada completa de NAREL**:

   ```powershell
   ollama list
   python scripts\smoke_ollama.py
   ```

6. Inicia la interfaz:

   ```powershell
   python -m streamlit run app.py
   ```

En otra terminal, `ollama ps` permite observar dónde se está ejecutando el
modelo. La Quadro M2000M tiene poca VRAM; las peticiones con mapas extensos
pueden necesitar RAM/CPU y ser lentas. `OLLAMA_NUM_CTX` controla el contexto
solicitado, **no** el contexto máximo del modelo.

### Comprobaciones y limitaciones

- La salida de Ollama se solicita como JSON Schema y se valida con el modelo
  Pydantic `AgentResponse` (`method="json_schema"`). Esto comprueba la
  estructura, **no** que las habitaciones estén geométricamente bien situadas.
- No se ha modificado el backend de validación geométrica en esta migración.
  Los defectos recogidos en la auditoría de NAREL-Mapper siguen pendientes.
- `scripts/smoke_ollama.py` requiere Ollama ejecutándose y el modelo descargado.
- `python -m pytest tests/test_ollama_configuration.py` ejecuta tests de
  configuración sin consumir tokens ni invocar al modelo.
- Puedes volver a los servicios comerciales cambiando `LLM_PROVIDER` a
  `openai` o `gemini` y configurando su clave; se conserva la misma fábrica.


## Restauración funcional: visualización y anidación (octubre 2026)

- El plano aparece ahora **en el panel principal** cuando existe un mapa. Los errores se muestran en pantalla en vez de ocultarse.
- Botón **Demo 2D sin LLM**: carga un dormitorio con baño anidado y cocina y dibuja el plano sin Ollama.
- `Room.nested_rooms` se conserva en Pydantic y se dibuja de manera recursiva, tomando coordenadas relativas al padre.
- Se pueden desplazar las habitaciones anidadas usando `move_intents` (traslaciones); mover el padre traslada visualmente a las hijas sin modificar sus coordenadas locales.
- Las rotaciones de un padre con habitaciones anidadas y las rotaciones de subhabitaciones se rechazan en esta restauración mínima: todavía no existe una transformación rígida general de los subframes.
- El gráfico se cachea para evitar reconstruirlo en cada `rerun` de Streamlit y no se fuerza a recrear LangGraph en cada interacción.
- Se informa en UI de la duración de la última llamada LLM. Ajusta `OLLAMA_NUM_CTX` y `MAX_CONTEXT_MESSAGES` en `.env`: 6144 y 2 son valores para pruebas rápidas, no parámetros ya validados científicamente.
- Esta revisión **no** repara aún todas las inconsistencias geométricas de la auditoría anterior, **no** añade polígonos, ni demuestra que Ollama genere correctamente mapas complejos.

**Para probar:** `python -m streamlit run app.py`, luego pulsa **Demo 2D sin LLM**. Si el plano se ve, prueba mensajes con el LLM por separado.
