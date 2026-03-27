# BC5 — Spotify Analytics Assistant

Business Case 5 del programa MDA13 de la escuela de negocios ISDI.

Este repositorio es un ejercicio docente. Los datos son sintéticos y no corresponden a ningún usuario real de Spotify. La estructura del dataset está inspirada en el formato de exportación de datos de Spotify, pero ha sido generada específicamente para este caso.

Ante cualquier duda: mtaboada@isdi.education

Este proyecto consiste en un asistente analítico conversacional sobre un historial de escucha de Spotify. El usuario formula preguntas en lenguaje natural y la aplicación genera automáticamente una visualización con Plotly a partir de los datos.

## Enlaces

- GitHub:https://github.com/jmarrero-web/BC5_MDA13
- Streamlit Cloud:https://bc5mda13-cxi6qeiqkau5awethzqcrk.streamlit.app/

## Arquitectura

La solución sigue un enfoque text-to-code:

1. El usuario escribe una pregunta en lenguaje natural.
2. El LLM recibe un system prompt con la estructura del dataset y las reglas de análisis.
3. El modelo devuelve un JSON con:
   - tipo de respuesta
   - código Python
   - interpretación breve
4. La aplicación ejecuta ese código en local sobre el DataFrame.
5. Streamlit renderiza el gráfico en pantalla.

## Puesta en marcha

1. Clona el repositorio o descarga los archivos
2. Crea un entorno virtual y actívalo
3. Instala dependencias: `pip install -r requirements.txt`
4. Copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y rellena la API key y tu contraseña
5. Ejecuta: `streamlit run app.py`

## Archivos

| Archivo | Descripción |
|---|---|
| `app.py` | Esqueleto de la aplicación. Tu trabajo está aquí. |
| `streaming_history.json` | Dataset del caso (~15.000 registros) |
| `requirements.txt` | Dependencias fijadas. No modificar. |
| `.gitignore` | Excluye secrets del repositorio |
| `.streamlit/secrets.toml.example` | Plantilla para API key y contraseña. Copiar como `secrets.toml`. |


