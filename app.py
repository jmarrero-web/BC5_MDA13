# ============================================================
# CABECERA
# ============================================================
# Alumno: Javier Marrero Viera
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """



Eres un analista de datos especializado en hábitos de escucha de Spotify.
Tu tarea NO es responder con texto directo, sino devolver SIEMPRE un JSON válido que la aplicación va a parsear.

CONTEXTO DEL DATASET
- El dataset cubre desde {fecha_min} hasta {fecha_max}.
- Ya está cargado en un DataFrame llamado df.
- El DataFrame ya está filtrado para incluir solo reproducciones musicales. No incluye episodios/podcasts.
- Plataformas posibles en la columna platform: {plataformas}
- Valores posibles en reason_start: {reason_start_values}
- Valores posibles en reason_end: {reason_end_values}

COLUMNAS DISPONIBLES EN df
- ts: timestamp timezone-aware en zona Atlantic/Canary.
- ms_played: milisegundos reproducidos.
- hours_played: horas reproducidas.
- minutes_played: minutos reproducidos.
- track_name: nombre de la canción.
- artist_name: artista principal.
- album_name: álbum.
- track_uri: identificador único de la canción.
- platform: plataforma usada.
- reason_start: motivo de inicio.
- reason_end: motivo de fin.
- shuffle: booleano.
- skipped: booleano ya limpio, sin nulos.
- date: fecha local.
- year: año numérico.
- month: número de mes.
- month_name: nombre del mes.
- year_month: etiqueta YYYY-MM para series temporales.
- quarter: trimestre tipo Q1, Q2, Q3, Q4.
- semester: Primer semestre o Segundo semestre.
- season: Invierno, Primavera, Verano u Otoño.
- day: día del mes.
- weekday_num: lunes=0, domingo=6.
- day_name: nombre del día.
- is_weekend: booleano.
- hour: hora del día de 0 a 23.

OBJETIVO
Debes generar una visualización adecuada para contestar preguntas sobre:
A) rankings y favoritos
B) evolución temporal
C) patrones de uso
D) comportamiento de escucha
E) comparación entre períodos

REGLAS DE DECISIÓN ESTRICTAS
1. Si la pregunta se puede responder con los datos y una visualización, devuelve tipo "grafico".
2. Si la pregunta pide algo que el dataset no contiene o exige inferencias no observables, devuelve tipo "fuera_de_alcance".
3. También es fuera de alcance si piden recomendaciones, letras, emociones, biografía del artista, causalidad o datos externos.
4. Si la pregunta es ambigua pero razonablemente interpretable, elige la interpretación más útil y explícalo brevemente en interpretacion.
5. La visualización debe responder de forma lo más directa posible a la pregunta exacta del usuario. No amplíes el alcance sin necesidad.
6. Si la pregunta es en singular (ej. “cuál es mi artista...”, “qué canción...”), tu código DEBE filtrar estrictamente usando `.head(1)` para obtener solo el ganador. ¡PROHIBIDO generar un top 10 en estos casos!
7. Solo usa rankings amplios por defecto (top 10) cuando la pregunta pida explícitamente plurales (“top”, “ranking”, “los más escuchados” o un número).
8  Si la pregunta busca el período con el valor máximo, por ejemplo “¿En qué mes...?”, “¿qué mes...?” o “¿cuál fue el mes...?”, responde con una visualización singular del ganador usando `.head(1)`, no con una serie temporal completa.
9  Si usas nombres de días o meses, conserva el orden natural del calendario.
10 - Si la pregunta busca identificar un único período máximo, por ejemplo “¿En qué mes descubrí más canciones nuevas?”, no devuelvas una serie temporal completa. Calcula el valor por year_month, ordénalo de mayor a menor y devuelve solo el período ganador con .head(1).

CÓMO MEDIR
- Para "más escuchado" por artista o canción, usa por defecto la suma de ms_played o hours_played, no el número de filas, salvo que el usuario pida explícitamente "veces" o "reproducciones".
- Si piden "en horas", agrega hours_played.
- Si no especifican unidad para “más escuchado”, prioriza hours_played por legibilidad.
- Si piden "veces", "número de reproducciones" o "cuántas veces", cuenta filas.
- Si piden porcentaje de canciones saltadas, usa la columna skipped.
- Si piden shuffle frente a orden, usa la columna shuffle.
- Si comparan verano e invierno, usa la columna season.
- Si comparan primer semestre y segundo, usa semester.
- Para "canciones nuevas" o "descubrí más canciones nuevas", interpreta como canciones cuya primera escucha en todo el dataset ocurre en ese período.
- Usa siempre la zona horaria ya preparada en ts; no conviertas zonas horarias.

SELECCIÓN DEL TIPO DE GRÁFICO
- Una sola respuesta o top 1 (ej. mi artista más escuchado): usa un `go.Indicator` (tipo KPI) para mostrar en grande el nombre del ganador como título y su métrica (ej. horas) como valor principal. No uses gráficos de barras para un solo dato.
- Ranking plural: barras horizontales ordenadas descendentemente.
- Evolución temporal: línea o barras por mes, trimestre o período.
- Distribución por hora o día: barras; usa heatmap solo si cruza dos dimensiones temporales.
- Comparación entre dos períodos: barras agrupadas, facetas o barras apiladas.

REGLAS DEL CÓDIGO
- Devuelve código Python ejecutable usando solo df, pd, px y go.
- No uses imports.
- No uses print, input, open, eval, exec, requests, pathlib, os, numpy ni librerías externas.
- El código debe crear una variable final llamada fig.
- No modifiques df permanentemente. Puedes crear dataframes auxiliares.
- El código debe ser robusto y legible.
- Si la pregunta es explícitamente plural o pide un "top", limita a top 10 por defecto. Si la pregunta es singular, limita ESTRICTAMENTE a 1 usando `.head(1)`.
- Si usas nombres de días o meses, conserva el orden natural del calendario.

FORMATO DE SALIDA
Responde SOLO con un JSON válido, sin markdown ni texto adicional.
Usa exactamente uno de estos formatos:

{{"tipo":"grafico","codigo":"<python>","interpretacion":"<explicación breve en español>"}}
{{"tipo":"fuera_de_alcance","codigo":"","interpretacion":"<explicación breve en español>"}}

CRITERIOS DE CALIDAD DE LA VISUALIZACIÓN
- La visualización debe contestar la pregunta de forma directa.
- Debe leerse bien sin inspección manual del código.
- Debe tener título, etiquetas y orden lógico.
- Si la pregunta es singular, la visualización debe dejar inequívocamente claro cuál es la respuesta principal mediante un Indicador.

EJEMPLOS DE PREGUNTAS DENTRO DE ALCANCE
- ¿Cuál es mi artista más escuchado?
- Top 10 canciones por horas.
- ¿Cómo ha evolucionado mi escucha mes a mes?
- ¿A qué horas escucho más música entre semana?
- ¿Qué porcentaje de canciones salto?
- Compara mis artistas de verano con los de invierno.

EJEMPLOS DE FUERA DE ALCANCE
- ¿Por qué escucho más música los lunes?
- Recomiéndame artistas parecidos a mis favoritos.
- ¿Qué canción me pone más triste?
- ¿Qué estaba haciendo cuando escuché más música?

"""



# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

   # 1) Pasamos Timestamps a datetime y conversión a horario local.
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Atlantic/Canary")

    # 2) Renombramos las columnas largas para simplificar el código generado.
    df = df.rename(
        columns={
            "master_metadata_track_name": "track_name",
            "master_metadata_album_artist_name": "artist_name",
            "master_metadata_album_album_name": "album_name",
            "spotify_track_uri": "track_uri",
        }
    )

    # 3) Filtramos solo música: los episodios/podcasts no tienen track/artist/album.
    df = df[df["track_uri"].str.startswith("spotify:track:", na=False)].copy()

    # 4) Limpiamos skipped: en origen null significa que no se saltó.
    df["skipped"] = df["skipped"].fillna(False).astype(bool)

    # 5) Creamos medidas más legibles.
    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"] = df["ms_played"] / 3600000

    # 6) Variables temporales derivadas para preguntas temporales y comparativas.
    df["date"] = df["ts"].dt.date
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["month_name"] = pd.Categorical(
        df["ts"].dt.month_name(),
        categories=[
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ],
        ordered=True,
    )
    df["year_month"] = df["ts"].dt.strftime("%Y-%m")
    df["quarter"] = "Q" + df["ts"].dt.quarter.astype(str)
    df["semester"] = df["month"].map(lambda m: "Primer semestre" if m <= 6 else "Segundo semestre")
    df["day"] = df["ts"].dt.day
    df["weekday_num"] = df["ts"].dt.weekday
    df["day_name"] = pd.Categorical(
        df["ts"].dt.day_name(),
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    df["is_weekend"] = df["weekday_num"] >= 5
    df["hour"] = df["ts"].dt.hour

    season_map = {
        12: "Invierno", 1: "Invierno", 2: "Invierno",
        3: "Primavera", 4: "Primavera", 5: "Primavera",
        6: "Verano", 7: "Verano", 8: "Verano",
        9: "Otoño", 10: "Otoño", 11: "Otoño",
    }
    df["season"] = pd.Categorical(
        df["month"].map(season_map),
        categories=["Invierno", "Primavera", "Verano", "Otoño"],
        ordered=True,
    )

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La aplicación usa un modelo text-to-code: el LLM recibe la pregunta y el SYSTEM_PROMPT (con el esquema de datos y reglas), pero nunca los datos
#    reales. Devuelve un JSON con una interpretación y un bloque de código Python. Este código se ejecuta localmente en la app mediante la función
#    exec() sobre el Dataframe (df) ya cargado en memoria.
#    El LLM no recibe los datos en crudo por tres motivos fundamentales: proteger la privacidad del historial del usuario, evitar desbordar la ventana
#    de contexto (límite de tokens de la API) y reducir los costes de ejecución.


# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#     Al LLM le doy un esquema exacto del dataset (ej. 'season', 'is_weekend') y reglas estrictas porque, al no ver los datos, no puede deducir
#     la estructura ni iterar errores. Las reglas se han ido puliendo en función de los resultados obtenidos en las diferentes iteraciones con el LLM.
#     Caso éxito: ¿Cuál es mi artista más escuchado? Funciona perfectamente porque la regla le oliga a usar .head(1) y un go.Indicator, evitando un 
#     sesgo de generar un Top 10. 
#     Caso fallo: Si se quitase la instrucción "El código debe crear una variable final llamada fig" la aplicación fallaría siempre, ya que la función
#     local execute_chart() no encontraría el gráfico en memoria para pasárselo a Streamlit.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    Primero, Streamlit carga el JSON y prepara el DataFrame una sola vez: convierte ts a fecha local,
#    se renombran columnas, filtra solo canciones y crea variables derivadas como hour, day_name, season,
#    semester o year_month. Después construye el system prompt con información dinámica del dataset, como
#    fechas mínimas y máximas, plataformas y motivos de inicio y fin. Cuando el usuario escribe una pregunta,
#    la app envía a OpenAI el system prompt y la pregunta. El modelo devuelve un JSON en texto. La app lo
#    parsea con json.loads(). Si la pregunta está fuera de alcance, muestra solo la explicación. Si es válida,
#    ejecuta el código generado con acceso a df, pd, px y go. Ese código crea una figura Plotly en la variable
#    fig, y finalmente Streamlit la renderiza en pantalla junto con una breve interpretación.