# =============================================================================
# src/ui/tab1_signal.py
# Tab 1 — Señal BVP y ciclo del sueño nocturno.
#
# Muestra:
#   1. Métricas del modelo (Accuracy, F1, Sensitivity, Specificity)
#   2. Señal BVP cruda vs filtrada (30 segundos)
#   3. Ciclo del sueño nocturno (hipnograma con colores por fase)
#   4. Distribución de fases en % con indicadores normal/anormal
# =============================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import LabelEncoder
import joblib

from src.utils import preprocesar_dataset, obtener_clases_ordenadas
from src.evaluation import calcular_metricas_completas


# Colores por fase del sueño — consistentes en toda la interfaz
COLORES_FASES = {
    'Wake':  '#534AB7',
    'Sleep': '#1D9E75',
    'NREM':  '#1D9E75',
    'Light': '#1D9E75',
    'Deep':  '#BA7517',
    'REM':   '#D85A30',
}

# Rangos clínicos normales para adultos (en porcentaje)
RANGOS_NORMALES = {
    'Wake':  (5,  10),
    'Sleep': (85, 95),
    'NREM':  (45, 75),
    'Light': (40, 55),
    'Deep':  (15, 25),
    'REM':   (20, 25),
}


# =============================================================================
# FUNCIÓN PRINCIPAL DEL TAB
# =============================================================================

# DESCRIPCIÓN: Controlador principal de la pestana de analisis individual (Tab 1). Gestiona el estado 
#              de la sesion de Streamlit (session_state) para evitar recargas innecesarias, evalua el disparador 
#              del boton de analisis, carga las predicciones/caracteristicas del paciente y coordina el renderizado 
#              secuencial de metricas, senal fotopletismografica (BVP/PPG), hipnograma y distribucion de fases.
# PARÁMETROS:
#   - config (dict): Ajustes seleccionados en la barra lateral (paciente, modelo, num_clases, ruta_proyecto, analizar).
# RETORNO:
#   - None: Inyecta secuencialmente las secciones visuales e indicadores en la interfaz del Tab 1.

def mostrar_tab1(config: dict):
    clave = f"tab1_{config['paciente']}_{config['modelo']}_{config['num_clases']}"

    if not config.get('analizar') and clave not in st.session_state:
        _mostrar_instrucciones()
        return

    if not config.get('paciente'):
        st.warning("Selecciona un paciente en el sidebar.")
        return

    if config.get('analizar') or clave not in st.session_state:
        with st.spinner("Cargando datos del paciente..."):
            datos = _cargar_datos_paciente(config)
        if datos is None:
            return
        st.session_state[clave] = datos
    else:
        datos = st.session_state[clave]

    _mostrar_metricas(datos)
    st.divider()
    _mostrar_senal_bvp(config)
    st.divider()
    _mostrar_ciclo_sueno(datos, config)
    st.divider()
    _mostrar_distribucion_fases(datos, config)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

# DESCRIPCIÓN: Carga el modelo clasificador serializado (.pkl) correspondiente al escenario actual, 
#              extrae y filtra las características (features) y etiquetas del paciente seleccionado, 
#              ejecuta la división y preprocesamiento de datos, verifica la integridad de clases del paciente 
#              y calcula las predicciones, métricas completas y distribución porcentual de fases del sueño.
# PARÁMETROS:
#   - config (dict): Diccionario de configuración con paciente, modelo, num_clases y ruta_proyecto.
# RETORNO:
#   - dict | None: Diccionario con los objetos procesados del paciente (modelo, X_train, X_test, y_test_str, 
#                  pred_str, clases, clases_ordenadas, metricas, distribucion, le) o None si ocurre una advertencia 
#                  o fallo en la carga/clases.

def _cargar_datos_paciente(config: dict) -> dict | None:
    ruta_proyecto = config['ruta_proyecto']
    modelo_key    = config['modelo']
    num_clases    = config['num_clases']

    # Cargar modelo entrenado
    ruta_pkl = os.path.join(
        ruta_proyecto, 'outputs', 'models',
        f"{modelo_key}_{num_clases}_clases.pkl"
    )
    if not os.path.exists(ruta_pkl):
        st.error(f"Modelo no encontrado: {ruta_pkl}")
        return None

    modelo = joblib.load(ruta_pkl)

    # Cargar features y etiquetas
    X_raw = pd.read_csv(os.path.join(ruta_proyecto, 'data', 'processed', 'X_features.csv'))
    y_raw = pd.read_csv(os.path.join(ruta_proyecto, 'data', 'processed', 'y_labels.csv'))
    y_raw = y_raw['Sleep_Stage'].values

    # Filtrar solo el paciente seleccionado si existe la columna
    paciente_id = config['paciente'].replace('.csv', '')
    if 'id_paciente' in X_raw.columns:
        mascara     = X_raw['id_paciente'] == paciente_id
        X_paciente  = X_raw[mascara].copy()
        y_paciente  = y_raw[mascara.values]
    else:
        X_paciente = X_raw.copy()
        y_paciente = y_raw

    # Eliminar columnas de metadatos
    cols_meta  = ['id_paciente']
    X_clean    = X_paciente.drop(columns=[c for c in cols_meta if c in X_paciente.columns])

    # Preprocesar y predecir
    X_train, X_val, X_test, y_train_str, y_val_str, y_test_str = preprocesar_dataset(
        X_clean, y_paciente, num_clases,
        test_size=0.20, val_size=0.10, random_state=42
    )

    clases_ordenadas  = obtener_clases_ordenadas(num_clases)
    clases_paciente   = np.unique(np.concatenate([y_train_str, y_test_str]))
    clases_faltantes  = [c for c in clases_ordenadas if c not in clases_paciente]

    if clases_faltantes:
        st.warning(
            f"⚠️ El paciente **{config['paciente']}** no tiene epochs "
            f"de la fase **{', '.join(clases_faltantes)}** "
            f"en el escenario de {num_clases} clases.\n\n"
            f"**Opciones:**\n"
            f"- Selecciona el escenario de **3 clases** (Wake / NREM / REM)\n"
            f"- Elige otro paciente que tenga todas las fases del sueño"
        )
        return None


    le      = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_test  = le.transform(y_test_str)
    clases  = le.classes_

    predicciones     = modelo.predict(X_test)
    y_test_str_local = le.inverse_transform(y_test)
    pred_str         = le.inverse_transform(predicciones)

    # Calcular métricas
    clases_ordenadas = obtener_clases_ordenadas(num_clases)
    metricas         = calcular_metricas_completas(
        y_test_str_local, pred_str, clases_ordenadas
    )

    # Calcular distribución de fases
    distribucion = _calcular_distribucion(pred_str, clases_ordenadas)

    return {
        'modelo':          modelo,
        'X_train':         X_train,
        'X_test':          X_test,
        'y_test_str':      y_test_str_local,
        'pred_str':        pred_str,
        'clases':          clases,
        'clases_ordenadas': clases_ordenadas,
        'metricas':        metricas,
        'distribucion':    distribucion,
        'le':              le,
    }

# DESCRIPCIÓN: Calcula el porcentaje de permanencia y distribución temporal en cada fase del sueño 
#              a partir del array de predicciones en texto, garantizando la presentación en el orden 
#              canónico definido por el escenario de clasificación.
# PARÁMETROS:
#   - pred_str (np.ndarray): Array unidimensional con las etiquetas predichas en formato texto por epoch.
#   - clases_ordenadas (list): Lista de cadenas con el orden canónico de las fases según el escenario.
# RETORNO:
#   - dict: Diccionario {fase: porcentaje_redondeado} respetando la secuencia de clases_ordenadas.

def _calcular_distribucion(pred_str: np.ndarray,
                           clases_ordenadas: list) -> dict:
    total = len(pred_str)
    return {
        clase: round(np.sum(pred_str == clase) / total * 100, 1)
        for clase in clases_ordenadas
    }

# =============================================================================
# COMPONENTE 1 — MÉTRICAS DEL MODELO
# =============================================================================

# DESCRIPCIÓN: Renderiza un panel de cuatro tarjetas métricas (st.metric) en Streamlit para visualizar 
#              los indicadores globales de rendimiento del modelo: Accuracy, F1-Score, Sensitivity 
#              (sensibilidad) y Specificity (especificidad), expresados en porcentaje con un decimal.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente que contiene la clave 'metricas' con los valores numéricos.
# RETORNO:
#   - None: Inyecta directamente las cuatro tarjetas informativas en la interfaz gráfica.

def _mostrar_metricas(datos: dict):
    st.markdown("#### Métricas del modelo")

    metricas = datos['metricas']
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Accuracy",
        value=f"{metricas['Accuracy']*100:.1f}%",
        help="Porcentaje total de epochs clasificados correctamente."
    )
    col2.metric(
        label="F1-Score",
        value=f"{metricas['F1-Score']*100:.1f}%",
        help="Media armónica entre Precision y Recall (weighted)."
    )
    col3.metric(
        label="Sensitivity",
        value=f"{metricas['Sensitivity']*100:.1f}%",
        help="Proporción de fases reales identificadas correctamente (macro)."
    )
    col4.metric(
        label="Specificity",
        value=f"{metricas['Specificity']*100:.1f}%",
        help="Proporción de negativos identificados correctamente (macro)."
    )


# =============================================================================
# COMPONENTE 2 — SEÑAL BVP CRUDA VS FILTRADA
# =============================================================================

# DESCRIPCIÓN: Carga un segmento de 30 segundos de la señal fotopletismográfica (BVP) del paciente 
#              desde el CSV en 'data/raw/', aplica un filtro suavizador de Savitzky-Golay para atenuar 
#              ruido de movimiento y visualiza ambas trazas (cruda vs filtrada) en un gráfico Plotly interactivo.
# PARÁMETROS:
#   - config (dict): Diccionario de configuración con las claves 'ruta_proyecto' y 'paciente'.
# RETORNO:
#   - None: Inyecta directamente el gráfico de la señal y la nota explicativa en Streamlit.

def _mostrar_senal_bvp(config: dict):
    st.markdown("#### Señal BVP — 30 segundos de ejemplo")

    ruta_csv = os.path.join(
        config['ruta_proyecto'], 'data', 'raw', config['paciente']
    )

    if not os.path.exists(ruta_csv):
        st.warning(f"No se encontró el CSV: {config['paciente']}")
        return

    df_raw    = pd.read_csv(ruta_csv)
    df_valido = df_raw[~df_raw['Sleep_Stage'].isin(['P', 'Missing'])].reset_index(drop=True)

    # Buscar un segmento representativo con algo de movimiento
    bvp_crudo    = df_valido['BVP'].values[15000:16920]
    bvp_filtrado = savgol_filter(bvp_crudo, window_length=9, polyorder=3)
    tiempo       = np.arange(len(bvp_crudo)) / 64

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tiempo, y=bvp_crudo,
        name='BVP crudo',
        line=dict(color='#B4B2A9', width=1),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=tiempo, y=bvp_filtrado,
        name='BVP filtrado (Savitzky-Golay)',
        line=dict(color='#534AB7', width=1.5),
        showlegend=True
    ))

    fig.update_layout(
        xaxis_title="Tiempo (segundos)",
        yaxis_title="Amplitud (u.a.)",
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=250
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "La línea azul (filtrada) elimina el ruido de movimiento "
        "preservando la morfología del pulso cardíaco."
    )


# =============================================================================
# COMPONENTE 3 — CICLO DEL SUEÑO NOCTURNO
# =============================================================================

# DESCRIPCIÓN: Muestra la arquitectura y progresión nocturna del sueño (hipnograma) como un gráfico 
#              escalonado interactivo en Plotly. Asigna a cada fase un nivel único en el eje Y y su 
#              correspondiente paleta de color, graficando la secuencia temporal convertida a horas.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente que contiene las predicciones ('pred_str') y las clases ordenadas ('clases_ordenadas').
#   - config (dict): Ajustes de la barra lateral con el nombre legible del modelo ('modelo_nombre') y el número de clases ('num_clases').
# RETORNO:
#   - None: Inyecta el hipnograma escalonado y el pie explicativo directamente en la interfaz de Streamlit.

def _mostrar_ciclo_sueno(datos: dict, config: dict):
    st.markdown("#### Ciclo del sueño nocturno")

    pred_str         = datos['pred_str']
    clases_ordenadas = datos['clases_ordenadas']

    # Convertir fases a valores numéricos para el eje Y
    mapa_y = {clase: i for i, clase in enumerate(clases_ordenadas[::-1])}
    y_vals = [mapa_y.get(f, 0) for f in pred_str]
    tiempo = [i * 30 / 3600 for i in range(len(pred_str))]

    # Crear colores para cada punto
    colores_linea = [COLORES_FASES.get(f, '#888780') for f in pred_str]

    fig = go.Figure()

    # Añadir una traza por fase para la leyenda
    for fase in clases_ordenadas:
        mascara = [f == fase for f in pred_str]
        x_fase  = [tiempo[i] for i, m in enumerate(mascara) if m]
        y_fase  = [y_vals[i]  for i, m in enumerate(mascara) if m]

        if x_fase:
            fig.add_trace(go.Scatter(
                x=x_fase, y=y_fase,
                mode='markers',
                marker=dict(color=COLORES_FASES.get(fase, '#888780'),
                            size=4, symbol='square'),
                name=fase,
                showlegend=True
            ))

    # Línea escalonada principal
    fig.add_trace(go.Scatter(
        x=tiempo, y=y_vals,
        mode='lines',
        line=dict(color='#888780', width=1, shape='hv'),
        showlegend=False,
        hovertemplate='Hora: %{x:.2f}h<br>Fase: %{text}<extra></extra>',
        text=pred_str
    ))

    fig.update_layout(
        xaxis_title="Tiempo (horas)",
        yaxis=dict(
            tickvals=list(range(len(clases_ordenadas))),
            ticktext=clases_ordenadas[::-1]
        ),
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=280
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Ciclo del sueño predicho por {config['modelo_nombre']} "
        f"({config['num_clases']} clases) — "
        f"{len(pred_str)} epochs de 30 segundos."
    )


# =============================================================================
# COMPONENTE 4 — DISTRIBUCIÓN DE FASES
# =============================================================================

# DESCRIPCIÓN: Despliega la métrica porcentual de cada fase del sueño comparándola contra los 
#              rangos clínicos de referencia para adultos (indicando si el valor es Bajo, Alto o Normal) 
#              y renderiza un gráfico de dona (Pie Plotly) interactivo para visualizar la proporción global del ciclo.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente que incluye la 'distribucion' y las 'clases_ordenadas'.
#   - config (dict): Ajustes de configuración globales del usuario.
# RETORNO:
#   - None: Inyecta directamente los indicadores de estado, el gráfico de dona y la nota clínica en Streamlit.

def _mostrar_distribucion_fases(datos: dict, config: dict):
    st.markdown("#### Distribución de fases del sueño")

    distribucion     = datos['distribucion']
    clases_ordenadas = datos['clases_ordenadas']

    cols = st.columns(len(clases_ordenadas))

    for col, fase in zip(cols, clases_ordenadas):
        pct   = distribucion.get(fase, 0)
        rango = RANGOS_NORMALES.get(fase, (0, 100))
        mins  = rango[0]
        maxs  = rango[1]

        # Determinar estado clínico
        if pct < mins:
            estado = "↓ Bajo"
            color  = "inverse"
        elif pct > maxs:
            estado = "↑ Alto"
            color  = "inverse"
        else:
            estado = "✓ Normal"
            color  = "normal"

        col.metric(
            label=fase,
            value=f"{pct}%",
            delta=estado,
            delta_color=color,
            help=f"Rango normal adulto: {mins}-{maxs}%"
        )

    # Gráfico de tarta interactivo
    fig = px.pie(
        values=list(distribucion.values()),
        names=list(distribucion.keys()),
        color=list(distribucion.keys()),
        color_discrete_map=COLORES_FASES,
        hole=0.4
    )
    fig.update_layout(
        legend=dict(orientation='h', y=-0.1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Rangos normales adulto — "
        "Wake: 5-10% · Light: 40-55% · Deep: 15-25% · REM: 20-25%"
    )


# =============================================================================
# INSTRUCCIONES INICIALES
# =============================================================================

# DESCRIPCIÓN: Despliega una alerta informativa inicial (st.info) en el área principal de la interfaz 
#              para guiar al usuario sobre los pasos previos requeridos en la barra lateral antes de 
#              ejecutar el pipeline de análisis individual.
# PARÁMETROS:
#   - Ninguno.
# RETORNO:
#   - None: Inyecta directamente el mensaje con las instrucciones de uso en Streamlit.

def _mostrar_instrucciones():
    st.info(
        "👈 Selecciona un paciente, modelo y escenario en el sidebar "
        "y pulsa **Analizar** para ver los resultados."
    )