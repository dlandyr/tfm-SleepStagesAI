# =============================================================================
# src/ui/dashboard.py
# Modo poblacional — análisis de todos los pacientes.
#
# Muestra 4 tabs:
#   Tab 1 — Estadísticas generales de los 100 pacientes
#   Tab 2 — Comparativa por subgrupos clínicos
#   Tab 3 — Ranking de modelos (RF vs KNN vs XGBoost vs SVM)
#   Tab 4 — Ranking de alertas por paciente
# =============================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.ui.tab1_signal import COLORES_FASES, RANGOS_NORMALES


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

# DESCRIPCIÓN: Controlador principal de la vista del cuadro de mando (dashboard) a nivel poblacional 
#              en Streamlit. Gestiona el flujo de análisis, la persistencia en caché (session_state) 
#              y el renderizado modular de las 4 pestañas analíticas principales (estadísticas generales, 
#              comparativa de subgrupos, ranking de modelos y alertas).
# PARÁMETROS:
#   - config (dict): Ajustes globales del usuario (ruta del proyecto, número de clases de sueño y estado del botón de análisis).
# RETORNO:
#   - None: Modifica directamente el estado de la aplicación Streamlit e inyecta la interfaz gráfica.

def mostrar_dashboard(config: dict):    
    st.markdown("## Dashboard poblacional — todos los pacientes")

    clave = f"datos_pob_{config.get('num_clases', 4)}"

    if config['analizar']:
        # Limpiar solo la clave activa para forzar recálculo
        if clave in st.session_state:
            del st.session_state[clave]

    if clave not in st.session_state:
        if not config['analizar']:
            # No hay datos y no se pulsó Analizar — no hacer nada
            return
        with st.spinner("Cargando datos de todos los pacientes..."):
            datos_pob = _cargar_datos_poblacion(config)
        if datos_pob is None:
            st.error("No se encontraron datos procesados. Ejecuta run_pipeline.py primero.")
            return
        st.session_state[clave] = datos_pob
    else:
        datos_pob = st.session_state[clave]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Estadísticas generales",
        "🔬 Comparativa subgrupos",
        "🏆 Ranking de modelos",
        "🚨 Ranking de alertas"
    ])

    with tab1:
        _mostrar_estadisticas_generales(datos_pob, config)
    with tab2:
        _mostrar_comparativa_subgrupos(datos_pob, config)
    with tab3:
        _mostrar_ranking_modelos(config)
    with tab4:
        _mostrar_ranking_alertas(datos_pob, config)


# =============================================================================
# CARGA DE DATOS POBLACIONALES
# =============================================================================

# DESCRIPCIÓN: Carga los conjuntos de datos procesados (características y etiquetas) y realiza la lectura 
#              y armonización de los metadatos de los pacientes (mapeo de identificadores y filtrado). 
#              Asimismo, calcula las métricas estadísticas agregadas a nivel poblacional y por paciente.
# PARÁMETROS:
#   - config (dict): Diccionario de configuración con la ruta raíz del proyecto y el número de clases objetivo.
# RETORNO:
#   - dict | None: Diccionario con la estructura {'X', 'y', 'meta', 'stats_paciente', ...} con los datos 
#                  cargados o None si no se encuentran los archivos CSV de entrada.

def _cargar_datos_poblacion(config: dict) -> dict | None:    
    ruta_proyecto = config['ruta_proyecto']

    ruta_X = os.path.join(ruta_proyecto, 'data', 'processed', 'X_features.csv')
    ruta_y = os.path.join(ruta_proyecto, 'data', 'processed', 'y_labels.csv')

    if not os.path.exists(ruta_X):
        return None

    X = pd.read_csv(ruta_X)
    y = pd.read_csv(ruta_y)['Sleep_Stage'].values

    # Cargar metadatos si existen
    ruta_meta = os.path.join(ruta_proyecto, 'data', 'raw', 'info_pacientes.csv')
    if os.path.exists(ruta_meta):
        df_meta = pd.read_csv(ruta_meta)

        # Renombrar columnas al formato interno del proyecto
        df_meta = df_meta.rename(columns={
            'SID':    'id_paciente',
            'AGE':    'age',
            'GENDER': 'gender',
            'BMI':    'bmi',
            'AHI':    'AHI'
        })

        # Mapeo correcto: S002→paciente01, S003→paciente02, etc.
        # basado en el renombrado original de los archivos del dataset
        mapeo_ids = {
            'S002': 'paciente01',
            'S003': 'paciente02',
            'S004': 'paciente03',
            'S005': 'paciente04',
            'S006': 'paciente05',
            'S007': 'paciente06',
            'S008': 'paciente07',
            'S009': 'paciente08',
            'S010': 'paciente09',
            'S011': 'paciente10',
        }
        df_meta['id_paciente'] = df_meta['id_paciente'].map(mapeo_ids)

        # Eliminar filas de pacientes no utilizados (S012-S031)
        df_meta = df_meta.dropna(subset=['id_paciente'])

    else:
        df_meta = None

    # Calcular estadísticas por paciente
    stats_paciente = None
    if 'id_paciente' in X.columns:        
        stats_paciente = _calcular_stats_por_paciente(X, y, num_clases=config.get('num_clases', 4))

    return {
        'X':              X,
        'y':              y,
        'meta':           df_meta,
        'stats_paciente': stats_paciente,
        'num_clases':     config.get('num_clases', 4), 
        'n_pacientes':    X['id_paciente'].nunique() if 'id_paciente' in X.columns else 1,
        'n_epochs':       len(X),
    }

# DESCRIPCIÓN: Agrupa y calcula la distribución porcentual de las fases del sueño para cada paciente, 
#              aplicando dinámicamente la reasignación de etiquetas (map) según el número de clases 
#              configurado (2, 3 o 4 clases) y pivotando el resultado en formato tabular.
# PARÁMETROS:
#   - X (pd.DataFrame): DataFrame de características que contiene la columna 'id_paciente'.
#   - y (np.ndarray): Vector de etiquetas o clases de sueño observadas por época.
#   - num_clases (int, opcional): Esquema de clasificación a aplicar (2: Wake/Sleep, 3: Wake/NREM/REM, 4: Wake/Light/Deep/REM).
# RETORNO:
#   - pd.DataFrame: DataFrame normalizado donde cada fila representa un paciente y las columnas los porcentajes por fase.

def _calcular_stats_por_paciente(X: pd.DataFrame,
                                  y: np.ndarray,
                                  num_clases: int = 4) -> pd.DataFrame:
    if num_clases == 2:
        mapeo = {'W': 'Wake', 'N1': 'Sleep', 'N2': 'Sleep',
                 'N3': 'Sleep', 'R': 'Sleep'}
    elif num_clases == 3:
        mapeo = {'W': 'Wake', 'N1': 'NREM', 'N2': 'NREM',
                 'N3': 'NREM', 'R': 'REM'}
    else:
        mapeo = {'W': 'Wake', 'N1': 'Light', 'N2': 'Light',
                 'N3': 'Deep', 'R': 'REM'}

    df = X[['id_paciente']].copy()
    df['fase'] = y

    # Aplicar mapeo siempre — map() ignora valores que no están en el dict
    df['fase'] = df['fase'].astype(str).map(mapeo).fillna(df['fase'])

    pivot = df.groupby(['id_paciente', 'fase']).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot = pivot.round(1).reset_index()

    return pivot


# =============================================================================
# TAB 1 — ESTADÍSTICAS GENERALES
# =============================================================================

# DESCRIPCIÓN: Renderiza en la interfaz de Streamlit el bloque de estadísticas generales del dataset 
#              poblacional, mostrando las métricas globales de registro (pacientes, épocas y horas), 
#              los gráficos interactivos de distribución de fases (barras y sectores) y los metadatos demográficos.
# PARÁMETROS:
#   - datos_pob (dict): Diccionario con los datos procesados, conteo de épocas/pacientes y metadatos del dataset.
#   - config (dict): Ajustes globales del usuario para la visualización.
# RETORNO:
#   - None: Inyecta directamente los componentes visuales y métricas en la interfaz de Streamlit.

def _mostrar_estadisticas_generales(datos_pob: dict, config: dict):   
    st.markdown("#### Resumen del dataset")

    # Métricas globales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total pacientes", datos_pob['n_pacientes'])
    col2.metric("Total epochs", f"{datos_pob['n_epochs']:,}")
    col3.metric("Horas registradas", f"{datos_pob['n_epochs'] * 30 / 3600:.0f}h")
    col4.metric("Duración media/noche", f"{datos_pob['n_epochs'] * 30 / 3600 / datos_pob['n_pacientes']:.1f}h")

    st.divider()

    # Distribución global de fases
    st.markdown("#### Distribución global de fases del sueño")
    y = datos_pob['y']
    fases, conteos = np.unique(y, return_counts=True)
    total = len(y)

    df_dist = pd.DataFrame({
        'Fase':       fases,
        'Epochs':     conteos,
        'Porcentaje': (conteos / total * 100).round(1)
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_dist, x='Fase', y='Porcentaje',
            color='Fase',
            color_discrete_map=COLORES_FASES,
            title="Distribución de fases (% del total)",
            text='Porcentaje'
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(showlegend=False, height=350,
                         margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            df_dist, values='Epochs', names='Fase',
            color='Fase',
            color_discrete_map=COLORES_FASES,
            title="Proporción de epochs por fase",
            hole=0.4
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Metadatos demográficos si están disponibles
    if datos_pob['meta'] is not None:
        st.divider()
        _mostrar_demograficos(datos_pob['meta'])

# DESCRIPCIÓN: Renderiza en la interfaz de Streamlit el perfil demográfico y clínico de los participantes 
#              del estudio, incluyendo histogramas para la edad y el índice de apnea-hipopnea (AHI con 
#              umbrales de severidad clínica) y un gráfico de sectores para la distribución por género.
# PARÁMETROS:
#   - df_meta (pd.DataFrame): DataFrame con los metadatos clínicos y demográficos de los pacientes.
# RETORNO:
#   - None: Inyecta directamente los componentes gráficos interactivos en la interfaz gráfica.

def _mostrar_demograficos(df_meta: pd.DataFrame):   
    st.markdown("#### Perfil demográfico de los participantes")

    col1, col2 = st.columns(2)

    with col1:
        if 'age' in df_meta.columns:
            fig = px.histogram(
                df_meta, x='age',
                title="Distribución de edad",
                color_discrete_sequence=['#534AB7'],
                nbins=15
            )
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'gender' in df_meta.columns:
            conteo = df_meta['gender'].value_counts()
            fig = px.pie(
                values=conteo.values,
                names=conteo.index,
                title="Distribución por género",
                color_discrete_sequence=['#534AB7', '#1D9E75'],
                hole=0.4
            )
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    if 'AHI' in df_meta.columns:
        fig = px.histogram(
            df_meta, x='AHI',
            title="Índice de Apnea-Hipopnea (AHI) — distribución",
            color_discrete_sequence=['#D85A30'],
            nbins=20
        )
        fig.add_vline(x=5,  line_dash='dash', line_color='gray',
                      annotation_text="Normal (<5)")
        fig.add_vline(x=15, line_dash='dash', line_color='orange',
                      annotation_text="Leve (5-15)")
        fig.add_vline(x=30, line_dash='dash', line_color='red',
                      annotation_text="Moderada (15-30)")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 2 — COMPARATIVA POR SUBGRUPOS
# =============================================================================

# DESCRIPCIÓN: Gestiona el flujo y renderizado de la comparativa de subgrupos clínicos en Streamlit.
#              Normaliza los porcentajes por fase del sueño, cruza las estadísticas por paciente 
#              con sus metadatos demográficos/clínicos y delega el análisis gráfico según el criterio 
#              seleccionado por el usuario (edad, severidad de apnea o género).
# PARÁMETROS:
#   - datos_pob (dict): Diccionario con las estadísticas por paciente ('stats_paciente') y metadatos ('meta').
#   - config (dict): Ajustes globales del usuario para la visualización.
# RETORNO:
#   - None: Inyecta directamente el selector y las visualizaciones comparativas en la interfaz gráfica.

def _mostrar_comparativa_subgrupos(datos_pob: dict, config: dict):
    st.markdown("#### Comparativa por subgrupos clínicos")

    df_meta = datos_pob.get('meta')
    stats   = datos_pob.get('stats_paciente')

    if stats is None or df_meta is None:
        st.warning("Metadatos necesarios no disponibles para esta comparativa.")
        return

    # stats ya viene con columnas mapeadas al escenario (Wake, NREM, REM, etc.)
    # No hay que remapear — usar directamente
    stats_mapped = stats.copy()

    # Normalizar a porcentajes (por si acaso no suman 100)
    cols_fases = [c for c in stats_mapped.columns if c != 'id_paciente']
    total = stats_mapped[cols_fases].sum(axis=1)
    for col in cols_fases:
        stats_mapped[col] = (stats_mapped[col] / total * 100).round(1)

    # Unir stats con metadatos
    df_merged = stats_mapped.merge(df_meta, on='id_paciente', how='inner')

    if df_merged.empty:
        st.warning("No se pudieron combinar estadísticas con metadatos.")
        return

    # Selector de comparativa
    subgrupo = st.selectbox(
        "Seleccionar comparativa",
        options=[
            "Edad (jóvenes vs mayores)",
            "Apnea (con apnea vs sin apnea)",
            "Género (hombres vs mujeres)"
        ]
    )

    if "Edad" in subgrupo and 'age' in df_merged.columns:
        _comparar_por_edad(df_merged)
    elif "Apnea" in subgrupo and 'AHI' in df_merged.columns:
        _comparar_por_apnea(df_merged)
    elif "Género" in subgrupo and 'gender' in df_merged.columns:
        _comparar_por_genero(df_merged)
    else:
        st.info("Metadatos necesarios no disponibles para esta comparativa.")

# DESCRIPCIÓN: Segmenta a los pacientes en tres cohortes de edad ('Joven', 'Adulto' y 'Mayor') 
#              mediante la discretización de la variable 'age' en rangos predefinidos, y ejecuta 
#              la función auxiliar para graficar la distribución comparativa de fases del sueño.
# PARÁMETROS:
#   - df (pd.DataFrame): DataFrame unificado que contiene las características por paciente y la columna 'age'.
# RETORNO:
#   - None: Modifica el DataFrame de entrada e inyecta la visualización comparativa en la interfaz.

def _comparar_por_edad(df: pd.DataFrame):
    df['grupo_edad'] = pd.cut(
        df['age'],
        bins=[0, 35, 55, 100],
        labels=['Joven (<35)', 'Adulto (35-55)', 'Mayor (>55)']
    )
    _graficar_comparativa_subgrupos(df, 'grupo_edad', "Comparativa por grupo de edad")

# DESCRIPCIÓN: Categoriza a los pacientes según su severidad de apnea utilizando el índice AHI 
#              (umbral en 5 eventos/hora: 'Sin apnea' vs 'Con apnea') y delega la generación 
#              del gráfico comparativo de la distribución de fases del sueño.
# PARÁMETROS:
#   - df (pd.DataFrame): DataFrame unificado que contiene las estadísticas por paciente y la columna 'AHI'.
# RETORNO:
#   - None: Inyecta la representación gráfica comparativa de la apnea en la interfaz.

def _comparar_por_apnea(df: pd.DataFrame):
    df['grupo_apnea'] = pd.cut(
        df['AHI'],
        bins=[0, 5, 100],
        labels=['Sin apnea (AHI<5)', 'Con apnea (AHI≥5)']
    )
    _graficar_comparativa_subgrupos(df, 'grupo_apnea', "Comparativa por apnea")

# DESCRIPCIÓN: Mapea la variable de género ('gender') y delega directamente la generación del gráfico 
#              comparativo de la distribución de fases del sueño entre hombres y mujeres.
# PARÁMETROS:
#   - df (pd.DataFrame): DataFrame unificado que contiene las estadísticas por paciente y la columna 'gender'.
# RETORNO:
#   - None: Inyecta la representación gráfica comparativa por género en la interfaz.

def _comparar_por_genero(df: pd.DataFrame):
    _graficar_comparativa_subgrupos(df, 'gender', "Comparativa por género")

# DESCRIPCIÓN: Función auxiliar que transforma la estructura de datos (melt) y calcula las medias 
#              por subgrupo para renderizar un gráfico de barras agrupadas interactivas (Plotly) 
#              junto con una tabla de datos pivoteada en Streamlit.
# PARÁMETROS:
#   - df (pd.DataFrame): DataFrame con las características de los pacientes y la variable de agrupación.
#   - columna_grupo (str): Nombre de la columna utilizada para la segmentación (ej. 'grupo_edad', 'gender').
#   - titulo (str): Título principal para la figura y la sección gráfica.
# RETORNO:
#   - None: Inyecta el gráfico interactivo de barras y la tabla de resumen en la interfaz gráfica.

def _graficar_comparativa_subgrupos(df: pd.DataFrame,
                                     columna_grupo: str,
                                     titulo: str):
    fases_disponibles = [f for f in ['Wake', 'Light', 'Deep', 'REM', 'NREM', 'Sleep']
                         if f in df.columns]

    if not fases_disponibles:
        st.warning("No hay columnas de fases disponibles para comparar.")
        return

    df_melt = df[[columna_grupo] + fases_disponibles].melt(
        id_vars=columna_grupo,
        var_name='Fase',
        value_name='Porcentaje'
    )
    df_melt = df_melt.dropna()

    df_agg = df_melt.groupby([columna_grupo, 'Fase'])['Porcentaje'].mean().reset_index()

    fig = px.bar(
        df_agg,
        x='Fase', y='Porcentaje',
        color=columna_grupo,
        barmode='group',
        title=titulo,
        color_discrete_sequence=['#534AB7', '#1D9E75', '#D85A30']
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df_agg.pivot(index=columna_grupo, columns='Fase', values='Porcentaje').round(1),
        use_container_width=True
    )


# =============================================================================
# TAB 3 — RANKING DE MODELOS
# =============================================================================

# DESCRIPCIÓN: Carga la tabla global de métricas (resultados_completos.csv) y construye una vista 
#              comparativa interactiva en Streamlit. Renderiza la tabla completa de resultados y un gráfico 
#              de barras agrupadas que evalúa el rendimiento (Accuracy %) de los 4 modelos (RF, KNN, XGBoost y SVM) 
#              a través de los 3 escenarios de clasificación (2, 3 y 4 clases).
# PARÁMETROS:
#   - config (dict): Ajustes globales del usuario conteniendo la ruta raíz del proyecto.
# RETORNO:
#   - None: Inyecta directamente la tabla de métricas y el gráfico de barras comparativo en la interfaz gráfica.

def _mostrar_ranking_modelos(config: dict):
    st.markdown("#### Ranking de modelos — RF vs KNN vs XGBoost vs SVM")

    ruta_csv = os.path.join(
        config['ruta_proyecto'], 'outputs', 'resultados_completos.csv'
    )

    if not os.path.exists(ruta_csv):
        st.warning(
            "No se encontró resultados_completos.csv. "
            "Ejecuta run_pipeline.py primero."
        )
        return

    df_res = pd.read_csv(ruta_csv, index_col=0)

    # Mostrar tabla completa
    st.dataframe(df_res, use_container_width=True)

    # Gráfico comparativo de accuracy por escenario
    st.markdown("#### Accuracy por modelo y escenario")

    modelos    = ['RF', 'KNN', 'XGBOOST', 'SVM']
    escenarios = ['2cls', '3cls', '4cls']
    titulos    = ['2 clases', '3 clases', '4 clases']

    fig = go.Figure()

    colores_modelos = {
        'RF':      '#534AB7',
        'KNN':     '#1D9E75',
        'XGBOOST': '#D85A30',
        'SVM':     '#BA7517'
    }

    for modelo in modelos:
        valores = []
        for esc in escenarios:
            clave = f"{modelo}_{esc}"
            if clave in df_res.index and 'Accuracy' in df_res.columns:
                val = df_res.loc[clave, 'Accuracy']
                val = float(str(val).replace('%', ''))
                valores.append(val)
            else:
                valores.append(0)

        fig.add_trace(go.Bar(
            name=modelo,
            x=titulos,
            y=valores,
            marker_color=colores_modelos.get(modelo, '#888780'),
            text=[f"{v:.1f}%" for v in valores],
            textposition='outside'
        ))

    fig.update_layout(
        barmode='group',
        yaxis_title="Accuracy (%)",
        yaxis_range=[50, 100],
        legend=dict(orientation='h', y=1.1),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 4 — RANKING DE ALERTAS
# =============================================================================

# DESCRIPCIÓN: Procesa y muestra la tabla de ranking de alertas clínicas a nivel poblacional en Streamlit. 
#              Permite filtrar iterativamente por nivel de severidad ('Crítico', 'Aviso', 'Normal'), 
#              ordenar según la puntuación de calidad o identificador del paciente, desplegar un resumen 
#              métrico de la cohorte y exportar los datos filtrados a formato CSV.
# PARÁMETROS:
#   - datos_pob (dict): Diccionario que incluye el DataFrame 'stats_paciente' con los porcentajes de fases.
#   - config (dict): Ajustes globales del usuario para la visualización.
# RETORNO:
#   - None: Inyecta directamente la tabla interactiva, los filtros, las métricas de resumen y el botón de descarga.

def _mostrar_ranking_alertas(datos_pob: dict, config: dict):
    st.markdown("#### Ranking de alertas por paciente")

    stats = datos_pob.get('stats_paciente')
    if stats is None:
        st.info(
            "Se necesita la columna id_paciente en X_features.csv "
            "para generar el ranking de alertas por paciente."
        )
        return

    # Calcular puntuación y nivel de alerta por paciente
    df_alertas = _calcular_alertas_poblacion(stats, config)

    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        nivel_filtro = st.multiselect(
            "Filtrar por nivel",
            options=['Crítico', 'Aviso', 'Normal'],
            default=['Crítico', 'Aviso', 'Normal']
        )
    with col2:
        ordenar_por = st.selectbox(
            "Ordenar por",
            options=['Puntuación (menor primero)', 'Puntuación (mayor primero)',
                     'Paciente']
        )

    # Aplicar filtros
    df_mostrar = df_alertas[df_alertas['Nivel'].isin(nivel_filtro)]

    if 'menor' in ordenar_por:
        df_mostrar = df_mostrar.sort_values('Puntuación')
    elif 'mayor' in ordenar_por:
        df_mostrar = df_mostrar.sort_values('Puntuación', ascending=False)
    else:
        df_mostrar = df_mostrar.sort_values('id_paciente')

    # Mostrar tabla
    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

    # Resumen
    col1, col2, col3 = st.columns(3)
    col1.metric("Pacientes críticos",
                len(df_alertas[df_alertas['Nivel'] == 'Crítico']))
    col2.metric("Pacientes con aviso",
                len(df_alertas[df_alertas['Nivel'] == 'Aviso']))
    col3.metric("Pacientes normales",
                len(df_alertas[df_alertas['Nivel'] == 'Normal']))

    # Exportar CSV
    csv = df_mostrar.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exportar ranking a CSV",
        data=csv,
        file_name="ranking_alertas_pacientes.csv",
        mime="text/csv"
    )

# DESCRIPCIÓN: Recorre las estadísticas de distribución de fases de cada paciente para evaluar sus 
#              desviaciones frente a los rangos normales de referencia, penalizar la puntuación global de 
#              sueño (0-100) y clasificar el nivel de alerta clínico resultante ('Crítico', 'Aviso' o 'Normal').
# PARÁMETROS:
#   - stats (pd.DataFrame): DataFrame con la distribución porcentual de fases del sueño agrupada por paciente.
#   - config (dict): Diccionario de configuración con los ajustes del entorno.
# RETORNO:
#   - pd.DataFrame: DataFrame con las métricas consolidadas por paciente (id_paciente, Puntuación, Nivel y % por fase).

def _calcular_alertas_poblacion(stats: pd.DataFrame,
                                 config: dict) -> pd.DataFrame:
    filas = []
    fases = [f for f in ['Wake', 'Light', 'Deep', 'REM', 'NREM', 'Sleep']
             if f in stats.columns]

    for _, fila in stats.iterrows():
        puntuacion = 100
        for fase in fases:
            pct   = fila.get(fase, 0)
            rango = RANGOS_NORMALES.get(fase, (0, 100))
            if pct < rango[0] * 0.5 or pct > rango[1] * 1.5:
                puntuacion -= 25
            elif pct < rango[0] or pct > rango[1]:
                puntuacion -= 10

        puntuacion = max(0, puntuacion)

        if puntuacion < 50:
            nivel = 'Crítico'
        elif puntuacion < 75:
            nivel = 'Aviso'
        else:
            nivel = 'Normal'

        fila_dict = {'id_paciente': fila.get('id_paciente', '—'),
                     'Puntuación':  puntuacion,
                     'Nivel':       nivel}
        for fase in fases:
            fila_dict[f"% {fase}"] = round(fila.get(fase, 0), 1)

        filas.append(fila_dict)

    return pd.DataFrame(filas)