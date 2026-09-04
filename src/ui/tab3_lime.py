# =============================================================================
# src/ui/tab3_lime.py
# Tab 3 — Explicabilidad LIME local y global.
#
# Muestra:
#   Subtab A — LIME local:
#     - Selector de epoch con botón de confirmación
#     - Información del epoch (fase real, predicha, probabilidades)
#     - Barras de contribución por característica
#   Subtab B — LIME global:
#     - Top-15 características más importantes
#     - Filtros por categoría de feature
#     - Interpretación automática
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.ui.tab1_signal import _cargar_datos_paciente
from src.explainers import generar_lime_global_paciente


# Categorías de features para los filtros del LIME global
CATEGORIAS_FEATURES = {
    'Ritmo y respiración': [
        'PPI_mean','PPI_sd','PPI_sdsd','PPI_rmssd',
        'RIAM_mean','RIAM_sd','RIAM_sdsd','RIAM_rmssd',
        'RIFM_mean','RIFM_sd','RIFM_sdsd','RIFM_rmssd'
    ],
    'Morfología': [
        'Mean_A','SD_A','Mean_A1','SD_A1','Mean_A2','SD_A2',
        'Mean_T1','SD_T1','Mean_T2','SD_T2',
        'Mean_IPAR','SD_IPAR','Mean_IPTR','SD_IPTR'
    ],
    'Estadísticas y energía': [
        'avgADPPG','stdADPPG','MADPPG','IQRPPG','nCMPPG',
        'avgEPPG','SFPPG','avgCLPPG','avgTEPPG',
        'GmPPG','HmPPG','TM25PPG','TM50PPG','SkewPPG','KurtPPG'
    ],
    'No lineales': [
        'HAPPG','HMPPG','HCPPG','HFDPPG','KFDPPG'
    ],
    'HRV Poincaré': [
        'SD1PPG','SD2PPG','RSD1SD2PPG','CCMPPG'
    ],
}


# =============================================================================
# FUNCIÓN PRINCIPAL DEL TAB
# =============================================================================

# DESCRIPCIÓN: Controlador principal de la pestaña de explicabilidad LIME (Tab 3). Gestiona el 
#              estado de la sesión (session_state) para el cálculo de interpretabilidad, asegura la 
#              carga previa de los datos del paciente y organiza la interfaz mediante dos subpestañas 
#              dedicadas a la explicación local (a nivel de epoch) y global (a nivel de toda la noche).
# PARÁMETROS:
#   - config (dict): Ajustes de la barra lateral con paciente, modelo, num_clases, ruta_proyecto y el botón analizar.
# RETORNO:
#   - None: Carga y distribuye las vistas de explicabilidad local y global dentro del Tab 3 en Streamlit.

def mostrar_tab3(config: dict):
    # Marcar que se analizó cuando se pulsa el botón del sidebar
    if config.get('analizar'):
        st.session_state['tab3_analizado'] = True
        with st.spinner("Cargando datos para LIME..."):
            datos = _cargar_datos_paciente(config)
        if datos is None:
            return
        st.session_state['tab3_datos'] = datos

    # Si nunca se ha analizado → mostrar mensaje
    if not st.session_state.get('tab3_analizado'):
        st.info("👈 Pulsa **Analizar** en el sidebar para ver la explicabilidad LIME.")
        return


    # Recuperar datos del session_state si existen
    datos = st.session_state.get('tab3_datos')
    if datos is None:
        return

    subtab_local, subtab_global = st.tabs([
        "🔍 LIME local — epoch concreto",
        "📊 LIME global — toda la noche"
    ])

    with subtab_local:
        _mostrar_lime_local(datos, config)

    with subtab_global:
        _mostrar_lime_global(datos, config)


# =============================================================================
# SUBTAB A — LIME LOCAL
# =============================================================================

# DESCRIPCIÓN: Renderiza la interfaz para la explicación local con LIME de un epoch específico. 
#              Permite al usuario seleccionar un epoch mediante un deslizador, muestra la equivalencia 
#              temporal (horas/minutos/segundos) y la concordancia entre la fase real (PSG) y la predicha, 
#              almacenando el estado en session_state para desencadenar el cálculo al pulsar el botón.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente con las matrices X_test, y_test_str, pred_str y clases.
#   - config (dict): Ajustes de la barra lateral con paciente, modelo, num_clases y ruta_proyecto.
# RETORNO:
#   - None: Inyecta los selectores, métricas comparativas y delega el cálculo visual de LIME local en Streamlit.

def _mostrar_lime_local(datos: dict, config: dict):
    st.markdown("#### LIME Local - Porqué el modelo clasificó este epoch así?")
    st.caption(
        "Selecciona un epoch de la noche y pulsa Analizar con LIME "
        "para ver qué características influyeron en esa predicción concreta."
    )

    X_test        = datos['X_test']
    y_test_str    = datos['y_test_str']
    pred_str      = datos['pred_str']
    clases        = datos['clases']
    n_epochs      = len(X_test)

    # Selector de epoch
    col1, col2 = st.columns([3, 1])
    with col1:
        epoch_idx = st.slider(
            "Epoch a analizar",
            min_value=0,
            max_value=n_epochs - 1,
            value=15,
            help="Cada epoch son 30 segundos de señal cardíaca."
        )
    with col2:
        segundos     = epoch_idx * 30
        horas        = segundos // 3600
        minutos      = (segundos % 3600) // 60
        segs         = segundos % 60
        st.metric("Hora", f"{horas:02d}:{minutos:02d}:{segs:02d}")

    # Información del epoch seleccionado
    fase_real     = y_test_str[epoch_idx]
    fase_predicha = pred_str[epoch_idx]
    acierto       = fase_real == fase_predicha

    col1, col2, col3 = st.columns(3)
    col1.metric("Fase real (PSG)", fase_real)
    col2.metric("Fase predicha",   fase_predicha)
    col3.metric("Resultado", "✓ Acierto" if acierto else "✗ Error")
    st.divider() 

    # Botón — al pulsar guarda en session_state
    if st.button("🔍 Analizar con LIME", type="primary"):
        st.session_state['lime_epoch_idx'] = epoch_idx   # ← epoch seleccionado
        st.session_state['lime_datos']     = datos        # ← datos del paciente
        st.session_state['lime_config']    = config       # ← configuración
        st.session_state['lime_calculado'] = True         # ← flag de activación

    # Mostrar resultado si ya se pulsó el botón alguna vez
    if st.session_state.get('lime_calculado'):
        st.divider()
        with st.spinner(
            f"Calculando LIME para epoch "
            f"{st.session_state['lime_epoch_idx']}..."
        ):
            _ejecutar_y_mostrar_lime_local(
                st.session_state['lime_datos'],
                st.session_state['lime_epoch_idx'],
                st.session_state['lime_config']
            )

# DESCRIPCIÓN: Analiza y limpia una cadena de condición generada por LIME (p. ej. 'HCPPG > 0.5' 
#              o '-0.67 < HCPPG') para aislar y retornar el nombre exacto de la característica (feature). 
#              Compara los tokens extraídos contra la lista de características conocidas, incluyendo un 
#              mecanismo de respaldo (fallback) que ignora valores numéricos.
# PARÁMETROS:
#   - condicion (str): Cadena de texto generada por LIME con la regla/intervalo de la característica.
#   - nombres_features (list): Lista de cadenas con los nombres canónicos de las características del dataset.
# RETORNO:
#   - str: Nombre aislado de la característica o, en su defecto, la cadena original sin procesar.

def extraer_nombre_feature(condicion: str, 
                             nombres_features: list) -> str:
    # Separar por espacios y operadores
    tokens = condicion.replace('<', ' ').replace('>', ' ').split()
    # Buscar el token que coincida con un nombre de feature conocido
    for token in tokens:
        if token in nombres_features:
            return token
    # Fallback — devolver el primer token que no sea un número
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return token
    return condicion

# DESCRIPCIÓN: Instancia un explicador tabular LIME (LimeTabularExplainer) para calcular e interpretar 
#              la predicción de un epoch específico. Visualiza la distribución de probabilidades por clase 
#              y renderiza un gráfico de barras horizontales en Plotly con los pesos de contribución de las 
#              15 características más relevantes (resaltando en verde los factores a favor y en rojo en contra), 
#              concluyendo con un análisis textual del acierto o la discrepancia.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente con las matrices de entrenamiento/prueba, modelo y etiquetas.
#   - epoch_idx (int): Índice numérico del epoch de 30 segundos que se desea interpretar.
#   - config (dict): Ajustes de configuración globales seleccionados en la barra lateral.
# RETORNO:
#   - None: Inyecta los indicadores de probabilidad, el gráfico de barras horizontales LIME y la nota interpretativa en Streamlit.    

def _ejecutar_y_mostrar_lime_local(datos: dict,
                                    epoch_idx: int,
                                    config: dict):
    modelo    = datos['modelo']
    X_train   = datos['X_train']
    X_test    = datos['X_test']
    clases    = datos['clases']

    # Crear explainer LIME
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=clases.tolist(),
        mode='classification',
        random_state=42
    )

    instancia = X_test.iloc[epoch_idx].values
    exp       = explainer.explain_instance(
        data_row=instancia,
        predict_fn=modelo.predict_proba,
        num_features=15
    )

    # Probabilidades por clase
    pred_proba     = modelo.predict_proba(X_test.iloc[[epoch_idx]])[0]
    clase_idx_pred = int(np.argmax(pred_proba))   # ← convertir a int Python
    clase_pred     = clases[clase_idx_pred]

    st.markdown("**PROBABILIDADES POR CLASE:**")
    cols = st.columns(len(clases))
    for col, (clase, prob) in zip(cols, zip(clases, pred_proba)):
        col.metric(clase, f"{prob*100:.1f}%")

    # Barras de contribución LIME
    col_tit, col_leyenda = st.columns([2, 1])
    with col_tit:
        st.markdown(f"#### CONTRIBUCIÓN PARA LA CLASE: {clase_pred}")
    with col_leyenda:
        st.markdown(
            "<div style='text-align:right;font-size:11px;margin-top:8px'>"
            "<span style='background:#1D9E75;color:white;padding:2px 8px;"
            "border-radius:4px;margin-right:6px'>Empuja hacia la clase</span>"
            "<span style='background:#E24B4A;color:white;padding:2px 8px;"
            "border-radius:4px'>Empuja en contra</span>"
            "</div>",
            unsafe_allow_html=True
        )

    # Obtener el label disponible en la explicación
    labels_disponibles = list(exp.local_exp.keys())
    label_usar = clase_idx_pred if clase_idx_pred in labels_disponibles else labels_disponibles[0]
    pesos_lista = exp.as_list(label=label_usar)   
    nombres_features = X_train.columns.tolist()
    nombres = [extraer_nombre_feature(p[0], nombres_features) 
           for p in pesos_lista]
    valores     = [p[1] for p in pesos_lista]

    fig = go.Figure(go.Bar(
        x=valores,
        y=nombres,
        orientation='h',
        marker_color=['#1D9E75' if v > 0 else '#E24B4A' for v in valores],
        text=[f"{v:.4f}" for v in valores],
        textposition='outside'
    ))
    fig.update_layout(
        xaxis_title="Contribución LIME",
        margin=dict(l=0, r=60, t=10, b=0),
        height=400,
        xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Nota de interpretación
    fase_real_local = datos['y_test_str'][epoch_idx]
    fase_pred_local = datos['pred_str'][epoch_idx]
    acierto_local   = fase_real_local == fase_pred_local

    st.divider()

    if acierto_local:
        st.info(
            f"El modelo predijo **{fase_pred_local}** correctamente. "
            f"Las barras verdes muestran qué características "
            f"confirmaron esta fase del sueño."
        )
    else:
        # Feature que más empujó en contra (valor más negativo)
        if valores:
            feature_principal = nombres[valores.index(min(valores))]
            st.warning(
                f"El modelo predijo **{fase_pred_local}** pero la fase "
                f"real era **{fase_real_local}**. "
                f"**{feature_principal}** fue la característica que más "
                f"dificultó la clasificación correcta."
            )


# =============================================================================
# SUBTAB B — LIME GLOBAL
# =============================================================================

# DESCRIPCIÓN: Renderiza el ranking de importancia global de características LIME acumulado sobre 
#              un muestreo de la noche (50 epochs). Almacena el resultado en session_state (incluyendo 
#              el top-5 para comparativas posteriores), ofrece un filtrado interactivo por categorías 
#              mediante un botón de radio, proyecta un gráfico de barras horizontales en Plotly y expone 
#              una tabla de datos desplegable junto a un análisis sintético del top-3.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente con los subconjuntos de datos y el modelo evaluado.
#   - config (dict): Ajustes de la barra lateral con paciente, modelo, num_clases y ruta_proyecto.
# RETORNO:
#   - None: Inyecta los controles de filtrado, el gráfico de barras horizontales, la tabla y la síntesis explicativa en Streamlit.

def _mostrar_lime_global(datos: dict, config: dict):
    st.markdown("#### LIME global — características más importantes de toda la noche")
    st.caption(
        "Promedio del peso absoluto de LIME sobre 50 epochs del paciente. "
        "Muestra qué características usa el modelo consistentemente "
        "para clasificar las fases del sueño de este paciente concreto."
    )

    clave = f"lime_global_{config['paciente']}_{config['num_clases']}"

    if clave not in st.session_state:
        with st.spinner("Calculando LIME global del paciente... (~1 minuto)"):
            df_lime = generar_lime_global_paciente(datos, n_muestras=50)
        st.session_state[clave] = df_lime
    else:
        df_lime = st.session_state[clave]

    # Guardar top5 para tabla comparativa en Tab 4
    st.session_state['lime_top5'] = df_lime['feature'].head(5).tolist()  

    # Filtros por categoría
    st.markdown("**Filtrar por categoría:**")
    categorias = ['Todas'] + list(CATEGORIAS_FEATURES.keys())
    categoria_sel = st.radio(
        label="categoria",
        options=categorias,
        horizontal=True,
        label_visibility="collapsed"
    )

    # Aplicar filtro
    if categoria_sel != 'Todas':
        features_cat = CATEGORIAS_FEATURES.get(categoria_sel, [])
        df_mostrar   = df_lime[df_lime['feature'].isin(features_cat)].head(15)
    else:
        df_mostrar = df_lime.head(15)

    if df_mostrar.empty:
        st.info(f"No hay features de la categoría '{categoria_sel}' en el top-15.")
        return

    # Gráfico de barras horizontal
    fig = go.Figure(go.Bar(
        x=df_mostrar['importancia'],
        y=df_mostrar['feature'],
        orientation='h',
        marker_color='#534AB7',
        text=[f"{v:.4f}" for v in df_mostrar['importancia']],
        textposition='outside'
    ))
    fig.update_layout(
        xaxis_title="Peso absoluto medio (LIME)",
        yaxis=dict(autorange='reversed'),
        margin=dict(l=0, r=60, t=10, b=0),
        height=max(300, len(df_mostrar) * 28)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla con los valores exactos
    with st.expander("Ver tabla de valores"):
        st.dataframe(
            df_mostrar.rename(columns={
                'feature':          'Característica',
                'importancia':  'Importancia media'
            }).round(5),
            use_container_width=True,
            hide_index=True
        )

    # Interpretación automática
    top3 = df_lime['feature'].head(3).tolist()
    st.info(
        f"Las 3 características más importantes para "
        f"**{config['paciente']}** son "
        f"**{top3[0]}** ({obtener_categoria(top3[0])}), "
        f"**{top3[1]}** ({obtener_categoria(top3[1])}) y "
        f"**{top3[2]}** ({obtener_categoria(top3[2])})."
    )

# DESCRIPCIÓN: Identifica y retorna la categoría funcional (dominio) a la que pertenece una 
#              característica específica buscando su presencia en el diccionario global 
#              'CATEGORIAS_FEATURES'. Si la característica no se encuentra en ninguna lista, 
#              asigna el valor por defecto "General".
# PARÁMETROS:
#   - feature (str): Nombre de la característica que se desea categorizar.
# RETORNO:
#   - str: Nombre de la categoría asignada ('Temporal', 'Frecuencial', 'No lineal', etc.) o 'General'.

def obtener_categoria(feature: str) -> str:
    for cat, lista in CATEGORIAS_FEATURES.items():
        if feature in lista:
            return cat
    return "General"