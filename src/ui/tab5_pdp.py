# =============================================================================
# src/ui/tab5_pdp.py
# Tab 5 — PDP (Partial Dependence Plot).
#
# Muestra cómo afecta el valor de cada feature a la probabilidad
# de predecir cada fase del sueño, manteniendo el resto constante.
#
# Solo disponible para XGBoost (más rápido con árboles).
# =============================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.inspection import PartialDependenceDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ui.tab1_signal import _cargar_datos_paciente, COLORES_FASES


# Features por defecto para el PDP — top-6 según LIME y SHAP
FEATURES_PDP_DEFAULT = [
    'HCPPG', 'PPI_mean', 'GmPPG', 'RIAM_sd', 'SD_A', 'TM50PPG'
]

# Interpretación fisiológica por feature y clase
INTERPRETACIONES = {
    'HCPPG': {
        'Wake':  "Valores altos de HCPPG (señal compleja) → mayor probabilidad de vigilia.",
        'Deep':  "Valores bajos de HCPPG (señal simple) → mayor probabilidad de sueño profundo.",
        'REM':   "HCPPG moderado-alto → asociado con sueño REM.",
        'Light': "HCPPG intermedio → característico del sueño ligero.",
        'NREM':  "HCPPG bajo → asociado con sueño NREM.",
        'Sleep': "HCPPG bajo → señal más predecible durante el sueño.",
    },
    'PPI_mean': {
        'Wake':  "Latidos más rápidos (PPI bajo) → mayor probabilidad de vigilia.",
        'Deep':  "Latidos lentos y espaciados (PPI alto) → característicos del sueño profundo.",
        'REM':   "PPI variable → el sueño REM tiene ritmo cardíaco irregular.",
        'Light': "PPI intermedio → característico del sueño ligero.",
        'NREM':  "PPI moderado → característico del sueño NREM.",
        'Sleep': "PPI alto → el corazón late más lento durante el sueño.",
    },
    'GmPPG': {
        'Wake':  "Mayor energía media del pulso → asociada con vigilia.",
        'Deep':  "Menor energía → el pulso es más débil en sueño profundo.",
        'REM':   "Energía moderada → característica del sueño REM.",
        'Light': "Energía media → característica del sueño ligero.",
        'NREM':  "Energía reducida → característica del sueño NREM.",
        'Sleep': "Energía reducida → el organismo consume menos energía al dormir.",
    },
}


# =============================================================================
# FUNCIÓN PRINCIPAL DEL TAB
# =============================================================================

# DESCRIPCIÓN: Controlador principal de la pestaña de Partial Dependence Plots (Tab 5). Verifica la 
#              compatibilidad del algoritmo (optimizado exclusivamente para XGBoost), gestiona la persistencia 
#              y reseteo del session_state ante cambios de escenario clínico o paciente, y delega la 
#              visualización del selector de clases y trazado de curvas PDP.
# PARÁMETROS:
#   - config (dict): Ajustes de la barra lateral con paciente, modelo, num_clases, ruta_proyecto y el estado de la acción analizar.
# RETORNO:
#   - None: Carga y orquesta los paneles de visualización de Partial Dependence Plots en Streamlit.

def mostrar_tab5(config: dict):
    if config.get('analizar'):
        st.session_state['tab5_analizado'] = True
        
        # ── Limpiar PDP si cambió el escenario ────────────────────────
        clave_escenario = f"{config['paciente']}_{config['num_clases']}"
        if st.session_state.get('tab5_escenario') != clave_escenario:
            st.session_state['tab5_escenario'] = clave_escenario
            st.session_state['pdp_calculado']  = False
            st.session_state['pdp_clase_sel']  = None
        # ─────────────────────────────────────────────────────────────

        with st.spinner("Preparando datos para PDP..."):
            datos = _cargar_datos_paciente(config)
        if datos is None:
            return
        st.session_state['tab5_datos'] = datos

    # Si nunca se ha analizado → salir sin mensaje
    if not st.session_state.get('tab5_analizado'):
        st.info("👈 Pulsa **Analizar** en el sidebar para ver los gráficos PDP.")
        return

    # Recuperar datos guardados — persisten aunque se cambie el radio button
    datos = st.session_state.get('tab5_datos')
    if datos is None:
        return

    if config['modelo'] != 'xgboost':
        st.warning(
            "PDP está optimizado para XGBoost. "
            "Selecciona XGBoost en el sidebar."
        )
        return

    _mostrar_selector_clase_y_pdp(datos, config)


# =============================================================================
# SELECTOR DE CLASE Y GENERACIÓN DE PDP
# =============================================================================

# DESCRIPCIÓN: Despliega la interfaz de control interactiva para Partial Dependence Plots (PDP). 
#              Permite al usuario seleccionar la fase del sueño de interés y filtrar hasta un máximo 
#              de 6 características, gestionando el cálculo bajo demanda mediante botón para persistir 
#              el estado en session_state y delegar la renderización de las curvas PDP.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente con las características 'X_test', clases y modelo.
#   - config (dict): Ajustes de la barra lateral con paciente, modelo, num_clases y ruta_proyecto.
# RETORNO:
#   - None: Inyecta los selectores de fase y características, los controles de disparo y llama a la rutina de trazado PDP en Streamlit.

def _mostrar_selector_clase_y_pdp(datos: dict, config: dict):
    st.markdown("#### PDP — ¿cómo afecta cada característica a la predicción?")
    st.caption(
        "Muestra cómo cambia la probabilidad de predecir una fase del sueño "
        "cuando el valor de una característica varía de mínimo a máximo, "
        "manteniendo el resto constante."
    )

    clases_ordenadas = datos['clases_ordenadas']

    # Selector de clase
    st.markdown("**Selecciona la fase del sueño a analizar:**")
    clase_sel = st.radio(
        label="clase_pdp",
        options=clases_ordenadas,
        horizontal=True,
        label_visibility="collapsed"
    )

    # Selector de features
    with st.expander("Seleccionar características a mostrar"):
        X_test         = datos['X_test']
        features_disp  = [f for f in FEATURES_PDP_DEFAULT if f in X_test.columns]
        features_extra = [f for f in X_test.columns if f not in FEATURES_PDP_DEFAULT]
        features_sel   = st.multiselect(
            "Características",
            options=features_disp + features_extra,
            default=features_disp,
            max_selections=6,
            help="Selecciona hasta 6 características y pulsa Generar PDP.",
            placeholder="Buscar característica...",
        )

    # ── Botón de confirmación ──────────────────────────────────────────
    if st.button("📊 Generar PDP", type="primary"):
        st.session_state['pdp_clase_sel']    = clase_sel
        st.session_state['pdp_features_sel'] = features_sel
        st.session_state['pdp_calculado']    = True
        st.session_state['pdp_mostrar_spinner'] = True

    # ── Mostrar resultado solo si se pulsó el botón ────────────────────
    if st.session_state.get('pdp_calculado'):
        clase_usar    = st.session_state['pdp_clase_sel']
        features_usar = st.session_state['pdp_features_sel']

        if not features_usar:
            st.warning("Selecciona al menos una característica.")
            return

        st.divider()
        if st.session_state.get('pdp_mostrar_spinner'):
            st.session_state['pdp_mostrar_spinner'] = False  # ← apagar tras mostrar
            with st.spinner(f"Calculando PDP para {clase_usar}..."):
                _generar_pdp(datos, clase_usar, features_usar, config)
        else:
            _generar_pdp(datos, clase_usar, features_usar, config)

# DESCRIPCIÓN: Calcula en tiempo real con scikit-learn (PartialDependenceDisplay) y grafica mediante 
#              matplotlib la dependencia parcial (PDP) para la clase y conjunto de características elegidos. 
#              Ajusta la disposición en rejilla (filas/columnas) de forma dinámica según el número de variables 
#              seleccionadas, oculta los subgráficos vacíos y renderiza los resultados en Streamlit junto 
#              a sus notas interpretativas clínicas.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente con el 'modelo', matriz 'X_train' y 'clases_ordenadas'.
#   - clase_sel (str): Nombre de la fase del sueño seleccionada para el cálculo (p. ej., 'Wake', 'REM').
#   - features_sel (list): Lista de nombres de características a evaluar en la rejilla de gráficos PDP.
#   - config (dict): Ajustes de la barra lateral con el número de clases y configuración global.
# RETORNO:
#   - None: Muestra la rejilla de gráficos de dependencia parcial en Streamlit y llama a la rutina de interpretación.

def _generar_pdp(datos: dict,
                 clase_sel: str,
                 features_sel: list,
                 config: dict):
    modelo           = datos['modelo']
    X_train          = datos['X_train']
    clases_ordenadas = datos['clases_ordenadas']

    try:
        clase_idx = clases_ordenadas.index(clase_sel)
    except ValueError:
        st.error(f"Clase '{clase_sel}' no encontrada.")
        return

    # ── Calcular siempre en tiempo real con las features seleccionadas ─
    # El PNG del pipeline usa features fijas — ignorarlo para respetar
    # la selección del usuario
    #with st.spinner(f"Calculando PDP para {clase_sel}..."):
    try:
            n = len(features_sel)
            ncols = 2 if n % 2 == 0 and n <= 4 else min(3, n)
            nrows = (n + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols,
                figsize=(14, 4 * nrows),
                constrained_layout=True
            )
            axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]

            PartialDependenceDisplay.from_estimator(
                modelo, X_train,
                features=features_sel,
                target=clase_idx,
                kind='average',
                n_jobs=1,
                ax=axes_flat[:n]
            )

            for i in range(n, len(axes_flat)):
                axes_flat[i].set_visible(False)

            fig.suptitle(
                f"PDP — XGBoost ({config['num_clases']} clases)\n"
                f"Dependencia parcial para: {clase_sel}",
                fontsize=12
            )

            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
            st.error(f"Error generando PDP: {e}")
            return        

    _mostrar_interpretaciones(clase_sel, features_sel)
    
# DESCRIPCIÓN: Despliega una guía textual con la interpretación fisiológica y clínica de cada 
#              característica seleccionada en el PDP para la fase del sueño bajo estudio. 
#              Recupera las explicaciones del diccionario global 'INTERPRETACIONES' y concluye 
#              con un cuadro aclaratorio sobre la naturaleza cualitativa de PDP frente a SHAP o Permutation Importance.
# PARÁMETROS:
#   - clase_sel (str): Nombre de la fase del sueño evaluada (p. ej., 'Wake', 'REM', 'N3').
#   - features_sel (list): Lista de nombres de características mostradas en los gráficos de PDP.
# RETORNO:
#   - None: Inyecta el desglose con viñetas explicativas y la nota metodológica en Streamlit.

def _mostrar_interpretaciones(clase_sel: str, features_sel: list):
    st.markdown("#### Interpretación fisiológica")

    for feature in features_sel:
        interp = INTERPRETACIONES.get(feature, {})       
        texto  = interp.get(
            clase_sel,
            f"{feature} — sin interpretación específica disponible "
            f"para esta característica."
        )
        st.markdown(f"- **{feature}**: {texto}")

    st.caption(
        "A diferencia de Permutation Importance (qué importa) y SHAP (cuánto importa), "
        "PDP muestra CÓMO afecta el valor concreto de cada característica a la predicción."
    )