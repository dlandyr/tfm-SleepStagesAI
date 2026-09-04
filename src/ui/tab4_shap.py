# =============================================================================
# src/ui/tab4_shap.py
# Tab 4 — SHAP global + Permutation Importance + tabla comparativa.
#
# Muestra:
#   1. SHAP global — importancia exacta por feature y por clase
#   2. Permutation Importance — caída en accuracy al barajar cada feature
#   3. Tabla comparativa top-5 LIME vs SHAP vs Permutation
# =============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap

from src.ui.tab1_signal import _cargar_datos_paciente, COLORES_FASES


# =============================================================================
# FUNCIÓN PRINCIPAL DEL TAB
# =============================================================================

# DESCRIPCIÓN: Controlador principal del módulo de comparativa avanzada de explicabilidad (Tab 4). 
#              Valida la compatibilidad del algoritmo (limitado a 'xgboost' y 'rf' para TreeExplainer), 
#              gestiona la persistencia de datos en session_state y orquestación visual dispuesta en 
#              dos columnas paralelas para SHAP Global y Permutation Importance, concluyendo con una 
#              sección unificada de tabla comparativa junto a LIME.
# PARÁMETROS:
#   - config (dict): Ajustes del sidebar con paciente, modelo, num_clases, ruta_proyecto y la bandera analizar.
# RETORNO:
#   - None: Renderiza la distribución en columnas de las métricas SHAP, Permutation Importance y la tabla cruzada en Streamlit.

def mostrar_tab4(config: dict):
    # Clave única por paciente, modelo y escenario
    clave = f"shap_{config['paciente']}_{config['modelo']}_{config['num_clases']}"

    if not config.get('analizar') and clave not in st.session_state:
        st.info("👈 Pulsa **Analizar** en el sidebar para ver SHAP y Permutation Importance.")
        return

    # Solo disponible para XGBoost y RF (TreeExplainer)
    if config['modelo'] not in ('xgboost', 'rf'):
        st.warning(
            "SHAP con TreeExplainer solo está disponible para XGBoost y Random Forest. "
            "Selecciona uno de estos modelos en el sidebar."
        )
        return

    if config.get('analizar') or clave not in st.session_state:
        with st.spinner("Calculando valores SHAP..."):
            datos = _cargar_datos_paciente(config)
        if datos is None:
            return
        st.session_state[clave] = datos
    else:
        datos = st.session_state[clave]

    # Mostrar los 3 componentes en columnas
    col1, col2 = st.columns(2)

    with col1:
        _mostrar_shap_global(datos, config)

    with col2:
        _mostrar_permutation_importance(datos, config)

    st.divider()
    _mostrar_tabla_comparativa(datos, config)


# =============================================================================
# COMPONENTE 1 — SHAP GLOBAL
# =============================================================================

# DESCRIPCIÓN: Calcula y proyecta la importancia global de las características mediante valores SHAP 
#              basados en teoría de juegos utilizando TreeExplainer sobre un subconjunto representativo. 
#              Maneja salidas multiclase (agregando la magnitud absoluta promedio sobre muestras y clases), 
#              registra el top-5 en session_state y representa un gráfico de barras horizontales en Plotly.
# PARÁMETROS:
#   - datos (dict): Diccionario cargado del paciente con la matriz 'X_test' y el 'modelo' entrenado.
#   - config (dict): Ajustes de configuración globales seleccionados en la barra lateral.
# RETORNO:
#   - None: Inyecta el gráfico de barras horizontales de valores SHAP y el resumen del top-3 en Streamlit.

def _mostrar_shap_global(datos: dict, config: dict):
    st.markdown("#### SHAP — importancia exacta")
    st.caption("Valores exactos calculados con TreeExplainer (teoría de juegos)")

    modelo    = datos['modelo']
    X_test    = datos['X_test']
    n_features = 15

    try:
        explainer = shap.TreeExplainer(modelo)

        # ── FIX: usar solo una muestra pequeña para evitar el error ──
        n_muestras  = min(100, len(X_test))
        X_muestra   = X_test.iloc[:n_muestras]

        shap_values = explainer.shap_values(X_muestra)

        # Multiclase — shap_values es lista de arrays
        if shap_values.ndim == 3:
            # Shape (muestras, features, clases) → media sobre muestras y clases
            importancia = np.abs(shap_values).mean(axis=(0, 2))
        elif shap_values.ndim == 2:
            # Shape (muestras, features) → media sobre muestras
            importancia = np.abs(shap_values).mean(axis=0)
        else:
            importancia = np.abs(shap_values).flatten()

        df_shap = pd.DataFrame({
            'feature':    X_test.columns,
            'importancia': importancia
        }).sort_values('importancia', ascending=False).head(n_features)

        # Guardar top5 para tabla comparativa
        st.session_state['shap_top5'] = df_shap['feature'].head(5).tolist()

        # Gráfico
        fig = go.Figure(go.Bar(
            x=df_shap['importancia'],
            y=df_shap['feature'],
            orientation='h',
            marker_color='#534AB7',
            text=[f"{v:.3f}" for v in df_shap['importancia']],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Importancia media |SHAP|",
            yaxis=dict(autorange='reversed'),
            margin=dict(l=0, r=60, t=10, b=0),
            height=max(300, n_features * 28)
        )
        st.plotly_chart(fig, use_container_width=True)

        top3 = df_shap['feature'].head(3).tolist()
        st.success(f"Top-3 SHAP: **{top3[0]}**, **{top3[1]}**, **{top3[2]}**")

    except Exception as e:
        st.error(f"Error calculando SHAP: {e}")

# =============================================================================
# COMPONENTE 2 — PERMUTATION IMPORTANCE
# =============================================================================

# DESCRIPCIÓN: Calcula y visualiza la importancia de características mediante Permutation Importance, 
#              evaluando la caída en la precisión (accuracy) al permutarlas. Utiliza un archivo CSV precalculado 
#              si existe o lo calcula en tiempo real con scikit-learn (10 repeticiones), guardando el top-5 
#              en session_state y renderizando un gráfico de barras horizontales con barras de error en Plotly.
# PARÁMETROS:
#   - datos (dict): Diccionario del paciente cargado con la matriz 'X_test', las etiquetas y el 'modelo'.
#   - config (dict): Ajustes de configuración globales del sidebar con las rutas y el modelo.
# RETORNO:
#   - None: Inyecta el gráfico de barras horizontales con desviaciones estándar y el resumen top-3 en Streamlit.

def _mostrar_permutation_importance(datos: dict, config: dict):
    st.markdown("#### Permutation Importance")
    st.caption("Disminución media en accuracy al barajar cada feature (30 repeticiones)")

    # Cargar CSV si ya existe (generado por run_pipeline.py)
    ruta_csv = os.path.join(
        config['ruta_proyecto'], 'outputs', 'plots',
        f"permutation_importance_{config['modelo']}_{config['num_clases']}_clases.csv"
    )

    if os.path.exists(ruta_csv):
        df_perm = pd.read_csv(ruta_csv).head(15)
    else:
        # Calcular en tiempo real si no existe el CSV
        st.info("Calculando Permutation Importance... (puede tardar 1-2 minutos)")
        from sklearn.inspection import permutation_importance as sklearn_perm

        modelo  = datos['modelo']
        X_test  = datos['X_test']
        y_test  = datos['le'].transform(datos['y_test_str'])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            resultado = sklearn_perm(
                modelo, X_test, y_test,
                n_repeats=10,
                random_state=42,
                scoring='accuracy',
                n_jobs=1
            )

        df_perm = pd.DataFrame({
            'feature':           X_test.columns,
            'importancia_media': resultado.importances_mean,
            'importancia_std':   resultado.importances_std
        }).sort_values('importancia_media', ascending=False).head(15)

    # Guardar top-5 para tabla comparativa
    st.session_state['perm_top5'] = df_perm['feature'].head(5).tolist()

    # Gráfico con barras de error
    fig = go.Figure(go.Bar(
        x=df_perm['importancia_media'],
        y=df_perm['feature'],
        orientation='h',
        marker_color='#1D9E75',
        error_x=dict(
            type='data',
            array=df_perm['importancia_std'].tolist() if 'importancia_std' in df_perm.columns else None,
            color='#888780'
        ),
        text=[f"{v:.4f}" for v in df_perm['importancia_media']],
        textposition='outside'
    ))
    fig.update_layout(
        xaxis_title="Disminución media en accuracy",
        yaxis=dict(autorange='reversed'),
        margin=dict(l=0, r=60, t=10, b=0),
        height=max(300, len(df_perm) * 28)
    )
    st.plotly_chart(fig, use_container_width=True)

    top3 = df_perm['feature'].head(3).tolist()
    st.success(
        f"Top-3 Permutation: **{top3[0]}**, **{top3[1]}**, **{top3[2]}**"
    )


# =============================================================================
# COMPONENTE 3 — TABLA COMPARATIVA
# =============================================================================

# DESCRIPCIÓN: Construye y visualiza una tabla comparativa cruzada con las 5 características (top-5) más 
#              relevantes obtenidas mediante LIME, SHAP y Permutation Importance. Recopila las métricas 
#              desde 'session_state', identifica solapamientos entre técnicas para determinar los indicadores 
#              fisiológicos más robustos y presenta un resumen de coincidencias (en 2 o en los 3 métodos).
# PARÁMETROS:
#   - datos (dict): Diccionario de datos del paciente cargado en la sesión.
#   - config (dict): Ajustes globales seleccionados en la barra lateral del cuadro de mando.
# RETORNO:
#   - None: Inyecta la tabla comparativa estructurada y los mensajes explicativos de coherencia en Streamlit.

def _mostrar_tabla_comparativa(datos: dict, config: dict):
    st.markdown("#### Comparativa top-5 — LIME vs SHAP vs Permutation Importance")
    st.caption(
        "Las características que aparecen en los tres métodos "
        "son los indicadores fisiológicos más robustos."
    )

    # Cargar top-5 de cada método desde session_state
    lime_top5 = st.session_state.get('lime_top5', None)
    shap_top5 = st.session_state.get('shap_top5', None)
    perm_top5 = st.session_state.get('perm_top5', ['—'] * 5)

    # Avisar si faltan datos de LIME o SHAP
    if lime_top5 is None and shap_top5 is None:
        st.info(
            "Para ver la tabla comparativa completa primero ve al "
            "**Tab LIME** y calcula el LIME global del paciente."
        )
    elif lime_top5 is None:
        st.info("Faltan datos de LIME — ve al Tab LIME y calcula el LIME global.")
    elif shap_top5 is None:
        st.info("Faltan datos de SHAP — serán calculados automáticamente.")

    # Usar valores por defecto si faltan
    lime_top5 = lime_top5 or ['—'] * 5
    shap_top5 = shap_top5 or ['—'] * 5

    # Crear tabla comparativa
    n = max(len(lime_top5), len(shap_top5), len(perm_top5), 5)
    lime_top5 = lime_top5 + ['—'] * (n - len(lime_top5))
    shap_top5 = shap_top5 + ['—'] * (n - len(shap_top5))
    perm_top5 = perm_top5 + ['—'] * (n - len(perm_top5))

    df_comp = pd.DataFrame({
        'Posición':     range(1, n + 1),
        'LIME':         lime_top5,
        'SHAP':         shap_top5,
        'Permutation':  perm_top5,
    })

    # Marcar coincidencias
    def marcar_coincidencia(row):
        features = [row['LIME'], row['SHAP'], row['Permutation']]
        features = [f for f in features if f != '—']
        if len(set(features)) < len(features):
            return True
        return False

    # Identificar features que coinciden en al menos 2 métodos
    todas_features = set(lime_top5 + shap_top5 + perm_top5) - {'—'}
    coincidentes   = {
        f for f in todas_features
        if sum([f in lime_top5, f in shap_top5, f in perm_top5]) >= 2
    }
    coinciden_3    = {
        f for f in todas_features
        if sum([f in lime_top5, f in shap_top5, f in perm_top5]) == 3
    }

    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Resumen de coincidencias
    if coinciden_3:
        st.success(
            f"Coinciden en los 3 métodos: **{', '.join(sorted(coinciden_3))}** "
            "— conclusión muy robusta y validada."
        )
    if coincidentes - coinciden_3:
        st.info(
            f"Coinciden en 2 métodos: **{', '.join(sorted(coincidentes - coinciden_3))}**"
        )