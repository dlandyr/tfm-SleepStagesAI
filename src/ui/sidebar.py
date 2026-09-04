# =============================================================================
# src/ui/sidebar.py
# Panel lateral de la interfaz con todos los controles de configuración.
# Devuelve un diccionario con la configuración seleccionada por el usuario.
# =============================================================================

import os
import glob
import streamlit as st
import pandas as pd
import numpy as np


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

# DESCRIPCIÓN: Construye y renderiza la barra lateral (sidebar) de configuración gráfica de Streamlit. 
#              Gestiona la selección del modo de análisis (Individual vs Poblacional), filtrado de pacientes, 
#              elección del modelo clasificador (RF, KNN, XGBoost, SVM), resolución del escenario de clases (2, 3 o 4) 
#              y el control de ejecución de los pipelines del sistema.
# PARÁMETROS:
#   - Ninguno: Detecta dinámicamente la estructura del entorno e interactúa con el usuario a través de widgets de Streamlit.
# RETORNO:
#   - dict: Diccionario de parámetros seleccionados {'modo', 'paciente', 'modelo', 'modelo_nombre', 'num_clases', 'ruta_proyecto', 'analizar'}.

def mostrar_sidebar() -> dict:  
    with st.sidebar:

        # Título del sidebar
        st.markdown("### ⚙️ Configuración")
        st.divider()

        # ── Modo de análisis ──────────────────────────────────────────────
        st.markdown("**Modo de análisis**")
        modo = st.radio(
            label="modo",
            options=["Individual", "Poblacional"],
            index=0,
            label_visibility="collapsed",
            help="Individual: analiza un paciente concreto. "
                 "Poblacional: estadísticas de todos los pacientes."
        )

        st.divider()

        # ── Ruta del proyecto ─────────────────────────────────────────────
        # Detecta automáticamente la raíz del proyecto buscando config.yaml
        ruta_proyecto = _detectar_ruta_proyecto()

        # ── Selector de paciente (solo en modo individual) ────────────────
        paciente = None
        if modo == "Individual":
            st.markdown("**Paciente**")
            pacientes_disponibles = _listar_pacientes(ruta_proyecto)

            if not pacientes_disponibles:
                st.error("No se encontraron CSVs en data/raw/")
                paciente = None
            else:
                paciente = st.selectbox(
                    label="paciente",
                    options=pacientes_disponibles,
                    label_visibility="collapsed",
                    help="Selecciona el registro nocturno del paciente a analizar."
                )

                # Indicador de fiabilidad del paciente seleccionado
                if paciente:
                    X_raw = pd.read_csv(os.path.join(ruta_proyecto, 'data', 'processed', 'X_features.csv'),
                                    usecols=['id_paciente'])
                    pacientes_todos = X_raw['id_paciente'].unique()
                    np.random.seed(42)
                    indices = np.random.permutation(len(pacientes_todos))
                    pac_test = set(pacientes_todos[indices[:10]])
                    paciente_id = paciente.replace('.csv', '')
                    if paciente_id in pac_test:
                        st.caption("🔬 Paciente de TEST — métricas rigurosas")
                    else:
                        st.caption("⚠️ Paciente de TRAIN — métricas orientativas")

            st.divider()

        # ── Selector de modelo ────────────────────────────────────────────
        if modo == "Individual":
            st.markdown("**Modelo de clasificación**")
            ruta_modelos = os.path.join(ruta_proyecto, 'outputs', 'models')
            mapeo = {'XGBoost': 'xgboost', 'Random Forest': 'rf', 'KNN': 'knn', 'SVM': 'svm'}
            nombres_disponibles = [
                nombre for nombre, clave in mapeo.items()
                if any(glob.glob(os.path.join(ruta_modelos, f"{clave}_*.pkl")))
            ]
            if not nombres_disponibles:
                nombres_disponibles = ['XGBoost']
            modelo_nombre = st.radio(
                label="modelo",
                options=nombres_disponibles,
                index=0,
                label_visibility="collapsed",
                help="Selecciona el algoritmo de clasificación a utilizar."
            )
            st.divider()
        else:
            modelo_nombre = "XGBoost"

        # Convertir nombre legible a clave interna
        modelo = _nombre_a_clave(modelo_nombre)

        st.divider()

        # ── Selector de escenario ─────────────────────────────────────────
        st.markdown("**Escenario de clasificación**")
        escenario = st.radio(
            label="escenario",
            options=["2 clases — Wake / Sleep",
                     "3 clases — Wake / NREM / REM",
                     "4 clases — Wake / Light / Deep / REM"],
            index=2,
            label_visibility="collapsed",
            help="El escenario de 4 clases es el más completo y el más desafiante."
        )
        num_clases = _escenario_a_clases(escenario)

        st.divider()

        # ── Botón de análisis ─────────────────────────────────────────────
        analizar = st.button(
            label="🔍 Analizar",
            use_container_width=True,
            type="primary",
            help="Pulsa para iniciar el análisis con la configuración seleccionada."
        )

        # ── Información del sistema ───────────────────────────────────────
        st.divider()
        _mostrar_info_sistema(ruta_proyecto)

    return {
        'modo':          modo,
        'paciente':      paciente,
        'modelo':        modelo,
        'modelo_nombre': modelo_nombre,
        'num_clases':    num_clases,
        'ruta_proyecto': ruta_proyecto,
        'analizar':      analizar,
    }


# =============================================================================
# FUNCIONES AUXILIARES PRIVADAS
# =============================================================================

# DESCRIPCIÓN: Localiza de forma dinámica la ruta absoluta del directorio raíz del proyecto mediante 
#              la búsqueda ascendente en la jerarquía de carpetas del archivo de configuración 'config.yaml'. 
#              Si no lo encuentra tras explorar 5 niveles superiores, aplica como estrategia de contingencia 
#              el directorio de trabajo actual.
# PARÁMETROS:
#   - Ninguno.
# RETORNO:
#   - str: Ruta absoluta a la raíz del proyecto.

def _detectar_ruta_proyecto() -> str:   
    ruta_actual = os.path.abspath(os.path.dirname(__file__))

    # Subir hasta encontrar config.yaml (máximo 5 niveles)
    for _ in range(5):
        if os.path.exists(os.path.join(ruta_actual, 'configs', 'config.yaml')):
            return ruta_actual
        ruta_actual = os.path.dirname(ruta_actual)

    # Fallback: directorio de trabajo actual
    return os.getcwd()

# DESCRIPCIÓN: Escanea el directorio de datos crudos ('data/raw/') del proyecto para identificar 
#              los archivos CSV de registro por paciente, excluyendo metadatos globales 
#              ('info_pacientes.csv', 'metadata.csv') y retornando una lista ordenada alfabéticamente.
# PARÁMETROS:
#   - ruta_proyecto (str): Ruta absoluta a la raíz del proyecto.
# RETORNO:
#   - list[str]: Lista con los nombres de archivo de los pacientes disponibles (ej. ['paciente01.csv', ...]).

def _listar_pacientes(ruta_proyecto: str) -> list:    
    patron = os.path.join(ruta_proyecto, 'data', 'raw', '*.csv')    
    archivos = sorted(glob.glob(patron))
    EXCLUIR = {'info_pacientes.csv', 'metadata.csv'}
    return [os.path.basename(a) for a in archivos if os.path.basename(a) not in EXCLUIR]

# DESCRIPCIÓN: Mapea la denominación legible de los algoritmos de clasificación mostrada en la interfaz 
#              a sus respectivas claves internas de identificación (utilizadas en src/models.py y en los 
#              ficheros de serialización .pkl), devolviendo 'xgboost' por defecto ante selecciones no reconocidas.
# PARÁMETROS:
#   - nombre (str): Etiqueta o nombre legible del modelo ('XGBoost', 'Random Forest', 'KNN', 'SVM').
# RETORNO:
#   - str: Identificador técnico interno ('xgboost', 'rf', 'knn', 'svm').

def _nombre_a_clave(nombre: str) -> str:
    mapa = {
        'XGBoost':       'xgboost',
        'Random Forest': 'rf',
        'KNN':           'knn',
        'SVM':           'svm',
    }
    return mapa.get(nombre, 'xgboost')

# DESCRIPCIÓN: Convierte la opción de texto seleccionada en la interfaz gráfica para el escenario 
#              de clasificación en un valor numérico entero correspondiente al número de clases objetivo.
# PARÁMETROS:
#   - escenario (str): Cadena de texto seleccionada en el widget ('2 clases — ...', '3 clases — ...', etc.).
# RETORNO:
#   - int: Número entero de clases resultantes (2, 3 o 4).

def _escenario_a_clases(escenario: str) -> int:    
    return int(escenario[0])

# DESCRIPCIÓN: Consulta el sistema de archivos local para contabilizar el número de registros de pacientes 
#              disponibles en 'data/raw/' y el número de modelos serializados (.pkl) en 'outputs/models/', 
#              desplegando estas métricas técnicas e información del dataset al pie de la barra lateral.
# PARÁMETROS:
#   - ruta_proyecto (str): Ruta absoluta al directorio raíz del proyecto.
# RETORNO:
#   - None: Inyecta los indicadores métricos y etiquetas textuales en el panel lateral de Streamlit.

def _mostrar_info_sistema(ruta_proyecto: str):
    n_pacientes = len(_listar_pacientes(ruta_proyecto))
    n_modelos   = len(glob.glob(
        os.path.join(ruta_proyecto, 'outputs', 'models', '*.pkl')
    ))

    st.markdown("**Información del sistema**")
    col1, col2 = st.columns(2)
    col1.metric("Pacientes", n_pacientes)
    col2.metric("Modelos", n_modelos)
    st.caption("Dataset DREAMT · PhysioNet 2025")
