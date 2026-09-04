# =============================================================================
# app.py
# Punto de entrada principal de la interfaz Streamlit.
# Orquesta el sidebar, el modo individual y el modo poblacional.
#
# Uso:
#   streamlit run app.py
# =============================================================================

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Importar módulos de la interfaz
from src.ui.sidebar   import mostrar_sidebar
from src.ui.tab1_signal import mostrar_tab1
from src.ui.tab2_alerts import mostrar_tab2
from src.ui.tab3_lime   import mostrar_tab3
from src.ui.tab4_shap   import mostrar_tab4
from src.ui.tab5_pdp    import mostrar_tab5
from src.ui.dashboard   import mostrar_dashboard


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Sistema Explicable de Clasificación de Fases del Sueño",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CABECERA PRINCIPAL
# =============================================================================

# DESCRIPCIÓN: Renderiza el título principal, subtítulo identificativo del dataset DREAMT y una 
#              línea divisora en la parte superior de la aplicación Streamlit. Sirve como 
#              cabecera visual global persistente independientemente de la navegación o las pestañas.
# PARÁMETROS:
#   - Ninguno.
# RETORNO:
#   - None: Inyecta directamente el título ('st.title'), la descripción ('st.caption') y la división ('st.divider') en la UI de Streamlit.

def mostrar_cabecera():
    st.title("🌙 Sistema Explicable de Clasificación de Fases del Sueño")
    st.caption(
        "Dataset DREAMT · "
        "Clasificación de fases del sueño mediante señal PPG · "
        "Aprendizaje automático con explicabilidad clínica"
    )
    st.divider()


# =============================================================================
# MODO INDIVIDUAL — 5 tabs para analizar un paciente concreto
# =============================================================================

# DESCRIPCIÓN: Controlador principal de la vista en modo individual. Despliega la navegación por pestañas 
#              (Tabs 1 a 5) para el análisis en profundidad de un paciente seleccionado en la barra lateral. 
#              Orquesta secuencialmente las llamadas a las funciones controladoras correspondientes a la señal 
#              fisiológica, alertas clínicas e interpretabilidad (LIME, SHAP + Permutation Importance y PDP), 
#              transmitiendo la configuración global.
# PARÁMETROS:
#   - config (dict): Ajustes del sidebar con los parámetros del paciente, modelo, número de clases y ruta del proyecto.
# RETORNO:
#   - None: Renderiza la estructura de 5 pestañas y delega el contenido de cada una en la UI de Streamlit.

def mostrar_modo_individual(config: dict):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Señal y ciclo del sueño",
        "🚨 Alertas clínicas",
        "🔍 LIME",
        "📈 SHAP + Permutation",
        "📉 PDP"
    ])

    with tab1:
        mostrar_tab1(config)

    with tab2:
        mostrar_tab2(config)

    with tab3:
        mostrar_tab3(config)

    with tab4:
        mostrar_tab4(config)

    with tab5:
        mostrar_tab5(config)


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

# DESCRIPCIÓN: Punto de entrada principal (main) de la aplicación Streamlit. Inicializa la cabecera 
#              global y la barra lateral de configuración (`mostrar_sidebar`), controlando el flujo de 
#              navegación entre el 'Modo Individual' (pestañas por paciente) y el 'Modo Poblacional' 
#              (dashboard global con estadísticas del cohorte). Gestiona el estado de la sesión 
#              y la persistencia en caché ante reejecuciones o cambios de escenario.
# PARÁMETROS:
#   - Ninguno.
# RETORNO:
#   - None: Ejecuta la aplicación e inyecta la interfaz correspondiente en Streamlit.

def main():
    mostrar_cabecera()
    config = mostrar_sidebar()

    if config['modo'] == 'Individual':
        mostrar_modo_individual(config)
    else:
        # Buscar si hay algún escenario ya analizado en caché
        clave_activa = None
        for n in [2, 3, 4]:
            if f"datos_pob_{n}" in st.session_state:
                clave_activa = n
                break

        if clave_activa is None and not config['analizar']:
            # Primera vez — no hay nada cacheado
            st.info("👆 Selecciona la configuración y pulsa **Analizar** para cargar el dashboard poblacional.")
        else:
            # Si cambió escenario sin pulsar Analizar, mostrar el último analizado
            if clave_activa is not None and clave_activa != config.get('num_clases') and not config['analizar']:
                mostrar_dashboard({**config, 'num_clases': clave_activa})
            else:
                mostrar_dashboard(config)


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main()
