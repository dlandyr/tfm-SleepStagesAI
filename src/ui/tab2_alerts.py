# =============================================================================
# src/ui/tab2_alerts.py
# Tab 2 — Alertas clínicas automáticas.
#
# Muestra:
#   1. Semáforo resumen (crítico / aviso / normal)
#   2. Puntuación de calidad del sueño (0-100)
#   3. Alerta detallada por cada fase con barra de progreso
#      y recomendación clínica personalizada
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
from src.ui.alerts import calcular_nivel_alerta
from src.ui.tab1_signal import COLORES_FASES, RANGOS_NORMALES, _cargar_datos_paciente


# =============================================================================
# UMBRALES CLÍNICOS DE ALERTA
# =============================================================================

# Cada fase tiene tres niveles: crítico, aviso y normal
# basados en literatura clínica de medicina del sueño
UMBRALES = {
    'Wake': {
        'critico': (20, 100),   # Wake > 20% → crítico
        'aviso':   (10, 20),    # Wake 10-20% → aviso
        'normal':  (5,  10),    # Wake 5-10% → normal
        'descripcion': (
            "Exceso de vigilia nocturna. Un valor superior al 10% "
            "indica despertares frecuentes que fragmentan el ciclo "
            "del sueño. Puede ser compatible con apnea obstructiva, "
            "insomnio o síndrome de piernas inquietas."
        ),
        'recomendacion': (
            "Realizar estudio de polisomnografía para descartar apnea. "
            "Valorar higiene del sueño, factores de estrés y "
            "medicación que pueda afectar al sueño."
        )
    },
    'Sleep': {
        'critico': (0,  70),
        'aviso':   (70, 85),
        'normal':  (85, 95),
        'descripcion': "Tiempo total de sueño reducido respecto al esperado.",
        'recomendacion': "Mantener horarios regulares de sueño y evitar pantallas antes de dormir."
    },
    'NREM': {
        'critico': (0,  40),
        'aviso':   (40, 45),
        'normal':  (45, 75),
        'descripcion': "Sueño NREM insuficiente. Afecta a la recuperación física.",
        'recomendacion': "Evitar alcohol y cafeína. Mantener temperatura fresca en el dormitorio."
    },
    'Light': {
        'critico': (60, 100),
        'aviso':   (55, 60),
        'normal':  (40, 55),
        'descripcion': (
            "Sueño ligero elevado. Puede deberse a compensación "
            "por déficit de sueño profundo — el organismo no consigue "
            "alcanzar las fases más profundas."
        ),
        'recomendacion': "Reducir el estrés antes de dormir. Practicar técnicas de relajación."
    },
    'Deep': {
        'critico': (0,  10),    # Deep < 10% → crítico
        'aviso':   (10, 15),    # Deep 10-15% → aviso
        'normal':  (15, 25),    # Deep 15-25% → normal
        'descripcion': (
            "Déficit de sueño profundo (N3). Este es el período de "
            "máxima recuperación física y consolidación de la memoria. "
            "Un déficit prolongado se asocia con deterioro cognitivo, "
            "mayor riesgo cardiovascular y debilitamiento inmune."
        ),
        'recomendacion': (
            "Consultar con especialista en medicina del sueño. "
            "Evitar alcohol y cafeína. Mantener horarios regulares."
        )
    },
    'REM': {
        'critico': (0,  10),
        'aviso':   (10, 15),
        'normal':  (20, 25),
        'descripcion': (
            "Sueño REM insuficiente. El sueño REM es fundamental "
            "para la consolidación de la memoria, el procesamiento "
            "emocional y la creatividad."
        ),
        'recomendacion': (
            "Evitar alcohol — suprime el sueño REM. "
            "Mantener horario de sueño regular. "
            "Reducir medicamentos que inhiben REM si es posible."
        )
    }
}


# =============================================================================
# FUNCIÓN PRINCIPAL DEL TAB
# =============================================================================

# DESCRIPCIÓN: Controlador principal de la pestaña de análisis clínico individual (Tab 2). Gestiona el 
#              estado de la sesión de Streamlit (session_state), invoca la carga/predicción de datos 
#              del paciente, calcula las desviaciones metabólicas y genera de forma secuencial los 
#              componentes de diagnóstico clínico: semáforo de riesgo, puntuación de calidad de sueño y 
#              desglose detallado de alertas médicas.
# PARÁMETROS:
#   - config (dict): Diccionario del sidebar con la selección del usuario ('paciente', 'modelo', 'num_clases', 'ruta_proyecto', 'analizar').
# RETORNO:
#   - None: Renderiza directamente el flujo visual de diagnóstico y alertas clínicas en el Tab 2.

def mostrar_tab2(config: dict):
    clave = f"tab2_{config['paciente']}_{config['modelo']}_{config['num_clases']}"

    if not config.get('analizar') and clave not in st.session_state:
        st.info("👈 Pulsa **Analizar** en el sidebar para ver las alertas clínicas.")
        return

    if config.get('analizar') or clave not in st.session_state:
        with st.spinner("Analizando fases del sueño..."):
            datos = _cargar_datos_paciente(config)
        if datos is None:
            return
        st.session_state[clave] = datos
    else:
        datos = st.session_state[clave]

    distribucion     = datos['distribucion']
    clases_ordenadas = datos['clases_ordenadas']

    alertas = _calcular_alertas(distribucion, clases_ordenadas)

    _mostrar_semaforo(alertas)
    st.divider()
    _mostrar_puntuacion_calidad(alertas, distribucion, clases_ordenadas)
    st.divider()
    _mostrar_alertas_detalladas(alertas, distribucion, clases_ordenadas)


# =============================================================================
# CÁLCULO DE ALERTAS
# =============================================================================

# DESCRIPCIÓN: Evalúa el estado clínico de cada fase del sueño para el paciente comparando su 
#              distribución porcentual con los umbrales de referencia estándar mediante la función 
#              centralizada de alerta, determinando la gravedad de la desviación ('critico', 'aviso' o 'normal').
# PARÁMETROS:
#   - distribucion (dict): Diccionario {fase: porcentaje} con los valores observados en el paciente.
#   - clases_ordenadas (list): Lista de fases del sueño en su orden canónico según el escenario activo.
# RETORNO:
#   - dict: Diccionario estructurado por fase {fase: {'nivel': str, 'pct': float, 'rango': tuple}}.

def _calcular_alertas(distribucion: dict,
                      clases_ordenadas: list) -> dict:
    alertas = {}

    for fase in clases_ordenadas:
        pct   = distribucion.get(fase, 0)
        rango = RANGOS_NORMALES.get(fase, (0, 100))

        # Usar la función centralizada en lugar de UMBRALES locales
        nivel = calcular_nivel_alerta(fase, pct)

        alertas[fase] = {
            'nivel': nivel,
            'pct':   pct,
            'rango': rango,
        }

    return alertas

# DESCRIPCIÓN: Quantifica la calidad global del sueño mediante un índice entero ponderado de 0 a 100, 
#              iniciando en una puntuación perfecta (100) y sustrayendo penalizaciones según el nivel de 
#              gravedad clínica registrado en cada fase (-25 por cada 'critico', -10 por cada 'aviso').
# PARÁMETROS:
#   - alertas (dict): Diccionario estructurado por fase con sus correspondientes niveles de alerta ('nivel').
#   - distribucion (dict): Diccionario con las proporciones porcentuales por fase del paciente.
#   - clases_ordenadas (list): Lista de fases del sueño en su orden canónico según el escenario activo.
# RETORNO:
#   - int: Puntuación entera del índice de calidad del sueño ajustada al rango [0, 100].

def _calcular_puntuacion(alertas: dict,
                         distribucion: dict,
                         clases_ordenadas: list) -> int:
    puntuacion = 100
    for fase, datos_alerta in alertas.items():
        if datos_alerta['nivel'] == 'critico':
            puntuacion -= 25
        elif datos_alerta['nivel'] == 'aviso':
            puntuacion -= 10
    return max(0, puntuacion)


# =============================================================================
# COMPONENTE 1 — SEMÁFORO RESUMEN
# =============================================================================

# DESCRIPCIÓN: Visualiza un resumen tripartito estilo semáforo (st.error, st.warning, st.success) 
#              que contabiliza y clasifica las fases del sueño del paciente según su gravedad clínica 
#              (alertas críticas, avisos y fases dentro de rango normal).
# PARÁMETROS:
#   - alertas (dict): Diccionario que contiene el nivel de alerta asignado a cada fase del sueño.
# RETORNO:
#   - None: Inyecta directamente las tres tarjetas de resumen cromático en la interfaz de Streamlit.

def _mostrar_semaforo(alertas: dict):
    st.markdown("#### Resumen de alertas")

    n_critico = sum(1 for a in alertas.values() if a['nivel'] == 'critico')
    n_aviso   = sum(1 for a in alertas.values() if a['nivel'] == 'aviso')
    n_normal  = sum(1 for a in alertas.values() if a['nivel'] == 'normal')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error(f"🔴 **{n_critico}** alerta{'s' if n_critico != 1 else ''} crítica{'s' if n_critico != 1 else ''}")

    with col2:
        st.warning(f"🟡 **{n_aviso}** aviso{'s' if n_aviso != 1 else ''}")

    with col3:
        st.success(f"🟢 **{n_normal}** fase{'s' if n_normal != 1 else ''} normal{'es' if n_normal != 1 else ''}")


# =============================================================================
# COMPONENTE 2 — PUNTUACIÓN DE CALIDAD
# =============================================================================

# DESCRIPCIÓN: Calcula el índice global de calidad del sueño (0-100) y renderiza en la interfaz un velocímetro 
#              interactivo (go.Indicator de Plotly) coloreado por tramos, acompañado de la clasificación 
#              cualitativa ('Buena', 'Aceptable', 'Deficiente', 'Muy deficiente') y recomendaciones clínicas.
# PARÁMETROS:
#   - alertas (dict): Diccionario estructurado con los niveles de alerta asignados por fase.
#   - distribucion (dict): Proporciones Porcentuales observadas en el paciente por fase.
#   - clases_ordenadas (list): Secuencia canónica de fases del sueño según el escenario.
# RETORNO:
#   - None: Inyecta el indicador tipo gauge y la síntesis diagnóstica en la interfaz de Streamlit.

def _mostrar_puntuacion_calidad(alertas: dict,
                                distribucion: dict,
                                clases_ordenadas: list):
    st.markdown("#### Puntuación de calidad del sueño")

    puntuacion = _calcular_puntuacion(alertas, distribucion, clases_ordenadas)

    # Determinar calificación textual
    if puntuacion >= 80:
        calificacion = "Buena"
        color_gauge  = "#1D9E75"
    elif puntuacion >= 60:
        calificacion = "Aceptable"
        color_gauge  = "#BA7517"
    elif puntuacion >= 40:
        calificacion = "Deficiente"
        color_gauge  = "#D85A30"
    else:
        calificacion = "Muy deficiente"
        color_gauge  = "#E24B4A"

    col1, col2 = st.columns([1, 2])

    with col1:
        # Gráfico gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=puntuacion,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': color_gauge},
                'steps': [
                    {'range': [0,  40], 'color': '#FCEBEB'},
                    {'range': [40, 60], 'color': '#FAEEDA'},
                    {'range': [60, 80], 'color': '#E1F5EE'},
                    {'range': [80, 100],'color': '#E1F5EE'},
                ],
            },
            number={'suffix': '/100'}
        ))
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=200,
            autosize=False,
            width=300
        )
        st.plotly_chart(fig, use_container_width=False)

    with col2:
        st.markdown(f"**Calidad: {calificacion}**")
        n_critico = sum(1 for a in alertas.values() if a['nivel'] == 'critico')
        if n_critico > 0:
            fases_crit = [f for f, a in alertas.items() if a['nivel'] == 'critico']
            st.write(
                f"El paciente presenta {n_critico} alerta{'s' if n_critico > 1 else ''} "
                f"crítica{'s' if n_critico > 1 else ''} en "
                f"{'las fases' if n_critico > 1 else 'la fase'}: "
                f"**{', '.join(fases_crit)}**. "
                "Se recomienda evaluación por especialista en medicina del sueño."
            )
        else:
            st.write(
                "No se detectaron alertas críticas. "
                "La arquitectura del sueño es aceptable para este registro."
            )


# =============================================================================
# COMPONENTE 3 — ALERTAS DETALLADAS POR FASE
# =============================================================================

# DESCRIPCIÓN: Despliega un desglose detallado e interactivo por cada fase del sueño en contenedores 
#              desplegables (st.expander). Muestra las métricas observadas vs. los rangos normales clínicos, 
#              una barra de progreso visual de la proporción y la descripción médica o recomendación personalizada 
#              asociada a la fase según el nivel de alerta determinado ('critico', 'aviso' o 'normal').
# PARÁMETROS:
#   - alertas (dict): Diccionario estructurado por fase con sus correspondientes niveles de alerta, porcentajes y rangos.
#   - distribucion (dict): Diccionario {fase: porcentaje} con los valores calculados para el paciente.
#   - clases_ordenadas (list): Secuencia de fases del sueño en su orden canónico según el escenario.
# RETORNO:
#   - None: Inyecta los contenedores expandibles y componentes visuales detallados en Streamlit.

def _mostrar_alertas_detalladas(alertas: dict,
                                distribucion: dict,
                                clases_ordenadas: list):
    st.markdown("#### Análisis detallado por fase")

    for fase in clases_ordenadas:
        alerta = alertas[fase]
        nivel  = alerta['nivel']
        pct    = alerta['pct']
        rango  = alerta['rango']
        umbral = UMBRALES.get(fase, {})

        # Icono y color según nivel
        if nivel == 'critico':
            icono    = "🔴"
            expander = st.expander(f"{icono} **{fase}** — {pct}% (crítico)", expanded=True)
        elif nivel == 'aviso':
            icono    = "🟡"
            expander = st.expander(f"{icono} **{fase}** — {pct}% (aviso)", expanded=False)
        else:
            icono    = "🟢"
            expander = st.expander(f"{icono} **{fase}** — {pct}% (normal)", expanded=False)

        with expander:
            col1, col2 = st.columns(2)
            col1.metric("Valor observado", f"{pct}%")
            col2.metric("Rango normal", f"{rango[0]}-{rango[1]}%")

            # Barra de progreso
            st.markdown(f"**Posición respecto al rango normal ({rango[0]}-{rango[1]}%)**")
            progreso = min(pct / 100, 1.0)
            st.progress(progreso)

            # Descripción clínica
            if umbral.get('descripcion'):
                st.info(umbral['descripcion'])

            # Recomendación
            if nivel in ('critico', 'aviso') and umbral.get('recomendacion'):
                st.markdown("**Recomendación:**")
                st.write(umbral['recomendacion'])