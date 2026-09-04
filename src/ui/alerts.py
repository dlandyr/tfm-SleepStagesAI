# =============================================================================
# src/ui/alerts.py
# Lógica compartida de alertas clínicas.
# Contiene las funciones que calculan alertas y puntuaciones
# usadas tanto en tab2_alerts.py como en dashboard.py.
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
# RANGOS CLÍNICOS NORMALES PARA ADULTOS
# Basados en literatura médica de medicina del sueño
# =============================================================================

RANGOS_NORMALES = {
    'Wake':  (5,  10),   # 5-10% del tiempo total
    'Sleep': (85, 95),   # 85-95% del tiempo total
    'NREM':  (45, 75),   # 45-75% del tiempo total
    'Light': (40, 55),   # 40-55% del tiempo total
    'Deep':  (15, 25),   # 15-25% del tiempo total
    'REM':   (20, 25),   # 20-25% del tiempo total
}

# Descripciones clínicas por fase
DESCRIPCIONES_CLINICAS = {
    'Wake': (
        "Vigilia nocturna excesiva. Un porcentaje superior al 10% "
        "indica despertares frecuentes que fragmentan el ciclo del sueño. "
        "Puede ser compatible con apnea obstructiva del sueño, "
        "insomnio o síndrome de piernas inquietas."
    ),
    'Deep': (
        "Déficit de sueño profundo (N3). Este es el período de máxima "
        "recuperación física y consolidación de la memoria declarativa. "
        "Un déficit prolongado se asocia con deterioro cognitivo, "
        "mayor riesgo cardiovascular y debilitamiento del sistema inmune."
    ),
    'REM': (
        "Sueño REM insuficiente. El sueño REM es fundamental para la "
        "consolidación de la memoria procedimental, el procesamiento "
        "emocional y la creatividad."
    ),
    'Light': (
        "Sueño ligero elevado. Puede deberse a compensación por déficit "
        "de sueño profundo — el organismo no consigue alcanzar las fases "
        "más profundas y permanece en fases superficiales."
    ),
    'NREM': (
        "Sueño NREM fuera del rango esperado. "
        "Afecta a la recuperación física nocturna."
    ),
    'Sleep': (
        "Tiempo total de sueño reducido. "
        "El organismo no está descansando el tiempo suficiente."
    ),
}

# Recomendaciones clínicas por fase
RECOMENDACIONES = {
    'Wake': (
        "Realizar estudio de polisomnografía para descartar apnea del sueño. "
        "Valorar higiene del sueño, factores de estrés y medicación "
        "que pueda interferir con el sueño."
    ),
    'Deep': (
        "Consultar con especialista en medicina del sueño. "
        "Evitar alcohol y cafeína antes de dormir. "
        "Mantener horarios regulares de sueño y temperatura fresca."
    ),
    'REM': (
        "Evitar alcohol — suprime el sueño REM. "
        "Mantener horario de sueño regular. "
        "Consultar si se toman medicamentos que inhiban el sueño REM."
    ),
    'Light': (
        "Reducir el estrés antes de dormir. "
        "Practicar técnicas de relajación como meditación o respiración profunda."
    ),
    'NREM': (
        "Mantener horarios regulares de sueño. "
        "Evitar pantallas y luz azul al menos 1 hora antes de dormir."
    ),
    'Sleep': (
        "Aumentar el tiempo dedicado al sueño. "
        "La mayoría de adultos necesitan entre 7 y 9 horas por noche."
    ),
}


# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

# DESCRIPCIÓN: Determina el nivel de alerta médica ('critico', 'aviso' o 'normal') para una fase 
#              del sueño específica evaluando el porcentaje observado respecto a los rangos de referencia.
# PARÁMETROS:
#   - fase (str): Nombre de la fase del sueño (ej. 'Wake', 'Deep', 'REM').
#   - porcentaje (float): Porcentaje de tiempo acumulado en dicha fase (0 a 100).
# RETORNO:
#   - str: Categoría de alerta ('critico', 'aviso' o 'normal').

def calcular_nivel_alerta(fase: str, porcentaje: float) -> str:   
    rango = RANGOS_NORMALES.get(fase, (0, 100))
    min_normal = rango[0]
    max_normal = rango[1]
    margen     = (max_normal - min_normal) * 0.3  # 30% del rango

    # Crítico: más del 50% fuera del rango
    if porcentaje < min_normal - margen or porcentaje > max_normal + margen:
        return 'critico'

    # Aviso: fuera del rango pero no crítico
    if porcentaje < min_normal or porcentaje > max_normal:
        return 'aviso'

    # Normal: dentro del rango
    return 'normal'


# DESCRIPCIÓN: Calcula un índice global de calidad del sueño (de 0 a 100) aplicando penalizaciones
#              acumulativas según los niveles de alerta ('critico': -25, 'aviso': -10) detectados
#              en la distribución de fases del paciente.
# PARÁMETROS:
#   - distribucion (dict): Estructura {nombre_fase: porcentaje_tiempo} con las proporciones observadas.
# RETORNO:
#   - int: Puntuación entera acotada en el rango [0, 100].

def calcular_puntuacion_calidad(distribucion: dict) -> int:
    puntuacion = 100

    for fase, porcentaje in distribucion.items():
        nivel = calcular_nivel_alerta(fase, porcentaje)
        if nivel == 'critico':
            puntuacion -= 25
        elif nivel == 'aviso':
            puntuacion -= 10

    return max(0, int(puntuacion))

# DESCRIPCIÓN: Convierte una puntuación numérica de calidad del sueño (0-100) en una categoría
#              cualitativa ('Buena', 'Aceptable', 'Deficiente', 'Muy deficiente') y le asigna
#              un código de color hexadecimal para el renderizado en la interfaz gráfica.
# PARÁMETROS:
#   - puntuacion (int): Puntuación de calidad calculada en el rango [0, 100].
# RETORNO:
#   - tuple[str, str]: Tupla con el texto de la calificación y el código de color hexadecimal (ej. ("Buena", "#1D9E75")).

def obtener_calificacion(puntuacion: int) -> tuple:    
    if puntuacion >= 80:
        return "Buena",         "#1D9E75"
    elif puntuacion >= 60:
        return "Aceptable",     "#BA7517"
    elif puntuacion >= 40:
        return "Deficiente",    "#D85A30"
    else:
        return "Muy deficiente","#E24B4A"


# DESCRIPCIÓN: Genera un informe clínico estructurado en lenguaje natural con las puntuaciones 
#              de calidad del sueño, la detección de anomalías por fase (alertas críticas y avisos) 
#              y las recomendaciones médicas correspondientes para el paciente analizado.
# PARÁMETROS:
#   - distribucion (dict): Estructura {nombre_fase: porcentaje_tiempo} con las proporciones del paciente.
#   - paciente_id (str, opcional): Identificador único del paciente para la cabecera del informe.
# RETORNO:
#   - str: Cadena de texto formateada con el contenido completo del informe clínico.

def generar_informe_texto(distribucion: dict,
                           paciente_id: str = None) -> str:
    puntuacion   = calcular_puntuacion_calidad(distribucion)
    calificacion = obtener_calificacion(puntuacion)[0]

    lineas = []
    if paciente_id:
        lineas.append(f"Informe de calidad del sueño — {paciente_id}")
        lineas.append("=" * 50)

    lineas.append(f"Puntuación de calidad: {puntuacion}/100 — {calificacion}")
    lineas.append("")

    # Alertas críticas
    alertas_crit = [
        f for f, p in distribucion.items()
        if calcular_nivel_alerta(f, p) == 'critico'
    ]
    if alertas_crit:
        lineas.append("ALERTAS CRÍTICAS:")
        for fase in alertas_crit:
            pct   = distribucion[fase]
            rango = RANGOS_NORMALES.get(fase, (0, 100))
            lineas.append(
                f"  • {fase}: {pct}% (normal: {rango[0]}-{rango[1]}%)"
            )
            lineas.append(f"    {RECOMENDACIONES.get(fase, '')}")

    # Avisos
    alertas_aviso = [
        f for f, p in distribucion.items()
        if calcular_nivel_alerta(f, p) == 'aviso'
    ]
    if alertas_aviso:
        lineas.append("")
        lineas.append("AVISOS:")
        for fase in alertas_aviso:
            pct   = distribucion[fase]
            rango = RANGOS_NORMALES.get(fase, (0, 100))
            lineas.append(
                f"  • {fase}: {pct}% (normal: {rango[0]}-{rango[1]}%)"
            )

    return "\n".join(lineas)


# DESCRIPCIÓN: Construye una lista estructurada de diccionarios con la información clínica detallada 
#              de cada fase del sueño (nivel de alerta, rangos de referencia, descripción y recomendaciones), 
#              ordenando los resultados según su prioridad o severidad ('critico' > 'aviso' > 'normal').
# PARÁMETROS:
#   - distribucion (dict): Estructura {nombre_fase: porcentaje_tiempo} con los valores observados.
#   - clases_ordenadas (list): Nombres de las fases del sueño en su secuencia canónica de procesamiento.
# RETORNO:
#   - list[dict]: Lista de diccionarios clasificados por gravedad con los datos clínicos de cada fase.

def calcular_alertas_lista(distribucion: dict,
                            clases_ordenadas: list) -> list:
    alertas = []

    for fase in clases_ordenadas:
        pct   = distribucion.get(fase, 0)
        nivel = calcular_nivel_alerta(fase, pct)
        rango = RANGOS_NORMALES.get(fase, (0, 100))

        alertas.append({
            'fase':           fase,
            'nivel':          nivel,
            'porcentaje':     pct,
            'rango_min':      rango[0],
            'rango_max':      rango[1],
            'descripcion':    DESCRIPCIONES_CLINICAS.get(fase, ''),
            'recomendacion':  RECOMENDACIONES.get(fase, ''),
        })

    # Ordenar: crítico primero, luego aviso, luego normal
    orden = {'critico': 0, 'aviso': 1, 'normal': 2}
    alertas.sort(key=lambda x: orden[x['nivel']])

    return alertas
