# =============================================================================
# Procesamiento de un archivo CSV de participante DREAMT:
#   1. Filtrado de etiquetas inválidas (P, Missing)
#   2. Segmentación en epochs de 30 s
#   3. Filtro Savitzky-Golay (ventana=9, orden=3)
#   4. Extracción de features BVP
# =============================================================================

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from src.filters import filtrar_estados_validos, validar_columnas_requeridas
from src.features import calcular_metricas_bvp

# DESCRIPCIÓN: Coordina el flujo completo de procesamiento y limpieza de datos 
#              para un paciente individual. Realiza la validación estructural, 
#              elimina registros inválidos, segmenta la señal continua en épocas 
#              de 30 segundos, aplica un suavizado digital y extrae el vector 
#              de características fisiológicas junto con su etiqueta de sueño.
# PARÁMETROS:
#   - df_crudo (pd.DataFrame): Matriz original leída directamente del archivo del paciente.
#   - id_paciente (str):       Identificador único del paciente (ej. "S001").
#   - frecuencia_hz (int):     Frecuencia de muestreo de la señal (64 Hz).
# RETORNO:
#   - Contiene los datos con los resultados finales del paciente:
#       1. X_sujeto (pd.DataFrame): Matriz con las características extraídas.
#       2. y_sujeto (np.ndarray):   Vector con las etiquetas de las fases del sueño.
#   Nota: Si el paciente no tiene información útil, devuelve ambos elementos vacíos.

def procesar_registro_paciente(df_crudo: pd.DataFrame,
                               id_paciente: str,
                               hertz_freq: int = 64) -> tuple[pd.DataFrame, np.ndarray]:
    # Validación de columnas mínimas
    if not validar_columnas_requeridas(df_crudo):
        print(f"[ERROR] Paciente {id_paciente}: columnas insuficientes, se omite.")
        return pd.DataFrame(), np.array([])

    # Limpieza de etiquetas
    df = filtrar_estados_validos(df_crudo)
    if df.empty:
        print(f"[AVISO] Paciente {id_paciente}: sin epochs válidos tras filtrado.")
        return pd.DataFrame(), np.array([])

    # Configuración de la ventana de 30 s
    window_size = hertz_freq * 30   # 1920 muestras a 64 Hz

    records = []

    for i in range(0, len(df) - window_size + 1, window_size):
        sub_df = df.iloc[i: i + window_size]

        # Eliminar ventanas incompletas (última ventana parcial)
        if len(sub_df) < window_size:
            break

        bvp_raw = sub_df['BVP'].values.astype(float)

        # Filtro Savitzky-Golay: ventana=9, polinomio orden=3
        bvp_filtrado = savgol_filter(bvp_raw, window_length=9, polyorder=3)

        # Extracción de features
        feat_dict = calcular_metricas_bvp(bvp_filtrado, hertz_freq)

        if feat_dict is not None:
            # Etiqueta del epoch: moda de las etiquetas dentro de la ventana.
            # En la práctica todos los samples de 30s tienen la misma etiqueta
            # porque el PSG anota por epochs de 30s, así que la moda es estable.
            etiqueta = sub_df['Sleep_Stage'].mode()[0]
            feat_dict['Sleep_Stage'] = etiqueta
            feat_dict['id_paciente'] = id_paciente
            records.append(feat_dict)

    if not records:
        print(f"[AVISO] Paciente {id_paciente}: ningún epoch superó el control de calidad.")
        return pd.DataFrame(), np.array([])

    df_sujeto = pd.DataFrame(records)
    y_sujeto  = df_sujeto['Sleep_Stage'].values
    X_sujeto  = df_sujeto.drop(columns=['Sleep_Stage'])

    print(f"Paciente {id_paciente}: {len(X_sujeto)} epochs válidos extraídos.")
    return X_sujeto, y_sujeto
