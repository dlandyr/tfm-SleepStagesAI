# =============================================================================
# Limpieza y validación de etiquetas PSG del dataset DREAMT.
# Elimina etapas de preparación (P) y epochs sin etiqueta (Missing).
# =============================================================================

import numpy as np
import pandas as pd

# DESCRIPCIÓN: Realiza el filtrado y depuración del flujo de etiquetas de la 
#              polisomnografía (PSG). Remueve los registros correspondientes a 
#              fases de preparación ('P'), valores ausentes o no registrados 
#              ('Missing'), y remueve cualquier valor nulo (NaN). 
#              Garantiza que el conjunto de datos contenga únicamente los estados 
#              estándar del ciclo del sueño (W, N1, N2, N3, R).
# PARÁMETROS:
#   - df (pd.DataFrame): Matriz de datos cruda extraída del registro biológico.
# RETORNO:
#   - pd.DataFrame: Matriz depurada, con índices reorganizados y listos para 
#                   las etapas de entrenamiento o evaluación.

def filtrar_estados_validos(df: pd.DataFrame) -> pd.DataFrame:
    df_limpio = df[~df['Sleep_Stage'].isin(['P', 'Missing'])].copy()
    df_limpio = df_limpio.dropna(subset=['Sleep_Stage'])
    df_limpio = df_limpio.reset_index(drop=True)
    return df_limpio

# DESCRIPCIÓN: Valida la integridad estructural del conjunto de datos de entrada. 
#              Verifica la existencia de las columnas mínimas obligatorias 
#              requeridas para la correcta ejecución del flujo de procesamiento 
#              fisiológico y la posterior extracción de características.
# PARÁMETROS:
#   - df (pd.DataFrame): Matriz de datos biológicos cuya estructura se va a verificar.
# RETORNO:
#   - bool: Retorna True si la matriz cumple con el diseño requerido. Retorna 
#           False y muestra una advertencia en consola si existe alguna omisión.

def validar_columnas_requeridas(df: pd.DataFrame) -> bool:
    columnas_requeridas = ['BVP', 'Sleep_Stage']
    columnas_faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if columnas_faltantes:
        print(f"   [ADVERTENCIA] Columnas faltantes: {columnas_faltantes}")
        return False
    return True
