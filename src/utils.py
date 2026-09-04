# =============================================================================
# Funciones de preprocesamiento compartidas por todas las pruebas:
#   - Mapeo de etiquetas PSG a 2, 3 ó 4 clases
#   - Split estratificado 80/10/10
#   - Imputación por mediana
#   - Escalado estándar
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Diccionarios estándar para las etiquetas:
MAPEOS_CLASES = {
    2: {'W': 'Wake', 'N1': 'Sleep', 'N2': 'Sleep', 'N3': 'Sleep', 'R': 'Sleep'},
    3: {'W': 'Wake', 'N1': 'NREM',  'N2': 'NREM',  'N3': 'NREM',  'R': 'REM'},
    4: {'W': 'Wake', 'N1': 'Light', 'N2': 'Light', 'N3': 'Deep',  'R': 'REM'},
}

# DESCRIPCIÓN: Convierte las anotaciones originales de la polisomnografía (W, N1, 
#              N2, N3, R) al formato simplificado de agrupación de fases del 
#              sueño elegido para el análisis (2, 3 o 4 clases).
# PARÁMETROS:
#   - y_raw (np.ndarray):  Vector con las etiquetas originales de las señales.
#   - num_clases (int):    Cantidad de clases objetivo seleccionadas (2, 3 o 4).
# RETORNO:
#   - np.ndarray: Vector de texto con las nuevas fases transformadas.
# EXCEPCIONES:
#   - ValueError: Se dispara si la cantidad de clases no es válida.

def formatear_etiquetas_psg(y_raw: np.ndarray, num_clases: int) -> np.ndarray:
    if num_clases not in MAPEOS_CLASES:
        raise ValueError(
            f"num_clases debe ser 2, 3 ó 4. Recibido: {num_clases}"
        )

    mapeo = MAPEOS_CLASES[num_clases]
    y_mapeado = np.array([mapeo.get(e, e) for e in y_raw])
    return y_mapeado


# DESCRIPCIÓN: Ejecuta el flujo secuencial de preparación de datos previo al 
#              entrenamiento. Realiza de forma ordenada la adaptación de 
#              etiquetas, la división estratificada del conjunto de datos, 
#              la sustitución de valores nulos y la normalización estadística.
# PARÁMETROS:
#   - X (pd.DataFrame):       Matriz con todas las características fisiológicas extraídas.
#   - y_raw (np.ndarray):     Vector con las anotaciones de sueño originales.
#   - num_clases (int):       Cantidad de clases seleccionadas para el análisis (2, 3 o 4).
#   - test_size (float):  Proporción de pacientes o registros destinada al conjunto de prueba.
#   - val_size (float):   Proporción de pacientes o registros destinada al conjunto de validación.
#   - random_state (int): Semilla aleatoria utilizada para garantizar la reproducibilidad del split.
# RETORNO:
#   - Grupo de 6 elementos listos para entrenar el modelo:
#       1. X_train_df: Características de entrenamiento normalizadas.
#       2. X_val_df:   Características de validación (ajuste de hiperparámetros).
#       3. X_test_df:  Características de prueba (evaluación final).
#       4. y_train:    Etiquetas correspondientes al entrenamiento.
#       5. y_val:      Etiquetas correspondientes a la validación.
#       6. y_test:     Etiquetas correspondientes a la prueba.

def preprocesar_dataset(X: pd.DataFrame,
                        y_raw: np.ndarray,
                        num_clases: int,
                        test_size: float = 0.10,      
                        val_size: float = 0.10,       
                        random_state: int = 42
                        ) -> tuple:

    y_task = formatear_etiquetas_psg(y_raw, num_clases)

    # Split por paciente para evitar data leakage
    if 'id_paciente' in X.columns:
        pacientes = X['id_paciente'].unique()
        np.random.seed(random_state)
        indices = np.random.permutation(len(pacientes))

        n = len(pacientes)
        n_test = max(1, int(n * test_size))        # 10 pacientes
        n_val  = max(1, int(n * val_size))         # 10 pacientes

        # Distribución de los grupos de pacientes usando los índices permutados
        pac_test  = pacientes[indices[:n_test]]
        pac_val   = pacientes[indices[n_test:n_test + n_val]]
        pac_train = pacientes[indices[n_test + n_val:]]

        # Creación de máscaras booleanas para seleccionar los registros correspondientes a cada grupo de pacientes
        mask_train = X['id_paciente'].isin(pac_train)
        mask_val   = X['id_paciente'].isin(pac_val)
        mask_test  = X['id_paciente'].isin(pac_test)

        # Aislamiento de características (X) eliminando el ID del paciente para que no entre al modelo
        X_train = X[mask_train].drop(columns=['id_paciente'])
        X_val   = X[mask_val].drop(columns=['id_paciente'])
        X_test  = X[mask_test].drop(columns=['id_paciente'])

        # Aislamiento de las variables objetivo (y) usando los vectores de las máscaras
        y_train = y_task[mask_train.values]
        y_val   = y_task[mask_val.values]
        y_test  = y_task[mask_test.values]

        # Consola informativa del balanceo y volumen de datos resultante del split por sujeto
        print(f"Split 80-10-10 por paciente:")
        print(f"  Train: {len(pac_train)} pacientes — {len(X_train)} epochs")
        print(f"  Val:   {len(pac_val)} pacientes — {len(X_val)} epochs")
        print(f"  Test:  {len(pac_test)} pacientes — {len(X_test)} epochs")

    else:
        # Fallback — si no hay id_paciente usar split por epoch
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_task,
            test_size=test_size,
            stratify=None,
            random_state=random_state
        )
        X_val, y_val = X_test, y_test  # sin validación separada

    # Imputación de valores faltantes
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)

    # Escalado estadístico decaracterísticas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled   = scaler.transform(X_val_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    # Reconstrucción de estructuras DataFrame
    cols = X_train.columns
    X_train_df = pd.DataFrame(X_train_scaled, columns=cols)
    X_val_df   = pd.DataFrame(X_val_scaled,   columns=cols)
    X_test_df  = pd.DataFrame(X_test_scaled,  columns=cols)

    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test

# DESCRIPCIÓN: Genera la lista de etiquetas de las fases del sueño dispuestas en 
#              un orden fijo y estándar. Esto garantiza que las matrices de 
#              confusión y los reportes de rendimiento de los tres modelos 
#              (RF, KNN y XGBoost) mantengan siempre la misma disposición visual 
#              y sean comparables entre sí.
# PARÁMETROS:
#   - num_clases (int): Cantidad de grupos de sueño seleccionados (2, 3 o 4).
# RETORNO:
#   - list[str]: Lista de etiquetas en el orden correcto (o vacía si hay un error).

def obtener_clases_ordenadas(num_clases: int) -> list[str]:
    orden = {
        2: ['Wake', 'Sleep'],
        3: ['Wake', 'NREM', 'REM'],
        4: ['Wake', 'Light', 'Deep', 'REM'],
    }
    return orden.get(num_clases, [])
