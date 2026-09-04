# =============================================================================
# Configuración de los tres modelos de clasificación seleccionados
# Parámetros del modelo en base a la literatura científica:
#   - Random Forest: 200 estimadores, max_depth=15
#   - KNN:           k=5, distancia euclídea
#   - XGBoost:       200 rondas, max_depth=8, eval_metric='mlogloss'
# =============================================================================

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# DESCRIPCIÓN: Inicializa y configura el algoritmo de clasificación seleccionado 
#              aplicando los parámetros óptimos establecidos en el estudio de 
#              referencia.
# PARÁMETROS:
#   - algoritmo (str):   Identificador del modelo a construir ('rf', 'knn', 'xgboost' o 'svm').
#   - num_clases (int):  Cantidad de fases de sueño configuradas (reservado para 
#                        futuras extensiones de la arquitectura).
# RETORNO:
#   - Devuelve el modelo seleccionado (RF, KNN o XGBoost) listo para ser entrenado.
# EXCEPCIONES:
#   - ValueError: Se dispara si el identificador del algoritmo no es válido.

def configurar_modelo(algoritmo: str, num_clases: int = None):
    if algoritmo == 'rf':
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'       
        )

    elif algoritmo == 'knn':
        return KNeighborsClassifier(
            n_neighbors=5,
            metric='euclidean',
            n_jobs=-1
        )

    elif algoritmo == 'xgboost':
        return XGBClassifier(
            n_estimators=200,
            max_depth=8,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
    
    elif algoritmo == 'svm':
        return SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )

    else:
        raise ValueError(
            f"Algoritmo '{algoritmo}' no soportado. "
            f"Opciones válidas: 'rf', 'knn', 'xgboost', 'svm'."
        )


# DESCRIPCIÓN: Almacena el modelo ya entrenado en el disco local utilizando la 
#              librería Joblib. Esto permite persistir el clasificador para 
#              poder reutilizarlo en el futuro sin necesidad de volver a entrenarlo.
# PARÁMETROS:
#   - modelo:             El modelo ya entrenado (RF, KNN o XGBoost).
#   - algoritmo (str):    Identificador del modelo utilizado ('rf', 'knn', 'xgboost').
#   - num_clases (int):   Cantidad de fases de sueño configuradas.
#   - ruta_modelos (str): Directorio o carpeta de destino donde se salvará el archivo.
# RETORNO:
#   - str: La ruta completa del archivo binario (.pkl) generado en el disco.

def guardar_modelo(modelo, algoritmo: str, num_clases: int, ruta_modelos: str) -> str:
    os.makedirs(ruta_modelos, exist_ok=True)
    nombre = f"{algoritmo}_{num_clases}_clases.pkl"
    ruta   = os.path.join(ruta_modelos, nombre)
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado: {ruta}")
    return ruta

# DESCRIPCIÓN: Recupera y carga desde el disco local un modelo previamente 
#              entrenado y guardado en formato binario (.pkl). Esto permite 
#              realizar predicciones de forma inmediata sobre nuevos datos de 
#              prueba sin necesidad de repetir el proceso de entrenamiento.
# PARÁMETROS:
#   - algoritmo (str):    Identificador del modelo a cargar ('rf', 'knn', 'xgboost').
#   - num_clases (int):   Cantidad de fases de sueño configuradas en el modelo.
#   - ruta_modelos (str): Directorio o carpeta donde se encuentra almacenado el archivo.
# RETORNO:
#   - Devuelve el modelo seleccionado (RF, KNN o XGBoost) listo para hacer inferencias.
# EXCEPCIONES:
#   - FileNotFoundError: Se dispara si el archivo solicitado no existe en la ruta especificada.

def cargar_modelo(algoritmo: str, num_clases: int, ruta_modelos: str):    
    nombre = f"{algoritmo}_{num_clases}_clases.pkl"
    ruta   = os.path.join(ruta_modelos, nombre)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Modelo no encontrado: {ruta}!")
    return joblib.load(ruta)
