# =============================================================================
# Punto de entrada para ejecutar una prueba de clasificación.
# Uso:
#   python main.py                      # usa configuración en configs/config.yaml
#   python main.py --clases 4           # sobreescribe número de clases
#   python main.py --skip-extraccion    # salta Fase 1 si ya existe X_features.csv
# =============================================================================

import os
import glob
import argparse
import pandas as pd
import numpy as np
import yaml
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from src.pipeline import procesar_registro_paciente
from src.utils import preprocesar_dataset, obtener_clases_ordenadas
from src.models import configurar_modelo, guardar_modelo
from src.evaluation import (
    calcular_metricas_completas,
    calcular_accuracy_individual,
    construir_tabla_resultados,
    exportar_resultados_csv,
    graficar_matriz_confusion
)
from src.explainers import (
    generar_analisis_lime,
    generar_lime_global,
    generar_shap_global, 
)


# =============================================================================
# CARGA DE CONFIGURACIÓN
# =============================================================================

# DESCRIPCIÓN: Abre y lee el archivo de configuración del proyecto para cargar 
#              los parámetros globales (rutas, opciones de modelos y carpetas).
# PARÁMETROS:
#   - ruta_yaml (str): Ubicación o ruta en el disco del archivo de texto.
# RETORNO:
#   - dict: Diccionario de Python con todas las opciones y variables cargadas.

def cargar_config(ruta_yaml: str) -> dict:
    with open(ruta_yaml, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# FASE 1: EXTRACCIÓN DE FEATURES
# =============================================================================

# DESCRIPCIÓN: Procesa de forma masiva los archivos de datos de los pacientes.
#              Lee las señales originales, calcula las variables fisiológicas
#              de cada una y consolida toda la información en dos únicos 
#              archivos finales para el entrenamiento de los modelos.
# PARÁMETROS:
#   - cfg (dict):        Diccionario que contiene las opciones y variables globales.
#   - ruta_base (str):   Dirección de la carpeta principal del proyecto en el disco.
# RETORNO:
#   - None: No devuelve nada en memoria, guarda los resultados directamente en archivos.

def extraer_caracteristicas_dataset(cfg: dict, ruta_base: str) -> None:
    carpeta_raw = os.path.join(ruta_base, "data", "raw")
    num_pacientes  = cfg['datos']['num_pacientes']
    frecuencia_hz  = cfg['datos']['frecuencia_hz']

    patron_busqueda = os.path.join(carpeta_raw, "paciente*.[cC][sS][vV]")
    archivos = sorted(glob.glob(patron_busqueda))[:num_pacientes]

    print(f"\n[FASE 1] Extracción de características - {len(archivos)} pacientes")

    X_lista, y_lista = [], []

    for ruta in archivos:
        nombre_archivo   = os.path.basename(ruta)
        id_paciente = nombre_archivo.split(".")[0]
        print(f"Procesando {nombre_archivo}...")

        df_crudo = pd.read_csv(ruta)
        X_paciente, y_paciente = procesar_registro_paciente(df_crudo, id_paciente, frecuencia_hz)

        if not X_paciente.empty:
            X_lista.append(X_paciente)
            y_lista.append(pd.DataFrame({'Sleep_Stage': y_paciente}))

    if not X_lista:
        raise RuntimeError("[ERROR] No se encontraron datos procesables para el análisis.")

    X_final = pd.concat(X_lista, ignore_index=True)
    y_final = pd.concat(y_lista, ignore_index=True)

    carpeta_guardado = os.path.join(ruta_base, "data", "processed")
    os.makedirs(carpeta_guardado, exist_ok=True)
    X_final.to_csv(os.path.join(carpeta_guardado, "X_features.csv"), index=False)
    y_final.to_csv(os.path.join(carpeta_guardado, "y_labels.csv"),   index=False)

    print(f"Características guardadas: {X_final.shape[0]} epochs × {X_final.shape[1]} columnas")
    print(f"Distribución de las fases del sueño:\n{y_final['Sleep_Stage'].value_counts()}")


# =============================================================================
# FASE 2: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# =============================================================================

# DESCRIPCIÓN: Ejecuta el ciclo completo de modelado estadístico para el número
#              de clases seleccionado. Carga los datos maestros, elimina variables
#              secundarias, entrena los tres algoritmos en paralelo, calcula sus
#              métricas de rendimiento, genera las matrices de confusión y
#              almacena los modelos optimizados en el disco.
# PARÁMETROS:
#   - cfg (dict):        Diccionario global con los parámetros de configuración.
#   - ruta_base (str):   Dirección de la carpeta principal del proyecto en el disco.
#   - num_clases (int):  Cantidad de grupos de sueño a evaluar (2, 3 o 4 clases).
# RETORNO:
#   - Contiene los datos con los resultados finales del modelado:
#       1. X_train:          Características usadas para el entrenamiento.
#       2. X_test:           Características usadas para la validación.
#       3. y_test_str:       Etiquetas reales de prueba en formato de texto.
#       4. clases:           Lista de clases únicas identificadas por el programa.
#       5. clases_ordenadas: Lista con los nombres de las fases en orden estándar.
#       6. modelos_train:    Diccionario con los tres modelos ya entrenados.

def evaluar_modelos(cfg: dict,
                    ruta_base: str,
                    num_clases: int) -> tuple:
    print(f"\n[FASE 2] Modelado - {num_clases} clases")

    # Cargamos los archivos maestros generados en la Fase 1
    X_raw = pd.read_csv(os.path.join(ruta_base, "data", "processed", "X_features.csv"))
    y_raw = pd.read_csv(os.path.join(ruta_base, "data", "processed", "y_labels.csv"))
    y_raw = y_raw['Sleep_Stage'].values

    # Limpieza de datos: Eliminamos columnas de metadatos no usadas como features
    cols_meta = ['Obstructive_Apnea', 'Central_Apnea',
                 'Hypopnea', 'Multiple_Events']
    X_clean = X_raw.drop(columns=[c for c in cols_meta if c in X_raw.columns])

    # Preparación integral del conjunto de datos (División, imputación y escala)
    X_train, X_val, X_test, y_train_str, y_val_str, y_test_str = preprocesar_dataset(
        X_clean, y_raw, num_clases,
        test_size=cfg['evaluacion']['test_size'],
        val_size=cfg['evaluacion'].get('val_size', 0.10), 
        random_state=cfg['proyecto']['semilla']
    )

    # Codificación numérica de las etiquetas de texto a números enteros para los modelos de clasificación
    codificador = LabelEncoder()
    y_train = codificador.fit_transform(y_train_str)
    y_val   = codificador.transform(y_val_str) 
    y_test  = codificador.transform(y_test_str)
    clases  = codificador.classes_

    clases_ordenadas = obtener_clases_ordenadas(num_clases)

    # Inicialización de rutas de guardado y variables de apoyo para resultados y modelos
    algoritmos    = cfg['modelos']['algoritmos']
    resultados    = {}
    modelos_train = {}
    ruta_graficas    = os.path.join(ruta_base, "outputs", "plots")
    ruta_modelos  = os.path.join(ruta_base, "outputs", "models")

    # Entrenamiento y evaluación de cada modelo en paralelo
    for alg in algoritmos:
        print(f"  -> {alg.upper()}...")
        modelo = configurar_modelo(alg, num_clases)

        if alg == 'xgboost':            
            weights = compute_sample_weight('balanced', y_train)
            modelo.fit(X_train, y_train, sample_weight=weights)
        else:
            modelo.fit(X_train, y_train)

        predicciones  = modelo.predict(X_test)

        # Calculamos las métricas globales e individuales (por cada fase del sueño)
        metricas = calcular_metricas_completas(y_test, predicciones)
        exactitud_por_clase  = calcular_accuracy_individual(y_test, predicciones, clases_ordenadas)
        resultados[alg] = {**metricas, **{f'Acc_{k}': v for k, v in exactitud_por_clase.items()}}
        modelos_train[alg] = modelo

        # Guardar el modelo entrenado en disco para su uso posterior en la Fase 3 de explicabilidad
        guardar_modelo(modelo, alg, num_clases, ruta_modelos)
        # Generamos y guardamos la gráfica de la matriz de confusión correspondiente
        graficar_matriz_confusion(
            y_test, predicciones, clases_ordenadas, alg, num_clases, ruta_graficas
        )

    # Consolidación y publicación de resultados finales en consola y CSV
    df_res = construir_tabla_resultados(
        {alg: {k: v for k, v in resultados[alg].items()
               if k in ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score']}
         for alg in algoritmos},
        num_clases
    )

    # Visualización del resumen de rendimiento en consola
    print("\n" + "=" * 60)
    print(df_res.to_string())
    print("=" * 60 + "\n")

    # Guardamos los resultados completos en un archivo CSV para su análisis posterior
    exportar_resultados_csv(df_res, num_clases, os.path.join(ruta_base, "outputs"))

    return X_train, X_val, X_test, y_test_str, clases, clases_ordenadas, modelos_train


# =============================================================================
# FASE 3: EXPLICABILIDAD (LIME LOCAL + GLOBAL)
# =============================================================================

# DESCRIPCIÓN: Coordina la fase de Explicabilidad de los modelos.
#              Explica el comportamiento del modelo seleccionado mediante dos vías:
#              un estudio detallado de un momento de sueño específico y un mapa 
#              general con la importancia de cada variable fisiológica.
# PARÁMETROS:
#   - cfg (dict):            Diccionario global con los parámetros de configuración.
#   - ruta_base (str):       Dirección de la carpeta principal del proyecto en el disco.
#   - modelos_train (dict):  Diccionario que contiene los modelos ya entrenados.
#   - X_train (pd.DataFrame):Características utilizadas para el entrenamiento.
#   - X_test (pd.DataFrame): Características utilizadas para la validación.
#   - y_test_str (np.ndarray):Etiquetas reales de las fases de sueño en texto.
#   - clases:                Lista de las fases de sueño identificadas.
#   - num_clases (int):      Cantidad de grupos de sueño evaluados (2, 3 o 4 clases).
# RETORNO:
#   - None: No devuelve elementos en memoria, genera y guarda los gráficos directamente.

def generar_explicabilidad(cfg: dict,
                           ruta_base: str,
                           modelos_train: dict,
                           X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           y_test_str: np.ndarray,
                           clases,
                           num_clases: int) -> None:

    print(f"\n[FASE 3] Explicabilidad LIME - {num_clases} clases")

    # Recuperamos los parámetros específicos para el análisis de explicabilidad desde el YAML
    algoritmo_elegido    = cfg['xai']['algoritmo']
    epoch_local  = cfg['xai']['epoch_local']
    n_muestras = cfg['xai']['n_lime_samples']
    n_features = cfg['xai']['n_lime_features']
    ruta_graficas = os.path.join(ruta_base, "outputs", "plots")

    modelo_xai = modelos_train[algoritmo_elegido]

    # Análisis de Explicabilidad Local con LIME.
    # Evalúa una única época de 30 segundos específica para analizar el comportamiento
    # detallado del modelo ante un evento concreto de un paciente.
    generar_analisis_lime(
        modelo=modelo_xai,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test_str,
        clases=clases,
        indice_epoch=epoch_local,
        ruta_guardada=ruta_graficas,
        n_features=n_features
    )

    # Análisis de Explicabilidad Global.
    # Promedia el impacto de las variables en una muestra amplia de datos para extraer 
    # conclusiones generales sobre qué métricas fisiológicas son las más relevantes.
    generar_lime_global(
        modelo=modelo_xai,
        X_train=X_train,
        X_test=X_test,
        clases=clases,
        ruta_guardada=ruta_graficas,
        n_muestras=n_muestras,
        n_features=n_features,
        num_clases=num_clases
    )

    # Análisis de Explicabilidad Global con SHAP.
    # Genera el gráfico de importancia global. Aporta validación cruzada y rigor
    # matemático exacto al análisis previo de LIME.
    generar_shap_global(
    modelo=modelo_xai,
    X_train=X_train,
    X_test=X_test,
    clases=clases,
    num_clases=num_clases,
    ruta_guardada=ruta_graficas,
    n_features=n_features
)


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL DEL PROGRAMA
# =============================================================================

if __name__ == "__main__":
    # Configuración del lector de comandos por terminal.
    lector_argumentos = argparse.ArgumentParser(
        description="Flujo de trabajo para la clasificación de fases del sueño"
    )
    lector_argumentos.add_argument(
        '--clases', type=int, default=None,
        help="Número de clases (2, 3 ó 4). Si se omite, usa el valor del archivo de configuración"
    )
    lector_argumentos.add_argument(
        '--skip-extraccion', action='store_true',
        help="Evita repetir la Fase 1 si los archivos procesados ya existen en el disco."
    )
    argumentos = lector_argumentos.parse_args()
    # Definimos las rutas principales del proyecto y cargamos la configuración global.
    RUTA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
    RUTA_CONFIG   = os.path.join(RUTA_PROYECTO, "configs", "config.yaml")

    cfg = cargar_config(RUTA_CONFIG)

    # Determinamos el número de clases a evaluar, priorizando el argumento de línea de comandos.
    num_clases = argumentos.clases or cfg['experimentos'][0]['num_clases']

    # EJECUCIÓN - FASE 1: Extracción de características fisiológicas
    ruta_datos_procesados = os.path.join(RUTA_PROYECTO, "data", "processed", "X_features.csv")
    if argumentos.skip_extraccion and os.path.exists(ruta_datos_procesados):
        print("[FASE 1] Omitida - Los datos procesados ya existen en el sistema.")
    else:
        extraer_caracteristicas_dataset(cfg, RUTA_PROYECTO)

    # EJECUCIÓN - FASE 2: Entrenamiento y evaluación de los modelos de IA
    X_train, X_val, X_test, y_test_str, clases, clases_ord, modelos = evaluar_modelos(
        cfg, RUTA_PROYECTO, num_clases
    )

    # EJECUCIÓN - FASE 3: Explicabilidad de los modelos mediante LIME (local y global)
    generar_explicabilidad(
        cfg, RUTA_PROYECTO, modelos,
        X_train, X_test, y_test_str, clases, num_clases
    )

    print("\n[FINALIZADO] El flujo de trabajo se ha completado con éxito.")
