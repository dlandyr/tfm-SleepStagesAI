# =============================================================================
# Ejecuta los 3 casos de estudio en secuencia (2, 3 y 4 clases)
# y consolida todos los resultados en una única tabla comparativa.
#
# Comandos de ejecución:
#   python run_pipeline.py                   # Ejecución completa (desde cero)
#   python run_pipeline.py --skip-extraccion # Ejecución rápida (datos ya procesados)
# =============================================================================

import os
import time
import argparse
import pandas as pd
import numpy as np
import yaml
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.preprocessing import LabelEncoder

from src.pipeline import procesar_registro_paciente
from src.utils import preprocesar_dataset, obtener_clases_ordenadas
from src.models import configurar_modelo, guardar_modelo, cargar_modelo
from src.evaluation import (
    calcular_metricas_completas,
    calcular_accuracy_individual,
    exportar_resultados_csv,
    graficar_matriz_confusion,
    graficar_ciclo_sueno,
    graficar_accuracy_categorias,
)
from src.explainers import generar_lime_global, generar_shap_global, generar_permutation_importance, generar_pdp
from main import cargar_config, extraer_caracteristicas_dataset


CATEGORIAS_FEATURES = {
    'Waveform morphology':         ['Mean_A', 'SD_A', 'Mean_A1', 'SD_A1', 'Mean_A2',
                                    'SD_A2', 'Mean_T1', 'SD_T1', 'Mean_T2', 'SD_T2',
                                    'Mean_IPAR', 'SD_IPAR', 'Mean_IPTR', 'SD_IPTR'],
    'Rhythm & respiratory':        ['PPI_mean', 'PPI_sd', 'PPI_sdsd', 'PPI_rmssd',
                                    'RIAM_mean', 'RIAM_sd', 'RIAM_sdsd', 'RIAM_rmssd',
                                    'RIFM_mean', 'RIFM_sd', 'RIFM_sdsd', 'RIFM_rmssd'],
    'Statistical & energy':        ['avgADPPG', 'stdADPPG', 'MADPPG', 'IQRPPG', 'nCMPPG',
                                    'avgEPPG', 'SFPPG', 'avgCLPPG', 'avgTEPPG',
                                    'GmPPG', 'HmPPG', 'TM25PPG', 'TM50PPG',
                                    'SkewPPG', 'KurtPPG'],
    'Heart rate variability':      ['SD1PPG', 'SD2PPG', 'RSD1SD2PPG', 'CCMPPG'],
    'Complexity & fractal':        ['HAPPG', 'HMPPG', 'HCPPG', 'HFDPPG', 'KFDPPG'],
}

# DESCRIPCIÓN: Coordina y ejecuta de forma automática el análisis comparativo 
#              global. Evalúa las tres configuraciones de clasificación (2, 3 
#              y 4 clases), entrena todos los modelos en cada caso, calcula 
#              el rendimiento por categorías de variables y exporta todas las 
#              tablas estadísticas y gráficos.
# PARÁMETROS:
#   - cfg (dict):      Diccionario global con los parámetros de configuración.
#   - ruta_base (str): Dirección de la carpeta principal del proyecto en el disco.
# RETORNO:
#   - None: No devuelve elementos a la memoria, almacena todo en archivos locales.

def ejecutar_analisis_completo(cfg: dict, ruta_base: str) -> None:
    # Carga de datos procesados (features y etiquetas) 
    X_raw = pd.read_csv(os.path.join(ruta_base, "data", "processed", "X_features.csv"))
    y_raw = pd.read_csv(os.path.join(ruta_base, "data", "processed", "y_labels.csv"))
    y_raw = y_raw['Sleep_Stage'].values

    # Elimina columnas de metadata si están presentes (id_paciente y eventos respiratorios)
    cols_meta = ['id_paciente', 'Obstructive_Apnea', 'Central_Apnea',
                 'Hypopnea', 'Multiple_Events']
    X_clean = X_raw.drop(columns=[c for c in cols_meta if c in X_raw.columns])

    # Definición de las carpetas de salida para los reportes y recursos gráficos
    ruta_graficas  = os.path.join(ruta_base, "outputs", "plots")
    ruta_modelos = os.path.join(ruta_base, "outputs", "models")
    ruta_salida    = os.path.join(ruta_base, "outputs")

    # Estructuras para acumular los resultados finales consolidados
    tabla_global_resultados = {}
    accuracy_por_categoria = {cat: {} for cat in CATEGORIAS_FEATURES}

    # Evalúa de forma consecutiva cada caso de estudio (2, 3 y 4 clases)
    for exp in cfg['experimentos']:
        num_clases = exp['num_clases']
        desc       = exp['descripcion']

        print(f"\n{'=' * 65}")
        print(f"CASO DE ESTUDIO: {num_clases} CLASES - {desc}")
        print(f"{'=' * 65}")

        # Preparación de los datos para el caso actual
        X_train, X_val, X_test, y_train_str, y_val_str, y_test_str = preprocesar_dataset(
            X_clean, y_raw, num_clases,
            test_size=cfg['evaluacion']['test_size'],
            val_size=cfg['evaluacion'].get('val_size', 0.10),
            random_state=cfg['proyecto']['semilla']
        )

        codificador = LabelEncoder()
        y_train = codificador.fit_transform(y_train_str)
        y_test  = codificador.transform(y_test_str)
        clases          = codificador.classes_
        clases_ordenadas = obtener_clases_ordenadas(num_clases)

        modelos_configuracion = {}

        y_test_str_local = codificador.inverse_transform(y_test)

        for alg in cfg['modelos']['algoritmos']:
            print(f"\n  [{alg.upper()}]")

            # SKIP si el modelo ya existe
            ruta_pkl = os.path.join(ruta_modelos, f"{alg}_{num_clases}_clases.pkl")
            if os.path.exists(ruta_pkl):
                print(f"    El Modelo ya existe, cargando desde disco...")
                modelo = cargar_modelo(alg, num_clases, ruta_modelos)
                modelos_configuracion[alg] = modelo
                continue

            modelo = configurar_modelo(alg, num_clases)
            modelo.fit(X_train, y_train)
            predicciones  = modelo.predict(X_test)

            predicciones_str = codificador.inverse_transform(predicciones)

            metricas = calcular_metricas_completas(y_test_str_local, predicciones_str, clases_ordenadas)
            acc_cls  = calcular_accuracy_individual(y_test_str_local, predicciones_str, clases_ordenadas)

            # Guardar en tabla global
            clave = f"{alg.upper()}_{num_clases}cls"
            tabla_global_resultados[clave] = {
                **{k: f"{v*100:.2f}%" for k, v in metricas.items()},
                **{f'Acc_{k}': f"{v*100:.2f}%" for k, v in acc_cls.items()},
            }

            print(f"Accuracy: {metricas['Accuracy']*100:.2f}% | "f"F1-Score: {metricas['F1-Score']*100:.2f}%")

            guardar_modelo(modelo, alg, num_clases, ruta_modelos)
            graficar_matriz_confusion(
                y_test_str_local, 
                predicciones_str, 
                clases_ordenadas, 
                alg, 
                num_clases, 
                ruta_graficas
            )

            generar_permutation_importance(
                modelo=modelo,
                X_test=X_test,
                y_test=y_test,
                num_clases=num_clases,
                algoritmo=alg,
                ruta_plots=ruta_graficas,
                n_repeats=30,
                n_features=15
            )

            modelos_configuracion[alg] = modelo
            print(f"  ⏸ Enfriamiento 45s...")
            time.sleep(45)

        # Análisis: accuracy por categoría de features (solo XGBoost)
        modelo_xgb = modelos_configuracion.get('xgboost')
        if modelo_xgb is not None:
            for cat, cols_cat in CATEGORIAS_FEATURES.items():
                cols_disp = [c for c in cols_cat if c in X_train.columns]
                if not cols_disp:
                    continue
                modelo_cat = configurar_modelo('xgboost', num_clases)
                modelo_cat.fit(X_train[cols_disp], y_train)
                predicciones_cat = modelo_cat.predict(X_test[cols_disp])
                predicciones_cat_str = codificador.inverse_transform(predicciones_cat)

                accuracy = calcular_metricas_completas(y_test_str_local, predicciones_cat_str, clases_ordenadas)['Accuracy']
                accuracy_por_categoria[cat][num_clases] = accuracy

            # Generar mapas de interpretabilidad globales con LIME (solo para 4 clases)
            if num_clases == cfg['xai'].get('n_lime_samples', 4) or num_clases == 4:
                print(f"\n  [EXPLICABILIDAD GLOBAL LIME] - {num_clases} clases...")
                generar_lime_global(
                    modelo=modelo_xgb,
                    X_train=X_train,
                    X_test=X_test,
                    clases=clases,
                    ruta_guardada=ruta_graficas,
                    n_muestras=cfg['xai']['n_lime_samples'],
                    n_features=cfg['xai']['n_lime_features'],
                    num_clases=num_clases
                )
            print(f"\n  [EXPLICABILIDAD GLOBAL SHAP] - {num_clases} clases...")
            generar_shap_global(
                    modelo=modelo_xgb,
                    X_test=X_test,
                    clases=clases,
                    num_clases=num_clases,
                    ruta_guardada=ruta_graficas,
                    n_features=cfg['xai']['n_shap_features']
                )
            
            print(f"\n  [PDP] - {num_clases} clases...")
            generar_pdp(
                modelo=modelo_xgb,
                X_train=X_train,
                clases_ordenadas=clases_ordenadas,
                num_clases=num_clases,
                algoritmo='xgboost',
                ruta_plots=ruta_graficas,
                n_features=6
            )

            # Hipnograma de control (para contrastar visualmente el ciclo real vs. el predicho)
            pred_str = codificador.inverse_transform(modelo_xgb.predict(X_test))
            graficar_ciclo_sueno(
                y_real=y_test_str,
                y_pred=pred_str,
                clases_ordenadas=clases_ordenadas,
                num_clases=num_clases,
                ruta_plots=ruta_graficas,
                id_paciente="test_sample"
            )
        # Pausa de enfriamiento entre escenarios
        print(f"\n  ⏸ Pausa de enfriamiento entre escenarios 2 min...")
        time.sleep(120)

    # Exportar tabla global de resultados
    df_global = pd.DataFrame(tabla_global_resultados).T
    df_global.to_csv(os.path.join(ruta_salida, "resultados_completos.csv"))
    print(f"\n Tabla global exportada: outputs/resultados_completos.csv")

    # Dibujar accuracy por categoría
    graficar_accuracy_categorias(accuracy_por_categoria, ruta_graficas)

    # Imprimir resumen final de accuracies XGBoost por número de clases
    print("\n" + "=" * 65)
    print("RESUMEN FINAL - Accuracy XGBoost")
    print("=" * 65)
    for exp in cfg['experimentos']:
        nc = exp['num_clases']
        clave = f"XGBOOST_{nc}cls"
        if clave in tabla_global_resultados:
            print(f"  {nc} clases: {tabla_global_resultados[clave]['Accuracy']}")
    print("=" * 65)

# =============================================================================
# BLOQUE DE ARRANQUE POR CONSOLA
# =============================================================================

# DESCRIPCIÓN: Punto de entrada principal para ejecutar el pipeline completo desde la consola.
if __name__ == "__main__":
    lector_argumentos = argparse.ArgumentParser(
        description="Ejecuta de forma secuencial los casos de prueba."
    )
    lector_argumentos.add_argument(
        '--skip-extraccion', action='store_true',
        help="Omite la extracción de variables si los archivos ya existen en el disco."
    )
    args = lector_argumentos.parse_args()

    # Trazado dinámico de las rutas locales del sistema
    RUTA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
    RUTA_CONFIG   = os.path.join(RUTA_PROYECTO, "configs", "config.yaml")
    cfg = cargar_config(RUTA_CONFIG)

    # Verificar si los datos ya han sido procesados para evitar repetir la extracción de features
    ruta_datos_procesados = os.path.join(RUTA_PROYECTO, "data", "processed", "X_features.csv")
    if args.skip_extraccion and os.path.exists(ruta_datos_procesados):
        print("[FASE 1] Omitida - Los datos procesados ya existen en el sistema.")
    else:
        extraer_caracteristicas_dataset(cfg, RUTA_PROYECTO)

    ejecutar_analisis_completo(cfg, RUTA_PROYECTO)

    print("\n[FINALIZADO] Todos los análisis se han completado con éxito.")
