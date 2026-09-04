# =============================================================================
# Análisis de explicabilidad con LIME:
#   - LIME local:  explica la predicción de un epoch específico.
#   - LIME global: promedia importancias sobre una muestra del test set,
#                  en base a la literatura científica (mean abs weight top-15).
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        module='sklearn.utils.parallel')

# ===============================================================================
# LIME LOCAL - Explica la predicción del modelo para una única época de registro
# ===============================================================================


# DESCRIPCIÓN: Genera una explicación agnóstica del modelo basada en LIME para 
#              una época de registro específica. Permite identificar de forma 
#              visual qué características fisiológicas pesaron más en la decisión 
#              del algoritmo.
# PARÁMETROS:
#   - modelo:                     Algoritmo entrenado con capacidad de estimación.
#   - X_train (pd.DataFrame):     Matriz de características de entrenamiento.
#   - X_test (pd.DataFrame):      Matriz de características de prueba.
#   - y_test (np.ndarray):        Etiquetas reales de control en la evaluación (strings).
#   - clases:                     Estructura (array/lista) con los nombres de las fases.
#   - indice_epoca (int):         Índice de la fila en X_test que se desea interpretar.
#   - ruta_guardada (str):        Directorio de destino para salvar la gráfica final.
#   - n_caracteristicas (int):    Cantidad de variables más influyentes a visualizar (por defecto 15).
# RETORNO:
#   - None: Renderiza el gráfico de importancia local y lo escribe en disco.

def generar_analisis_lime(modelo,
                          X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_test: np.ndarray,
                          clases,
                          indice_epoch: int,
                          ruta_guardada: str,
                          n_features: int = 15) -> None:
   
    os.makedirs(ruta_guardada, exist_ok=True)

    clases_lista = clases.tolist() if hasattr(clases, 'tolist') else list(clases)
    num_clases   = len(clases_lista)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=clases_lista,
        mode='classification',
        random_state=42
    )

    instancia   = X_test.iloc[indice_epoch].values
    etiq_real   = y_test[indice_epoch]

    print(f"[LIME local] Epoch {indice_epoch} | Clase real: {etiq_real}")

    exp = explainer.explain_instance(
        data_row=instancia,
        predict_fn=modelo.predict_proba,
        num_features=n_features
    )

    fig = exp.as_pyplot_figure()
    plt.title(
        f"LIME local - Epoch {indice_epoch} | Clase real: {etiq_real}",
        fontsize=10, pad=8
    )
    plt.tight_layout()

    nombre = f"lime_local_epoch_{indice_epoch}_{num_clases}_clases.png"
    fig.savefig(os.path.join(ruta_guardada, nombre), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Figura LIME local guardada: {nombre}")


# =============================================================================
# LIME GLOBAL - Calcula la importancia media de las características mediante el
#               promedio de explicaciones locales en el conjunto de test.
# =============================================================================


# DESCRIPCIÓN: Calcula el impacto general de las variables mediante la agregación 
#              y el promedio del valor absoluto de los pesos locales de LIME 
#              extraídos de un subconjunto aleatorio de prueba. Genera un reporte
#              de datos en formato CSV y delega la construcción del gráfico de 
#              barras horizontales (top-15).
# PARÁMETROS:
#   - modelo:                     Algoritmo entrenado con capacidad de estimación.
#   - X_train (pd.DataFrame):     Matriz de características de entrenamiento (calibración LIME).
#   - X_test (pd.DataFrame):      Matriz de características de prueba para extraer la muestra.
#   - clases:                     Nombres de las fases del sueño analizadas.
#   - ruta_guardada (str):        Directorio de destino para el CSV y el gráfico resultante.
#   - n_muestras (int):           Cantidad de instancias a evaluar para el promedio (defecto 200).
#   - n_caracteristicas (int):    Cantidad de variables principales a graficar (defecto 15).
#   - num_clases (int):           Cantidad de clases configuradas en la evaluación (defecto 4).
# RETORNO:
#   - pd.DataFrame: Contiene el listado de características ordenadas según su peso absoluto medio.

def generar_lime_global(modelo,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        clases,
                        ruta_guardada: str,
                        n_muestras: int = 200,
                        n_features: int = 15,
                        num_clases: int = 4) -> pd.DataFrame:

    os.makedirs(ruta_guardada, exist_ok=True)

    clases_lista = clases.tolist() if hasattr(clases, 'tolist') else list(clases)
    feature_names = X_train.columns.tolist()

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=clases_lista,
        mode='classification',
        random_state=42
    )

    # Muestreo aleatorio reproducible del set de prueba
    rng = np.random.default_rng(42)
    n_disponibles = len(X_test)
    n_usar = min(n_muestras, n_disponibles)
    indices = rng.choice(n_disponibles, size=n_usar, replace=False)

    print(f"[LIME global] Procesando {n_usar} registros del conjunto de prueba...")

    # Acumulador: suma de |peso| por feature
    pesos_acum = {f: 0.0 for f in feature_names}

    for idx, i in enumerate(indices):
        if idx % 50 == 0:
            print(f"      Instancia {idx + 1}/{n_usar}...")
        try:
            exp = explainer.explain_instance(
                data_row=X_test.iloc[i].values,
                predict_fn=modelo.predict_proba,
                num_features=len(feature_names),   # Todas las features
                num_samples=500                    # Muestras por explicación
            )
            for feat_name, peso in exp.as_list():
                # LIME puede añadir condiciones al nombre (p.ej. "TM25PPG > 0.5")
                # Extraemos solo el nombre base de la feature
                nombre_base = extraer_nombre_feature(feat_name, feature_names)
                if nombre_base in pesos_acum:
                    pesos_acum[nombre_base] += abs(peso)
        except Exception as e:
            print(f"[AVISO] Instancia {i} omitida: {e}")
            continue

    # Calculamos la media y ordenamos por importancia
    df_importancia = pd.DataFrame([
        {'feature': f, 'mean_abs_weight': pesos_acum[f] / n_usar}
        for f in feature_names
    ]).sort_values('mean_abs_weight', ascending=False).reset_index(drop=True)

    # Guardar CSV de importancias completas
    df_importancia.to_csv(
        os.path.join(ruta_guardada, f'lime_global_importancias_{num_clases}_clases.csv'),
        index=False
    )

    # Gráfico top-N
    graficar_lime_global(df_importancia, n_features, num_clases, ruta_guardada)

    return df_importancia


# DESCRIPCIÓN: Limpia y normaliza el identificador de texto generado por LIME. 
#              Dado que LIME añade operadores lógicos y umbrales numéricos a las 
#              variables (ej. 'Frecuencia_Cardiaca <= 72.5'), este método extrae 
#              exclusivamente el nombre base original para poder mapear y acumular 
#              correctamente los pesos en el DataFrame global.
# PARÁMETROS:
#   - texto_lime (str):         Cadena de texto con la condición generada por LIME.
#   - nombres_variables (list): Lista con los nombres originales de las características.
# RETORNO:
#   - str: Nombre base limpio de la característica (ej. 'Frecuencia_Cardiaca').

def extraer_nombre_feature(texto_lime: str, feature_names: list[str]) -> str:

    for nombre in feature_names:
        if texto_lime.startswith(nombre):
            return nombre
    # Fallback: tomar los caracteres antes del primer espacio o comparador
    for sep in [' ', '<', '>', '=']:
        if sep in texto_lime:
            return texto_lime.split(sep)[0].strip()
    return texto_lime.strip()


# DESCRIPCIÓN: Renderiza y salva un gráfico de barras horizontales que representa 
#              el ranking de las variables más influyentes (top-N) según su peso 
#              absoluto medio.
# PARÁMETROS:
#   - df_importancia (pd.DataFrame): DataFrame con el ranking ordenado de variables.
#   - n_caracteristicas (int):       Cantidad de variables principales a mostrar.
#   - num_clases (int):              Cantidad de clases configuradas en la evaluación.
#   - ruta_guardada (str):           Directorio de destino para salvar la imagen.
# RETORNO:
#   - None: Genera el gráfico y escribe físicamente la imagen en disco.

def graficar_lime_global(df_importancia: pd.DataFrame,
                          n_features: int,
                          num_clases: int,
                          ruta_guardada: str) -> None:

    top_df = df_importancia.head(n_features).iloc[::-1]   # Invertimos para que el top quede arriba

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top_df['feature'],
        top_df['mean_abs_weight'],
        color='steelblue',
        edgecolor='white',
        height=0.7
    )
    ax.set_xlabel('Importancia media LIME', fontsize=11)
    ax.set_title(
        f'LIME global - Top {n_features} features ({num_clases} clases, XGBoost)',
        fontsize=11, pad=10
    )
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    nombre = f"lime_global_{num_clases}_clases.png"
    fig.savefig(os.path.join(ruta_guardada, nombre), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figura LIME global guardada: {nombre}")

# DESCRIPCIÓN: Calcula el impacto global de LIME para un paciente específico. Promedia el 
#              peso absoluto de las características a lo largo de un subconjunto aleatorio 
#              de epochs del conjunto de prueba del paciente.               
# PARÁMETROS:
#   - datos (dict):     Diccionario de configuración que encapsula el modelo entrenado, 
#                       los conjuntos de datos (X_train, X_test) y la lista de clases.
#   - n_muestras (int): Cantidad máxima de epochs a evaluar de forma aleatoria para 
#                       construir la aproximación global (por defecto 50).
# RETORNO:
#   - pd.DataFrame:     DataFrame estructurado con las columnas 'feature' e 'importancia', 
#                       ordenado de forma descendente según el peso predictivo medio.

def generar_lime_global_paciente(datos: dict,
                                    n_muestras: int = 50) -> pd.DataFrame:
    modelo  = datos['modelo']
    X_train = datos['X_train']
    X_test  = datos['X_test']
    clases  = datos['clases']

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=clases.tolist(),
        mode='classification',
        random_state=42
    )

    # Seleccionamos epochs representativos del paciente
    n_muestras   = min(n_muestras, len(X_test))
    indices      = np.random.choice(len(X_test), n_muestras, replace=False)
    importancias = np.zeros(len(X_train.columns))

    for idx in indices:
        exp         = explainer.explain_instance(
            data_row=X_test.iloc[idx].values,
            predict_fn=modelo.predict_proba,
            num_features=len(X_train.columns)
        )
        labels_disp = list(exp.local_exp.keys())
        for label in labels_disp:
            for feat_idx, peso in exp.local_exp[label]:
                importancias[feat_idx] += abs(peso)

    # Normalizamos por número de muestras y clases
    importancias /= (n_muestras * len(clases))

    return pd.DataFrame({
        'feature':    X_train.columns,
        'importancia': importancias
    }).sort_values('importancia', ascending=False).reset_index(drop=True)

# DESCRIPCIÓN:
#   Genera y guarda un gráfico de barras global basado en los "Valores Shapley" 
#   para los modelos de árboles (como XGBoost o Random Forest). El gráfico muestra 
#   cuáles son las variables biomédicas que más peso e impacto tienen a la hora 
#   de clasificar y diferenciar las distintas fases del sueño.
#
# PARÁMETROS:
#   - modelo:                Contiene el modelo ya entrenado (ej. XGBoost).
#   - X_test:                Conjunto de datos de prueba con el que se calculará la explicabilidad.
#   - clases:                Lista o arreglo con los nombres reales de las fases del sueño (ej. W, N1, N2...).
#   - num_clases:            Número entero que indica el escenario actual (2, 3 o 4 clases).
#   - ruta_guardada: Dirección de la carpeta del disco duro donde se almacenará la imagen.
#   - n_features (Opcional): Cantidad máxima de variables que se pintarán en el gráfico 
#                            (por defecto se muestran las 15 más importantes).
#   - X_train (Opcional):    Conjunto de datos de entrenamiento para calibrar el explainer de SHAP
#
# RETORNO:
#   - None: El método no devuelve ninguna variable y su función es exportar el 
#           gráfico directamente como un archivo de imagen (.png) en el disco duro.

def generar_shap_global(modelo, X_test, clases,
                        num_clases, ruta_guardada, 
                        n_features=15, X_train=None):
    os.makedirs(ruta_guardada, exist_ok=True)
    # Convertir el formato de las clases (Wake, Sleep, etc.) a una lista estándar de Python
    clases_lista = clases.tolist() if hasattr(clases, 'tolist') else list(clases)

    print(f"[SHAP] Calculando valores para {len(X_test)} epochs...")

    # Explainer optimizado para modelos basados en árboles (Random Forest y XGBoost)
    explainer   = shap.TreeExplainer(
        modelo,
        data=X_train if X_train is not None else None
    )
    shap_values = explainer.shap_values(X_test)

    # Gráfico de barras de importancia global
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=X_test.columns.tolist(),
        class_names=clases_lista,
        max_display=n_features,
        plot_type='bar',
        show=False
    )
    plt.title(f'SHAP - Importancia global ({num_clases} clases)', fontsize=11)
    plt.gca().set_xlabel('Importancia media del valor SHAP', fontsize=11)
    plt.tight_layout()
    nombre = f"shap_global_{num_clases}_clases.png"
    plt.savefig(os.path.join(ruta_guardada, nombre), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Figura SHAP guardada: {nombre}")


# DESCRIPCIÓN:
#   Calcula y grafica la importancia de cada variable aplicando el método de permutación.
#   El algoritmo rompe de forma aislada la información de una variable reorganizando sus 
#   filas al azar; si el acierto (Accuracy) del modelo cae significativamente tras esa permutación,
#   significa que esa variable era importante para el modelo.
#   Funciona de forma agnóstica para cualquier modelo (RF, KNN, XGBoost).
#
# PARÁMETROS:
#   - modelo:                Clasificador ya entrenado (RF, KNN, XGBoost o SVM).
#   - X_test:                DataFrame de características (features) del conjunto de prueba.
#   - y_test:                Etiquetas reales en formato numérico (0=Wake, 1=NREM...)
#   - num_clases:            Número entero que indica el escenario actual (2, 3 o 4 clases).
#   - algoritmo:             Cadena de texto con el nombre del modelo (ej. 'rf', 'xgboost') para el reporte.
#   - ruta_plots:            Dirección de la carpeta en el disco donde se exportará el gráfico y el CSV.
#   - n_repeats (Opcional):  Número de veces que se baraja cada columna para asegurar estabilidad estadística.
#   - n_features (Opcional): Cantidad de variables que se mostrarán en el ranking visual del gráfico.
#
# RETORNO:
#   - df_importancia:        DataFrame de Pandas ordenado de mayor a menor con las columnas: 
#                            ['feature', 'importancia_media', 'importancia_std'].

def generar_permutation_importance(modelo,
                                   X_test: pd.DataFrame,
                                   y_test: np.ndarray,
                                   num_clases: int,
                                   algoritmo: str,
                                   ruta_plots: str,
                                   n_repeats: int = 30,
                                   n_features: int = 15) -> pd.DataFrame:      

        os.makedirs(ruta_plots, exist_ok=True)

        print(f"   [Permutation Importance] {algoritmo.upper()} "
            f"— {num_clases} clases — {n_repeats} repeticiones...")

        resultado = permutation_importance(
            modelo,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring='accuracy',
            n_jobs=1
        )

        df_importancia = pd.DataFrame({
            'feature':           X_test.columns,
            'importancia_media': resultado.importances_mean,
            'importancia_std':   resultado.importances_std
        }).sort_values('importancia_media', ascending=False).reset_index(drop=True)

        nombre_csv = f"permutation_importance_{algoritmo}_{num_clases}_clases.csv"
        df_importancia.to_csv(os.path.join(ruta_plots, nombre_csv), index=False)
        print(f"CSV guardado: {nombre_csv}")

        graficar_permutation_importance(
            df_importancia, n_features, num_clases, algoritmo, ruta_plots
        )

        return df_importancia
    

# DESCRIPCIÓN:
#   Genera y guarda un gráfico de barras horizontal que representa el ranking de 
#   las variables más importantes según el método de permutación.  El gráfico
#   incluye barras de error (intervalos de confianza) para mostrar la estabilidad 
#   de cada variable a lo largo de las múltiples permutaciones realizadas, ordenando 
#   las señales de mayor a menor impacto visual.
#
# PARÁMETROS:
#   - df:                    DataFrame de Pandas que contiene el ranking calculado con las columnas 
#                            ['feature', 'importancia_media', 'importancia_std'].
#   - n_features:            Número entero que define cuántas variables del top se mostrarán en la gráfica.
#   - num_clases:            Número entero que indica el escenario actual del estudio (2, 3 o 4 clases).
#   - algoritmo:             Cadena de texto con el nombre del modelo (ej. 'rf', 'knn') para el título.
#   - ruta_plots:            Dirección de la carpeta en el disco duro donde se guardará la imagen final.
#
# RETORNO:
#   - None: El método no devuelve ninguna variable; su función es exportar directamente el gráfico
#           de barras horizontal con los resultados de la permutación y sus intervalos de confianza. 
#           El gráfico se guarda físicamente en el disco duro como un archivo de imagen (.png) de alta resolución.

def graficar_permutation_importance(df: pd.DataFrame,
                                     n_features: int,
                                     num_clases: int,
                                     algoritmo: str,
                                     ruta_plots: str) -> None:

        top_df = df.head(n_features).iloc[::-1]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            top_df['feature'],
            top_df['importancia_media'],
            xerr=top_df['importancia_std'],
            color='#534AB7',
            ecolor='#AFA9EC',
            edgecolor='white',
            height=0.7,
            capsize=3
        )
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Disminución media en accuracy', fontsize=11)
        ax.set_title(
            f'Permutation importance — {algoritmo.upper()} ({num_clases} clases)\n'
            f'Top {n_features} features más importantes',
            fontsize=11, pad=10
        )
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        nombre = f"permutation_importance_{algoritmo}_{num_clases}_clases.png"
        fig.savefig(os.path.join(ruta_plots, nombre), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Figura guardada: {nombre}")


# DESCRIPCIÓN:
#   Identifica automáticamente las variables más influyentes del modelo y orquesta 
#   la creación de gráficos de Dependencia Parcial (PDP) para cada fase del sueño. 
#   Los gráficos PDP permiten analizar el comportamiento "marginal" de las señales, 
#   mostrando si la probabilidad de clasificar una fase del sueño sube, baja o se 
#   mantiene estable a medida que aumenta el valor de una variable concreta.
#
# PARÁMETROS:
#   - modelo:                Clasificador entrenado (ej. Random Forest o XGBoost).
#   - X_train:               DataFrame de Pandas con las características usadas en el entrenamiento.
#   - clases_ordenadas:      Lista con los nombres de las fases del sueño (ej. ['Wake', 'Sleep']).
#   - num_clases:            Número entero que indica el escenario actual (2, 3 o 4 clases).
#   - algoritmo:             Cadena de texto con el nombre del modelo (ej. 'rf', 'xgboost') para los reportes.
#   - ruta_plots:            Dirección de la carpeta en el disco duro donde se almacenarán las imágenes.
#   - n_features (Opcional): Cantidad de variables principales a las que se les aplicará el análisis estadístico.
#
# RETORNO:
#   - None: El método no devuelve variables; coordina el flujo e invoca la función secundaria 
#           graficar_pdp() para exportar los archivos visuales directamente al disco duro.

def generar_pdp(modelo,
                X_train: pd.DataFrame,
                clases_ordenadas: list,
                num_clases: int,
                algoritmo: str,
                ruta_plots: str,
                n_features: int = 6) -> None:
    os.makedirs(ruta_plots, exist_ok=True)

    print(f"   [PDP] {algoritmo.upper()} — {num_clases} clases "
          f"— top {n_features} features...")

    # Paso 1: Calculamos el ranking de features más importantes sobre train
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning,
                                module='sklearn.utils.parallel')
        resultado = permutation_importance(
            modelo, X_train,
            modelo.predict(X_train),
            n_repeats=10,
            random_state=42,
            scoring='accuracy',
            n_jobs=-1
        )

    # Seleccionamos top-N features por importancia
    indices_top = np.argsort(resultado.importances_mean)[::-1][:n_features]
    features_seleccionadas = [X_train.columns[i] for i in indices_top]
    print(f"Features seleccionadas: {features_seleccionadas}")

    # Paso 2: Generamos un gráfico PDP por cada clase del sueño
    # Limitamos a 3 clases máximo para no generar demasiados archivos
    n_clases_plot = min(num_clases, 3)

    for clase_idx in range(n_clases_plot):
        nombre_clase = clases_ordenadas[clase_idx]
        graficar_pdp(
            modelo=modelo,
            X_train=X_train,
            lst_variables=features_seleccionadas,
            clase_idx=clase_idx,
            nombre_clase=nombre_clase,
            num_clases=num_clases,
            algoritmo=algoritmo,
            ruta_plots=ruta_plots
        )



# DESCRIPCIÓN:
#   Genera y guarda un Gráfico de Dependencia Parcial (PDP - Partial Dependence Plot) 
#   para una fase del sueño específica. El gráfico muestra de forma visual el impacto 
#   aislado y la tendencia (ascendente, descendente o lineal) que tienen las variables 
#   seleccionadas sobre la probabilidad matemática de que el modelo clasifique la fase
#   del sueño analizada a medida que sus valores cambian del mínimo al máximo.
#
# PARÁMETROS:
#   - modelo:                 Clasificador entrenado (ej. Random Forest o XGBoost).
#   - X_train:                DataFrame de Pandas con el conjunto de características de entrenamiento.
#   - features_seleccionadas: Lista de cadenas de texto con los nombres de las variables a graficar.
#   - clase_idx:              Índice numérico entero de la clase objetivo según el LabelEncoder (0, 1, 2...).
#   - nombre_clase:           Cadena de texto con el nombre legible de la fase del sueño (ej. 'Wake', 'N2').
#   - num_clases:             Número entero que indica el escenario del experimento (2, 3 o 4 clases).
#   - algoritmo:              Cadena de texto con el nombre del modelo (ej. 'rf', 'xgboost') para el reporte.
#   - ruta_plots:             Dirección de la carpeta en el disco donde se exportará la figura final.
#
# RETORNO:
#   - None: El método no devuelve ninguna variable; exporta el panel de gráficos directamente 
#           como una imagen (.png) de alta resolución en la ruta especificada.

def graficar_pdp(modelo,
                 X_train: pd.DataFrame,
                 lst_variables: list,
                 clase_idx: int,
                 nombre_clase: str,
                 num_clases: int,
                 algoritmo: str,
                 ruta_plots: str):      

    n = len(lst_variables)
    ncols = 3
    nrows = (n + ncols - 1) // ncols   # Calculamos las filas necesarias para acomodar n gráficos

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(14, 4 * nrows),
        constrained_layout=True
    )

    # Aplanamos el array de ejes para iterar fácilmente aunque haya solo una fila o columna
    axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]

    try:
        PartialDependenceDisplay.from_estimator(
            modelo,
            X_train,
            features=lst_variables,
            target=clase_idx,     # Clase que se analiza
            kind='average',       # Linea de tendencia promedio
            ax=axes_flat[:n]
        )
    except Exception as e:
        print(f"[AVISO] PDP {nombre_clase} omitida: {e}")
        plt.close(fig)
        return

    # Ocultar subplots vacíos si n no es múltiplo de ncols
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        f'PDP — {algoritmo.upper()} ({num_clases} clases)\n'
        f'Dependencia parcial — clase: {nombre_clase}',
        fontsize=12
    )

    nombre = f"pdp_{algoritmo}_{num_clases}_clases_{nombre_clase.lower()}.png"
    fig.savefig(os.path.join(ruta_plots, nombre), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Figura PDP guardada: {nombre}")