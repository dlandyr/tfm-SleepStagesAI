# =============================================================================
# Cálculo de métricas, generación de figuras y exportación de resultados.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Backend sin pantalla para entornos de servidor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix)

# ====================================================================================
# MÉTRICAS PRINCIPALES
# ====================================================================================

# DESCRIPCIÓN: Calcula la especificidad macro-promediada para clasificación multiclase.
# PARÁMETROS:
#   - y_true: Etiquetas reales.
#   - y_pred: Etiquetas predichas.
# RETORNO:
#   - Especificidad promediada sobre todas las clases (float).

def calcular_especificidad_promedio(y_true: np.ndarray, y_pred: np.ndarray, clases: list = None) -> float:    
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    n  = len(cm)
    especificidades = []

    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        especificidades.append(spec)

    return float(np.mean(especificidades))

# DESCRIPCIÓN: Calcula las métricas en base a la literatura científica:
#              accuracy, precision, sensitivity (recall), specificity y F1-score.
# PARÁMETROS:
#   - y_true: Etiquetas reales.
#   - y_pred: Valores estimados.
# RETORNO:
#   - Diccionario con las 5 métricas como floats entre 0 y 1.

def calcular_metricas_completas(y_true: np.ndarray,
                                y_pred: np.ndarray, 
                                clases: list = None) -> dict:
    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Precision':   precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Sensitivity': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Specificity': calcular_especificidad_promedio(y_true, y_pred, clases),
        'F1-Score':    f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

# DESCRIPCIÓN: Calcula la precisión individual de cada fase del sueño (la diagonal de la matriz de confusión normalizada),
#              replicando las columnas 'Acc(Wake)', 'Acc(Sleep)', etc.
# PARÁMETROS:
#   - y_true: Etiquetas reales.
#   - y_pred: Valores estimados.
#   - clases (list): Nombres de las fases del sueño (ej. ['Wake', 'Sleep']).
# RETORNO:
#   - dict: Estructura {nombre_fase: valor_accuracy}.

def calcular_accuracy_individual(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 clases: list[str]) -> dict:
    clases_labels = list(range(len(clases))) if np.issubdtype(
        np.array(y_true).dtype, np.integer
    ) else clases
    
    cm_norm = confusion_matrix(y_true, y_pred, 
                               labels=clases_labels, 
                               normalize='true')
    return {clase: float(cm_norm[i, i]) 
            for i, clase in enumerate(clases)}


# =============================================================================
# TABLAS DE RESULTADOS
# =============================================================================


# DESCRIPCIÓN: Construye un DataFrame estructurado con diferentes valores de
#              rendimiento obtenidos para las clasificaciones de 2, 3 y 
#              4 clases, similar a la tabla de la literatura científica.
# PARÁMETROS:
#   - resultados_por_alg (dict): Estructura anidada {algoritmo: {métrica: valor_ratio}}.
#   - num_clases (int):          Cantidad de clases evaluadas (2, 3 o 4).
# RETORNO:
#   - pd.DataFrame: Tabla con métricas en columnas y valores porcentuales de tipo string.

def construir_tabla_resultados(resultados_por_alg: dict,
                               num_clases: int) -> pd.DataFrame:
    filas = {}
    for alg, metricas in resultados_por_alg.items():
        filas[alg.upper()] = {
            k: f"{v * 100:.2f}%" for k, v in metricas.items()
        }
    df = pd.DataFrame(filas).T
    df.index.name = f'Algoritmo ({num_clases} clases)'
    return df


# DESCRIPCIÓN: Exporta la tabla estructurada de rendimiento a un archivo en 
#              formato CSV para su almacenamiento y análisis posterior.
# PARÁMETROS:
#   - df_resultados (pd.DataFrame): DataFrame que contiene las métricas formateadas.
#   - num_clases (int):             Cantidad de clases configuradas en la evaluación (2, 3 o 4).
#   - ruta_salida (str):            Directorio de destino donde se guardará el archivo.
# RETORNO:
#   - None: Realiza una escritura física en disco y muestra un aviso en consola.

def exportar_resultados_csv(df_resultados: pd.DataFrame,
                            num_clases: int,
                            ruta_salida: str) -> None:
    os.makedirs(ruta_salida, exist_ok=True)
    nombre = f"resultados_evaluacion_{num_clases}_clases.csv"
    df_resultados.to_csv(os.path.join(ruta_salida, nombre))
    print(f"Tabla de resultados guardada: {nombre}")


# =============================================================================
# FIGURAS: MATRIZ DE CONFUSIÓN
# =============================================================================

# DESCRIPCIÓN: Genera y guarda visualmente la matriz de confusión normalizada por
#              filas. Los valores internos se calculan y muestran de forma 
#              directa en formato de porcentaje numérico continuo (0% a 100%).
# PARÁMETROS:
#   - y_true (np.ndarray):  Etiquetas reales de control (enteros).
#   - y_pred (np.ndarray):  Valores estimados por el modelo (enteros).
#   - clases (list):        Nombres de las fases del sueño (ej. ['Wake', 'Sleep']).
#   - algoritmo (str):      Nombre del algoritmo evaluado.
#   - num_clases (int):     Cantidad de clases configuradas en la evaluación.
#   - ruta_plots (str):     Directorio de destino para la imagen final.
# RETORNO:
#   - None: Realiza el renderizado y la escritura física del gráfico en disco.

def graficar_matriz_confusion(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              clases: list[str],
                              algoritmo: str,
                              num_clases: int,
                              ruta_plots: str) -> None:
    
    os.makedirs(ruta_plots, exist_ok=True)

    cm_norm = confusion_matrix(y_true, y_pred, labels=clases, normalize='true') * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=clases,
        yticklabels=clases,
        ax=ax,
        cbar_kws={'label': 'Porcentaje (%)'}
    )
    ax.invert_yaxis()
    ax.set_xlabel('Predicción', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    ax.set_title(
        f'Matriz de confusión - {algoritmo.upper()} ({num_clases} clases)',
        fontsize=11, pad=10
    )
    plt.tight_layout()

    nombre = f"matriz_{algoritmo}_{num_clases}_clases.png"
    fig.savefig(os.path.join(ruta_plots, nombre), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Matriz de confusión guardada: {nombre}")


# =============================================================================
# FIGURA: ACCURACY POR CATEGORÍA DE FEATURES
# =============================================================================


# DESCRIPCIÓN: Genera un gráfico de barras múltiples que compara el rendimiento
#              (Accuracy) según los distintos conjuntos de características para 
#              evaluaciones de 2, 3 y 4 clases.
# PARÁMETROS:
#   - resultados_categoria (dict): Estructura anidada {categoria: {n_clases: ratio}}.
#   - ruta_plots (str):            Directorio de destino para el gráfico final.
# RETORNO:
#   - None: Renderiza y guarda físicamente la imagen del gráfico en el disco.

def graficar_accuracy_categorias(resultados_categoria: dict,
                                    ruta_plots: str) -> None:
    
    os.makedirs(ruta_plots, exist_ok=True)

    categorias = list(resultados_categoria.keys())
    x = np.arange(len(categorias))
    ancho = 0.25
    colores = ['#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (n_cls, color) in enumerate(zip([2, 3, 4], colores)):
        vals = [resultados_categoria[c].get(n_cls, 0) * 100 for c in categorias]
        ax.bar(x + idx * ancho, vals, ancho, label=f'{n_cls} clases', color=color)

    ax.set_xticks(x + ancho)
    ax.set_xticklabels(
        [c.replace(' ', '\n') for c in categorias],
        fontsize=9
    )
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy por categoría de features (XGBoost)', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    fig.savefig(
        os.path.join(ruta_plots, 'accuracy_por_categoria.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig)
    print("Gráfico de accuracy por categoría guardado.")


# =============================================================================
# FIGURA: CICLO DEL SUEÑO (HIPNOGRAMA)
# =============================================================================

# DESCRIPCIÓN: Genera y guarda un gráfico secuencial comparativo que superpone 
#              las fases reales de sueño frente a las estimaciones del modelo
#              a lo largo de la línea temporal.
# PARÁMETROS:
#   - y_real (np.ndarray):      Array con las etiquetas reales de control (strings).
#   - y_pred (np.ndarray):      Array con los valores estimados por el modelo (strings).
#   - clases_ordenadas (list):  Orden formal de las fases del sueño en el eje Y.
#   - num_clases (int):         Cantidad de clases configuradas en la evaluación.
#   - ruta_plots (str):         Directorio de destino para salvar el gráfico final.
#   - id_paciente (str):        Identificador único del paciente (por defecto: "ejemplo").
# RETORNO:
#   - None: Crea y escribe de forma física el archivo de imagen en el disco duro.

def graficar_ciclo_sueno(y_real: np.ndarray,
                        y_pred: np.ndarray,
                        clases_ordenadas: list[str],
                        num_clases: int,
                        ruta_plots: str,
                        id_paciente: str = "ejemplo") -> None:

    os.makedirs(ruta_plots, exist_ok=True)

    # Codificación numérica para el eje Y
    mapa = {clase: i for i, clase in enumerate(clases_ordenadas)}
    y_real_num = np.array([mapa.get(e, -1) for e in y_real])
    y_pred_num = np.array([mapa.get(e, -1) for e in y_pred])
    tiempo_h   = np.arange(len(y_real)) * 30 / 3600   # segundos → horas

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for ax, y_num, titulo in zip(
        axes,
        [y_real_num, y_pred_num],
        ['Ciclo del sueño real (PSG)', f'Ciclo del sueño estimado (XGBoost, {num_clases} clases)']
    ):
        ax.step(tiempo_h, y_num, where='post', color='steelblue', linewidth=1.0)
        ax.set_yticks(range(len(clases_ordenadas)))
        ax.set_yticklabels(clases_ordenadas, fontsize=9)
        ax.set_ylabel('Fase del sueño', fontsize=9)
        ax.set_title(titulo, fontsize=10)
        ax.grid(axis='x', alpha=0.3)

    axes[-1].set_xlabel('Tiempo (horas)', fontsize=10)
    plt.suptitle(f'Comparación ciclo del sueño - Paciente {id_paciente}', fontsize=11, y=1.01)
    plt.tight_layout()

    nombre = f"ciclo_sueno_{num_clases}_clases_{id_paciente}.png"
    fig.savefig(os.path.join(ruta_plots, nombre), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Ciclo del sueño guardado: {nombre}")
