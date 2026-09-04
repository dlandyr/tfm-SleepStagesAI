# =============================================================================
# Extracción de las 50 variables fisiológicas a partir de la señal BVP/PPG, 
# fundamentada en la literatura científica de referencia.
#
# Categorías:
#   1. Waveform Morphology      (14 features): mean y SD de A, A1, A2, T1, T2, IPAR, IPTR
#   2. Rhythm & Respiratory     (12 features): mean, SD, SDSD y RMSSD de PPI, RIAM, RIFM
#   3. Statistical & Energy     (15 features): avgAD, stdAD, MAD, IQR, nCM, avgE, SF,
#                                              avgCL, avgTE, Gm, Hm, TM25, TM50, Skew, Kurt
#   4. Heart Rate Variability   (4 features) : SD1, SD2, RSD1SD2, CCM
#   5. Complexity & Fractals    (5 features) : HA, HM, HC, HFD, KFD
#
# Total: 14 + 12 + 15 + 4 + 5 = 50.
# =============================================================================

import numpy as np
import pandas as pd
import antropy as ant
from scipy.signal import find_peaks
from scipy.stats import gmean, hmean, iqr, kurtosis, moment, skew, trim_mean


# =============================================================================
# BLOQUE 1: FEATURES DE MORFOLOGÍA DE ONDA (14 features)
# Mean y SD de: A (área total), A1 (sistólica), A2 (diastólica),
#               T1 (tiempo sistólico), T2 (tiempo diastólico), IPAR, IPTR
# =============================================================================

# DESCRIPCIÓN: Extrae características morfológicas y métricas hemodinámicas ciclo a ciclo a partir de la 
#              señal BVP (Blood Volume Pulse) e identificación de picos. Localiza el punto de inicio (onset), 
#              el fin del ciclo (offset) y la muesca dicrota (notch) mediante la segunda derivada, calculando 
#              áreas de volumen (A1, A2, A), tiempos de tránsito (T1, T2) y ratios de inflexión (IPAR, IPTR). 
#              Devuelve la media y desviación estándar de cada métrica para toda la ventana.
# PARÁMETROS:
#   - positive_peaks (np.ndarray): Índices de posición de los picos sistólicos (máximos locales).
#   - negative_peaks (np.ndarray): Índices de posición de los valles o inicios/fines del pulso (mínimos locales).
#   - filtered_bvp (np.ndarray): Arreglo 1D con la señal BVP filtrada.
#   - second_deriv (np.ndarray): Arreglo 1D con la segunda derivada de la señal BVP (para ubicar el notch).
#   - hertz_freq (int): Frecuencia de muestreo de la señal en Hz.
# RETORNO:
#   - dict | None: Diccionario con la media (Mean_) y desviación estándar (SD_) de cada parámetro morfológico 
#                  ('A', 'A1', 'A2', 'T1', 'T2', 'IPAR', 'IPTR'), o None si no se pudieron procesar pulsos válidos.

def extraer_morfologia(positive_peaks: np.ndarray,
                        negative_peaks: np.ndarray,
                        filtered_bvp: np.ndarray,
                        second_deriv: np.ndarray,
                        hertz_freq: int) -> dict:

    resultados = []

    for peak in positive_peaks:
        prev_neg = negative_peaks[negative_peaks < peak]
        next_neg = negative_peaks[negative_peaks > peak]
        if len(prev_neg) == 0 or len(next_neg) == 0:
            continue

        onset_idx = prev_neg[-1]
        offset_idx = next_neg[0]
        baseline = filtered_bvp[onset_idx]

        # Nodo dicrótico: punto de máxima aceleración entre pico y offset
        seg_deriv = second_deriv[peak:offset_idx]
        if len(seg_deriv) == 0:
            continue
        notch_idx = peak + int(np.argmax(seg_deriv))

        # Áreas (integral discreta centrada en baseline)
        A1 = float(np.sum(filtered_bvp[onset_idx:notch_idx] - baseline))
        A2 = float(np.sum(filtered_bvp[notch_idx:offset_idx] - baseline))
        A  = A1 + A2

        # Tiempos fisiológicos (segundos)
        T1 = (notch_idx - onset_idx) / hertz_freq
        T2 = (offset_idx - notch_idx) / hertz_freq

        # Ratios de inflexión
        IPAR = A2 / A1 if A1 > 0 else 0.0
        IPTR = T2 / T1 if T1 > 0 else 0.0

        resultados.append({
            'A': A, 'A1': A1, 'A2': A2,
            'T1': T1, 'T2': T2,
            'IPAR': IPAR, 'IPTR': IPTR
        })

    if not resultados:
        return None

    df_lat = pd.DataFrame(resultados)
    feat = {}
    for col in ['A', 'A1', 'A2', 'T1', 'T2', 'IPAR', 'IPTR']:
        feat[f'Mean_{col}'] = float(df_lat[col].mean())
        feat[f'SD_{col}']   = float(df_lat[col].std(ddof=1) if len(df_lat) > 1 else 0.0)
    return feat


# =============================================================================
# BLOQUE 2: RHYTHM & RESPIRATORY MODULATION (12 features)
# PPI, RIAM, RIFM → mean, SD, SDSD (SD de diferencias sucesivas), RMSSD
# =============================================================================

# DESCRIPCIÓN: Deriva modulaciones de la señal de pulso volumétrico (BVP) inducidas por la respiración 
#              (EDR - Electrocardiogram/PPG Derived Respiration). Calcula las series temporales del intervalo 
#              pico a pico (PPI), la modulación de amplitud (RIAM) y la modulación de frecuencia (RIFM). 
#              Extrae estadísticos descriptivos y de variabilidad temporal ('mean', 'sd', 'sdsd', 'rmssd') 
#              para cada una de las series.
# PARÁMETROS:
#   - positive_peaks (np.ndarray): Índices de posición de los picos sistólicos en la señal.
#   - filtered_bvp (np.ndarray): Arreglo 1D con la señal BVP filtrada.
#   - hertz_freq (int): Frecuencia de muestreo de la señal en Hz.
# RETORNO:
#   - dict | None: Diccionario con los 12 indicadores estadísticos resultantes (PPI_*, RIAM_*, RIFM_*), 
#                  o None si la cantidad de picos es insuficiente (menor a 3).

def extraer_ritmo_respiratorio(positive_peaks: np.ndarray,
                                filtered_bvp: np.ndarray,
                                hertz_freq: int) -> dict:

    if len(positive_peaks) < 3:
        return None

    # PPI: intervalo entre picos consecutivos (s)
    ppi_series = np.diff(positive_peaks) / hertz_freq

    # RIAM: desviación relativa de amplitud pico a pico
    amplitudes = filtered_bvp[positive_peaks]
    amp_mean = np.mean(amplitudes)
    riam_series = np.abs(amplitudes[1:] - amplitudes[:-1]) / (amp_mean if amp_mean != 0 else 1.0)

    # RIFM: desviación relativa de frecuencia (1/PPI normalizada)
    freqs = 1.0 / (ppi_series + 1e-9)
    freq_mean = np.mean(freqs)
    rifm_series = np.abs(np.diff(freqs)) / (freq_mean if freq_mean != 0 else 1.0)

    # Alineamos las longitudes
    n = min(len(ppi_series), len(riam_series), len(rifm_series))
    ppi_s  = ppi_series[:n]
    riam_s = riam_series[:n]
    rifm_s = rifm_series[:n]

    def _estadisticos(serie):
        diff_s = np.diff(serie)
        return {
            'mean': float(np.mean(serie)),
            'sd':   float(np.std(serie, ddof=1) if len(serie) > 1 else 0.0),
            'sdsd': float(np.std(diff_s, ddof=1) if len(diff_s) > 1 else 0.0),
            'rmssd': float(np.sqrt(np.mean(diff_s**2)) if len(diff_s) > 0 else 0.0)
        }

    ppi_stats  = _estadisticos(ppi_s)
    riam_stats = _estadisticos(riam_s)
    rifm_stats = _estadisticos(rifm_s)

    feat = {}
    for stat in ['mean', 'sd', 'sdsd', 'rmssd']:
        feat[f'PPI_{stat}']  = ppi_stats[stat]
        feat[f'RIAM_{stat}'] = riam_stats[stat]
        feat[f'RIFM_{stat}'] = rifm_stats[stat]

    return feat


# =============================================================================
# BLOQUE 3: STATISTICAL & ENERGY FEATURES (15 features)
# =============================================================================

# DESCRIPCIÓN: Calcula métricas avanzadas de energía, dispersión, momentos estadísticos y morfología 
#              global de la señal BVP (Blood Volume Pulse) filtrada. Determina desviaciones absolutas, 
#              rango intercuartílico, momentos centrales, energía media, factor de forma, longitud de curva, 
#              energía de Teager (TEO), medias recortadas (12.5% y 25%) y formas de la distribución (asimetría y curtosis).
# PARÁMETROS:
#   - filtered_bvp (np.ndarray): Arreglo 1D con la señal BVP preprocesada y filtrada.
# RETORNO:
#   - dict: Diccionario con 15 características estadísticas y energéticas calculadas sobre la ventana (avgADPPG, 
#           stdADPPG, MADPPG, IQRPPG, nCMPPG, avgEPPG, SFPPG, avgCLPPG, avgTEPPG, GmPPG, HmPPG, TM25PPG, TM50PPG, SkewPPG, KurtPPG).

def extraer_estadisticas_energia(filtered_bvp: np.ndarray) -> dict:
    bvp_mean = np.mean(filtered_bvp)
    abs_dev   = np.abs(filtered_bvp - bvp_mean)

    # Media armónica y geométrica requieren valores positivos.
    pos_bvp = filtered_bvp - np.min(filtered_bvp) + 0.01

    feat = {
        'avgADPPG':  float(np.mean(abs_dev)),
        'stdADPPG':  float(np.std(abs_dev, ddof=1)),
        'MADPPG':    float(np.median(abs_dev)),
        'IQRPPG':    float(iqr(filtered_bvp)),
        'nCMPPG':    float(moment(filtered_bvp, moment=2)),
        'avgEPPG':   float(np.mean(filtered_bvp ** 2)),
        'SFPPG':     float(
            np.sqrt(np.mean(filtered_bvp ** 2)) / np.mean(np.abs(filtered_bvp))
            if np.mean(np.abs(filtered_bvp)) > 0 else 0.0
        ),
        'avgCLPPG':  float(np.mean(np.abs(np.diff(filtered_bvp)))),
        'avgTEPPG':  float(np.mean(
            filtered_bvp[1:-1] ** 2 - filtered_bvp[:-2] * filtered_bvp[2:]
        )),
        'GmPPG':     float(gmean(pos_bvp)),
        'HmPPG':     float(hmean(pos_bvp)),
        'TM25PPG':   float(trim_mean(filtered_bvp, 0.125)),
        'TM50PPG':   float(trim_mean(filtered_bvp, 0.25)),
        'SkewPPG':   float(skew(filtered_bvp)),
        'KurtPPG':   float(kurtosis(filtered_bvp)),
    }
    return feat


# =============================================================================
# BLOQUE 4: HEART RATE VARIABILITY - POINCARÉ (4 features)
# =============================================================================

# DESCRIPCIÓN: Calcula los descriptores no lineales del mapa de Poincaré (SD1, SD2 y su ratio RSD1SD2) 
#              junto a la medida de dispersión CCM (Complex Correlation Measure) directamente sobre la señal 
#              BVP filtrada. Evalúa la variabilidad a corto plazo (perpendicular a la línea de identidad) 
#              y a largo plazo (a lo largo de la línea de identidad) para capturar la dinámica no lineal 
#              y la complejidad del sistema cardiovascular.
# PARÁMETROS:
#   - filtered_bvp (np.ndarray): Arreglo 1D con la señal BVP filtrada de la ventana analizada.
# RETORNO:
#   - dict: Diccionario con los 4 descriptores no lineales resultantes ('SD1PPG', 'SD2PPG', 
#           'RSD1SD2PPG', 'CCMPPG').

def extraer_hrv_poincare(filtered_bvp: np.ndarray) -> dict:
    x = filtered_bvp[:-1]
    y = filtered_bvp[1:]

    sd1 = float(np.std((x - y) / np.sqrt(2), ddof=1))
    sd2 = float(np.std((x + y) / np.sqrt(2), ddof=1))
    rsd = sd1 / sd2 if sd2 > 0 else 0.0

    diff_sig = np.abs(np.diff(filtered_bvp))
    ccm = float(
        np.mean(diff_sig) / (sd1 * sd2) if (sd1 * sd2) > 0 else 0.0
    )

    return {
        'SD1PPG':      sd1,
        'SD2PPG':      sd2,
        'RSD1SD2PPG':  rsd,
        'CCMPPG':      ccm,
    }


# =============================================================================
# BLOQUE 5: COMPLEXITY & FRACTAL DYNAMICS (5 features)
# Hjorth Activity, Mobility, Complexity, Higuchi FD, Katz FD
# =============================================================================

# DESCRIPCIÓN: Calcula los parámetros de movilidad y complejidad de Hjorth (Actividad, Movilidad y Complejidad) 
#              junto con las dimensiones fractales de Higuchi (HFD) y Katz (KFD) sobre la señal BVP filtrada. 
#              Evalúa la varianza, la frecuencia media y los cambios de forma de la señal, cuantificando 
#              la autosimilitud y la complejidad no lineal de la dinámica cardiovascular.
# PARÁMETROS:
#   - filtered_bvp (np.ndarray): Arreglo 1D con la señal BVP preprocesada y filtrada.
# RETORNO:
#   - dict: Diccionario con las 5 métricas no lineales y fractales calculadas ('HAPPG', 'HMPPG', 'HCPPG', 
#           'HFDPPG', 'KFDPPG').

def extraer_complejidad_fractal(filtered_bvp: np.ndarray) -> dict:
    # Parámetros de Hjorth
    var_x   = float(np.var(filtered_bvp, ddof=1))
    var_dx  = float(np.var(np.diff(filtered_bvp), ddof=1))
    var_ddx = float(np.var(np.diff(np.diff(filtered_bvp)), ddof=1))

    ha = var_x
    hm = float(np.sqrt(var_dx / var_x)) if var_x > 0 else 0.0
    hc = float(
        (np.sqrt(var_ddx / var_dx) / hm)
        if (hm > 0 and var_dx > 0) else 0.0
    )

    # Dimensiones fractales (antropy)
    hfd = float(ant.higuchi_fd(filtered_bvp))
    kfd = float(ant.katz_fd(filtered_bvp))

    return {
        'HAPPG':  ha,
        'HMPPG':  hm,
        'HCPPG':  hc,
        'HFDPPG': hfd,
        'KFDPPG': kfd,
    }


# =============================================================================
# FUNCIÓN PRINCIPAL: calcular_metricas_bvp
# Orquesta los 5 bloques y devuelve exactamente 50 features o None.
# =============================================================================


# DESCRIPCIÓN: Coordina la extracción de las características fisiológicas indexadas 
#              a partir de un segmento temporal de 30 segundos de la señal BVP/PPG,
#              previamente suavizado mediante el filtro de Savitzky-Golay. Coordina 
#              la ejecución de 5 subgrupos analíticos y consolida el vector final.
# PARÁMETROS:
#   - bvp_filtrado (np.ndarray): Matriz con la señal BVP suavizada (ej. 1920 muestras).
#   - frecuencia_hz (int):       Frecuencia de muestreo del registro (ej. 64 Hz).
# RETORNO:
#   - dict:  Diccionario con las variables calculadas bajo su nomenclatura.
#   - None:  Retorna nulo si el segmento biológico no posee la calidad o el número 
#            mínimo de picos válidos para realizar un análisis matemático estable.

def calcular_metricas_bvp(filtered_bvp: np.ndarray, hertz_freq: int) -> dict | None:

    # --- Detección de picos ---
    std_bvp = np.std(filtered_bvp)
    prom_threshold = std_bvp * 0.2

    positive_peaks, _ = find_peaks(
        filtered_bvp,
        distance=int(hertz_freq * 0.4),   # ~40 % de 1 s → mínimo 40 BPM
        prominence=prom_threshold
    )
    negative_peaks, _ = find_peaks(
        -filtered_bvp,
        distance=int(hertz_freq * 0.4),
        prominence=prom_threshold
    )

    if len(positive_peaks) < 3 or len(negative_peaks) < 3:
        return None

    second_deriv = np.gradient(np.gradient(filtered_bvp))

    # --- Bloque 1: Morfología (14 features) ---
    feat_morf = extraer_morfologia(
        positive_peaks, negative_peaks, filtered_bvp, second_deriv, hertz_freq
    )
    if feat_morf is None:
        return None

    # --- Bloque 2: Ritmo y respiración (12 features) ---
    feat_ritmo = extraer_ritmo_respiratorio(positive_peaks, filtered_bvp, hertz_freq)
    if feat_ritmo is None:
        return None

    # --- Bloque 3: Estadísticas y energía (15 features) ---
    feat_stat = extraer_estadisticas_energia(filtered_bvp)

    # --- Bloque 4: HRV Poincaré (4 features) ---
    feat_hrv = extraer_hrv_poincare(filtered_bvp)

    # --- Bloque 5: Complejidad y fractales (5 features) ---
    feat_frac = extraer_complejidad_fractal(filtered_bvp)

    # --- Consolidar (14 + 12 + 15 + 4 + 5 = 50) ---
    feat_total = {**feat_morf, **feat_ritmo, **feat_stat, **feat_hrv, **feat_frac}

    # Verificación interna: debe haber exactamente 50 keys antes del filtro
    assert len(feat_total) == 50, (
        f"Se esperaban 50 features intermedias, se obtuvieron {len(feat_total)}"
    )
    return feat_total

# DESCRIPCIÓN: Devuelve el listado ordenado y completo de los nombres de las 50 
#              características fisiológicas calculadas. Este método es crucial 
#              para auditar y garantizar la perfecta alineación estructural 
#              entre las matrices de los conjuntos de entrenamiento y prueba.
# RETORNO:
#   - list[str]: Lista con las etiquetas de texto de todas las variables.

def nombre_features() -> list[str]:
    morf  = [f'{s}_{c}' for s in ('Mean', 'SD') for c in ('A', 'A1', 'A2', 'T1', 'T2', 'IPAR', 'IPTR')]
    ritmo = [f'{s}_{m}' for m in ('PPI', 'RIAM', 'RIFM') for s in ('mean', 'sd', 'sdsd', 'rmssd')]
    stat  = ['avgADPPG', 'stdADPPG', 'MADPPG', 'IQRPPG', 'nCMPPG', 'avgEPPG',
             'SFPPG', 'avgCLPPG', 'avgTEPPG', 'GmPPG', 'HmPPG',
             'TM25PPG', 'TM50PPG', 'SkewPPG', 'KurtPPG']
    hrv   = ['SD1PPG', 'SD2PPG', 'RSD1SD2PPG', 'CCMPPG']
    frac  = ['HAPPG', 'HMPPG', 'HCPPG', 'HFDPPG', 'KFDPPG']
    return morf + ritmo + stat + hrv + frac