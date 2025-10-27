#!/usr/bin/env python3
"""
Pipeline de Preprocesamiento: GPD con 1 Componente (Z)

Este script implementa el pipeline de 9 pasos para transformar archivos miniSEED
de componente vertical (Z) a vectores listos para entrenamiento de GPD.

Input:  miniSEED @ 64 Hz, ~6s, 1 componente (Z)
Output: NumPy arrays @ 100 Hz, 4s, shape (400,)

Autor: Claude Code
Fecha: 2025-10-27
Version: 1.0.0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings

import numpy as np
import pandas as pd
from obspy import read
from obspy.core.stream import Stream
from obspy.core.trace import Trace


# ==================== CONFIGURACIÓN ====================

# Clase a código numérico
CLASS_MAPPING = {
    'P': 0,
    'S': 1,
    'Noise': 2
}

# Parámetros del pipeline
ORIGINAL_SAMPLING_RATE = 64.0  # Hz
TARGET_SAMPLING_RATE = 100.0   # Hz
TARGET_DURATION = 4.0          # segundos
TARGET_SAMPLES = 400           # muestras (100 Hz * 4 s)
ORIGINAL_DURATION_APPROX = 6.0 # segundos

# Parámetros de filtrado
FILTER_FREQMIN = 1.0   # Hz
FILTER_FREQMAX = 30.0  # Hz (limitado por Nyquist de 64 Hz = 32 Hz)
FILTER_CORNERS = 4
FILTER_ZEROPHASE = True

# Tolerancias de validación
MIN_SAMPLES_EXPECTED = 380
MAX_SAMPLES_EXPECTED = 390
NORMALIZATION_EPSILON = 1e-10
MAXABS_TOLERANCE = 1.0 + 1e-6


# ==================== CONFIGURACIÓN DE LOGGING ====================

def setup_logging(log_level: str = 'INFO') -> None:
    """
    Configura el sistema de logging.

    Args:
        log_level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Suprimir warnings de ObsPy si no estamos en modo DEBUG
    if numeric_level > logging.DEBUG:
        warnings.filterwarnings('ignore', category=UserWarning)


# ==================== FUNCIONES DE VALIDACIÓN ====================

def validate_environment() -> None:
    """
    Valida que el entorno tenga las versiones correctas de dependencias.
    """
    # Validar Python 3.9
    if sys.version_info[:2] != (3, 9):
        logging.warning(f"Se recomienda Python 3.9, se detectó {sys.version_info[:2]}")

    # Validar NumPy
    np_version = np.__version__
    if not np_version.startswith('1.26'):
        logging.warning(f"Se recomienda NumPy 1.26.x, se detectó {np_version}")

    logging.info(f"Entorno validado: Python {sys.version_info[:2]}, NumPy {np_version}")


def infer_class_from_filename(filepath: Path) -> Optional[str]:
    """
    Infiere la clase del archivo basándose en su nombre.

    Args:
        filepath: Path al archivo miniSEED

    Returns:
        'P', 'S', 'Noise' o None si no se puede inferir
    """
    filename = filepath.name

    if filename.endswith('_P.mseed'):
        return 'P'
    elif filename.endswith('_S.mseed'):
        return 'S'
    elif filename.startswith('noise_'):
        return 'Noise'
    else:
        return None


def validate_trace(trace: Trace, filepath: Path) -> Tuple[bool, str]:
    """
    Valida que la traza cumpla con las especificaciones esperadas.

    Args:
        trace: Traza de ObsPy
        filepath: Path al archivo (para logging)

    Returns:
        (es_válido, mensaje_error)
    """
    # Validar sampling rate
    if not np.isclose(trace.stats.sampling_rate, ORIGINAL_SAMPLING_RATE):
        return False, f"Sampling rate esperado {ORIGINAL_SAMPLING_RATE} Hz, encontrado {trace.stats.sampling_rate} Hz"

    # Validar número de muestras
    npts = trace.stats.npts
    if not (MIN_SAMPLES_EXPECTED <= npts <= MAX_SAMPLES_EXPECTED):
        return False, f"Número de muestras fuera de rango [{MIN_SAMPLES_EXPECTED}, {MAX_SAMPLES_EXPECTED}], encontrado {npts}"

    return True, ""


def validate_processed_data(data: np.ndarray) -> Tuple[bool, str]:
    """
    Valida que los datos procesados cumplan con las especificaciones.

    Args:
        data: Array procesado

    Returns:
        (es_válido, mensaje_error)
    """
    # Validar longitud
    if len(data) != TARGET_SAMPLES:
        return False, f"Longitud esperada {TARGET_SAMPLES}, encontrado {len(data)}"

    # Validar NaN
    if np.isnan(data).any():
        return False, "Contiene valores NaN"

    # Validar Inf
    if np.isinf(data).any():
        return False, "Contiene valores Inf"

    # Validar rango de normalización
    maxabs = np.max(np.abs(data))
    if maxabs > MAXABS_TOLERANCE:
        return False, f"Valor máximo absoluto {maxabs} excede tolerancia {MAXABS_TOLERANCE}"

    return True, ""


# ==================== PIPELINE DE PROCESAMIENTO ====================

def process_single_file(filepath: Path) -> Optional[Dict]:
    """
    Procesa un único archivo miniSEED siguiendo el pipeline de 9 pasos.

    Pipeline:
        1. Lectura con ObsPy
        2. Detrend (demean)
        3. Detrend (linear)
        4. Filtrado pasa-banda 1-30 Hz
        5. Remuestreo 64 Hz → 100 Hz
        6. Normalización max-abs
        7. Recorte temporal (6s → 4s, centrado)
        8. Validaciones
        9. Etiquetado y metadata

    Args:
        filepath: Path al archivo miniSEED

    Returns:
        Dict con claves: 'data', 'label', 'label_name', 'filename'
        o None si el procesamiento falla
    """
    try:
        # ===== PASO 1: Lectura con ObsPy =====
        stream: Stream = read(str(filepath))

        if len(stream) == 0:
            logging.error(f"{filepath.name}: Stream vacío")
            return None

        trace: Trace = stream[0]

        # Validar traza
        is_valid, error_msg = validate_trace(trace, filepath)
        if not is_valid:
            logging.error(f"{filepath.name}: {error_msg}")
            return None

        # Inferir clase
        class_name = infer_class_from_filename(filepath)
        if class_name is None:
            logging.error(f"{filepath.name}: No se pudo inferir la clase del nombre del archivo")
            return None

        # ===== PASO 2: Detrend - Remover offset DC =====
        trace.detrend('demean')

        # ===== PASO 3: Detrend - Remover tendencia lineal =====
        trace.detrend('linear')

        # ===== PASO 4: Filtrado Pasa-Banda 1-30 Hz =====
        trace.filter(
            'bandpass',
            freqmin=FILTER_FREQMIN,
            freqmax=FILTER_FREQMAX,
            corners=FILTER_CORNERS,
            zerophase=FILTER_ZEROPHASE
        )

        # ===== PASO 5: Remuestreo 64 Hz → 100 Hz =====
        trace.interpolate(
            sampling_rate=TARGET_SAMPLING_RATE,
            method='linear'
        )

        # ===== PASO 6: Normalización Max-Abs =====
        data = trace.data.astype(np.float32)
        maxabs = np.max(np.abs(data))
        data = data / max(maxabs, NORMALIZATION_EPSILON)

        # ===== PASO 7: Recorte Temporal (6s → 4s, centrado) =====
        total_samples = len(data)
        start_idx = (total_samples - TARGET_SAMPLES) // 2
        end_idx = start_idx + TARGET_SAMPLES

        if start_idx < 0 or end_idx > total_samples:
            logging.error(
                f"{filepath.name}: No se puede recortar a {TARGET_SAMPLES} muestras "
                f"desde {total_samples} muestras disponibles"
            )
            return None

        data_crop = data[start_idx:end_idx]

        # ===== PASO 8: Validaciones =====
        is_valid, error_msg = validate_processed_data(data_crop)
        if not is_valid:
            logging.error(f"{filepath.name}: Validación fallida - {error_msg}")
            return None

        # ===== PASO 9: Etiquetado y Metadata =====
        label = CLASS_MAPPING[class_name]

        return {
            'data': data_crop,
            'label': label,
            'label_name': class_name,
            'filename': filepath.name
        }

    except Exception as e:
        logging.error(f"{filepath.name}: Error durante el procesamiento - {str(e)}")
        return None


# ==================== PROCESAMIENTO DE DATASET ====================

def process_dataset(input_dir: Path, output_dir: Path) -> None:
    """
    Procesa todos los archivos .mseed en el directorio de entrada.

    Args:
        input_dir: Directorio con archivos miniSEED
        output_dir: Directorio donde guardar los resultados
    """
    logging.info(f"Iniciando procesamiento de dataset")
    logging.info(f"Directorio de entrada: {input_dir}")
    logging.info(f"Directorio de salida: {output_dir}")

    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar todos los archivos .mseed
    mseed_files = sorted(input_dir.glob('*.mseed'))
    total_files = len(mseed_files)

    if total_files == 0:
        logging.error(f"No se encontraron archivos .mseed en {input_dir}")
        sys.exit(1)

    logging.info(f"Archivos encontrados: {total_files}")

    # Listas para acumular resultados
    data_list = []
    label_list = []
    metadata_list = []

    # Contadores
    processed_count = 0
    failed_count = 0

    # Procesar cada archivo
    for idx, filepath in enumerate(mseed_files, 1):
        # Logging de progreso cada 1000 archivos o 10%
        if idx % 1000 == 0 or idx % max(1, total_files // 10) == 0:
            progress = (idx / total_files) * 100
            logging.info(f"Progreso: {idx}/{total_files} ({progress:.1f}%) - Procesados: {processed_count}, Fallidos: {failed_count}")

        result = process_single_file(filepath)

        if result is not None:
            data_list.append(result['data'])
            label_list.append(result['label'])
            metadata_list.append({
                'filename': result['filename'],
                'label': result['label'],
                'label_name': result['label_name']
            })
            processed_count += 1
        else:
            failed_count += 1

    # Resumen final
    logging.info(f"Procesamiento completado:")
    logging.info(f"  - Total archivos: {total_files}")
    logging.info(f"  - Procesados exitosamente: {processed_count}")
    logging.info(f"  - Fallidos: {failed_count}")

    if processed_count == 0:
        logging.error("No se procesó ningún archivo exitosamente")
        sys.exit(1)

    # ===== GUARDAR RESULTADOS =====

    # Convertir a arrays NumPy
    X = np.array(data_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int8)

    logging.info(f"Shape de X: {X.shape}")
    logging.info(f"Shape de y: {y.shape}")
    logging.info(f"Dtype de X: {X.dtype}")
    logging.info(f"Dtype de y: {y.dtype}")

    # Estadísticas de clases
    unique, counts = np.unique(y, return_counts=True)
    logging.info("Distribución de clases:")
    for class_code, count in zip(unique, counts):
        class_name = [k for k, v in CLASS_MAPPING.items() if v == class_code][0]
        percentage = (count / len(y)) * 100
        logging.info(f"  - {class_name} (código {class_code}): {count} muestras ({percentage:.2f}%)")

    # Guardar X.npy
    X_path = output_dir / 'X.npy'
    np.save(X_path, X)
    logging.info(f"Guardado X.npy en {X_path}")

    # Guardar y.npy
    y_path = output_dir / 'y.npy'
    np.save(y_path, y)
    logging.info(f"Guardado y.npy en {y_path}")

    # Guardar metadata.csv
    metadata_df = pd.DataFrame(metadata_list)
    metadata_path = output_dir / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    logging.info(f"Guardado metadata.csv en {metadata_path}")

    # Estadísticas de datos
    logging.info("Estadísticas de datos procesados:")
    logging.info(f"  - Media de X: {X.mean():.6f}")
    logging.info(f"  - Desviación estándar de X: {X.std():.6f}")
    logging.info(f"  - Min de X: {X.min():.6f}")
    logging.info(f"  - Max de X: {X.max():.6f}")

    logging.info("¡Preprocesamiento completado exitosamente!")


# ==================== MAIN ====================

def main():
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description='Preprocesamiento de archivos miniSEED para GPD (1 componente Z)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python preprocess_dataset.py -i /data/mseed_files -o /data/processed

El script espera archivos miniSEED con la siguiente nomenclatura:
  - Ondas P: <event>_<sampleidx>_P.mseed
  - Ondas S: <event>_<sampleidx>_S.mseed
  - Ruido:   noise_<sampleidx>.mseed

Output:
  - X.npy: Array de forma (N, 400) con datos normalizados
  - y.npy: Array de forma (N,) con etiquetas (P:0, S:1, Noise:2)
  - metadata.csv: CSV con filename, label, label_name
        """
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='Directorio con archivos miniSEED de entrada'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directorio donde guardar los resultados (X.npy, y.npy, metadata.csv)'
    )

    parser.add_argument(
        '-l', '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Nivel de logging (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Validar entorno
    validate_environment()

    # Convertir a Path
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validar que el directorio de entrada existe
    if not input_dir.exists():
        logging.error(f"El directorio de entrada no existe: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        logging.error(f"La ruta de entrada no es un directorio: {input_dir}")
        sys.exit(1)

    # Procesar dataset
    process_dataset(input_dir, output_dir)


if __name__ == '__main__':
    main()
