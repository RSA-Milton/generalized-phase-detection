#!/usr/bin/env python3
"""
Script de División Estratificada del Dataset para Reentrenamiento GPD

Este script divide el dataset preprocesado en conjuntos de entrenamiento,
validación y prueba manteniendo la proporción de clases (estratificación).

Funcionalidad:
    1. Carga dataset preprocesado (X.npy, y.npy)
    2. Realiza split estratificado 70/15/15 (train/val/test)
    3. Valida balance de clases en cada subset
    4. Calcula estadísticas globales y por clase
    5. Genera archivo dataset_stats.json con trazabilidad
    6. Guarda subsets en formato .npy

Input:
    - X.npy: Array (N, 400) con datos normalizados
    - y.npy: Array (N,) con etiquetas (0: P, 1: S, 2: Noise)
    - metadata.csv: CSV con información de archivos originales

Output:
    - X_train.npy, y_train.npy (~70%)
    - X_val.npy, y_val.npy (~15%)
    - X_test.npy, y_test.npy (~15%)
    - dataset_stats.json (estadísticas y trazabilidad)

Autor: Claude Code
Fecha: 2025-10-29
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import entropy


# ==================== CONFIGURACIÓN ====================

CLASS_NAMES = {0: 'P', 1: 'S', 2: 'Noise'}
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


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


# ==================== FUNCIONES DE CARGA ====================

def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carga el dataset preprocesado.

    Args:
        data_dir: Directorio con X.npy, y.npy, metadata.csv

    Returns:
        (X, y, metadata_df)
    """
    X_path = data_dir / 'X.npy'
    y_path = data_dir / 'y.npy'
    metadata_path = data_dir / 'metadata.csv'

    # Verificar que existen los archivos
    for path in [X_path, y_path, metadata_path]:
        if not path.exists():
            logging.error(f"No se encontró el archivo: {path}")
            sys.exit(1)

    logging.info(f"Cargando dataset desde {data_dir}")
    X = np.load(X_path)
    y = np.load(y_path)
    metadata_df = pd.read_csv(metadata_path)

    logging.info(f"  - X shape: {X.shape}, dtype: {X.dtype}")
    logging.info(f"  - y shape: {y.shape}, dtype: {y.dtype}")
    logging.info(f"  - metadata: {len(metadata_df)} filas")

    return X, y, metadata_df


# ==================== FUNCIONES DE SPLIT ====================

def stratified_split(X: np.ndarray, y: np.ndarray,
                     train_ratio: float = TRAIN_RATIO,
                     val_ratio: float = VAL_RATIO,
                     test_ratio: float = TEST_RATIO,
                     random_state: int = RANDOM_STATE) -> Dict[str, np.ndarray]:
    """
    Realiza split estratificado del dataset en train/val/test.

    Args:
        X: Array de datos (N, 400)
        y: Array de etiquetas (N,)
        train_ratio: Proporción de entrenamiento (default: 0.70)
        val_ratio: Proporción de validación (default: 0.15)
        test_ratio: Proporción de prueba (default: 0.15)
        random_state: Semilla para reproducibilidad

    Returns:
        Diccionario con claves: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Validar proporciones
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        logging.error(f"Las proporciones deben sumar 1.0, suma actual: {total_ratio}")
        sys.exit(1)

    logging.info(f"Realizando split estratificado:")
    logging.info(f"  - Train: {train_ratio*100:.1f}%")
    logging.info(f"  - Val: {val_ratio*100:.1f}%")
    logging.info(f"  - Test: {test_ratio*100:.1f}%")
    logging.info(f"  - Random state: {random_state}")

    # Primera división: train vs (val+test)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_state
    )

    logging.info(f"Primera división completada:")
    logging.info(f"  - Train: {len(y_train)} muestras")
    logging.info(f"  - Temp (val+test): {len(y_temp)} muestras")

    # Segunda división: val vs test (del conjunto temporal)
    # test_ratio relativo al conjunto temporal
    test_ratio_relative = test_ratio / temp_ratio

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio_relative,
        stratify=y_temp,
        random_state=random_state
    )

    logging.info(f"Segunda división completada:")
    logging.info(f"  - Val: {len(y_val)} muestras")
    logging.info(f"  - Test: {len(y_test)} muestras")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


# ==================== FUNCIONES DE VALIDACIÓN ====================

def calculate_class_distribution(y: np.ndarray) -> Dict[str, Dict]:
    """
    Calcula la distribución de clases.

    Args:
        y: Array de etiquetas

    Returns:
        Diccionario con conteos y porcentajes por clase
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    distribution = {
        'counts': {},
        'percentages': {},
        'total': total
    }

    for class_code, count in zip(unique, counts):
        class_name = CLASS_NAMES[class_code]
        percentage = (count / total) * 100

        distribution['counts'][class_name] = int(count)
        distribution['percentages'][class_name] = float(percentage)

    return distribution


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calcula la divergencia KL entre dos distribuciones.

    Args:
        p: Primera distribución (probabilidades)
        q: Segunda distribución (probabilidades)

    Returns:
        Divergencia KL (en nats)
    """
    # Evitar división por cero
    p = np.asarray(p) + 1e-10
    q = np.asarray(q) + 1e-10

    # Normalizar
    p = p / p.sum()
    q = q / q.sum()

    return entropy(p, q)


def validate_stratification(y_dict: Dict[str, np.ndarray]) -> Dict:
    """
    Valida que la estratificación mantenga las proporciones de clases.

    Args:
        y_dict: Diccionario con y_train, y_val, y_test

    Returns:
        Diccionario con métricas de validación
    """
    logging.info("\n" + "="*60)
    logging.info("VALIDACIÓN DE ESTRATIFICACIÓN")
    logging.info("="*60)

    # Calcular distribuciones
    distributions = {}
    for subset_name, y_subset in y_dict.items():
        distributions[subset_name] = calculate_class_distribution(y_subset)

    # Calcular distribución global
    y_all = np.concatenate([y_dict['y_train'], y_dict['y_val'], y_dict['y_test']])
    distributions['all'] = calculate_class_distribution(y_all)

    # Imprimir distribuciones
    logging.info("\nDistribución de clases por subset:")
    logging.info("-" * 60)

    for subset_name in ['all', 'y_train', 'y_val', 'y_test']:
        dist = distributions[subset_name]
        logging.info(f"\n{subset_name.upper()}:")
        logging.info(f"  Total: {dist['total']} muestras")

        for class_name in ['P', 'S', 'Noise']:
            count = dist['counts'].get(class_name, 0)
            percentage = dist['percentages'].get(class_name, 0.0)
            logging.info(f"  - {class_name:6s}: {count:6d} muestras ({percentage:5.2f}%)")

    # Calcular divergencias KL respecto a la distribución global
    logging.info("\nDivergencia KL respecto a distribución global:")
    logging.info("-" * 60)

    # Obtener probabilidades de la distribución global
    p_global = np.array([
        distributions['all']['percentages']['P'],
        distributions['all']['percentages']['S'],
        distributions['all']['percentages']['Noise']
    ]) / 100.0

    kl_divergences = {}
    for subset_name in ['y_train', 'y_val', 'y_test']:
        p_subset = np.array([
            distributions[subset_name]['percentages']['P'],
            distributions[subset_name]['percentages']['S'],
            distributions[subset_name]['percentages']['Noise']
        ]) / 100.0

        kl_div = calculate_kl_divergence(p_subset, p_global)
        kl_divergences[subset_name] = float(kl_div)
        logging.info(f"  {subset_name}: {kl_div:.6f} nats")

    # Calcular diferencias máximas de porcentaje
    logging.info("\nDiferencias máximas de porcentaje vs global:")
    logging.info("-" * 60)

    max_diffs = {}
    for subset_name in ['y_train', 'y_val', 'y_test']:
        diffs = []
        for class_name in ['P', 'S', 'Noise']:
            global_pct = distributions['all']['percentages'][class_name]
            subset_pct = distributions[subset_name]['percentages'][class_name]
            diff = abs(subset_pct - global_pct)
            diffs.append(diff)

        max_diff = max(diffs)
        max_diffs[subset_name] = float(max_diff)
        logging.info(f"  {subset_name}: {max_diff:.3f}%")

    # Verificar criterios de éxito
    logging.info("\nCriterios de validación:")
    logging.info("-" * 60)

    all_valid = True

    # Criterio 1: Diferencias menores al 2%
    for subset_name, max_diff in max_diffs.items():
        status = "✓" if max_diff < 2.0 else "✗"
        logging.info(f"  {status} {subset_name}: max diff = {max_diff:.3f}% (criterio: <2%)")
        if max_diff >= 2.0:
            all_valid = False

    # Criterio 2: KL divergence < 0.005
    for subset_name, kl_div in kl_divergences.items():
        status = "✓" if kl_div < 0.005 else "⚠"
        logging.info(f"  {status} {subset_name}: KL div = {kl_div:.6f} (esperado: <0.005)")

    if all_valid:
        logging.info("\n✓ Estratificación EXITOSA: Todos los criterios cumplidos")
    else:
        logging.warning("\n⚠ Estratificación ACEPTABLE: Algunos criterios no cumplidos")

    logging.info("="*60 + "\n")

    return {
        'distributions': distributions,
        'kl_divergences': kl_divergences,
        'max_differences': max_diffs,
        'validation_passed': all_valid
    }


# ==================== FUNCIONES DE ESTADÍSTICAS ====================

def calculate_statistics(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Calcula estadísticas globales y por clase.

    Args:
        X: Array de datos (N, 400)
        y: Array de etiquetas (N,)

    Returns:
        Diccionario con estadísticas
    """
    stats = {
        'global': {
            'mean': float(X.mean()),
            'std': float(X.std()),
            'min': float(X.min()),
            'max': float(X.max()),
            'median': float(np.median(X))
        },
        'per_class': {}
    }

    # Estadísticas por clase
    for class_code in [0, 1, 2]:
        class_name = CLASS_NAMES[class_code]
        class_indices = np.where(y == class_code)[0]
        class_data = X[class_indices]

        stats['per_class'][class_name] = {
            'mean': float(class_data.mean()),
            'std': float(class_data.std()),
            'min': float(class_data.min()),
            'max': float(class_data.max()),
            'median': float(np.median(class_data)),
            'abs_mean': float(np.mean(np.abs(class_data)))
        }

    return stats


def generate_dataset_stats(splits: Dict[str, np.ndarray],
                           validation_results: Dict,
                           input_path: Path,
                           output_path: Path) -> Dict:
    """
    Genera el diccionario completo de estadísticas del dataset.

    Args:
        splits: Diccionario con X_train, y_train, etc.
        validation_results: Resultados de validación de estratificación
        input_path: Path del dataset original
        output_path: Path donde se guardará el dataset dividido

    Returns:
        Diccionario con todas las estadísticas
    """
    logging.info("Calculando estadísticas del dataset...")

    stats = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'input_dataset_path': str(input_path),
        'output_dataset_path': str(output_path),
        'random_state': RANDOM_STATE,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'sizes': {
            'train': int(len(splits['y_train'])),
            'val': int(len(splits['y_val'])),
            'test': int(len(splits['y_test'])),
            'total': int(len(splits['y_train']) + len(splits['y_val']) + len(splits['y_test']))
        },
        'balance': {},
        'stats_global': {},
        'stats_per_class': {},
        'validation': {
            'kl_divergences': validation_results['kl_divergences'],
            'max_differences': validation_results['max_differences'],
            'passed': validation_results['validation_passed']
        }
    }

    # Balance de clases
    for subset_name in ['train', 'val', 'test']:
        y_key = f'y_{subset_name}'
        dist = validation_results['distributions'][y_key]
        stats['balance'][subset_name] = dist['percentages']

    # Estadísticas globales y por clase
    for subset_name in ['train', 'val', 'test']:
        X_key = f'X_{subset_name}'
        y_key = f'y_{subset_name}'

        subset_stats = calculate_statistics(splits[X_key], splits[y_key])
        stats['stats_global'][subset_name] = subset_stats['global']
        stats['stats_per_class'][subset_name] = subset_stats['per_class']

    logging.info("Estadísticas calculadas exitosamente")

    return stats


# ==================== FUNCIONES DE GUARDADO ====================

def save_splits(splits: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Guarda los subsets en archivos .npy.

    Args:
        splits: Diccionario con X_train, y_train, etc.
        output_dir: Directorio de salida
    """
    logging.info(f"\nGuardando subsets en {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for key, array in splits.items():
        output_path = output_dir / f'{key}.npy'
        np.save(output_path, array)
        logging.info(f"  - Guardado {key}.npy: shape {array.shape}, dtype {array.dtype}")


def save_stats_json(stats: Dict, output_dir: Path) -> None:
    """
    Guarda las estadísticas en formato JSON.

    Args:
        stats: Diccionario con estadísticas
        output_dir: Directorio de salida
    """
    output_path = output_dir / 'dataset_stats.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logging.info(f"\nEstadísticas guardadas en {output_path}")


# ==================== MAIN ====================

def main():
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description='División estratificada del dataset para reentrenamiento GPD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python split_dataset.py \\
    -i /home/rsa/projects/gpd/data/processed/datasets/processed-dataset \\
    -o /home/rsa/projects/gpd/data/processed/datasets/retrain-set

El script carga X.npy, y.npy y metadata.csv, realiza un split estratificado
70/15/15 (train/val/test) y genera:
  - X_train.npy, y_train.npy
  - X_val.npy, y_val.npy
  - X_test.npy, y_test.npy
  - dataset_stats.json (estadísticas y trazabilidad)
        """
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='Directorio con dataset preprocesado (X.npy, y.npy, metadata.csv)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directorio donde guardar los subsets divididos'
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

    # Convertir a Path
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validar directorio de entrada
    if not input_dir.exists():
        logging.error(f"El directorio de entrada no existe: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        logging.error(f"La ruta de entrada no es un directorio: {input_dir}")
        sys.exit(1)

    logging.info("="*60)
    logging.info("DIVISIÓN ESTRATIFICADA DEL DATASET")
    logging.info("="*60)
    logging.info(f"Input: {input_dir}")
    logging.info(f"Output: {output_dir}")

    # Paso 1: Cargar dataset
    X, y, metadata_df = load_dataset(input_dir)

    # Paso 2: Realizar split estratificado
    splits = stratified_split(X, y)

    # Paso 3: Validar estratificación
    y_dict = {
        'y_train': splits['y_train'],
        'y_val': splits['y_val'],
        'y_test': splits['y_test']
    }
    validation_results = validate_stratification(y_dict)

    # Paso 4: Generar estadísticas
    stats = generate_dataset_stats(splits, validation_results, input_dir, output_dir)

    # Paso 5: Guardar subsets
    save_splits(splits, output_dir)

    # Paso 6: Guardar estadísticas
    save_stats_json(stats, output_dir)

    logging.info("\n" + "="*60)
    logging.info("DIVISIÓN COMPLETADA EXITOSAMENTE")
    logging.info("="*60)
    logging.info(f"Dataset dividido guardado en: {output_dir}")
    logging.info(f"Total de archivos generados: 7")
    logging.info(f"  - 6 archivos .npy (X_train, y_train, X_val, y_val, X_test, y_test)")
    logging.info(f"  - 1 archivo JSON (dataset_stats.json)")


if __name__ == '__main__':
    main()
