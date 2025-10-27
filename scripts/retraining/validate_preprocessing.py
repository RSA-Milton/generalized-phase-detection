#!/usr/bin/env python3
"""
Script de Validación Visual del Preprocesamiento

Este script carga los datos preprocesados (X.npy, y.npy, metadata.csv) y genera
visualizaciones para verificar la calidad del preprocesamiento.

Verificaciones:
    a) Comparar visualmente 5 trazas al azar de cada clase
    b) Histograma de amplitud (después de normalizar) por clase
    c) Verificar que no hay padding chueco (inicio y final de trazas)

Autor: Claude Code
Fecha: 2025-10-27
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


# ==================== CONFIGURACIÓN ====================

CLASS_NAMES = {0: 'P', 1: 'S', 2: 'Noise'}
CLASS_COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}  # Azul, Naranja, Verde
SAMPLING_RATE = 100.0  # Hz
DURATION = 4.0  # segundos
N_SAMPLES = 400
EDGE_SAMPLES = 50  # Primeros y últimos 50 samples (~0.5s)


# ==================== FUNCIONES DE CARGA ====================

def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carga los datos preprocesados.

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
            print(f"ERROR: No se encontró el archivo {path}")
            sys.exit(1)

    print(f"Cargando datos desde {data_dir}")
    X = np.load(X_path)
    y = np.load(y_path)
    metadata_df = pd.read_csv(metadata_path)

    print(f"  - X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  - y shape: {y.shape}, dtype: {y.dtype}")
    print(f"  - metadata: {len(metadata_df)} filas")

    return X, y, metadata_df


# ==================== VERIFICACIÓN A: TRAZAS ALEATORIAS POR CLASE ====================

def plot_random_traces_per_class(X: np.ndarray, y: np.ndarray, n_traces: int = 5,
                                  output_path: Path = None) -> None:
    """
    Plotea n_traces aleatorias por cada clase.

    Args:
        X: Array de datos (N, 400)
        y: Array de etiquetas (N,)
        n_traces: Número de trazas a plotear por clase
        output_path: Path donde guardar la figura (opcional)
    """
    print(f"\nGenerando gráfico de {n_traces} trazas aleatorias por clase...")

    # Tiempo en segundos
    time_axis = np.linspace(0, DURATION, N_SAMPLES)

    # Crear figura
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Trazas Aleatorias por Clase (Preprocesadas)', fontsize=14, fontweight='bold')

    for class_code in [0, 1, 2]:
        ax = axes[class_code]
        class_name = CLASS_NAMES[class_code]
        color = CLASS_COLORS[class_code]

        # Obtener índices de esta clase
        class_indices = np.where(y == class_code)[0]
        n_available = len(class_indices)

        print(f"  - Clase {class_name}: {n_available} muestras disponibles")

        if n_available < n_traces:
            print(f"    ADVERTENCIA: Solo hay {n_available} muestras, se plotearan todas")
            selected_n = n_available
        else:
            selected_n = n_traces

        # Seleccionar aleatoriamente
        np.random.seed(42)  # Para reproducibilidad
        selected_indices = np.random.choice(class_indices, size=selected_n, replace=False)

        # Plotear cada traza
        for i, idx in enumerate(selected_indices):
            trace = X[idx]
            alpha = 0.7 if i < selected_n - 1 else 1.0  # Última traza más oscura
            linewidth = 1.0 if i < selected_n - 1 else 1.5
            ax.plot(time_axis, trace, color=color, alpha=alpha, linewidth=linewidth)

        # Configuración del subplot
        ax.set_title(f'Clase: {class_name} ({selected_n} trazas)', fontweight='bold')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Amplitud normalizada')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlim(0, DURATION)
        ax.set_ylim(-1.1, 1.1)

        # Estadísticas de amplitud
        class_data = X[class_indices]
        mean_amp = np.mean(np.abs(class_data))
        std_amp = np.std(np.abs(class_data))
        max_amp = np.max(np.abs(class_data))

        stats_text = f'|Amp|: μ={mean_amp:.3f}, σ={std_amp:.3f}, max={max_amp:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  - Gráfico guardado en {output_path}")
    else:
        plt.show()

    plt.close()


# ==================== VERIFICACIÓN B: HISTOGRAMAS DE AMPLITUD ====================

def plot_amplitude_histograms(X: np.ndarray, y: np.ndarray, output_path: Path = None) -> None:
    """
    Plotea histogramas de amplitud por clase.

    Args:
        X: Array de datos (N, 400)
        y: Array de etiquetas (N,)
        output_path: Path donde guardar la figura (opcional)
    """
    print(f"\nGenerando histogramas de amplitud por clase...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribución de Amplitudes por Clase (Post-Normalización)',
                 fontsize=14, fontweight='bold')

    # Subplot 1: Histogramas superpuestos
    ax = axes[0, 0]
    bins = np.linspace(-1, 1, 100)

    for class_code in [0, 1, 2]:
        class_name = CLASS_NAMES[class_code]
        color = CLASS_COLORS[class_code]
        class_indices = np.where(y == class_code)[0]
        class_data = X[class_indices].flatten()

        ax.hist(class_data, bins=bins, alpha=0.6, color=color, label=class_name,
                density=True, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Amplitud normalizada')
    ax.set_ylabel('Densidad')
    ax.set_title('Histogramas Superpuestos (Todas las muestras)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Subplot 2: Histogramas de valor absoluto
    ax = axes[0, 1]
    bins_abs = np.linspace(0, 1, 100)

    for class_code in [0, 1, 2]:
        class_name = CLASS_NAMES[class_code]
        color = CLASS_COLORS[class_code]
        class_indices = np.where(y == class_code)[0]
        class_data = np.abs(X[class_indices]).flatten()

        ax.hist(class_data, bins=bins_abs, alpha=0.6, color=color, label=class_name,
                density=True, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('|Amplitud normalizada|')
    ax.set_ylabel('Densidad')
    ax.set_title('Histogramas de Valor Absoluto')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: Histogramas separados (pequeños)
    for i, class_code in enumerate([0, 1, 2]):
        if i == 0:
            ax = axes[1, 0]
        elif i == 1:
            ax = axes[1, 1]
        else:
            # Para la tercera clase, crear un subplot adicional
            continue

        class_name = CLASS_NAMES[class_code]
        color = CLASS_COLORS[class_code]
        class_indices = np.where(y == class_code)[0]
        class_data = X[class_indices].flatten()

        ax.hist(class_data, bins=bins, alpha=0.8, color=color,
                edgecolor='black', linewidth=0.5)

        # Estadísticas
        mean = np.mean(class_data)
        std = np.std(class_data)
        median = np.median(class_data)

        ax.axvline(x=mean, color='red', linestyle='--', linewidth=2, label=f'μ={mean:.3f}')
        ax.axvline(x=median, color='blue', linestyle='--', linewidth=2, label=f'med={median:.3f}')

        ax.set_xlabel('Amplitud normalizada')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Clase: {class_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        stats_text = f'μ={mean:.4f}\nσ={std:.4f}\nmed={median:.4f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Para la clase Noise, usar espacio restante
    ax = plt.subplot(2, 3, 6)
    class_code = 2
    class_name = CLASS_NAMES[class_code]
    color = CLASS_COLORS[class_code]
    class_indices = np.where(y == class_code)[0]
    class_data = X[class_indices].flatten()

    ax.hist(class_data, bins=bins, alpha=0.8, color=color,
            edgecolor='black', linewidth=0.5)

    mean = np.mean(class_data)
    std = np.std(class_data)
    median = np.median(class_data)

    ax.axvline(x=mean, color='red', linestyle='--', linewidth=2, label=f'μ={mean:.3f}')
    ax.axvline(x=median, color='blue', linestyle='--', linewidth=2, label=f'med={median:.3f}')

    ax.set_xlabel('Amplitud normalizada')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Clase: {class_name}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    stats_text = f'μ={mean:.4f}\nσ={std:.4f}\nmed={median:.4f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  - Gráfico guardado en {output_path}")
    else:
        plt.show()

    plt.close()


# ==================== VERIFICACIÓN C: PADDING CHUECO ====================

def plot_edge_verification(X: np.ndarray, y: np.ndarray, n_samples: int = 6,
                           output_path: Path = None) -> None:
    """
    Verifica que no haya padding chueco mostrando los primeros y últimos 0.5s.

    Args:
        X: Array de datos (N, 400)
        y: Array de etiquetas (N,)
        n_samples: Número de muestras aleatorias a verificar
        output_path: Path donde guardar la figura (opcional)
    """
    print(f"\nGenerando verificación de padding (primeros y últimos {EDGE_SAMPLES} samples)...")

    # Seleccionar muestras aleatorias (2 por clase)
    np.random.seed(42)
    selected_indices = []

    for class_code in [0, 1, 2]:
        class_indices = np.where(y == class_code)[0]
        selected = np.random.choice(class_indices, size=2, replace=False)
        selected_indices.extend(selected)

    # Tiempo para edges
    time_start = np.linspace(0, 0.5, EDGE_SAMPLES)
    time_end = np.linspace(DURATION - 0.5, DURATION, EDGE_SAMPLES)

    # Crear figura
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Verificación de Padding: Primeros y Últimos 0.5s (~50 muestras)',
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(n_samples, 2, figure=fig, hspace=0.4, wspace=0.3)

    for i, idx in enumerate(selected_indices):
        trace = X[idx]
        class_code = y[idx]
        class_name = CLASS_NAMES[class_code]
        color = CLASS_COLORS[class_code]

        # Extraer edges
        start_segment = trace[:EDGE_SAMPLES]
        end_segment = trace[-EDGE_SAMPLES:]

        # Subplot: Inicio (primeros 0.5s)
        ax_start = fig.add_subplot(gs[i, 0])
        ax_start.plot(time_start, start_segment, color=color, linewidth=2, marker='o', markersize=3)
        ax_start.set_title(f'Muestra {i+1} ({class_name}) - INICIO (0.0-0.5s)', fontsize=10)
        ax_start.set_xlabel('Tiempo (s)', fontsize=8)
        ax_start.set_ylabel('Amplitud', fontsize=8)
        ax_start.grid(True, alpha=0.3, linestyle='--')
        ax_start.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

        # Resaltar región si hay valores anómalos
        if np.any(np.abs(start_segment) > 1.0):
            ax_start.add_patch(Rectangle((0, -1.1), 0.5, 2.2, facecolor='red', alpha=0.2))
            ax_start.text(0.25, 0.9, '⚠ FUERA DE RANGO', transform=ax_start.transAxes,
                         color='red', fontweight='bold', ha='center')

        # Subplot: Final (últimos 0.5s)
        ax_end = fig.add_subplot(gs[i, 1])
        ax_end.plot(time_end, end_segment, color=color, linewidth=2, marker='o', markersize=3)
        ax_end.set_title(f'Muestra {i+1} ({class_name}) - FINAL (3.5-4.0s)', fontsize=10)
        ax_end.set_xlabel('Tiempo (s)', fontsize=8)
        ax_end.set_ylabel('Amplitud', fontsize=8)
        ax_end.grid(True, alpha=0.3, linestyle='--')
        ax_end.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

        # Resaltar región si hay valores anómalos
        if np.any(np.abs(end_segment) > 1.0):
            ax_end.add_patch(Rectangle((DURATION - 0.5, -1.1), 0.5, 2.2, facecolor='red', alpha=0.2))
            ax_end.text(0.25, 0.9, '⚠ FUERA DE RANGO', transform=ax_end.transAxes,
                       color='red', fontweight='bold', ha='center')

        # Estadísticas
        start_mean = np.mean(np.abs(start_segment))
        end_mean = np.mean(np.abs(end_segment))

        ax_start.text(0.02, 0.98, f'|μ|={start_mean:.3f}', transform=ax_start.transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax_end.text(0.02, 0.98, f'|μ|={end_mean:.3f}', transform=ax_end.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  - Gráfico guardado en {output_path}")
    else:
        plt.show()

    plt.close()


# ==================== ESTADÍSTICAS GENERALES ====================

def print_dataset_statistics(X: np.ndarray, y: np.ndarray, metadata_df: pd.DataFrame) -> None:
    """
    Imprime estadísticas descriptivas del dataset.

    Args:
        X: Array de datos (N, 400)
        y: Array de etiquetas (N,)
        metadata_df: DataFrame con metadata
    """
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL DATASET PREPROCESADO")
    print("="*60)

    print(f"\nDimensiones:")
    print(f"  - Número de muestras: {X.shape[0]}")
    print(f"  - Muestras por traza: {X.shape[1]}")
    print(f"  - Duración por traza: {DURATION} s")
    print(f"  - Sampling rate: {SAMPLING_RATE} Hz")

    print(f"\nDistribución de clases:")
    unique, counts = np.unique(y, return_counts=True)
    for class_code, count in zip(unique, counts):
        class_name = CLASS_NAMES[class_code]
        percentage = (count / len(y)) * 100
        print(f"  - {class_name:6s} (código {class_code}): {count:6d} muestras ({percentage:5.2f}%)")

    print(f"\nEstadísticas de amplitud global:")
    print(f"  - Media: {X.mean():.6f}")
    print(f"  - Desviación estándar: {X.std():.6f}")
    print(f"  - Mínimo: {X.min():.6f}")
    print(f"  - Máximo: {X.max():.6f}")
    print(f"  - Mediana: {np.median(X):.6f}")

    print(f"\nEstadísticas de amplitud por clase:")
    for class_code in [0, 1, 2]:
        class_name = CLASS_NAMES[class_code]
        class_indices = np.where(y == class_code)[0]
        class_data = X[class_indices]

        print(f"  {class_name}:")
        print(f"    - Media: {class_data.mean():.6f}")
        print(f"    - Std: {class_data.std():.6f}")
        print(f"    - Min: {class_data.min():.6f}")
        print(f"    - Max: {class_data.max():.6f}")
        print(f"    - |Amp| media: {np.mean(np.abs(class_data)):.6f}")

    # Verificar valores fuera de rango
    print(f"\nVerificación de rangos:")
    out_of_range = np.sum(np.abs(X) > 1.0 + 1e-6)
    if out_of_range > 0:
        print(f"  ⚠ ADVERTENCIA: {out_of_range} valores fuera de [-1, 1]")
    else:
        print(f"  ✓ Todos los valores están en el rango [-1, 1]")

    nan_count = np.sum(np.isnan(X))
    if nan_count > 0:
        print(f"  ⚠ ADVERTENCIA: {nan_count} valores NaN detectados")
    else:
        print(f"  ✓ No se detectaron valores NaN")

    inf_count = np.sum(np.isinf(X))
    if inf_count > 0:
        print(f"  ⚠ ADVERTENCIA: {inf_count} valores Inf detectados")
    else:
        print(f"  ✓ No se detectaron valores Inf")

    print("\n" + "="*60)


# ==================== MAIN ====================

def main():
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description='Validación visual del preprocesamiento de datos para GPD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python validate_preprocessing.py -i /data/processed -o /data/validation_plots

Este script genera tres tipos de visualizaciones:
  a) Trazas aleatorias por clase (5 por clase)
  b) Histogramas de amplitud por clase
  c) Verificación de padding (primeros y últimos 0.5s)

Input esperado en el directorio:
  - X.npy: Array (N, 400) con datos normalizados
  - y.npy: Array (N,) con etiquetas
  - metadata.csv: CSV con filename, label, label_name
        """
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='Directorio con X.npy, y.npy, metadata.csv'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Directorio donde guardar los gráficos (opcional, si no se especifica se muestran en pantalla)'
    )

    parser.add_argument(
        '-n', '--n-traces',
        type=int,
        default=5,
        help='Número de trazas aleatorias a plotear por clase (default: 5)'
    )

    args = parser.parse_args()

    # Convertir a Path
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"ERROR: El directorio de entrada no existe: {input_dir}")
        sys.exit(1)

    # Cargar datos
    X, y, metadata_df = load_dataset(input_dir)

    # Imprimir estadísticas
    print_dataset_statistics(X, y, metadata_df)

    # Preparar directorio de salida si se especificó
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nLos gráficos se guardarán en: {output_dir}")
    else:
        print(f"\nLos gráficos se mostrarán en pantalla (no se guardarán)")

    # a) Trazas aleatorias por clase
    traces_output = output_dir / 'traces_per_class.png' if output_dir else None
    plot_random_traces_per_class(X, y, n_traces=args.n_traces, output_path=traces_output)

    # b) Histogramas de amplitud
    hist_output = output_dir / 'amplitude_histograms.png' if output_dir else None
    plot_amplitude_histograms(X, y, output_path=hist_output)

    # c) Verificación de padding
    padding_output = output_dir / 'padding_verification.png' if output_dir else None
    plot_edge_verification(X, y, n_samples=6, output_path=padding_output)

    print("\n¡Validación visual completada!")
    if output_dir:
        print(f"Los gráficos han sido guardados en: {output_dir}")


if __name__ == '__main__':
    main()
