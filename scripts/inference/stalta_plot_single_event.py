#!/usr/bin/env python
"""
Visualización de detección de fases P y S usando STA/LTA para un evento sísmico único
Muestra la traza sísmica y las funciones características STA/LTA para P y S

Uso:
python stalta_plot_single_event.py --input evento.mseed --output evento.png
python stalta_plot_single_event.py --input evento.mseed --output evento.png --threshold-p-on 3.0 --threshold-s-on 2.5
"""

import numpy as np
import obspy.core as oc
from obspy.signal.trigger import trigger_onset, classic_sta_lta
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add config module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
import config

# =================== CONFIGURACIÓN ===================
# Directorios por defecto
DEFAULT_MSEED_DIR = "/home/rsa/projects/gpd/data/processed/mseed/events/test_1000"
DEFAULT_OUTPUT_DIR = "/home/rsa/projects/gpd/data/analysis/plots/stalta-evaluation-event"

# Parámetros STA/LTA por defecto
DEFAULT_THRESHOLD_P_ON = 3.0
DEFAULT_THRESHOLD_P_OFF = 1.5
DEFAULT_THRESHOLD_S_ON = 2.5
DEFAULT_THRESHOLD_S_OFF = 1.2

# Ventanas STA/LTA (en segundos)
DEFAULT_STA_P = 0.5
DEFAULT_LTA_P = 10.0
DEFAULT_STA_S = 1.0
DEFAULT_LTA_S = 20.0

# Gap mínimo P-S
DEFAULT_GAP_MIN = 0.5

def compute_sta_lta_trace(data, sampling_rate, sta_len, lta_len):
    """
    Calcula la función característica STA/LTA para una traza

    Returns:
    --------
    array: Valores STA/LTA para cada muestra
    """
    nsta = int(sta_len * sampling_rate)
    nlta = int(lta_len * sampling_rate)
    sta_lta = classic_sta_lta(data, nsta, nlta)
    return sta_lta

def detect_picks_stalta(sta_lta, threshold_on, threshold_off, sampling_rate, start_time):
    """
    Detecta picks en la función STA/LTA

    Returns:
    --------
    list: Lista de tuplas (tiempo_absoluto, valor_sta_lta, índice)
    """
    triggers = trigger_onset(sta_lta, threshold_on, threshold_off)

    picks = []
    for trig in triggers:
        if trig[1] == trig[0]:
            continue

        # Buscar el máximo dentro del trigger
        trig_start, trig_end = trig[0], trig[1]
        max_idx = np.argmax(sta_lta[trig_start:trig_end]) + trig_start

        # Convertir índice a tiempo
        pick_time = start_time + max_idx / sampling_rate
        max_value = sta_lta[max_idx]

        picks.append((pick_time, max_value, max_idx))

    return picks

def process_and_detect(mseed_file,
                      threshold_p_on, threshold_p_off, sta_p, lta_p,
                      threshold_s_on, threshold_s_off, sta_s, lta_s,
                      gap_min=0.5, verbose=False):
    """
    Procesa un archivo mseed y retorna toda la información necesaria para graficar

    Returns:
    --------
    dict con:
        'st': Stream de ObsPy con las trazas
        'times': array de tiempos (en segundos desde inicio)
        'sta_lta_p': array de valores STA/LTA para P
        'sta_lta_s': array de valores STA/LTA para S
        'p_detections': lista de tuplas (tiempo_absoluto, valor, índice)
        's_detections': lista de tuplas (tiempo_absoluto, valor, índice)
        's_detections_filtered': lista después de aplicar gap mínimo
        'best_p': tupla (tiempo_absoluto, valor, índice) o None
        'best_s': tupla (tiempo_absoluto, valor, índice) o None
        'start_time': UTCDateTime del inicio
        'sampling_rate': frecuencia de muestreo
        'data_z': datos componente Z (para P)
        'data_h': datos componente horizontal (para S)
        'component_s': nombre de componente usada para S ('N' o 'E')
    """

    # Cargar archivo mseed
    st = oc.read(mseed_file)

    # Verificar que tenemos 3 trazas
    if len(st) != 3:
        raise ValueError(f"Se esperaban 3 trazas, pero se encontraron {len(st)}")

    # Obtener información básica
    sampling_rate = st[0].stats.sampling_rate
    start_time = st[0].stats.starttime

    if verbose:
        print(f"Archivo: {mseed_file}")
        print(f"Inicio: {start_time}")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Duración: {len(st[0].data) / sampling_rate:.2f} s")
        print(f"Estación: {st[0].stats.station}")

    # Preparar datos - componentes
    data_z = st[2].data  # Componente vertical para P
    data_n = st[0].data  # Componente norte
    data_e = st[1].data  # Componente este

    # Para S, usar la componente horizontal con mayor energía
    energy_n = np.sum(data_n**2)
    energy_e = np.sum(data_e**2)

    if energy_n >= energy_e:
        data_h = data_n
        component_s = 'N'
    else:
        data_h = data_e
        component_s = 'E'

    if verbose:
        print(f"Componente seleccionada para S: {component_s} (energía N={energy_n:.2e}, E={energy_e:.2e})")

    # Calcular STA/LTA para P (componente Z)
    sta_lta_p = compute_sta_lta_trace(data_z, sampling_rate, sta_p, lta_p)

    # Calcular STA/LTA para S (componente horizontal)
    sta_lta_s = compute_sta_lta_trace(data_h, sampling_rate, sta_s, lta_s)

    # Array de tiempos
    times = np.arange(len(data_z)) / sampling_rate

    # Detectar picks P
    p_detections = detect_picks_stalta(sta_lta_p, threshold_p_on, threshold_p_off,
                                       sampling_rate, start_time)

    # Seleccionar primera P (criterio temporal)
    best_p = None
    t_p = None
    if len(p_detections) > 0:
        p_detections.sort(key=lambda x: x[0])  # Ordenar por tiempo
        best_p = p_detections[0]
        t_p = best_p[0]
        if verbose:
            print(f"\nDetecciones P: {len(p_detections)}")
            print(f"Primera P seleccionada: t={t_p}, STA/LTA={best_p[1]:.3f}")

    # Detectar picks S
    s_detections = detect_picks_stalta(sta_lta_s, threshold_s_on, threshold_s_off,
                                       sampling_rate, start_time)

    # Aplicar filtro de gap mínimo si hay P detectada
    s_detections_filtered = s_detections.copy()
    if t_p is not None:
        t_min_s = t_p + gap_min
        s_detections_filtered = [s for s in s_detections if s[0] >= t_min_s]
        if verbose and len(s_detections) != len(s_detections_filtered):
            print(f"\nDetecciones S: {len(s_detections)}")
            print(f"S después de filtro gap: {len(s_detections_filtered)}")

    # Seleccionar mejor S (mayor STA/LTA después de gap)
    best_s = None
    if len(s_detections_filtered) > 0:
        s_detections_filtered.sort(key=lambda x: x[1], reverse=True)  # Por valor STA/LTA
        best_s = s_detections_filtered[0]
        if verbose:
            print(f"Mejor S seleccionada: t={best_s[0]}, STA/LTA={best_s[1]:.3f}")

    return {
        'st': st,
        'times': times,
        'sta_lta_p': sta_lta_p,
        'sta_lta_s': sta_lta_s,
        'p_detections': p_detections,
        's_detections': s_detections,
        's_detections_filtered': s_detections_filtered,
        'best_p': best_p,
        'best_s': best_s,
        'start_time': start_time,
        'sampling_rate': sampling_rate,
        'data_z': data_z,
        'data_h': data_h,
        'component_s': component_s
    }

def plot_stalta_detection(result, threshold_p_on, threshold_p_off,
                         threshold_s_on, threshold_s_off,
                         sta_p, lta_p, sta_s, lta_s, gap_min, output_file=None):
    """
    Crea un gráfico de dos paneles mostrando:
    - Panel superior: traza sísmica (componente Z para P)
    - Panel inferior: funciones STA/LTA para P y S con umbrales y picks
    """

    st = result['st']
    times = result['times']
    sta_lta_p = result['sta_lta_p']
    sta_lta_s = result['sta_lta_s']
    p_detections = result['p_detections']
    s_detections = result['s_detections']
    s_detections_filtered = result['s_detections_filtered']
    best_p = result['best_p']
    best_s = result['best_s']
    start_time = result['start_time']
    sampling_rate = result['sampling_rate']
    data_z = result['data_z']
    component_s = result['component_s']

    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={'height_ratios': [1, 1.5]})

    # ============= PANEL SUPERIOR: TRAZA SÍSMICA (Z) =============
    trace = st[2]  # Componente Z

    ax1.plot(times, data_z, 'k-', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Amplitud', fontsize=11, fontweight='bold')
    ax1.set_title(f'Evento Sísmico - {trace.stats.station} - {trace.stats.channel} - {start_time}',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, times[-1])

    # Marcar picks en la traza
    if best_p:
        pick_time_rel = best_p[2] / sampling_rate
        ax1.axvline(pick_time_rel, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'P pick (STA/LTA={best_p[1]:.2f})')

    if best_s:
        pick_time_rel = best_s[2] / sampling_rate
        ax1.axvline(pick_time_rel, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'S pick (STA/LTA={best_s[1]:.2f})')

    if best_p or best_s:
        ax1.legend(loc='upper right', fontsize=9)

    # ============= PANEL INFERIOR: FUNCIONES STA/LTA =============
    # Graficar curvas STA/LTA
    ax2.plot(times, sta_lta_p, 'r-', linewidth=1.5, label='STA/LTA P (comp. Z)', alpha=0.8)
    ax2.plot(times, sta_lta_s, 'b-', linewidth=1.5, label=f'STA/LTA S (comp. {component_s})', alpha=0.8)

    # Líneas de umbral
    ax2.axhline(threshold_p_on, color='red', linestyle=':', linewidth=2,
               alpha=0.6, label=f'Umbral P ON = {threshold_p_on}')
    ax2.axhline(threshold_p_off, color='red', linestyle='--', linewidth=1,
               alpha=0.4, label=f'Umbral P OFF = {threshold_p_off}')
    ax2.axhline(threshold_s_on, color='blue', linestyle=':', linewidth=2,
               alpha=0.6, label=f'Umbral S ON = {threshold_s_on}')
    ax2.axhline(threshold_s_off, color='blue', linestyle='--', linewidth=1,
               alpha=0.4, label=f'Umbral S OFF = {threshold_s_off}')

    # Marcar TODAS las detecciones P
    for i, (pick_time, pick_val, pick_idx) in enumerate(p_detections):
        pick_time_rel = pick_idx / sampling_rate
        if i == 0 and best_p and pick_idx == best_p[2]:
            # Primera P (mejor pick) - marcador más grande
            ax2.plot(pick_time_rel, pick_val, 'r*', markersize=20,
                    markeredgecolor='darkred', markeredgewidth=1.5,
                    zorder=10)
        else:
            # Otras detecciones P
            ax2.plot(pick_time_rel, pick_val, 'ro', markersize=8,
                    markeredgecolor='darkred', markeredgewidth=1,
                    alpha=0.6, zorder=5)

    # Marcar detecciones S (mostrar todas, pero distinguir filtradas)
    for pick_time, pick_val, pick_idx in s_detections:
        pick_time_rel = pick_idx / sampling_rate
        is_filtered = (pick_time, pick_val, pick_idx) in s_detections_filtered

        if is_filtered and best_s and pick_idx == best_s[2]:
            # Mejor S - marcador más grande
            ax2.plot(pick_time_rel, pick_val, 'b*', markersize=20,
                    markeredgecolor='darkblue', markeredgewidth=1.5,
                    zorder=10)
        elif is_filtered:
            # Otras S válidas (después de gap)
            ax2.plot(pick_time_rel, pick_val, 'bo', markersize=8,
                    markeredgecolor='darkblue', markeredgewidth=1,
                    alpha=0.6, zorder=5)
        else:
            # S descartadas por gap (marcador gris)
            ax2.plot(pick_time_rel, pick_val, 'o', color='gray', markersize=6,
                    markeredgecolor='dimgray', markeredgewidth=1,
                    alpha=0.4, zorder=3)

    # Marcar zona de gap mínimo si hay P
    if best_p:
        gap_start = best_p[2] / sampling_rate
        gap_end = gap_start + gap_min
        ax2.axvspan(gap_start, min(gap_end, times[-1]), alpha=0.1, color='red',
                   label=f'Gap mínimo P-S ({gap_min}s)')

    ax2.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Ratio STA/LTA', fontsize=11, fontweight='bold')
    ax2.set_title('Funciones Características STA/LTA para Detección de Fases',
                  fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlim(0, times[-1])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)

    # Calcular y mostrar límite y del panel inferior
    max_val = max(np.max(sta_lta_p), np.max(sta_lta_s))
    ax2.set_ylim(-0.5, max(max_val * 1.1, max(threshold_p_on, threshold_s_on) * 1.2))

    # Información de detecciones
    num_p_total = len(p_detections)
    num_s_total = len(s_detections)
    num_s_valid = len(s_detections_filtered)
    info_text = f'Detecciones: {num_p_total} P, {num_s_total} S ({num_s_valid} válidas)'
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Parámetros STA/LTA
    params_text = f'P: STA={sta_p}s, LTA={lta_p}s\nS: STA={sta_s}s, LTA={lta_s}s'
    ax2.text(0.98, 0.98, params_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Guardar o mostrar
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {output_file}")
    else:
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Visualización de detección de fases P y S usando STA/LTA para un evento único',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Procesar archivo desde directorio por defecto
  python stalta_plot_single_event.py --input evento.mseed --output evento.png

  # Con umbrales personalizados
  python stalta_plot_single_event.py --input evento.mseed --output evento.png --threshold-p-on 3.5 --threshold-s-on 2.8

  # Usar ruta absoluta
  python stalta_plot_single_event.py --input /ruta/completa/evento.mseed --output /ruta/figura.png

  # Ventanas personalizadas
  python stalta_plot_single_event.py --input evento.mseed --output evento.png --sta-p 0.3 --lta-p 15.0
        """)

    parser.add_argument('--input', '-i', type=str, required=True,
                       help=f'Nombre del archivo mseed (se busca en {DEFAULT_MSEED_DIR}) o ruta absoluta')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help=f'Nombre del archivo de salida (se guarda en {DEFAULT_OUTPUT_DIR}) o ruta absoluta')

    # Parámetros P
    parser.add_argument('--threshold-p-on', type=float, default=DEFAULT_THRESHOLD_P_ON,
                       help=f'Umbral activación P (default: {DEFAULT_THRESHOLD_P_ON})')
    parser.add_argument('--threshold-p-off', type=float, default=DEFAULT_THRESHOLD_P_OFF,
                       help=f'Umbral desactivación P (default: {DEFAULT_THRESHOLD_P_OFF})')
    parser.add_argument('--sta-p', type=float, default=DEFAULT_STA_P,
                       help=f'Ventana STA para P en segundos (default: {DEFAULT_STA_P})')
    parser.add_argument('--lta-p', type=float, default=DEFAULT_LTA_P,
                       help=f'Ventana LTA para P en segundos (default: {DEFAULT_LTA_P})')

    # Parámetros S
    parser.add_argument('--threshold-s-on', type=float, default=DEFAULT_THRESHOLD_S_ON,
                       help=f'Umbral activación S (default: {DEFAULT_THRESHOLD_S_ON})')
    parser.add_argument('--threshold-s-off', type=float, default=DEFAULT_THRESHOLD_S_OFF,
                       help=f'Umbral desactivación S (default: {DEFAULT_THRESHOLD_S_OFF})')
    parser.add_argument('--sta-s', type=float, default=DEFAULT_STA_S,
                       help=f'Ventana STA para S en segundos (default: {DEFAULT_STA_S})')
    parser.add_argument('--lta-s', type=float, default=DEFAULT_LTA_S,
                       help=f'Ventana LTA para S en segundos (default: {DEFAULT_LTA_S})')

    # Parámetros físicos
    parser.add_argument('--gap-min', type=float, default=DEFAULT_GAP_MIN,
                       help=f'Gap mínimo P-S en segundos (default: {DEFAULT_GAP_MIN}s)')

    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada')

    args = parser.parse_args()

    # Resolver la ruta del archivo mseed de entrada
    input_path = Path(args.input)
    if not input_path.is_absolute():
        # Si no es ruta absoluta, buscar en directorio por defecto
        mseed_file = Path(DEFAULT_MSEED_DIR) / args.input
    else:
        mseed_file = input_path

    # Resolver la ruta del archivo de salida
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # Si no es ruta absoluta, guardar en directorio por defecto
        output_file = Path(DEFAULT_OUTPUT_DIR) / args.output
        # Crear directorio si no existe
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = output_path
        # Crear directorio si no existe
        output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=== Visualización de Detección de Fases STA/LTA ===")
    print(f"Archivo MSEED entrada: {mseed_file}")
    print(f"Archivo figura salida: {output_file}")
    print(f"Umbrales P: ON={args.threshold_p_on}, OFF={args.threshold_p_off}")
    print(f"Umbrales S: ON={args.threshold_s_on}, OFF={args.threshold_s_off}")
    print(f"Ventanas P: STA={args.sta_p}s, LTA={args.lta_p}s")
    print(f"Ventanas S: STA={args.sta_s}s, LTA={args.lta_s}s")
    print(f"Gap mínimo: {args.gap_min}s")

    # Verificar archivo mseed
    if not mseed_file.is_file():
        print(f"ERROR: Archivo no encontrado: {mseed_file}")
        return 1

    # Procesar evento
    print("\nProcesando evento...")
    try:
        result = process_and_detect(str(mseed_file),
                                   args.threshold_p_on, args.threshold_p_off,
                                   args.sta_p, args.lta_p,
                                   args.threshold_s_on, args.threshold_s_off,
                                   args.sta_s, args.lta_s,
                                   args.gap_min, args.verbose)
    except Exception as e:
        print(f"ERROR procesando evento: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Crear gráfico
    print("\nGenerando gráfico...")
    try:
        plot_stalta_detection(result,
                            args.threshold_p_on, args.threshold_p_off,
                            args.threshold_s_on, args.threshold_s_off,
                            args.sta_p, args.lta_p, args.sta_s, args.lta_s,
                            args.gap_min, str(output_file))
    except Exception as e:
        print(f"ERROR generando gráfico: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nProceso completado exitosamente")
    print(f"Figura guardada en: {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
