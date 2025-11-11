#!/usr/bin/env python
"""
Visualización de detección de fases P y S para un evento sísmico único
Muestra la traza sísmica y las probabilidades P/S predichas por el modelo GPD

Uso:
python gpd_plot_evaluation_event.py ruta/al/archivo.mseed
python gpd_plot_evaluation_event.py ruta/al/archivo.mseed --min-proba-p 0.55 --min-proba-s 0.85
python gpd_plot_evaluation_event.py ruta/al/archivo.mseed --save figura.png
"""

import numpy as np
import obspy.core as oc
from tensorflow.keras.models import load_model
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from obspy.signal.trigger import trigger_onset

# Add config module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
import config

# =================== CONFIGURACIÓN ===================
# Directorios por defecto
DEFAULT_MSEED_DIR = "/home/rsa/projects/gpd/data/processed/mseed/events/test_1000"
DEFAULT_OUTPUT_DIR = "/home/rsa/projects/gpd/data/analysis/plots/gpd-evaluation-event"

# Parámetros del modelo
DEFAULT_MIN_PROBA_P = 0.55
DEFAULT_MIN_PROBA_S = 0.85
n_shift = 10
batch_size = 100
half_dur = 2.00
only_dt = 0.01
n_win = int(half_dur/only_dt)
n_feat = 2*n_win

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """Función sliding window para crear ventanas deslizantes"""
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")
    if stepsize < 1:
        raise ValueError("Stepsize may not be zero or negative")
    if size > data.shape[axis]:
        raise ValueError("Sliding window size may not exceed size of selected axis")

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return strided.copy() if copy else strided

def process_and_predict(model, mseed_file, min_proba_P, min_proba_S, verbose=False):
    """
    Procesa un archivo mseed y retorna toda la información necesaria para graficar

    Returns:
    --------
    dict con:
        'st': Stream de ObsPy con las trazas
        'tt': array de tiempos (en segundos desde inicio)
        'prob_P': array de probabilidades P
        'prob_S': array de probabilidades S
        'p_detections': lista de tuplas (tiempo_absoluto, probabilidad, índice)
        's_detections': lista de tuplas (tiempo_absoluto, probabilidad, índice)
        'best_p': tupla (tiempo_absoluto, probabilidad, índice) o None
        'best_s': tupla (tiempo_absoluto, probabilidad, índice) o None
        'start_time': UTCDateTime del inicio
        'dt': delta de muestreo
    """

    # Cargar archivo mseed
    st = oc.read(mseed_file)

    # Verificar que tenemos 3 trazas
    if len(st) != 3:
        raise ValueError(f"Se esperaban 3 trazas, pero se encontraron {len(st)}")

    # Extraer información básica
    dt = st[0].stats.delta
    start_time = st[0].stats.starttime

    if verbose:
        print(f"Archivo: {mseed_file}")
        print(f"Inicio: {start_time}")
        print(f"Delta: {dt} s")
        print(f"Duración: {len(st[0].data) * dt:.2f} s")
        print(f"Estación: {st[0].stats.station}")

    # Crear ventanas deslizantes
    data_length = len(st[0].data)
    tt = (np.arange(0, data_length, n_shift) + n_win) * dt

    sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)
    sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)
    sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)

    # Verificar que todas las ventanas tienen el mismo tamaño
    min_windows = min(sliding_N.shape[0], sliding_E.shape[0], sliding_Z.shape[0])
    if min_windows == 0:
        raise ValueError("No se pudieron crear ventanas deslizantes")

    # Apilar ventanas
    tr_win = np.zeros((min_windows, n_feat, 3), dtype=np.float32)
    tr_win[:,:,0] = sliding_N[:min_windows]
    tr_win[:,:,1] = sliding_E[:min_windows]
    tr_win[:,:,2] = sliding_Z[:min_windows]

    # Normalizar por canal (por ventana)
    max_vals = np.max(np.abs(tr_win), axis=1, keepdims=True) + 1e-9
    tr_win = tr_win / max_vals

    tt = tt[:min_windows]

    # Predicción
    if verbose:
        print(f"Realizando predicción sobre {min_windows} ventanas...")

    ts = model.predict(tr_win, verbose=False, batch_size=batch_size)

    prob_P = ts[:,0]
    prob_S = ts[:,1]

    # Detectar picks P
    trigs_p = trigger_onset(prob_P, min_proba_P, 0.1)

    p_detections = []
    for trig in trigs_p:
        if trig[1] == trig[0]:
            continue
        pick_idx = np.argmax(ts[trig[0]:trig[1], 0]) + trig[0]
        pick_time = start_time + tt[pick_idx]
        pick_prob = ts[pick_idx, 0]
        p_detections.append((pick_time, pick_prob, pick_idx))

    # Detectar picks S
    trigs_s = trigger_onset(prob_S, min_proba_S, 0.1)

    s_detections = []
    for trig in trigs_s:
        if trig[1] == trig[0]:
            continue
        pick_idx = np.argmax(ts[trig[0]:trig[1], 1]) + trig[0]
        pick_time = start_time + tt[pick_idx]
        pick_prob = ts[pick_idx, 1]
        s_detections.append((pick_time, pick_prob, pick_idx))

    # Seleccionar mejores picks
    best_p = None
    if len(p_detections) > 0:
        p_detections.sort(key=lambda x: x[1], reverse=True)
        best_p = p_detections[0]

    best_s = None
    if len(s_detections) > 0:
        s_detections.sort(key=lambda x: x[1], reverse=True)
        best_s = s_detections[0]

    if verbose:
        print(f"\nDetecciones:")
        print(f"  Fases P: {len(p_detections)}")
        if best_p:
            print(f"    Mejor P: {best_p[0]} (prob={best_p[1]:.3f})")
        print(f"  Fases S: {len(s_detections)}")
        if best_s:
            print(f"    Mejor S: {best_s[0]} (prob={best_s[1]:.3f})")

    return {
        'st': st,
        'tt': tt,
        'prob_P': prob_P,
        'prob_S': prob_S,
        'p_detections': p_detections,
        's_detections': s_detections,
        'best_p': best_p,
        'best_s': best_s,
        'start_time': start_time,
        'dt': dt
    }

def plot_event_detection(result, min_proba_P, min_proba_S, output_file=None):
    """
    Crea un gráfico de dos paneles mostrando:
    - Panel superior: traza sísmica (primera componente)
    - Panel inferior: probabilidades P y S con umbrales y picks
    """

    st = result['st']
    tt = result['tt']
    prob_P = result['prob_P']
    prob_S = result['prob_S']
    p_detections = result['p_detections']
    s_detections = result['s_detections']
    best_p = result['best_p']
    best_s = result['best_s']
    start_time = result['start_time']
    dt = result['dt']

    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={'height_ratios': [1, 1.5]})

    # ============= PANEL SUPERIOR: TRAZA SÍSMICA =============
    trace = st[0]
    times_trace = np.arange(len(trace.data)) * dt

    ax1.plot(times_trace, trace.data, 'k-', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Amplitud', fontsize=11, fontweight='bold')
    ax1.set_title(f'Evento Sísmico - {trace.stats.station} - {trace.stats.channel} - {start_time}',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, times_trace[-1])

    # Marcar picks en la traza
    if best_p:
        pick_time_rel = best_p[2] * n_shift * dt + n_win * dt
        ax1.axvline(pick_time_rel, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'P pick (prob={best_p[1]:.3f})')

    if best_s:
        pick_time_rel = best_s[2] * n_shift * dt + n_win * dt
        ax1.axvline(pick_time_rel, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'S pick (prob={best_s[1]:.3f})')

    if best_p or best_s:
        ax1.legend(loc='upper right', fontsize=9)

    # ============= PANEL INFERIOR: PROBABILIDADES =============
    # Graficar curvas de probabilidad
    ax2.plot(tt, prob_P, 'r-', linewidth=1.5, label='Probabilidad P', alpha=0.8)
    ax2.plot(tt, prob_S, 'b-', linewidth=1.5, label='Probabilidad S', alpha=0.8)

    # Líneas de umbral
    ax2.axhline(min_proba_P, color='red', linestyle=':', linewidth=2,
               alpha=0.6, label=f'Umbral P = {min_proba_P}')
    ax2.axhline(min_proba_S, color='blue', linestyle=':', linewidth=2,
               alpha=0.6, label=f'Umbral S = {min_proba_S}')

    # Marcar TODAS las detecciones (no solo la mejor)
    for i, (pick_time, pick_prob, pick_idx) in enumerate(p_detections):
        pick_time_rel = tt[pick_idx]
        if i == 0 and best_p and pick_idx == best_p[2]:
            # Mejor pick P - marcador más grande
            ax2.plot(pick_time_rel, pick_prob, 'r*', markersize=20,
                    markeredgecolor='darkred', markeredgewidth=1.5,
                    zorder=10)
        else:
            # Otros picks P
            ax2.plot(pick_time_rel, pick_prob, 'ro', markersize=8,
                    markeredgecolor='darkred', markeredgewidth=1,
                    alpha=0.6, zorder=5)

    for i, (pick_time, pick_prob, pick_idx) in enumerate(s_detections):
        pick_time_rel = tt[pick_idx]
        if i == 0 and best_s and pick_idx == best_s[2]:
            # Mejor pick S - marcador más grande
            ax2.plot(pick_time_rel, pick_prob, 'b*', markersize=20,
                    markeredgecolor='darkblue', markeredgewidth=1.5,
                    zorder=10)
        else:
            # Otros picks S
            ax2.plot(pick_time_rel, pick_prob, 'bo', markersize=8,
                    markeredgecolor='darkblue', markeredgewidth=1,
                    alpha=0.6, zorder=5)

    ax2.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probabilidad', fontsize=11, fontweight='bold')
    ax2.set_title('Probabilidades de Fase P y S del Modelo GPD',
                  fontsize=11, fontweight='bold', pad=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(0, tt[-1])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)

    # Información de detecciones
    info_text = f'Detecciones: {len(p_detections)} P, {len(s_detections)} S'
    ax2.text(0.02, 0.02, info_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
        description='Visualización de detección de fases P y S para un evento único',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Procesar archivo desde directorio por defecto
  python gpd_plot_evaluation_event.py --input evento.mseed --output evento.png

  # Con umbrales personalizados
  python gpd_plot_evaluation_event.py --input evento.mseed --output evento.png --min-proba-p 0.6 --min-proba-s 0.9

  # Usar ruta absoluta
  python gpd_plot_evaluation_event.py --input /ruta/completa/evento.mseed --output /ruta/figura.png

  # Modelo específico
  python gpd_plot_evaluation_event.py --input evento.mseed --output evento.png --model-path gpd_v1.keras
        """)

    parser.add_argument('--input', '-i', type=str, required=True,
                       help=f'Nombre del archivo mseed (se busca en {DEFAULT_MSEED_DIR}) o ruta absoluta')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help=f'Nombre del archivo de salida (se guarda en {DEFAULT_OUTPUT_DIR}) o ruta absoluta')
    parser.add_argument('--min-proba-p', type=float, default=DEFAULT_MIN_PROBA_P,
                       help=f'Umbral de probabilidad para fases P (default: {DEFAULT_MIN_PROBA_P})')
    parser.add_argument('--min-proba-s', type=float, default=DEFAULT_MIN_PROBA_S,
                       help=f'Umbral de probabilidad para fases S (default: {DEFAULT_MIN_PROBA_S})')
    parser.add_argument('--model-path', type=str, default=None,
                       help=f'Nombre del modelo GPD (default: {config.get_default_model_name()}). '
                            f'Se busca en {config.get_models_dir()}')
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

    # Resolver la ruta del modelo
    if args.model_path is None:
        model_path = config.get_default_model_path()
        model_name = config.get_default_model_name()
    else:
        model_name = args.model_path
        model_path = config.get_models_dir() / model_name

    print("=== Visualización de Detección de Fases GPD ===")
    print(f"Archivo MSEED entrada: {mseed_file}")
    print(f"Archivo figura salida: {output_file}")
    print(f"Umbral P: {args.min_proba_p}")
    print(f"Umbral S: {args.min_proba_s}")
    print(f"Modelo: {model_name}")

    # Verificar archivo mseed
    if not mseed_file.is_file():
        print(f"ERROR: Archivo no encontrado: {mseed_file}")
        return 1

    # Verificar modelo
    if not model_path.is_file():
        print(f"ERROR: Modelo no encontrado: {model_path}")
        print(f"Modelos disponibles en {config.get_models_dir()}:")
        try:
            available_models = list(config.get_models_dir().glob('*.keras')) + \
                             list(config.get_models_dir().glob('*.hdf5')) + \
                             list(config.get_models_dir().glob('*.h5'))
            for model_file in available_models:
                print(f"  - {model_file.name}")
        except:
            print("  No se pudo listar los modelos disponibles")
        return 1

    # Cargar modelo
    print("\nCargando modelo GPD...")
    try:
        model = load_model(model_path, compile=False)
        if args.verbose:
            print(f"OK: Modelo cargado - input_shape={model.input_shape}, output_shape={model.output_shape}")
    except Exception as e:
        print(f"ERROR cargando modelo: {e}")
        return 1

    # Procesar evento
    print("\nProcesando evento...")
    try:
        result = process_and_predict(model, str(mseed_file),
                                     args.min_proba_p, args.min_proba_s,
                                     args.verbose)
    except Exception as e:
        print(f"ERROR procesando evento: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Crear gráfico
    print("\nGenerando gráfico...")
    try:
        plot_event_detection(result, args.min_proba_p, args.min_proba_s, str(output_file))
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
