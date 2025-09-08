#!/usr/bin/env python
"""
Evaluación de eventos sísmicos usando modelo GPD
Procesa archivos mseed preprocesados y evalúa detecciones contra picks manuales

Uso: 
python evaluate_gpd_events.py -V
python evaluate_gpd_events.py --stations CHAI LABR -V
"""

import numpy as np
import obspy.core as oc
from tensorflow.keras.models import load_model
import pandas as pd
import argparse
import os
import gc
from collections import defaultdict

# =================== CONFIGURACIÓN ===================
# Rutas por defecto
MSEED_DIR = "/home/rsa/projects/gpd/data/mseed/test_100/"
CSV_INPUT = "/home/rsa/projects/gpd/data/dataset/dataset_estratificado_100.csv"
CSV_OUTPUT = "/home/rsa/projects/gpd/data/out/resultados_evaluacion_100.csv"

# Estaciones por defecto
DEFAULT_STATIONS = ['LABR', 'CUSH', 'CHAI', 'UVER', 'PORT']

# Parámetros del modelo GPD
min_proba = 0.95
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

def process_event_file(model, mseed_file, verbose=False):
    """
    Procesa un archivo de evento sísmico y devuelve las detecciones
    
    Returns:
    --------
    dict con:
        'num_p': número de fases P detectadas
        'num_s': número de fases S detectadas  
        't_p': tiempo de fase P (o None si != 1)
        't_s': tiempo de fase S (o None si != 1)
        'prob_p': probabilidad de fase P (o None si != 1)
        'prob_s': probabilidad de fase S (o None si != 1)
    """
    
    result = {
        'num_p': 0, 'num_s': 0,
        't_p': None, 't_s': None, 
        'prob_p': None, 'prob_s': None
    }
    
    try:
        # Cargar archivo mseed (sin preprocesamiento adicional)
        st = oc.read(mseed_file)
        
        # Verificar que tenemos 3 trazas
        if len(st) != 3:
            if verbose:
                print(f"    ERROR: {len(st)} trazas encontradas, se esperaban 3")
            return result
        
        # Los datos ya están preprocesados, solo extraemos la información básica
        dt = st[0].stats.delta
        start_time = st[0].stats.starttime
        
        # Crear ventanas deslizantes
        data_length = len(st[0].data)
        tt = (np.arange(0, data_length, n_shift) + n_win) * dt
        
        sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)
        sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)
        sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)
        
        # Verificar que todas las ventanas tienen el mismo tamaño
        min_windows = min(sliding_N.shape[0], sliding_E.shape[0], sliding_Z.shape[0])
        if min_windows == 0:
            return result
            
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
        ts = model.predict(tr_win, verbose=False, batch_size=batch_size)
        
        prob_P = ts[:,0]
        prob_S = ts[:,1]
        
        # Detectar picks P
        from obspy.signal.trigger import trigger_onset
        trigs_p = trigger_onset(prob_P, min_proba, 0.1)
        
        p_detections = []
        for trig in trigs_p:
            if trig[1] == trig[0]:
                continue
            pick_idx = np.argmax(ts[trig[0]:trig[1], 0]) + trig[0]
            pick_time = start_time + tt[pick_idx]
            pick_prob = ts[pick_idx, 0]
            p_detections.append((pick_time, pick_prob))
        
        # Detectar picks S
        trigs_s = trigger_onset(prob_S, min_proba, 0.1)
        
        s_detections = []
        for trig in trigs_s:
            if trig[1] == trig[0]:
                continue
            pick_idx = np.argmax(ts[trig[0]:trig[1], 1]) + trig[0]
            pick_time = start_time + tt[pick_idx]
            pick_prob = ts[pick_idx, 1]
            s_detections.append((pick_time, pick_prob))
        
        # Llenar resultados
        result['num_p'] = len(p_detections)
        result['num_s'] = len(s_detections)
        
        # Solo llenar tiempos y probabilidades si hay exactamente 1 detección
        if result['num_p'] == 1:
            result['t_p'] = p_detections[0][0].isoformat()
            result['prob_p'] = float(p_detections[0][1])
        
        if result['num_s'] == 1:
            result['t_s'] = s_detections[0][0].isoformat()
            result['prob_s'] = float(s_detections[0][1])
        
        # Limpiar memoria
        del tr_win, ts, sliding_N, sliding_E, sliding_Z
        gc.collect()
        
    except Exception as e:
        if verbose:
            print(f"    ERROR procesando {mseed_file}: {e}")
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description='Evaluación de eventos sísmicos con modelo GPD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Evaluación estándar con todas las estaciones por defecto
  python evaluate_gpd_events.py -V
  
  # Solo evaluar estaciones específicas
  python evaluate_gpd_events.py --stations CHAI LABR UVER -V
  
  # Cambiar directorios de entrada y salida
  python evaluate_gpd_events.py --mseed-dir /path/to/mseed/ --csv-input /path/to/input.csv --csv-output /path/to/output.csv
        """)
    
    parser.add_argument('--mseed-dir', type=str, default=MSEED_DIR,
                       help=f'Directorio con archivos mseed (default: {MSEED_DIR})')
    parser.add_argument('--csv-input', type=str, default=CSV_INPUT,
                       help=f'Archivo CSV de entrada (default: {CSV_INPUT})')
    parser.add_argument('--csv-output', type=str, default=CSV_OUTPUT,
                       help=f'Archivo CSV de salida (default: {CSV_OUTPUT})')
    parser.add_argument('--stations', nargs='+', default=DEFAULT_STATIONS,
                       help=f'Estaciones a procesar (default: {DEFAULT_STATIONS})')
    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada')
    parser.add_argument('--model-path', type=str, default="./models/gpd_v2.keras",
                       help='Ruta al modelo GPD')
    
    args = parser.parse_args()
    
    print("=== Evaluación de Eventos Sísmicos con GPD ===")
    print(f"Directorio MSEED: {args.mseed_dir}")
    print(f"CSV entrada: {args.csv_input}")
    print(f"CSV salida: {args.csv_output}")
    print(f"Estaciones: {args.stations}")
    print(f"Modelo: {args.model_path}")
    
    # Verificar directorios y archivos
    if not os.path.isdir(args.mseed_dir):
        print(f"ERROR: Directorio no encontrado: {args.mseed_dir}")
        return
    
    if not os.path.isfile(args.csv_input):
        print(f"ERROR: Archivo CSV no encontrado: {args.csv_input}")
        return
    
    if not os.path.isfile(args.model_path):
        print(f"ERROR: Modelo no encontrado: {args.model_path}")
        return
    
    # Cargar modelo GPD
    print("Cargando modelo GPD...")
    try:
        model = load_model(args.model_path, compile=False)
        print(f"OK: Modelo cargado - input_shape={model.input_shape}, output_shape={model.output_shape}")
    except Exception as e:
        print(f"ERROR cargando modelo: {e}")
        return
    
    # Cargar dataset de referencia
    print("Cargando dataset de referencia...")
    try:
        df_ref = pd.read_csv(args.csv_input)
        print(f"OK: {len(df_ref)} eventos en dataset de referencia")
        print(f"Columnas: {list(df_ref.columns)}")
    except Exception as e:
        print(f"ERROR cargando CSV: {e}")
        return
    
    # Filtrar por estaciones seleccionadas
    df_filtered = df_ref[df_ref['Estacion'].isin(args.stations)]
    print(f"Eventos filtrados para estaciones {args.stations}: {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("No hay eventos para procesar con las estaciones seleccionadas")
        return
    
    # Preparar datos de salida
    results = []
    station_stats = defaultdict(lambda: {'total': 0, 'multi_p': 0, 'multi_s': 0})
    
    print(f"\nProcesando {len(df_filtered)} eventos...")
    
    # Procesar cada evento
    for idx, row in df_filtered.iterrows():
        estacion = row['Estacion']
        mseed_name = row['mseed']
        mseed_path = os.path.join(args.mseed_dir, mseed_name)
        
        station_stats[estacion]['total'] += 1
        
        # Verificar que el archivo existe
        if not os.path.isfile(mseed_path):
            if args.verbose:
                print(f"{mseed_name}: ARCHIVO NO ENCONTRADO")
            continue
        
        # Procesar evento
        detection = process_event_file(model, mseed_path, args.verbose)
        
        # Registrar estadísticas
        if detection['num_p'] > 1:
            station_stats[estacion]['multi_p'] += 1
        if detection['num_s'] > 1:
            station_stats[estacion]['multi_s'] += 1
        
        # Preparar resultado
        result_row = {
            'Estacion': estacion,
            'mseed': mseed_name,
            'Num-P': detection['num_p'],
            'Num-S': detection['num_s'],
            'T-P': detection['t_p'] if detection['t_p'] else 'NA',
            'T-S': detection['t_s'] if detection['t_s'] else 'NA',
            'Pond T-P': detection['prob_p'] if detection['prob_p'] else 'NA',
            'Pond T-S': detection['prob_s'] if detection['prob_s'] else 'NA'
        }
        
        results.append(result_row)
        
        # Mostrar progreso si verbose
        if args.verbose:
            prob_p_str = f"{detection['prob_p']:.3f}" if detection['prob_p'] else "NA"
            prob_s_str = f"{detection['prob_s']:.3f}" if detection['prob_s'] else "NA"
            print(f"{mseed_name}: Num-P={detection['num_p']}, Num-S={detection['num_s']}, "
                  f"Pond T-P={prob_p_str}, Pond T-S={prob_s_str}")
    
    # Guardar resultados
    print(f"\nGuardando resultados en {args.csv_output}...")
    df_results = pd.DataFrame(results)
    
    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(args.csv_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df_results.to_csv(args.csv_output, index=False)
    print(f"OK: {len(results)} resultados guardados")
    
    # =======================================================
    # ANÁLISIS ESTADÍSTICO BASADO EN EL CSV GENERADO
    # =======================================================
    
    print(f"\n=== ANÁLISIS ESTADÍSTICO DEL CSV GENERADO ===")
    
    # Estadísticas globales
    total_events = len(df_results)
    
    # Contar por categorías
    zero_p = (df_results['Num-P'] == 0).sum()
    one_p = (df_results['Num-P'] == 1).sum()
    multi_p = (df_results['Num-P'] > 1).sum()
    
    zero_s = (df_results['Num-S'] == 0).sum()
    one_s = (df_results['Num-S'] == 1).sum()
    multi_s = (df_results['Num-S'] > 1).sum()
    
    # Estadísticas por estación
    print(f"\n=== ESTADÍSTICAS POR ESTACIÓN ===")
    for estacion in sorted(df_results['Estacion'].unique()):
        subset = df_results[df_results['Estacion'] == estacion]
        est_total = len(subset)
        est_zero_p = (subset['Num-P'] == 0).sum()
        est_multi_p = (subset['Num-P'] > 1).sum()
        est_zero_s = (subset['Num-S'] == 0).sum()
        est_multi_s = (subset['Num-S'] > 1).sum()
        
        print(f"{estacion:>6}: Total={est_total:>3}, "
              f"Zero-P={est_zero_p:>3} ({100*est_zero_p/est_total:.1f}%), "
              f"Multi-P={est_multi_p:>3} ({100*est_multi_p/est_total:.1f}%), "
              f"Zero-S={est_zero_s:>3} ({100*est_zero_s/est_total:.1f}%), "
              f"Multi-S={est_multi_s:>3} ({100*est_multi_s/est_total:.1f}%)")
    
    # Estadísticas globales
    print(f"\n=== ESTADÍSTICAS GLOBALES ===")
    print(f"Total eventos: {total_events}")
    print(f"Eventos sin P detectadas: {zero_p} ({100*zero_p/total_events:.1f}%)")
    print(f"Eventos con exactamente 1 P: {one_p} ({100*one_p/total_events:.1f}%)")
    print(f"Eventos con múltiples P: {multi_p} ({100*multi_p/total_events:.1f}%)")
    print(f"Eventos sin S detectadas: {zero_s} ({100*zero_s/total_events:.1f}%)")
    print(f"Eventos con exactamente 1 S: {one_s} ({100*one_s/total_events:.1f}%)")
    print(f"Eventos con múltiples S: {multi_s} ({100*multi_s/total_events:.1f}%)")
    
    # Eventos válidos (exactamente 1 P y 1 S)
    valid_both = ((df_results['Num-P'] == 1) & (df_results['Num-S'] == 1)).sum()
    
    print(f"\n=== EVENTOS VÁLIDOS ===")
    print(f"Eventos con exactamente 1 P: {one_p} ({100*one_p/total_events:.1f}%)")
    print(f"Eventos con exactamente 1 S: {one_s} ({100*one_s/total_events:.1f}%)")
    print(f"Eventos con exactamente 1 P y 1 S: {valid_both} ({100*valid_both/total_events:.1f}%)")
    
    # Estadísticas adicionales de calidad
    print(f"\n=== CALIDAD DE DETECCIÓN ===")
    if one_p > 0:
        p_detected_with_s = ((df_results['Num-P'] == 1) & (df_results['Num-S'] >= 1)).sum()
        print(f"Eventos con P detectada que también tienen S: {p_detected_with_s}/{one_p} ({100*p_detected_with_s/one_p:.1f}%)")
    
    if one_s > 0:
        s_detected_with_p = ((df_results['Num-S'] == 1) & (df_results['Num-P'] >= 1)).sum()
        print(f"Eventos con S detectada que también tienen P: {s_detected_with_p}/{one_s} ({100*s_detected_with_p/one_s:.1f}%)")
    
    # Resumen de utilidad del modelo
    print(f"\n=== RESUMEN DE UTILIDAD ===")
    print(f"Eventos completamente inútiles (0 P, 0 S): {((df_results['Num-P'] == 0) & (df_results['Num-S'] == 0)).sum()}")
    print(f"Eventos parcialmente útiles (solo P o solo S): {((df_results['Num-P'] == 1) & (df_results['Num-S'] != 1)).sum() + ((df_results['Num-P'] != 1) & (df_results['Num-S'] == 1)).sum()}")
    print(f"Eventos completamente útiles (1 P y 1 S): {valid_both}")
    print(f"Eventos problemáticos (múltiples detecciones): {multi_p + multi_s}")
    
    # Distribución de probabilidades para detecciones válidas
    valid_p_probs = df_results[(df_results['Num-P'] == 1) & (df_results['Pond T-P'] != 'NA')]['Pond T-P']
    valid_s_probs = df_results[(df_results['Num-S'] == 1) & (df_results['Pond T-S'] != 'NA')]['Pond T-S']
    
    if len(valid_p_probs) > 0:
        print(f"\n=== DISTRIBUCIÓN DE PROBABILIDADES ===")
        print(f"Probabilidades P válidas: media={float(valid_p_probs.mean()):.3f}, "
              f"min={float(valid_p_probs.min()):.3f}, max={float(valid_p_probs.max()):.3f}")
    
    if len(valid_s_probs) > 0:
        if len(valid_p_probs) == 0:
            print(f"\n=== DISTRIBUCIÓN DE PROBABILIDADES ===")
        print(f"Probabilidades S válidas: media={float(valid_s_probs.mean()):.3f}, "
              f"min={float(valid_s_probs.min()):.3f}, max={float(valid_s_probs.max()):.3f}")
    
    print(f"\n=== Evaluación completada ===")
    print(f"Resultados disponibles en: {args.csv_output}")

if __name__ == "__main__":
    main()