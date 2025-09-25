#!/usr/bin/env python
"""
Evaluación de eventos sísmicos usando modelo GPD
Procesa archivos mseed preprocesados y evalúa detecciones contra picks manuales

Uso: 
python gpd_keras_inference_events.py -V
python gpd_keras_inference_events.py --min-proba-p 0.55 --min-proba-s 0.85 -V
"""

import numpy as np
import obspy.core as oc
from tensorflow.keras.models import load_model
import pandas as pd
import argparse
import os
import gc
import sys
from collections import defaultdict
from pathlib import Path

# Add config module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
import config

# =================== CONFIGURACIÓN ===================
# Rutas por defecto usando configuración
MSEED_DIR = str(config.get_data_dir('dataset/test_1000'))
CSV_INPUT = str(config.get_data_dir('dataset/dataset_estratificado_1000_with_snr.csv'))
CSV_OUTPUT = str(config.get_data_dir('results/resultados_evaluacion_1000_agente.csv'))

# Estaciones por defecto
DEFAULT_STATIONS = ['LABR', 'CHAI', 'CUSH', 'UVER', 'PORT']
#DEFAULT_STATIONS = ['LABR']

# Parámetros del modelo GPD
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

def process_event_file(model, mseed_file, min_proba_P, min_proba_S, verbose=False):
    """
    Procesa un archivo de evento sísmico y devuelve las detecciones
    
    Returns:
    --------
    dict con:
        'num_p': número de fases P detectadas
        'num_s': número de fases S detectadas  
        't_p': tiempo de mejor fase P (o None si no hay)
        't_s': tiempo de mejor fase S (o None si no hay)
        'prob_p': probabilidad de mejor fase P (o None si no hay)
        'prob_s': probabilidad de mejor fase S (o None si no hay)
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
        
        # Detectar picks P con umbral específico
        from obspy.signal.trigger import trigger_onset
        trigs_p = trigger_onset(prob_P, min_proba_P, 0.1)
        
        p_detections = []
        for trig in trigs_p:
            if trig[1] == trig[0]:
                continue
            pick_idx = np.argmax(ts[trig[0]:trig[1], 0]) + trig[0]
            pick_time = start_time + tt[pick_idx]
            pick_prob = ts[pick_idx, 0]
            p_detections.append((pick_time, pick_prob, pick_idx))
        
        # Detectar picks S con umbral específico
        trigs_s = trigger_onset(prob_S, min_proba_S, 0.1)
        
        s_detections = []
        for trig in trigs_s:
            if trig[1] == trig[0]:
                continue
            pick_idx = np.argmax(ts[trig[0]:trig[1], 1]) + trig[0]
            pick_time = start_time + tt[pick_idx]
            pick_prob = ts[pick_idx, 1]
            s_detections.append((pick_time, pick_prob, pick_idx))
        
        # Llenar resultados
        result['num_p'] = len(p_detections)
        result['num_s'] = len(s_detections)
        
        # Para P: si hay detecciones, tomar la de mayor probabilidad
        if result['num_p'] > 0:
            p_detections.sort(key=lambda x: x[1], reverse=True)
            best_p = p_detections[0]
            result['t_p'] = best_p[0].isoformat()
            result['prob_p'] = float(best_p[1])
        
        # Para S: si hay detecciones, tomar la de mayor probabilidad
        if result['num_s'] > 0:
            s_detections.sort(key=lambda x: x[1], reverse=True)
            best_s = s_detections[0]
            result['t_s'] = best_s[0].isoformat()
            result['prob_s'] = float(best_s[1])
        
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
  # Evaluación con modelo y umbrales por defecto
  python gpd_keras_inference_events.py -V

  # Umbrales personalizados
  python gpd_keras_inference_events.py --min-proba-p 0.55 --min-proba-s 0.90 -V

  # Modelo específico (solo nombre)
  python gpd_keras_inference_events.py --model-path gpd_v1.keras -V

  # Modelo HDF5 legacy
  python gpd_keras_inference_events.py --model-path model_pol_best.hdf5 -V

  # Solo estaciones específicas
  python gpd_keras_inference_events.py --stations CHAI LABR -V
        """)
    
    parser.add_argument('--mseed-dir', type=str, default=MSEED_DIR,
                       help=f'Directorio con archivos mseed (default: {MSEED_DIR})')
    parser.add_argument('--csv-input', type=str, default=CSV_INPUT,
                       help=f'Archivo CSV de entrada (default: {CSV_INPUT})')
    parser.add_argument('--csv-output', type=str, default=CSV_OUTPUT,
                       help=f'Archivo CSV de salida (default: {CSV_OUTPUT})')
    parser.add_argument('--stations', nargs='+', default=DEFAULT_STATIONS,
                       help=f'Estaciones a procesar (default: {DEFAULT_STATIONS})')
    parser.add_argument('--min-proba-p', type=float, default=DEFAULT_MIN_PROBA_P,
                       help=f'Umbral de probabilidad para fases P (default: {DEFAULT_MIN_PROBA_P})')
    parser.add_argument('--min-proba-s', type=float, default=DEFAULT_MIN_PROBA_S,
                       help=f'Umbral de probabilidad para fases S (default: {DEFAULT_MIN_PROBA_S})')
    parser.add_argument('--model-path', type=str, default=None,
                       help=f'Nombre del modelo GPD (default: {config.get_default_model_name()}). '
                            f'Se busca en {config.get_models_dir()}. '
                            f'Ejemplos: gpd_v2.keras, gpd_v1.keras, model_pol_best.hdf5')
    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()

    # Resolver la ruta del modelo
    if args.model_path is None:
        # Usar modelo por defecto
        model_path = config.get_default_model_path()
        model_name = config.get_default_model_name()
    else:
        # Usar modelo especificado (solo nombre, construir ruta completa)
        model_name = args.model_path
        model_path = config.get_models_dir() / model_name

    print("=== Evaluación de Eventos Sísmicos con GPD ===")
    print(f"Directorio MSEED: {args.mseed_dir}")
    print(f"CSV entrada: {args.csv_input}")
    print(f"CSV salida: {args.csv_output}")
    print(f"Estaciones: {args.stations}")
    print(f"Umbral P: {args.min_proba_p}")
    print(f"Umbral S: {args.min_proba_s}")
    print(f"Modelo: {model_name}")
    print(f"Ruta del modelo: {model_path}")
    
    # Verificar directorios y archivos
    if not os.path.isdir(args.mseed_dir):
        print(f"ERROR: Directorio no encontrado: {args.mseed_dir}")
        return
    
    if not os.path.isfile(args.csv_input):
        print(f"ERROR: Archivo CSV no encontrado: {args.csv_input}")
        return
    
    if not os.path.isfile(model_path):
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
        return

    # Cargar modelo GPD
    print("Cargando modelo GPD...")
    try:
        model = load_model(model_path, compile=False)
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
    
    print(f"\nProcesando {len(df_filtered)} eventos...")
    
    # Procesar cada evento
    for idx, row in df_filtered.iterrows():
        estacion = row['Estacion']
        mseed_name = row['mseed']
        mseed_path = os.path.join(args.mseed_dir, mseed_name)
        
        # Verificar que el archivo existe
        if not os.path.isfile(mseed_path):
            if args.verbose:
                print(f"{mseed_name}: ARCHIVO NO ENCONTRADO")
            continue
        
        # Procesar evento
        detection = process_event_file(model, mseed_path, args.min_proba_p, args.min_proba_s, args.verbose)
        
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
    print(f"Umbrales utilizados: P={args.min_proba_p}, S={args.min_proba_s}")
    
    # Estadísticas globales
    total_events = len(df_results)
    
    # Contar por categorías
    zero_p = (df_results['Num-P'] == 0).sum()
    one_p = (df_results['Num-P'] == 1).sum()
    multi_p = (df_results['Num-P'] > 1).sum()
    
    zero_s = (df_results['Num-S'] == 0).sum()
    one_s = (df_results['Num-S'] == 1).sum()
    multi_s = (df_results['Num-S'] > 1).sum()
    
    # Contar eventos con datos válidos (no NA) - NUEVA LÓGICA
    valid_data_p = (df_results['T-P'] != 'NA').sum()
    valid_data_s = (df_results['T-S'] != 'NA').sum()
    valid_data_both = ((df_results['T-P'] != 'NA') & (df_results['T-S'] != 'NA')).sum()
    
    # Estadísticas por estación
    print(f"\n=== ESTADÍSTICAS POR ESTACIÓN ===")
    for estacion in sorted(df_results['Estacion'].unique()):
        subset = df_results[df_results['Estacion'] == estacion]
        est_total = len(subset)
        est_zero_p = (subset['Num-P'] == 0).sum()
        est_multi_p = (subset['Num-P'] > 1).sum()
        est_zero_s = (subset['Num-S'] == 0).sum()
        est_multi_s = (subset['Num-S'] > 1).sum()
        est_valid_p = (subset['T-P'] != 'NA').sum()
        est_valid_s = (subset['T-S'] != 'NA').sum()
        
        print(f"{estacion:>6}: Total={est_total:>3}, "
              f"Zero-P={est_zero_p:>3} ({100*est_zero_p/est_total:.1f}%), "
              f"Multi-P={est_multi_p:>3} ({100*est_multi_p/est_total:.1f}%), "
              f"Valid-P={est_valid_p:>3} ({100*est_valid_p/est_total:.1f}%), "
              f"Zero-S={est_zero_s:>3} ({100*est_zero_s/est_total:.1f}%), "
              f"Multi-S={est_multi_s:>3} ({100*est_multi_s/est_total:.1f}%), "
              f"Valid-S={est_valid_s:>3} ({100*est_valid_s/est_total:.1f}%)")
    
    # Estadísticas globales
    print(f"\n=== ESTADÍSTICAS GLOBALES ===")
    print(f"Total eventos: {total_events}")
    print(f"Eventos sin P detectadas: {zero_p} ({100*zero_p/total_events:.1f}%)")
    print(f"Eventos con exactamente 1 P: {one_p} ({100*one_p/total_events:.1f}%)")
    print(f"Eventos con múltiples P: {multi_p} ({100*multi_p/total_events:.1f}%)")
    print(f"Eventos con datos P válidos: {valid_data_p} ({100*valid_data_p/total_events:.1f}%)")
    print(f"Eventos sin S detectadas: {zero_s} ({100*zero_s/total_events:.1f}%)")
    print(f"Eventos con exactamente 1 S: {one_s} ({100*one_s/total_events:.1f}%)")
    print(f"Eventos con múltiples S: {multi_s} ({100*multi_s/total_events:.1f}%)")
    print(f"Eventos con datos S válidos: {valid_data_s} ({100*valid_data_s/total_events:.1f}%)")
    
    # Eventos válidos
    print(f"\n=== EVENTOS CON DATOS UTILIZABLES ===")
    print(f"Eventos con datos P utilizables: {valid_data_p} ({100*valid_data_p/total_events:.1f}%)")
    print(f"Eventos con datos S utilizables: {valid_data_s} ({100*valid_data_s/total_events:.1f}%)")
    print(f"Eventos con ambos datos utilizables: {valid_data_both} ({100*valid_data_both/total_events:.1f}%)")
    
    # Análisis de correcciones aplicadas
    print(f"\n=== ANÁLISIS DE CORRECCIONES P-S ===")
    if 'Corregido' in df_results.columns and correcciones_aplicadas > 0:
        eventos_corregidos = df_results[df_results['Corregido'] == True]
        
        print(f"Total correcciones aplicadas: {correcciones_aplicadas}")
        print(f"Porcentaje de eventos corregidos: {100*correcciones_aplicadas/total_events:.1f}%")
        print(f"Porcentaje de eventos con ambas fases corregidos: {100*correcciones_aplicadas/valid_data_both:.1f}%")
        
        print(f"\nEventos corregidos por estación:")
        for estacion in sorted(eventos_corregidos['Estacion'].unique()):
            count = len(eventos_corregidos[eventos_corregidos['Estacion'] == estacion])
            total_est = len(df_results[df_results['Estacion'] == estacion])
            print(f"  {estacion}: {count} de {total_est} eventos ({100*count/total_est:.1f}%)")
        
        # Distribución de probabilidades en eventos corregidos
        prob_p_corr = eventos_corregidos[(eventos_corregidos['Pond T-P'] != 'NA')]['Pond T-P']
        prob_s_corr = eventos_corregidos[(eventos_corregidos['Pond T-S'] != 'NA')]['Pond T-S']
        
        if len(prob_p_corr) > 0:
            print(f"\nProbabilidades en eventos corregidos:")
            print(f"  P corregidas: media={float(prob_p_corr.mean()):.3f}, "
                  f"min={float(prob_p_corr.min()):.3f}, max={float(prob_p_corr.max()):.3f}")
        
        if len(prob_s_corr) > 0:
            print(f"  S corregidas: media={float(prob_s_corr.mean()):.3f}, "
                  f"min={float(prob_s_corr.min()):.3f}, max={float(prob_s_corr.max()):.3f}")
    
    elif 'Corregido' in df_results.columns:
        print("No se aplicaron correcciones P-S en este procesamiento")
    else:
        print("Información de correcciones no disponible (columna 'Corregido' faltante)")
        print("Esto puede indicar que el programa no detectó casos que requirieran corrección")
        print("o que hubo un problema en el procesamiento de correcciones")
    
    # Estadísticas de calidad mejoradas
    print(f"\n=== CALIDAD DE DETECCIÓN ===")
    if valid_data_p > 0:
        p_with_s = ((df_results['T-P'] != 'NA') & (df_results['T-S'] != 'NA')).sum()
        print(f"Eventos con P detectada que también tienen S: {p_with_s}/{valid_data_p} ({100*p_with_s/valid_data_p:.1f}%)")
    
    if valid_data_s > 0:
        s_with_p = ((df_results['T-S'] != 'NA') & (df_results['T-P'] != 'NA')).sum()
        print(f"Eventos con S detectada que también tienen P: {s_with_p}/{valid_data_s} ({100*s_with_p/valid_data_s:.1f}%)")
    
    # Análisis de múltiples detecciones pero con mejor pick
    multi_p_with_data = ((df_results['Num-P'] > 1) & (df_results['T-P'] != 'NA')).sum()
    multi_s_with_data = ((df_results['Num-S'] > 1) & (df_results['T-S'] != 'NA')).sum()
    
    print(f"\n=== EFECTIVIDAD DE SELECCIÓN DE MEJOR PICK ===")
    if multi_p > 0:
        print(f"Múltiples P con mejor pick seleccionado: {multi_p_with_data}/{multi_p} ({100*multi_p_with_data/multi_p:.1f}%)")
    if multi_s > 0:
        print(f"Múltiples S con mejor pick seleccionado: {multi_s_with_data}/{multi_s} ({100*multi_s_with_data/multi_s:.1f}%)")
    
    # Resumen de utilidad actualizado
    no_detections = ((df_results['Num-P'] == 0) & (df_results['Num-S'] == 0)).sum()
    partial_detections = ((df_results['T-P'] != 'NA') & (df_results['T-S'] == 'NA')).sum() + \
                        ((df_results['T-P'] == 'NA') & (df_results['T-S'] != 'NA')).sum()
    
    print(f"\n=== RESUMEN DE UTILIDAD ACTUALIZADO ===")
    print(f"Eventos sin detecciones: {no_detections} ({100*no_detections/total_events:.1f}%)")
    print(f"Eventos con una fase utilizable: {partial_detections} ({100*partial_detections/total_events:.1f}%)")
    print(f"Eventos con ambas fases utilizables: {valid_data_both} ({100*valid_data_both/total_events:.1f}%)")
    print(f"Total eventos utilizables: {valid_data_p + valid_data_s - valid_data_both} ({100*(valid_data_p + valid_data_s - valid_data_both)/total_events:.1f}%)")
    
    # Distribución de probabilidades para picks seleccionados
    valid_p_probs = df_results[(df_results['T-P'] != 'NA') & (df_results['Pond T-P'] != 'NA')]['Pond T-P']
    valid_s_probs = df_results[(df_results['T-S'] != 'NA') & (df_results['Pond T-S'] != 'NA')]['Pond T-S']
    
    if len(valid_p_probs) > 0:
        print(f"\n=== DISTRIBUCIÓN DE PROBABILIDADES DE PICKS SELECCIONADOS ===")
        print(f"Probabilidades P seleccionadas: N={len(valid_p_probs)}, "
              f"media={float(valid_p_probs.mean()):.3f}, "
              f"min={float(valid_p_probs.min()):.3f}, "
              f"max={float(valid_p_probs.max()):.3f}")
    
    if len(valid_s_probs) > 0:
        if len(valid_p_probs) == 0:
            print(f"\n=== DISTRIBUCIÓN DE PROBABILIDADES DE PICKS SELECCIONADOS ===")
        print(f"Probabilidades S seleccionadas: N={len(valid_s_probs)}, "
              f"media={float(valid_s_probs.mean()):.3f}, "
              f"min={float(valid_s_probs.min()):.3f}, "
              f"max={float(valid_s_probs.max()):.3f}")
    
    print(f"\n=== Evaluación completada ===")
    print(f"Resultados disponibles en: {args.csv_output}")

if __name__ == "__main__":
    main()