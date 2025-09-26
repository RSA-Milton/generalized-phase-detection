#!/usr/bin/env python
"""
Evaluación de eventos sísmicos usando STA/LTA
Procesa archivos mseed preprocesados y evalúa detecciones contra picks manuales
Formato compatible con evaluate_gpd_events.py para comparación directa

Uso: 
python stalta_inference_events.py -V
python stalta_inference_events.py --threshold-p-on 3.0 --threshold-s-on 2.5 -V
"""

import numpy as np
import obspy.core as oc
from obspy.signal.trigger import trigger_onset, classic_sta_lta
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
# Rutas por defecto usando nueva estructura
MSEED_DIR = str(config.get_processed_mseed_events_dir('test_1000'))
CSV_INPUT = str(config.get_processed_datasets_dir() / 'test' / 'dataset_estratificado_1000_with_snr.csv')
CSV_OUTPUT = str(config.get_results_stalta_dir() / 'resultados_evaluacion_stalta_1000_agente_labr.csv')

# Estaciones por defecto
DEFAULT_STATIONS = ['LABR']

# Parámetros STA/LTA
DEFAULT_THRESHOLD_P_ON = 3.0
DEFAULT_THRESHOLD_P_OFF = 1.5
DEFAULT_THRESHOLD_S_ON = 2.5
DEFAULT_THRESHOLD_S_OFF = 1.2

# Ventanas STA/LTA (en segundos)
DEFAULT_STA_P = 0.5    # Ventana corta para P
DEFAULT_LTA_P = 10.0   # Ventana larga para P
DEFAULT_STA_S = 1.0    # Ventana corta para S
DEFAULT_LTA_S = 20.0   # Ventana larga para S

# Gap mínimo P-S
DEFAULT_GAP_MIN = 0.5
DEFAULT_MAX_CORRECTION = 10.0

# Variable global para correcciones
correcciones_aplicadas = 0

def sta_lta_picks(data, sampling_rate, sta_len, lta_len, threshold_on, threshold_off, 
                  start_time, phase_name="", verbose=False):
    """
    Detecta picks usando STA/LTA con histéresis
    
    Parameters:
    -----------
    data : array
        Datos sísmicos (una componente)
    sampling_rate : float
        Frecuencia de muestreo
    sta_len : float
        Longitud ventana STA en segundos
    lta_len : float
        Longitud ventana LTA en segundos
    threshold_on : float
        Umbral de activación
    threshold_off : float
        Umbral de desactivación
    start_time : obspy.UTCDateTime
        Tiempo de inicio del registro
    phase_name : str
        Nombre de la fase ("P" o "S") para verbose
    verbose : bool
        Mostrar información detallada
    
    Returns:
    --------
    list: Lista de (tiempo_pick, valor_sta_lta, índice)
    """
    
    # Convertir longitudes a muestras
    nsta = int(sta_len * sampling_rate)
    nlta = int(lta_len * sampling_rate)
    
    # Calcular STA/LTA
    sta_lta = classic_sta_lta(data, nsta, nlta)
    
    # Detectar triggers con histéresis
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
        
        if verbose:
            print(f"    {phase_name} trigger: t={pick_time}, STA/LTA={max_value:.2f}")
    
    return picks

def process_event_file_stalta(mseed_file, 
                             threshold_p_on, threshold_p_off, sta_p, lta_p,
                             threshold_s_on, threshold_s_off, sta_s, lta_s,
                             gap_min=0.5, max_correction=10.0, verbose=False):
    """
    Procesa un archivo de evento sísmico usando STA/LTA
    
    Returns:
    --------
    dict con formato compatible con evaluate_gpd_events.py:
        'num_p': número de fases P detectadas
        'num_s': número de fases S detectadas  
        't_p': tiempo de mejor fase P (o None si no hay)
        't_s': tiempo de mejor fase S (o None si no hay)
        'prob_p': valor STA/LTA de mejor fase P (o None si no hay)
        'prob_s': valor STA/LTA de mejor fase S (o None si no hay)
        'corrected': True si se aplicó corrección física
    """
    global correcciones_aplicadas
    
    result = {
        'num_p': 0, 'num_s': 0,
        't_p': None, 't_s': None, 
        'prob_p': None, 'prob_s': None,
        'corrected': False
    }
    
    try:
        # Cargar archivo mseed
        st = oc.read(mseed_file)
        
        # Verificar que tenemos 3 trazas
        if len(st) != 3:
            if verbose:
                print(f"    ERROR: {len(st)} trazas encontradas, se esperaban 3")
            return result
        
        # Obtener información básica
        sampling_rate = st[0].stats.sampling_rate
        start_time = st[0].stats.starttime
        
        if verbose:
            print(f"    Archivo: {os.path.basename(mseed_file)}")
            print(f"    Sampling rate: {sampling_rate} Hz")
        
        # Preparar datos - usar componente vertical (Z) para P y horizontal para S
        # Asumiendo orden N-E-Z en las trazas
        data_z = st[2].data  # Componente vertical para P
        data_n = st[0].data  # Componente norte para S
        data_e = st[1].data  # Componente este para S
        
        # Para S, usar la componente horizontal con mayor energía
        energy_n = np.sum(data_n**2)
        energy_e = np.sum(data_e**2)
        data_h = data_n if energy_n >= energy_e else data_e
        
        # =================== DETECCIÓN P (componente Z) ===================
        p_picks = sta_lta_picks(data_z, sampling_rate, sta_p, lta_p, 
                               threshold_p_on, threshold_p_off, start_time, "P", verbose)
        
        result['num_p'] = len(p_picks)
        
        # APLICAR CRITERIO: PRIMERA P (no la de mayor STA/LTA)
        if len(p_picks) > 0:
            # Ordenar por tiempo (primera en llegar)
            p_picks.sort(key=lambda x: x[0])
            best_p = p_picks[0]  # Primera P
            t_p = best_p[0]
            prob_p = best_p[1]
            if verbose:
                print(f"    Primera P seleccionada: t={t_p}, STA/LTA={prob_p:.3f}")
        else:
            t_p = None
            prob_p = None
        
        # =================== DETECCIÓN S (componente horizontal) ===================
        s_picks = sta_lta_picks(data_h, sampling_rate, sta_s, lta_s, 
                               threshold_s_on, threshold_s_off, start_time, "S", verbose)
        
        result['num_s'] = len(s_picks)
        
        # APLICAR CRITERIO: GAP MÍNIMO desde P
        if t_p is not None:
            t_min_s = t_p + gap_min
            s_picks_filtered = [s for s in s_picks if s[0] >= t_min_s]
            if verbose and len(s_picks) != len(s_picks_filtered):
                print(f"    S filtradas por gap: {len(s_picks)} -> {len(s_picks_filtered)}")
        else:
            s_picks_filtered = s_picks
        
        # Tomar la S de mayor STA/LTA después del filtro de gap
        if len(s_picks_filtered) > 0:
            s_picks_filtered.sort(key=lambda x: x[1], reverse=True)  # Por valor STA/LTA
            best_s = s_picks_filtered[0]
            t_s = best_s[0]
            prob_s = best_s[1]
            if verbose:
                print(f"    Mejor S después de gap: t={t_s}, STA/LTA={prob_s:.3f}")
        else:
            t_s = None
            prob_s = None
        
        # =================== CORRECCIÓN FÍSICA P-S ===================
        corrected = False
        if t_p is not None and t_s is not None:
            dt = (t_s - t_p).total_seconds()
            
            if dt < 0:  # Inversión: S antes que P
                abs_dt = abs(dt)
                
                if abs_dt <= max_correction:
                    # Intercambiar P y S
                    t_p, t_s = t_s, t_p
                    prob_p, prob_s = prob_s, prob_p
                    corrected = True
                    correcciones_aplicadas += 1
                    
                    if verbose:
                        print(f"    CORRECCIÓN aplicada: inversión de {abs_dt:.2f}s")
                
                else:
                    # Inversión demasiado grande: marcar ambas como NA
                    t_p = None
                    t_s = None
                    prob_p = None
                    prob_s = None
                    
                    if verbose:
                        print(f"    INVALIDACIÓN: inversión de {abs_dt:.2f}s > {max_correction}s")
        
        # Llenar resultados finales
        result['t_p'] = t_p.isoformat() if t_p else None
        result['t_s'] = t_s.isoformat() if t_s else None
        result['prob_p'] = float(prob_p) if prob_p is not None else None
        result['prob_s'] = float(prob_s) if prob_s is not None else None
        result['corrected'] = corrected
        
        # Limpiar memoria
        del st
        gc.collect()
        
    except Exception as e:
        if verbose:
            print(f"    ERROR procesando {mseed_file}: {e}")
    
    return result

def main():
    global correcciones_aplicadas
    correcciones_aplicadas = 0  # Reset contador
    
    parser = argparse.ArgumentParser(
        description='Evaluación de eventos sísmicos con STA/LTA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parámetros STA/LTA:
  - Utiliza componente Z para detección P
  - Utiliza componente horizontal (N o E) para detección S  
  - Implementa criterios físicos: primera P, gap mínimo S, corrección inversiones

Ejemplos de uso:
  # Evaluación con umbrales por defecto
  python evaluate_stalta_events.py -V
  
  # Umbrales personalizados
  python evaluate_stalta_events.py --threshold-p-on 3.5 --threshold-s-on 2.8 -V
  
  # Ventanas personalizadas
  python evaluate_stalta_events.py --sta-p 0.3 --lta-p 15.0 -V
        """)
    
    parser.add_argument('--mseed-dir', type=str, default=MSEED_DIR,
                       help=f'Directorio con archivos mseed (default: {MSEED_DIR})')
    parser.add_argument('--csv-input', type=str, default=CSV_INPUT,
                       help=f'Archivo CSV de entrada (default: {CSV_INPUT})')
    parser.add_argument('--csv-output', type=str, default=CSV_OUTPUT,
                       help=f'Archivo CSV de salida (default: {CSV_OUTPUT})')
    parser.add_argument('--stations', nargs='+', default=DEFAULT_STATIONS,
                       help=f'Estaciones a procesar (default: {DEFAULT_STATIONS})')
    
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
    parser.add_argument('--max-correction', type=float, default=DEFAULT_MAX_CORRECTION,
                       help=f'Máxima corrección de inversión P-S (default: {DEFAULT_MAX_CORRECTION}s)')
    
    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    print("=== Evaluación de Eventos Sísmicos con STA/LTA ===")
    print(f"Directorio MSEED: {args.mseed_dir}")
    print(f"CSV entrada: {args.csv_input}")
    print(f"CSV salida: {args.csv_output}")
    print(f"Estaciones: {args.stations}")
    print(f"Umbrales P: ON={args.threshold_p_on}, OFF={args.threshold_p_off}")
    print(f"Umbrales S: ON={args.threshold_s_on}, OFF={args.threshold_s_off}")
    print(f"Ventanas P: STA={args.sta_p}s, LTA={args.lta_p}s")
    print(f"Ventanas S: STA={args.sta_s}s, LTA={args.lta_s}s")
    print(f"Gap mínimo P-S: {args.gap_min}s")
    print(f"Máx corrección: {args.max_correction}s")
    
    # Verificar directorios y archivos
    if not os.path.isdir(args.mseed_dir):
        print(f"ERROR: Directorio no encontrado: {args.mseed_dir}")
        return
    
    if not os.path.isfile(args.csv_input):
        print(f"ERROR: Archivo CSV no encontrado: {args.csv_input}")
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
        detection = process_event_file_stalta(
            mseed_path,
            args.threshold_p_on, args.threshold_p_off, args.sta_p, args.lta_p,
            args.threshold_s_on, args.threshold_s_off, args.sta_s, args.lta_s,
            args.gap_min, args.max_correction, args.verbose
        )
        
        # Preparar resultado (formato compatible con GPD)
        result_row = {
            'Estacion': estacion,
            'mseed': mseed_name,
            'Num-P': detection['num_p'],
            'Num-S': detection['num_s'],
            'T-P': detection['t_p'] if detection['t_p'] else 'NA',
            'T-S': detection['t_s'] if detection['t_s'] else 'NA',
            'Pond T-P': detection['prob_p'] if detection['prob_p'] else 'NA',  # STA/LTA value
            'Pond T-S': detection['prob_s'] if detection['prob_s'] else 'NA',  # STA/LTA value
            'Corregido': detection['corrected']
        }
        
        results.append(result_row)
        
        # Mostrar progreso si verbose
        if args.verbose:
            prob_p_str = f"{detection['prob_p']:.3f}" if detection['prob_p'] else "NA"
            prob_s_str = f"{detection['prob_s']:.3f}" if detection['prob_s'] else "NA"
            corr_str = " [CORREGIDO]" if detection['corrected'] else ""
            print(f"{mseed_name}: Num-P={detection['num_p']}, Num-S={detection['num_s']}, "
                  f"STA/LTA-P={prob_p_str}, STA/LTA-S={prob_s_str}{corr_str}")
    
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
    # ANÁLISIS ESTADÍSTICO (formato compatible con GPD)
    # =======================================================
    
    print(f"\n=== ANÁLISIS ESTADÍSTICO STA/LTA ===")
    print(f"Parámetros utilizados:")
    print(f"  P: STA={args.sta_p}s, LTA={args.lta_p}s, ON={args.threshold_p_on}, OFF={args.threshold_p_off}")
    print(f"  S: STA={args.sta_s}s, LTA={args.lta_s}s, ON={args.threshold_s_on}, OFF={args.threshold_s_off}")
    print(f"  Gap mínimo: {args.gap_min}s, Máx corrección: {args.max_correction}s")
    
    # Estadísticas globales
    total_events = len(df_results)
    
    # Contar eventos con datos válidos
    valid_data_p = (df_results['T-P'] != 'NA').sum()
    valid_data_s = (df_results['T-S'] != 'NA').sum()
    valid_data_both = ((df_results['T-P'] != 'NA') & (df_results['T-S'] != 'NA')).sum()
    
    print(f"\n=== RESULTADOS PRINCIPALES ===")
    print(f"Total eventos: {total_events}")
    print(f"Eventos con datos P válidos: {valid_data_p} ({100*valid_data_p/total_events:.1f}%)")
    print(f"Eventos con datos S válidos: {valid_data_s} ({100*valid_data_s/total_events:.1f}%)")
    print(f"Eventos con ambas fases válidas: {valid_data_both} ({100*valid_data_both/total_events:.1f}%)")
    
    if correcciones_aplicadas > 0:
        print(f"Correcciones P-S aplicadas: {correcciones_aplicadas} ({100*correcciones_aplicadas/total_events:.1f}%)")
    else:
        print(f"Correcciones P-S aplicadas: 0 (orden temporal correcto)")
    
    # Estadísticas por estación
    print(f"\n=== POR ESTACIÓN ===")
    for estacion in sorted(df_results['Estacion'].unique()):
        subset = df_results[df_results['Estacion'] == estacion]
        est_total = len(subset)
        est_valid_p = (subset['T-P'] != 'NA').sum()
        est_valid_s = (subset['T-S'] != 'NA').sum()
        est_valid_both = ((subset['T-P'] != 'NA') & (subset['T-S'] != 'NA')).sum()
        
        print(f"{estacion}: {est_total} eventos, P={est_valid_p} ({100*est_valid_p/est_total:.1f}%), "
              f"S={est_valid_s} ({100*est_valid_s/est_total:.1f}%), "
              f"Ambas={est_valid_both} ({100*est_valid_both/est_total:.1f}%)")
    
    # Distribución de valores STA/LTA
    valid_p_vals = df_results[(df_results['T-P'] != 'NA') & (df_results['Pond T-P'] != 'NA')]['Pond T-P']
    valid_s_vals = df_results[(df_results['T-S'] != 'NA') & (df_results['Pond T-S'] != 'NA')]['Pond T-S']
    
    if len(valid_p_vals) > 0:
        print(f"\n=== DISTRIBUCIÓN DE VALORES STA/LTA ===")
        print(f"Valores STA/LTA P: N={len(valid_p_vals)}, "
              f"media={float(valid_p_vals.mean()):.2f}, "
              f"min={float(valid_p_vals.min()):.2f}, "
              f"max={float(valid_p_vals.max()):.2f}")
    
    if len(valid_s_vals) > 0:
        if len(valid_p_vals) == 0:
            print(f"\n=== DISTRIBUCIÓN DE VALORES STA/LTA ===")
        print(f"Valores STA/LTA S: N={len(valid_s_vals)}, "
              f"media={float(valid_s_vals.mean()):.2f}, "
              f"min={float(valid_s_vals.min()):.2f}, "
              f"max={float(valid_s_vals.max()):.2f}")
    
    print(f"\n=== CARACTERÍSTICAS DEL ALGORITMO STA/LTA ===")
    print(f"✓ Detección P en componente vertical (Z)")
    print(f"✓ Detección S en componente horizontal (N o E)")
    print(f"✓ Histéresis: umbrales diferenciados ON/OFF")
    print(f"✓ Primera cresta P (no máxima)")
    print(f"✓ Gap mínimo S después de P")
    print(f"✓ Corrección física de inversiones")
    
    # Recomendaciones
    print(f"\n=== RECOMENDACIONES PARA OPTIMIZACIÓN ===")
    if valid_data_p/total_events < 0.5:
        print(f"⚠ Baja detección P ({100*valid_data_p/total_events:.1f}%) → Reducir threshold-p-on")
    if valid_data_s/total_events < 0.5:
        print(f"⚠ Baja detección S ({100*valid_data_s/total_events:.1f}%) → Reducir threshold-s-on")
    if correcciones_aplicadas/total_events > 0.1:
        print(f"⚠ Muchas correcciones ({100*correcciones_aplicadas/total_events:.1f}%) → Ajustar ventanas STA/LTA")
    
    print(f"\n=== Evaluación completada ===")
    print(f"Resultados guardados en: {args.csv_output}")
    print(f"Formato compatible con evaluate_gpd_events.py para comparación directa")

if __name__ == "__main__":
    main()