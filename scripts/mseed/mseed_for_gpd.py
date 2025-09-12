#!/usr/bin/env python
"""
Script para preprocesar archivos mseed para uso con modelo GPD
Combina remuestreo, filtrado y preprocesamiento completo
Maneja archivos de un solo canal replicándolo a 3 canales
Procesa archivos definidos en un CSV y calcula SNR para fases P y S

Uso: 

# Procesamiento usando archivo CSV
python preprocess_mseed_for_gpd.py -I dataset.csv -O /path/to/output/ --input-freq 250

# Con información detallada
python preprocess_mseed_for_gpd.py -I dataset.csv -O /path/to/output/ --input-freq 100 -V

# Con parámetros personalizados
python preprocess_mseed_for_gpd.py -I dataset.csv -O /path/to/output/ --input-freq 64 --target-freq 100 --freq-min 2.0 --freq-max 25.0

"""

import os
import argparse
import glob
import pandas as pd
import obspy.core as oc
import numpy as np
from obspy import UTCDateTime

# Ruta por defecto donde están los archivos mseed
MSEED_DIR = "/home/rsa/projects/gpd/data/dataset/test/"

def calculate_snr(trace, arrival_time, pre_window=5.0, post_window=5.0):
    """
    Calcula el SNR (Signal-to-Noise Ratio) para una fase sísmica
    
    Parameters:
    -----------
    trace : obspy.Trace
        Traza sísmica
    arrival_time : UTCDateTime
        Tiempo de arribo de la fase
    pre_window : float
        Ventana antes del arribo para calcular el ruido (segundos)
    post_window : float
        Ventana después del arribo para calcular la señal (segundos)
    
    Returns:
    --------
    float : SNR en dB
    """
    try:
        # Convertir arrival_time a UTCDateTime si es string
        if isinstance(arrival_time, str):
            arrival_time = UTCDateTime(arrival_time)
        elif pd.isna(arrival_time) or arrival_time == '':
            return np.nan
            
        # Verificar que el tiempo de arribo esté dentro de la traza
        if arrival_time < trace.stats.starttime or arrival_time > trace.stats.endtime:
            return np.nan
        
        # Calcular ventanas temporales
        noise_start = arrival_time - pre_window
        noise_end = arrival_time
        signal_start = arrival_time
        signal_end = arrival_time + post_window
        
        # Verificar que las ventanas estén dentro de los datos
        if (noise_start < trace.stats.starttime or 
            signal_end > trace.stats.endtime):
            return np.nan
        
        # Obtener datos de ruido y señal
        trace_copy = trace.copy()
        noise_trace = trace_copy.slice(noise_start, noise_end)
        signal_trace = trace_copy.slice(signal_start, signal_end)
        
        if len(noise_trace.data) == 0 or len(signal_trace.data) == 0:
            return np.nan
        
        # Calcular RMS del ruido y la señal
        noise_rms = np.sqrt(np.mean(noise_trace.data**2))
        signal_rms = np.sqrt(np.mean(signal_trace.data**2))
        
        # Evitar división por cero
        if noise_rms == 0:
            return np.inf if signal_rms > 0 else np.nan
        
        # SNR en dB
        snr_db = 20 * np.log10(signal_rms / noise_rms)
        
        return snr_db
        
    except Exception as e:
        print(f"    Error calculando SNR: {e}")
        return np.nan

def preprocess_mseed_file(input_file, output_file, input_freq, target_freq=100.0, 
                         freq_min=1.0, freq_max=20.0, apply_resample=True, 
                         apply_filter=True, verbose=False):
    """
    Preprocesa un archivo mseed para uso con GPD
    
    Parameters:
    -----------
    input_file : str
        Archivo mseed de entrada
    output_file : str  
        Archivo mseed de salida
    input_freq : float
        Frecuencia de muestreo de entrada
    target_freq : float
        Frecuencia de muestreo objetivo (default: 100.0 Hz)
    freq_min, freq_max : float
        Frecuencias del filtro pasa banda
    apply_resample : bool
        Si aplicar remuestreo
    apply_filter : bool
        Si aplicar filtro pasa banda
    verbose : bool
        Mostrar información detallada
    
    Returns:
    --------
    tuple: (success, num_channels, processed_stream)
    """
    
    try:
        if verbose:
            print(f"Procesando: {input_file}")
        
        # Leer archivo
        st = oc.read(input_file)
        
        if verbose:
            print(f"  Trazas encontradas: {len(st)}")
            for i, tr in enumerate(st):
                print(f"    {i+1}: {tr.stats.channel} - {tr.stats.sampling_rate} Hz - {len(tr.data)} muestras")
        
        # Verificar si necesitamos replicar canal único a 3 canales
        if len(st) == 1:
            if verbose:
                print("  Detectado archivo de un solo canal, replicando a 3 canales (N, E, Z)")
            
            # Obtener la traza original
            original_trace = st[0].copy()
            
            # Crear 3 trazas con diferentes códigos de canal
            st.clear()  # Limpiar stream
            
            # Canal North (N)
            tr_n = original_trace.copy()
            tr_n.stats.channel = tr_n.stats.channel[:-1] + 'N'  # Cambiar último carácter a N
            st.append(tr_n)
            
            # Canal East (E)
            tr_e = original_trace.copy()
            tr_e.stats.channel = tr_e.stats.channel[:-1] + 'E'  # Cambiar último carácter a E
            st.append(tr_e)
            
            # Canal Z (vertical)
            tr_z = original_trace.copy()
            tr_z.stats.channel = tr_z.stats.channel[:-1] + 'Z'  # Cambiar último carácter a Z
            st.append(tr_z)
            
            if verbose:
                print(f"    Canales creados: {[tr.stats.channel for tr in st]}")
        
        elif len(st) < 3:
            if verbose:
                print(f"  ADVERTENCIA: Solo {len(st)} canales disponibles, puede afectar el procesamiento GPD")
        
        # Verificar frecuencias de muestreo
        sampling_rates = [tr.stats.sampling_rate for tr in st]
        if not all(abs(rate - input_freq) < 0.1 for rate in sampling_rates):
            if verbose:
                print(f"  ADVERTENCIA: Frecuencias no coinciden con --input-freq {input_freq}")
                print(f"    Frecuencias encontradas: {sampling_rates}")
        
        # Sincronizar trazas (importante para GPD)
        if len(st) > 1:
            if verbose:
                print("  Sincronizando trazas...")
            
            # Encontrar ventana común
            latest_start = max([tr.stats.starttime for tr in st])
            earliest_stop = min([tr.stats.endtime for tr in st])
            
            if latest_start >= earliest_stop:
                raise ValueError("No hay superposición temporal entre las trazas")
            
            # Recortar a ventana común
            st.trim(latest_start, earliest_stop)
            
            if verbose:
                print(f"    Ventana sincronizada: {latest_start} - {earliest_stop}")
                print(f"    Duración: {earliest_stop - latest_start} s")
        
        # Aplicar preprocesamiento (similar a gpd_chunked_processing)
        if verbose:
            print("  Aplicando preprocesamiento...")
        
        # 1. Detrend (eliminar tendencia lineal)
        st.detrend(type='linear')
        if verbose:
            print("    ✓ Detrend lineal aplicado")
        
        # 2. Filtro pasa banda
        if apply_filter:
            if verbose:
                print(f"    Aplicando filtro pasa-banda: {freq_min}-{freq_max} Hz")
            
            # Verificar que las frecuencias del filtro sean válidas
            nyquist = min(sampling_rates) / 2.0
            if freq_max >= nyquist:
                freq_max_adj = nyquist * 0.9
                if verbose:
                    print(f"    ADVERTENCIA: freq_max ajustada de {freq_max} a {freq_max_adj} Hz")
                freq_max = freq_max_adj
            
            st.filter(type='bandpass', freqmin=freq_min, freqmax=freq_max)
            if verbose:
                print("    ✓ Filtro pasa-banda aplicado")
        
        # 3. Remuestreo (si es necesario)
        if apply_resample and abs(input_freq - target_freq) > 0.1:
            if verbose:
                print(f"    Remuestreando: {input_freq} Hz → {target_freq} Hz")
            
            # Aplicar filtro antialiasing antes del remuestreo si es necesario
            if input_freq > target_freq:
                nyquist_target = target_freq / 2.0
                fmax_antialiasing = nyquist_target * 0.9
                
                if freq_max > fmax_antialiasing:
                    if verbose:
                        print(f"    Aplicando filtro antialiasing adicional: < {fmax_antialiasing} Hz")
                    st.filter(type='lowpass', freq=fmax_antialiasing)
            
            # Remuestrear
            st.resample(target_freq)
            
            if verbose:
                print(f"    ✓ Remuestreo completado")
                for i, tr in enumerate(st):
                    print(f"      Canal {tr.stats.channel}: {len(tr.data)} muestras")
        
        elif not apply_resample:
            if verbose:
                print("    Remuestreo deshabilitado")
        else:
            if verbose:
                print(f"    Ya está a {target_freq} Hz, no requiere remuestreo")
        
        # Verificar calidad final de los datos
        if verbose:
            print("  Verificando calidad final:")
            for tr in st:
                data_stats = {
                    'mean': np.mean(tr.data),
                    'std': np.std(tr.data),
                    'min': np.min(tr.data),
                    'max': np.max(tr.data),
                    'samples': len(tr.data)
                }
                print(f"    {tr.stats.channel}: mean={data_stats['mean']:.2e}, "
                      f"std={data_stats['std']:.2e}, "
                      f"range=[{data_stats['min']:.2e}, {data_stats['max']:.2e}], "
                      f"N={data_stats['samples']}")
        
        # Guardar archivo procesado
        st.write(output_file, format='MSEED')
        
        if verbose:
            print(f"  ✓ Guardado en: {output_file}")
            
        return True, len(st), st
        
    except Exception as e:
        print(f"ERROR procesando {input_file}: {e}")
        return False, 0, None

def calculate_snr_for_phases(stream, tp_time, ts_time, verbose=False):
    """
    Calcula SNR para las fases P y S usando el canal con mejor SNR
    
    Parameters:
    -----------
    stream : obspy.Stream
        Stream procesado
    tp_time : str o UTCDateTime
        Tiempo de arribo de la fase P
    ts_time : str o UTCDateTime  
        Tiempo de arribo de la fase S
    verbose : bool
        Mostrar información detallada
        
    Returns:
    --------
    tuple: (snr_p, snr_s)
    """
    
    if stream is None or len(stream) == 0:
        return np.nan, np.nan
    
    snr_p_values = []
    snr_s_values = []
    
    # Calcular SNR para cada canal
    for trace in stream:
        if verbose:
            print(f"    Calculando SNR para canal {trace.stats.channel}")
        
        # SNR para fase P
        if not pd.isna(tp_time) and tp_time != '':
            snr_p = calculate_snr(trace, tp_time)
            if not np.isnan(snr_p):
                snr_p_values.append(snr_p)
                if verbose:
                    print(f"      SNR-P: {snr_p:.2f} dB")
        
        # SNR para fase S
        if not pd.isna(ts_time) and ts_time != '':
            snr_s = calculate_snr(trace, ts_time)
            if not np.isnan(snr_s):
                snr_s_values.append(snr_s)
                if verbose:
                    print(f"      SNR-S: {snr_s:.2f} dB")
    
    # Usar el mejor SNR (máximo) de todos los canales
    final_snr_p = np.max(snr_p_values) if snr_p_values else np.nan
    final_snr_s = np.max(snr_s_values) if snr_s_values else np.nan
    
    if verbose and snr_p_values:
        print(f"    SNR-P final (máximo): {final_snr_p:.2f} dB")
    if verbose and snr_s_values:
        print(f"    SNR-S final (máximo): {final_snr_s:.2f} dB")
    
    return final_snr_p, final_snr_s

def process_csv_dataset(csv_file, output_dir, input_freq, target_freq=100.0, 
                       freq_min=1.0, freq_max=20.0, apply_resample=True, 
                       apply_filter=True, verbose=False):
    """
    Procesa todos los archivos definidos en el CSV
    """
    
    try:
        # Leer archivo CSV
        df = pd.read_csv(csv_file)
        print(f"Cargado CSV con {len(df)} registros")
        
        # Verificar columnas requeridas
        required_columns = ['Estacion', 'mseed', 'T-P', 'T-S']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas en CSV: {missing_columns}")
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar columnas de SNR si no existen
        if 'SNR-P' not in df.columns:
            df['SNR-P'] = np.nan
        if 'SNR-S' not in df.columns:
            df['SNR-S'] = np.nan
        
        success_count = 0
        fail_count = 0
        total_channels = 0
        
        print(f"Directorio base MSEED: {MSEED_DIR}")
        print(f"Directorio de salida: {output_dir}")
        
        # Procesar cada archivo
        for idx, row in df.iterrows():
            mseed_filename = row['mseed']
            estacion = row['Estacion']
            tp_time = row['T-P']
            ts_time = row['T-S']
            
            if verbose:
                print(f"\n--- Procesando registro {idx+1}/{len(df)} ---")
                print(f"Estación: {estacion}")
                print(f"Archivo: {mseed_filename}")
                print(f"T-P: {tp_time}")
                print(f"T-S: {ts_time}")
            
            # Construir ruta completa del archivo de entrada
            input_file = os.path.join(MSEED_DIR, mseed_filename)
            
            if not os.path.exists(input_file):
                print(f"ERROR: No se encuentra el archivo {input_file}")
                fail_count += 1
                continue
            
            # Generar nombre de archivo de salida
            output_file = os.path.join(output_dir, mseed_filename)
            
            # Procesar archivo
            success, channels, processed_stream = preprocess_mseed_file(
                input_file, output_file, input_freq, target_freq, 
                freq_min, freq_max, apply_resample, apply_filter, verbose
            )
            
            if success:
                success_count += 1
                total_channels += channels
                
                # Calcular SNR para las fases P y S
                if verbose:
                    print("  Calculando SNR...")
                
                snr_p, snr_s = calculate_snr_for_phases(
                    processed_stream, tp_time, ts_time, verbose
                )
                
                # Actualizar DataFrame con los valores de SNR
                df.at[idx, 'SNR-P'] = snr_p
                df.at[idx, 'SNR-S'] = snr_s
                
                if verbose:
                    if not np.isnan(snr_p):
                        print(f"  ✓ SNR-P: {snr_p:.2f} dB")
                    if not np.isnan(snr_s):
                        print(f"  ✓ SNR-S: {snr_s:.2f} dB")
                
            else:
                fail_count += 1
        
        # Guardar CSV actualizado con SNR
        output_csv = csv_file.replace('.csv', '_with_snr.csv')
        df.to_csv(output_csv, index=False)
        
        print(f"\n=== Procesamiento completado ===")
        print(f"  Éxito: {success_count} archivos")
        print(f"  Fallos: {fail_count} archivos")
        print(f"  Total canales procesados: {total_channels}")
        print(f"  CSV actualizado guardado en: {output_csv}")
        
        # Estadísticas de SNR
        snr_p_valid = df['SNR-P'].dropna()
        snr_s_valid = df['SNR-S'].dropna()
        
        if len(snr_p_valid) > 0:
            print(f"  SNR-P: media={snr_p_valid.mean():.2f} dB, "
                  f"mediana={snr_p_valid.median():.2f} dB, "
                  f"rango=[{snr_p_valid.min():.2f}, {snr_p_valid.max():.2f}] dB")
        
        if len(snr_s_valid) > 0:
            print(f"  SNR-S: media={snr_s_valid.mean():.2f} dB, "
                  f"mediana={snr_s_valid.median():.2f} dB, "
                  f"rango=[{snr_s_valid.min():.2f}, {snr_s_valid.max():.2f}] dB")
        
    except Exception as e:
        print(f"ERROR procesando CSV: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Preprocesa archivos mseed definidos en CSV para modelo GPD con cálculo de SNR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Procesamiento básico usando CSV
  python preprocess_mseed_for_gpd.py -I dataset.csv -O /path/to/output/ --input-freq 250 -V
  
  # Con parámetros personalizados
  python preprocess_mseed_for_gpd.py -I dataset.csv -O /path/to/output/ --input-freq 64 --target-freq 50 --freq-min 2.0 --freq-max 15.0
  
  # Solo preprocesamiento sin remuestreo
  python preprocess_mseed_for_gpd.py -I dataset.csv -O /path/to/output/ --input-freq 100 --no-resample
        """)
    
    parser.add_argument('-I', '--input', type=str, required=True, 
                       help='Archivo CSV con definición de archivos a procesar')
    parser.add_argument('-O', '--output', type=str, required=True,
                       help='Directorio de salida para archivos procesados')
    parser.add_argument('--input-freq', type=float, required=True,
                       help='Frecuencia de muestreo de entrada en Hz (ej: 64, 100, 200, 250)')
    parser.add_argument('--target-freq', type=float, default=100.0,
                       help='Frecuencia de muestreo objetivo en Hz (default: 100.0)')
    parser.add_argument('--freq-min', type=float, default=1.0,
                       help='Frecuencia mínima del filtro en Hz (default: 1.0)')
    parser.add_argument('--freq-max', type=float, default=20.0,
                       help='Frecuencia máxima del filtro en Hz (default: 20.0)')
    parser.add_argument('--no-resample', action='store_true',
                       help='No aplicar remuestreo (solo preprocesamiento)')
    parser.add_argument('--no-filter', action='store_true',
                       help='No aplicar filtro pasa-banda')
    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    print("=== Preprocesador MSEED para GPD con cálculo de SNR ===")
    print(f"Directorio base MSEED: {MSEED_DIR}")
    print(f"Archivo CSV de entrada: {args.input}")
    print(f"Frecuencia de entrada: {args.input_freq} Hz")
    print(f"Frecuencia objetivo: {args.target_freq} Hz")
    
    apply_resample = not args.no_resample
    apply_filter = not args.no_filter
    
    if apply_filter:
        print(f"Filtro pasa-banda: {args.freq_min}-{args.freq_max} Hz")
    else:
        print("Filtro pasa-banda: DESHABILITADO")
    
    if apply_resample and abs(args.input_freq - args.target_freq) > 0.1:
        print(f"Remuestreo: {args.input_freq} → {args.target_freq} Hz")
    else:
        print("Remuestreo: NO REQUERIDO o DESHABILITADO")
    
    print("Preprocesamiento: detrend + sincronización + manejo canales + cálculo SNR")
    
    # Verificar que el archivo CSV existe
    if not os.path.isfile(args.input):
        print(f"ERROR: No se encuentra el archivo CSV {args.input}")
        return
    
    # Procesar dataset definido en CSV
    process_csv_dataset(args.input, args.output, args.input_freq, args.target_freq,
                       args.freq_min, args.freq_max, apply_resample, apply_filter, args.verbose)

if __name__ == "__main__":
    main()