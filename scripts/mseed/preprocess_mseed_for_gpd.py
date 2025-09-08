#!/usr/bin/env python
"""
Script para preprocesar archivos mseed para uso con modelo GPD
Combina remuestreo, filtrado y preprocesamiento completo
Maneja archivos de un solo canal replicándolo a 3 canales

Uso: 

# Archivo individual básico
python preprocess_mseed_for_gpd.py -I data_250hz.mseed -O data_processed.mseed --input-freq 250

# Con información detallada
python preprocess_mseed_for_gpd.py -I data_100hz.mseed -O data_processed.mseed --input-freq 100 -V

# Procesamiento en lote de un directorio
python preprocess_mseed_for_gpd.py -I /datos/entrada/ -O /datos/salida/ --input-freq 200 --batch -V

# Con parámetros personalizados
python preprocess_mseed_for_gpd.py -I input.mseed -O output.mseed --input-freq 64 --target-freq 100 --freq-min 2.0 --freq-max 25.0

# Sin remuestreo (solo preprocesamiento)
python preprocess_mseed_for_gpd.py -I input.mseed -O output.mseed --input-freq 100 --target-freq 100 --no-resample

"""

import os
import argparse
import glob
import obspy.core as oc
import numpy as np

def preprocess_mseed_file(input_file, output_file, input_freq, target_freq=100.0, 
                         freq_min=3.0, freq_max=20.0, apply_resample=True, 
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
            
        return True, len(st)
        
    except Exception as e:
        print(f"ERROR procesando {input_file}: {e}")
        return False, 0

def process_directory(input_dir, output_dir, input_freq, target_freq=100.0, 
                     freq_min=3.0, freq_max=20.0, apply_resample=True, 
                     apply_filter=True, verbose=False):
    """
    Procesa todos los archivos .mseed de un directorio
    """
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar archivos .mseed
    pattern = os.path.join(input_dir, "*.mseed")
    mseed_files = glob.glob(pattern)
    
    if not mseed_files:
        print(f"No se encontraron archivos .mseed en {input_dir}")
        return
    
    print(f"Encontrados {len(mseed_files)} archivos .mseed")
    
    success_count = 0
    fail_count = 0
    total_channels = 0
    
    for input_file in sorted(mseed_files):
        # Generar nombre de archivo de salida
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, base_name)
        
        # Procesar archivo
        success, channels = preprocess_mseed_file(
            input_file, output_file, input_freq, target_freq, 
            freq_min, freq_max, apply_resample, apply_filter, verbose
        )
        
        if success:
            success_count += 1
            total_channels += channels
        else:
            fail_count += 1
    
    print(f"\n=== Procesamiento completado ===")
    print(f"  Éxito: {success_count} archivos")
    print(f"  Fallos: {fail_count} archivos")
    print(f"  Total canales procesados: {total_channels}")

def main():
    parser = argparse.ArgumentParser(
        description='Preprocesa archivos mseed para modelo GPD (remuestreo + filtrado + preprocesamiento)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Archivo individual con frecuencia específica
  python preprocess_mseed_for_gpd.py -I data_250hz.mseed -O data_processed.mseed --input-freq 250 -V
  
  # Procesamiento en lote
  python preprocess_mseed_for_gpd.py -I /path/to/entrada/ -O /path/to/salida/ --input-freq 200 --batch -V
  
  # Solo preprocesamiento sin remuestreo
  python preprocess_mseed_for_gpd.py -I input.mseed -O output.mseed --input-freq 100 --no-resample
  
  # Con parámetros personalizados
  python preprocess_mseed_for_gpd.py -I input.mseed -O output.mseed --input-freq 64 --target-freq 50 --freq-min 1.0 --freq-max 10.0
        """)
    
    parser.add_argument('-I', '--input', type=str, required=True, 
                       help='Archivo mseed de entrada o directorio (con --batch)')
    parser.add_argument('-O', '--output', type=str, required=True,
                       help='Archivo mseed de salida o directorio (con --batch)')
    parser.add_argument('--input-freq', type=float, required=True,
                       help='Frecuencia de muestreo de entrada en Hz (ej: 64, 100, 200, 250)')
    parser.add_argument('--target-freq', type=float, default=100.0,
                       help='Frecuencia de muestreo objetivo en Hz (default: 100.0)')
    parser.add_argument('--freq-min', type=float, default=3.0,
                       help='Frecuencia mínima del filtro en Hz (default: 3.0)')
    parser.add_argument('--freq-max', type=float, default=20.0,
                       help='Frecuencia máxima del filtro en Hz (default: 20.0)')
    parser.add_argument('--batch', action='store_true',
                       help='Procesar todos los archivos .mseed de un directorio')
    parser.add_argument('--no-resample', action='store_true',
                       help='No aplicar remuestreo (solo preprocesamiento)')
    parser.add_argument('--no-filter', action='store_true',
                       help='No aplicar filtro pasa-banda')
    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    print("=== Preprocesador MSEED para GPD ===")
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
    
    print("Preprocesamiento: detrend + sincronización + manejo canales")
    
    if args.batch:
        # Procesamiento en lote
        if not os.path.isdir(args.input):
            print(f"ERROR: {args.input} no es un directorio")
            return
        
        print(f"Modo lote: {args.input} → {args.output}")
        process_directory(args.input, args.output, args.input_freq, args.target_freq,
                         args.freq_min, args.freq_max, apply_resample, apply_filter, args.verbose)
    
    else:
        # Archivo individual
        if not os.path.isfile(args.input):
            print(f"ERROR: No se encuentra el archivo {args.input}")
            return
        
        print(f"Archivo individual: {args.input} → {args.output}")
        
        # Crear directorio de salida si es necesario
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        success, channels = preprocess_mseed_file(
            args.input, args.output, args.input_freq, args.target_freq,
            args.freq_min, args.freq_max, apply_resample, apply_filter, args.verbose
        )
        
        if success:
            print(f"✓ Conversión exitosa - {channels} canales procesados")
        else:
            print("✗ Error en la conversión")

if __name__ == "__main__":
    main()