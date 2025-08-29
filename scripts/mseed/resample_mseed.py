#!/usr/bin/env python
"""
Script para remuestrear archivos mseed de 250Hz a 100Hz para uso con modelo GPD
Uso: 

# Archivo individual básico
python resample_mseed.py -I data_250hz.mseed -O data_100hz.mseed

# Con información detallada
python resample_mseed.py -I data_250hz.mseed -O data_100hz.mseed -V

# Procesamiento en lote de un directorio
python resample_mseed.py -I /datos/250hz/ -O /datos/100hz/ --batch -V

# Sin filtro antialiasing (no recomendado)
python resample_mseed.py -I input.mseed -O output.mseed --no-filter

# Filtro personalizado
python resample_mseed.py -I input.mseed -O output.mseed --freq-min 2.0 --freq-max 25.0

"""

import os
import argparse
import glob
import obspy.core as oc
from obspy.signal.filter import bandpass
import numpy as np

def resample_mseed_file(input_file, output_file, target_rate=100.0, 
                       apply_filter=True, freq_min=3.0, freq_max=20.0, verbose=False):
    """
    Remuestrea un archivo mseed de alta frecuencia a 100Hz
    
    Parameters:
    -----------
    input_file : str
        Archivo mseed de entrada
    output_file : str  
        Archivo mseed de salida
    target_rate : float
        Frecuencia de muestreo objetivo (default: 100.0 Hz)
    apply_filter : bool
        Si aplicar filtro antialiasing antes del remuestreo
    freq_min, freq_max : float
        Frecuencias del filtro pasa banda
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
        
        # Procesar cada traza
        for tr in st:
            original_rate = tr.stats.sampling_rate
            
            if verbose:
                print(f"  Procesando {tr.stats.channel}: {original_rate} Hz → {target_rate} Hz")
            
            # Solo procesar si la frecuencia es diferente
            if abs(original_rate - target_rate) > 0.1:
                
                # Aplicar filtro antialiasing si está habilitado
                if apply_filter and original_rate > target_rate:
                    # Calcular frecuencia de Nyquist del objetivo
                    nyquist_target = target_rate / 2.0
                    
                    # Ajustar frecuencias del filtro para evitar aliasing
                    fmax_filter = min(freq_max, nyquist_target * 0.9)  # 90% de Nyquist
                    fmin_filter = freq_min
                    
                    if verbose:
                        print(f"    Aplicando filtro: {fmin_filter}-{fmax_filter} Hz")
                    
                    # Aplicar filtro pasa banda
                    tr.filter('bandpass', freqmin=fmin_filter, freqmax=fmax_filter, 
                             corners=4, zerophase=True)
                
                # Remuestrear
                if verbose:
                    print(f"    Remuestreando a {target_rate} Hz...")
                tr.resample(target_rate)
                
                if verbose:
                    print(f"    Nueva longitud: {len(tr.data)} muestras")
            else:
                if verbose:
                    print(f"    Ya está a {target_rate} Hz, saltando...")
        
        # Guardar archivo remuestreado
        st.write(output_file, format='MSEED')
        
        if verbose:
            print(f"  Guardado en: {output_file}")
            
        return True
        
    except Exception as e:
        print(f"ERROR procesando {input_file}: {e}")
        return False

def process_directory(input_dir, output_dir, target_rate=100.0, 
                     apply_filter=True, freq_min=3.0, freq_max=20.0, verbose=False):
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
    
    for input_file in sorted(mseed_files):
        # Generar nombre de archivo de salida
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, base_name)
        
        # Procesar archivo
        if resample_mseed_file(input_file, output_file, target_rate, 
                              apply_filter, freq_min, freq_max, verbose):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nProcesamiento completado:")
    print(f"  Éxito: {success_count} archivos")
    print(f"  Fallos: {fail_count} archivos")

def main():
    parser = argparse.ArgumentParser(
        description='Remuestrea archivos mseed de alta frecuencia a 100Hz para modelo GPD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Archivo individual
  python resample_mseed.py -I data_250hz.mseed -O data_100hz.mseed -V
  
  # Procesamiento en lote de un directorio
  python resample_mseed.py -I /path/to/250hz_files/ -O /path/to/100hz_files/ --batch -V
  
  # Sin filtro antialiasing
  python resample_mseed.py -I input.mseed -O output.mseed --no-filter
        """)
    
    parser.add_argument('-I', '--input', type=str, required=True, 
                       help='Archivo mseed de entrada o directorio (con --batch)')
    parser.add_argument('-O', '--output', type=str, required=True,
                       help='Archivo mseed de salida o directorio (con --batch)')
    parser.add_argument('--target-rate', type=float, default=100.0,
                       help='Frecuencia de muestreo objetivo en Hz (default: 100.0)')
    parser.add_argument('--batch', action='store_true',
                       help='Procesar todos los archivos .mseed de un directorio')
    parser.add_argument('--no-filter', action='store_true',
                       help='No aplicar filtro antialiasing (no recomendado)')
    parser.add_argument('--freq-min', type=float, default=3.0,
                       help='Frecuencia mínima del filtro en Hz (default: 3.0)')
    parser.add_argument('--freq-max', type=float, default=20.0,
                       help='Frecuencia máxima del filtro en Hz (default: 20.0)')
    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    print("=== Remuestreador MSEED para GPD ===")
    print(f"Frecuencia objetivo: {args.target_rate} Hz")
    
    if not args.no_filter:
        print(f"Filtro antialiasing: {args.freq_min}-{args.freq_max} Hz")
    else:
        print("Filtro antialiasing: DESHABILITADO")
        print("ADVERTENCIA: Sin filtro puede causar aliasing!")
    
    apply_filter = not args.no_filter
    
    if args.batch:
        # Procesamiento en lote
        if not os.path.isdir(args.input):
            print(f"ERROR: {args.input} no es un directorio")
            return
        
        print(f"Modo lote: {args.input} → {args.output}")
        process_directory(args.input, args.output, args.target_rate,
                         apply_filter, args.freq_min, args.freq_max, args.verbose)
    
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
        
        success = resample_mseed_file(args.input, args.output, args.target_rate,
                                    apply_filter, args.freq_min, args.freq_max, args.verbose)
        
        if success:
            print("✓ Conversión exitosa")
        else:
            print("✗ Error en la conversión")

if __name__ == "__main__":
    main()