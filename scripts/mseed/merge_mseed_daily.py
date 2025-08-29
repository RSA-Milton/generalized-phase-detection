#!/usr/bin/env python
"""
Script para fusionar archivos mseed por día y optimizar el tamaño
Formato entrada: CODIGO_AAAAMMDD_hhmmss.mseed
Formato salida:  CODIGO_AAAAMMDD.mseed

Uso: 

# Uso básico con información detallada
python merge_mseed_daily.py -I datos_originales/ -O datos_fusionados/ -V

# Solo mostrar qué se haría (recomendado primero)
python merge_mseed_daily.py -I datos/ -O fusionados/ --dry-run -V

# Guardar todo en un directorio sin subdirectorios
python merge_mseed_daily.py -I datos/ -O . --no-subdir -V

# Sobrescribir archivos existentes
python merge_mseed_daily.py -I datos/ -O fusionados/ --overwrite -V

"""

import os
import glob
import argparse
from collections import defaultdict
import obspy.core as oc
from obspy import UTCDateTime
import re

def parse_filename(filename):
    """
    Parsea el nombre del archivo para extraer código, fecha y hora
    Formato: CODIGO_AAAAMMDD_hhmmss.mseed
    
    Returns:
    --------
    tuple: (codigo, fecha_str, datetime_obj) o None si no coincide
    """
    basename = os.path.basename(filename)
    
    # Pattern: CODIGO_YYYYMMDD_hhmmss.mseed
    pattern = r'^([A-Z0-9]+)_(\d{8})_(\d{6})\.mseed$'
    match = re.match(pattern, basename)
    
    if not match:
        return None
    
    codigo = match.group(1)
    fecha_str = match.group(2)  # YYYYMMDD
    hora_str = match.group(3)   # hhmmss
    
    try:
        # Crear objeto datetime
        año = int(fecha_str[:4])
        mes = int(fecha_str[4:6])
        dia = int(fecha_str[6:8])
        hora = int(hora_str[:2])
        minuto = int(hora_str[2:4])
        segundo = int(hora_str[4:6])
        
        datetime_obj = UTCDateTime(año, mes, dia, hora, minuto, segundo)
        
        return codigo, fecha_str, datetime_obj
        
    except (ValueError, TypeError):
        return None

def group_files_by_day(input_dir, verbose=False):
    """
    Agrupa archivos por código de estación y día
    
    Returns:
    --------
    dict: {(codigo, fecha_str): [lista_archivos_ordenados]}
    """
    
    # Buscar todos los archivos .mseed
    pattern = os.path.join(input_dir, "*.mseed")
    mseed_files = glob.glob(pattern)
    
    if not mseed_files:
        print(f"No se encontraron archivos .mseed en {input_dir}")
        return {}
    
    if verbose:
        print(f"Encontrados {len(mseed_files)} archivos .mseed")
    
    # Agrupar por (codigo, fecha)
    groups = defaultdict(list)
    skipped_files = []
    
    for filepath in mseed_files:
        parsed = parse_filename(filepath)
        
        if parsed is None:
            skipped_files.append(os.path.basename(filepath))
            continue
        
        codigo, fecha_str, datetime_obj = parsed
        key = (codigo, fecha_str)
        groups[key].append((filepath, datetime_obj))
    
    if skipped_files and verbose:
        print(f"Archivos con formato incorrecto (saltados): {len(skipped_files)}")
        for f in skipped_files[:5]:  # Mostrar solo los primeros 5
            print(f"  - {f}")
        if len(skipped_files) > 5:
            print(f"  ... y {len(skipped_files) - 5} más")
    
    # Ordenar archivos dentro de cada grupo por timestamp
    for key in groups:
        groups[key].sort(key=lambda x: x[1])  # Ordenar por datetime_obj
        groups[key] = [x[0] for x in groups[key]]  # Quedarse solo con rutas
    
    if verbose:
        print(f"Grupos encontrados: {len(groups)}")
        for (codigo, fecha), files in groups.items():
            print(f"  {codigo}_{fecha}: {len(files)} archivos")
    
    return groups

def merge_day_files(file_list, output_file, verbose=False):
    """
    Fusiona archivos de un día en un solo archivo optimizado
    
    Parameters:
    -----------
    file_list : list
        Lista de archivos mseed ordenados por tiempo
    output_file : str
        Ruta del archivo de salida
    verbose : bool
        Mostrar información detallada
    
    Returns:
    --------
    bool: True si fue exitoso, False si hubo error
    """
    
    try:
        if verbose:
            print(f"    Fusionando {len(file_list)} archivos...")
        
        # Leer todos los archivos y combinar en un Stream
        st_combined = oc.Stream()
        
        for i, filepath in enumerate(file_list):
            if verbose and i % 10 == 0:  # Progreso cada 10 archivos
                print(f"      Procesando archivo {i+1}/{len(file_list)}")
            
            try:
                st_temp = oc.read(filepath)
                st_combined += st_temp
                
            except Exception as e:
                print(f"      ERROR leyendo {os.path.basename(filepath)}: {e}")
                continue
        
        if len(st_combined) == 0:
            print(f"      ERROR: No se pudieron leer datos de los archivos")
            return False
        
        if verbose:
            print(f"    Trazas totales antes de fusionar: {len(st_combined)}")
        
        # Fusionar trazas del mismo canal
        # Esto combina automáticamente segmentos contiguos del mismo canal
        st_combined.merge(method=1, fill_value='interpolate', interpolation_samples=100)
        
        if verbose:
            print(f"    Trazas después de merge: {len(st_combined)}")
        
        # Ordenar trazas por canal para mantener consistencia
        st_combined.sort(['channel'])
        
        # Mostrar información de las trazas finales
        if verbose:
            print(f"    Trazas finales:")
            for tr in st_combined:
                duration = tr.stats.endtime - tr.stats.starttime
                print(f"      {tr.stats.channel}: {tr.stats.starttime} - {tr.stats.endtime} ({duration:.1f}s)")
        
        # Crear directorio de salida si no existe
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Escribir archivo fusionado
        # MSEED format automáticamente optimiza el almacenamiento
        st_combined.write(output_file, format='MSEED', encoding='STEIM2', reclen=512)
        
        # Calcular estadísticas de compresión
        if verbose:
            total_input_size = sum(os.path.getsize(f) for f in file_list if os.path.exists(f))
            output_size = os.path.getsize(output_file)
            compression_ratio = (1 - output_size / total_input_size) * 100 if total_input_size > 0 else 0
            
            print(f"    Tamaño entrada: {total_input_size / 1024 / 1024:.1f} MB")
            print(f"    Tamaño salida: {output_size / 1024 / 1024:.1f} MB")
            print(f"    Compresión: {compression_ratio:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"    ERROR fusionando archivos: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Fusiona archivos mseed por día optimizando el tamaño',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Fusionar archivos del directorio actual
  python merge_mseed_daily.py -I datos/ -O fusionados/ -V
  
  # Fusionar sin crear subdirectorios
  python merge_mseed_daily.py -I datos/ -O . --no-subdir
  
  # Solo mostrar qué se haría sin ejecutar
  python merge_mseed_daily.py -I datos/ -O fusionados/ --dry-run -V

Formato esperado de archivos:
  CODIGO_YYYYMMDD_hhmmss.mseed
  
Formato de archivos generados:
  CODIGO_YYYYMMDD.mseed
        """)
    
    parser.add_argument('-I', '--input', type=str, required=True,
                       help='Directorio con archivos mseed de entrada')
    parser.add_argument('-O', '--output', type=str, required=True,
                       help='Directorio donde guardar archivos fusionados')
    parser.add_argument('-V', '--verbose', action='store_true',
                       help='Mostrar información detallada del proceso')
    parser.add_argument('--dry-run', action='store_true',
                       help='Mostrar qué se haría sin ejecutar realmente')
    parser.add_argument('--no-subdir', action='store_true',
                       help='No crear subdirectorios por estación')
    parser.add_argument('--overwrite', action='store_true',
                       help='Sobrescribir archivos existentes')
    
    args = parser.parse_args()
    
    print("=== Fusionador MSEED Diario ===")
    
    # Verificar directorio de entrada
    if not os.path.isdir(args.input):
        print(f"ERROR: {args.input} no es un directorio válido")
        return
    
    print(f"Directorio entrada: {args.input}")
    print(f"Directorio salida: {args.output}")
    
    if args.dry_run:
        print("MODO DRY-RUN: Solo mostrando qué se haría")
    
    # Agrupar archivos por día
    groups = group_files_by_day(args.input, args.verbose)
    
    if not groups:
        print("No se encontraron archivos válidos para procesar")
        return
    
    # Crear directorio de salida
    if not args.dry_run:
        os.makedirs(args.output, exist_ok=True)
    
    # Procesar cada grupo
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for (codigo, fecha), file_list in groups.items():
        print(f"\n--- Procesando {codigo}_{fecha} ---")
        print(f"  Archivos a fusionar: {len(file_list)}")
        
        if args.verbose:
            print(f"  Rango temporal:")
            print(f"    Primer archivo: {os.path.basename(file_list[0])}")
            print(f"    Último archivo: {os.path.basename(file_list[-1])}")
        
        # Determinar archivo de salida
        if args.no_subdir:
            output_file = os.path.join(args.output, f"{codigo}_{fecha}.mseed")
        else:
            # Crear subdirectorio por estación
            station_dir = os.path.join(args.output, codigo)
            output_file = os.path.join(station_dir, f"{codigo}_{fecha}.mseed")
        
        # Verificar si ya existe
        if os.path.exists(output_file) and not args.overwrite:
            print(f"  SALTADO: {output_file} ya existe (usar --overwrite para forzar)")
            skip_count += 1
            continue
        
        print(f"  Archivo salida: {output_file}")
        
        if args.dry_run:
            print(f"  [DRY-RUN] Se fusionarían {len(file_list)} archivos")
            success_count += 1
            continue
        
        # Fusionar archivos
        if merge_day_files(file_list, output_file, args.verbose):
            print(f"  ✓ Fusión exitosa")
            success_count += 1
        else:
            print(f"  ✗ Error en la fusión")
            error_count += 1
    
    # Resumen final
    print(f"\n=== Resumen del procesamiento ===")
    print(f"Grupos procesados exitosamente: {success_count}")
    print(f"Grupos saltados: {skip_count}")
    print(f"Grupos con errores: {error_count}")
    print(f"Total grupos: {len(groups)}")
    
    if not args.dry_run and success_count > 0:
        print(f"\nArchivos fusionados guardados en: {args.output}")

if __name__ == "__main__":
    main()