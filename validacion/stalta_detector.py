#!/usr/bin/env python
"""
Detector de eventos sísmicos usando técnica STA/LTA
Procesa archivos mseed de 3 canales con procesamiento por chunks
Uso: 

# Uso básico (detección en cualquier canal)
python stalta_detector.py -I archivo_3ch.mseed -O eventos.out -V

# Detección conservadora (requiere 2 canales)
python stalta_detector.py -I archivo_3ch.mseed -O eventos.out --coincidence 2 -V

# Parámetros personalizados para microsismos
python stalta_detector.py -I archivo_3ch.mseed -O eventos.out \
    --sta 0.2 --lta 10.0 --thr-on 4.0 --min-dur 0.3 -V

# Detección regional (frecuencias más bajas)
python stalta_detector.py -I archivo_3ch.mseed -O eventos.out \
    --freq-min 0.5 --freq-max 10.0 --sta 1.0 --lta 60.0 -V

# Solo procesar 2 horas
python stalta_detector.py -I archivo_3ch.mseed -O eventos.out --hours 2 -V

"""

import numpy as np
import obspy.core as oc
from obspy.signal.trigger import classic_sta_lta, trigger_onset, coincidence_trigger
import argparse
import os
import gc

# Configuración STA/LTA por defecto
sta_len = 0.5      # Ventana corta en segundos
lta_len = 30.0     # Ventana larga en segundos
thr_on = 3.5       # Umbral para activación
thr_off = 1.0      # Umbral para desactivación
min_dur = 0.5      # Duración mínima del evento en segundos

# Configuración de preprocesamiento
freq_min = 1.0     # Frecuencia mínima del filtro
freq_max = 25.0    # Frecuencia máxima del filtro
filter_data = True
decimate_data = False

# Tamaño de chunk (muestras por chunk)
CHUNK_SIZE = 600000  # ~100 minutos a 100 Hz (más grande para STA/LTA)
OVERLAP_SIZE = 6000  # 60 segundos de overlap (importante para ventana LTA)

def process_chunk(chunk_data, chunk_start_time, dt, net, sta, output_file, 
                 sta_sec, lta_sec, thr_on_val, thr_off_val, min_dur_sec, 
                 coincidence_channels=1, verbose=False):
    """
    Procesar un chunk de datos con STA/LTA y escribir detecciones
    
    Parameters:
    -----------
    chunk_data : list
        Lista con arrays de datos de los 3 canales [N, E, Z]
    chunk_start_time : UTCDateTime
        Tiempo de inicio del chunk
    dt : float
        Delta tiempo (1/sampling_rate)
    net, sta : str
        Código de red y estación
    output_file : file object
        Archivo donde escribir las detecciones
    sta_sec, lta_sec : float
        Ventanas STA y LTA en segundos
    thr_on_val, thr_off_val : float
        Umbrales de activación y desactivación
    min_dur_sec : float
        Duración mínima del evento en segundos
    coincidence_channels : int
        Número mínimo de canales que deben disparar para considerar evento
    verbose : bool
        Mostrar información detallada
    """
    
    detections_found = 0
    
    try:
        # Verificar que todos los canales tengan la misma longitud
        lengths = [len(data) for data in chunk_data]
        if len(set(lengths)) > 1:
            min_length = min(lengths)
            chunk_data = [data[:min_length] for data in chunk_data]
            if verbose:
                print(f"      Ajustando longitudes a {min_length} muestras")
        
        if len(chunk_data[0]) == 0:
            return detections_found
        
        # Calcular frecuencia de muestreo
        fs = 1.0 / dt
        
        # Convertir ventanas de tiempo a muestras
        sta_samples = int(sta_sec * fs)
        lta_samples = int(lta_sec * fs)
        
        # Verificar que hay suficientes muestras para LTA
        min_samples_needed = lta_samples + sta_samples
        if len(chunk_data[0]) < min_samples_needed:
            if verbose:
                print(f"      Chunk muy pequeño para LTA ({len(chunk_data[0])} < {min_samples_needed})")
            return detections_found
        
        # Lista para almacenar triggers de cada canal
        channel_triggers = []
        channel_names = ['N', 'E', 'Z']  # Orden asumido
        
        # Procesar cada canal
        for i, (data, ch_name) in enumerate(zip(chunk_data, channel_names)):
            if verbose and i == 0:  # Solo mostrar una vez por chunk
                print(f"      Calculando STA/LTA: STA={sta_sec}s, LTA={lta_sec}s")
            
            # Calcular función característica (envelope o datos originales)
            # Para mejor detección, usamos el valor absoluto
            cf = np.abs(data.astype(np.float64))
            
            # Calcular STA/LTA
            sta_lta = classic_sta_lta(cf, sta_samples, lta_samples)
            
            # Detectar triggers
            triggers = trigger_onset(sta_lta, thr_on_val, thr_off_val)
            
            # Filtrar por duración mínima
            min_samples = int(min_dur_sec * fs)
            valid_triggers = []
            
            for trigger in triggers:
                trigger_samples = trigger[1] - trigger[0]
                if trigger_samples >= min_samples:
                    # Convertir índices a tiempos
                    start_time = chunk_start_time + trigger[0] * dt
                    end_time = chunk_start_time + trigger[1] * dt
                    
                    # Encontrar el pico dentro del trigger
                    peak_idx = np.argmax(sta_lta[trigger[0]:trigger[1]]) + trigger[0]
                    peak_time = chunk_start_time + peak_idx * dt
                    peak_value = sta_lta[peak_idx]
                    
                    valid_triggers.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'peak_time': peak_time,
                        'peak_value': peak_value,
                        'channel': ch_name,
                        'start_idx': trigger[0],
                        'end_idx': trigger[1],
                        'peak_idx': peak_idx
                    })
            
            channel_triggers.append(valid_triggers)
            
            if verbose and len(valid_triggers) > 0:
                print(f"        Canal {ch_name}: {len(valid_triggers)} triggers válidos")
        
        # Aplicar lógica de coincidencia si se requiere
        if coincidence_channels > 1:
            # Buscar eventos que ocurran en múltiples canales dentro de una ventana de tiempo
            coincidence_window = 2.0  # segundos
            events = find_coincident_events(channel_triggers, coincidence_channels, 
                                           coincidence_window, verbose)
        else:
            # Tomar todos los triggers de todos los canales
            events = []
            for ch_triggers in channel_triggers:
                for trigger in ch_triggers:
                    events.append(trigger)
            
            # Ordenar por tiempo de pico
            events.sort(key=lambda x: x['peak_time'])
        
        # Escribir detecciones al archivo
        for event in events:
            # Formato similar a GPD: RED ESTACION TIPO TIEMPO
            # Para STA/LTA no distinguimos P/S, usamos 'E' (Event)
            output_file.write(f"{net} {sta} E {event['peak_time'].isoformat()}\n")
            detections_found += 1
            
            if verbose:
                duration = event['end_time'] - event['start_time']
                channel_info = event.get('channels', event['channel'])
                print(f"        Evento: {event['peak_time'].isoformat()} "
                      f"(duración: {duration:.1f}s, canal(es): {channel_info}, "
                      f"STA/LTA: {event['peak_value']:.2f})")
        
        # Limpiar memoria
        del chunk_data
        gc.collect()
        
    except Exception as e:
        print(f"ERROR procesando chunk: {e}")
    
    return detections_found

def find_coincident_events(channel_triggers, min_channels, time_window, verbose=False):
    """
    Encuentra eventos que ocurren en múltiples canales dentro de una ventana de tiempo
    
    Returns:
    --------
    list: Lista de eventos coincidentes con información combinada
    """
    
    # Combinar todos los triggers con información de canal
    all_triggers = []
    for ch_idx, triggers in enumerate(channel_triggers):
        for trigger in triggers:
            trigger['channel_idx'] = ch_idx
            all_triggers.append(trigger)
    
    # Ordenar por tiempo de pico
    all_triggers.sort(key=lambda x: x['peak_time'])
    
    # Buscar coincidencias
    coincident_events = []
    used_triggers = set()
    
    for i, trigger in enumerate(all_triggers):
        if i in used_triggers:
            continue
        
        # Buscar triggers coincidentes dentro de la ventana de tiempo
        coincident_triggers = [trigger]
        trigger_channels = {trigger['channel_idx']}
        
        for j in range(i + 1, len(all_triggers)):
            if j in used_triggers:
                continue
            
            other_trigger = all_triggers[j]
            time_diff = abs(other_trigger['peak_time'] - trigger['peak_time'])
            
            if time_diff > time_window:
                break  # Los triggers están ordenados, no habrá más coincidencias
            
            # Verificar que no sea del mismo canal
            if other_trigger['channel_idx'] not in trigger_channels:
                coincident_triggers.append(other_trigger)
                trigger_channels.add(other_trigger['channel_idx'])
        
        # Verificar si cumple el criterio de coincidencia
        if len(coincident_triggers) >= min_channels:
            # Crear evento combinado
            # Usar el trigger con mayor STA/LTA como referencia
            best_trigger = max(coincident_triggers, key=lambda x: x['peak_value'])
            
            # Calcular tiempos combinados
            start_times = [t['start_time'] for t in coincident_triggers]
            end_times = [t['end_time'] for t in coincident_triggers]
            
            combined_event = {
                'start_time': min(start_times),
                'end_time': max(end_times),
                'peak_time': best_trigger['peak_time'],
                'peak_value': best_trigger['peak_value'],
                'channels': [t['channel'] for t in coincident_triggers],
                'channel_count': len(coincident_triggers)
            }
            
            coincident_events.append(combined_event)
            
            # Marcar triggers como usados
            for k, t in enumerate(all_triggers):
                if t in coincident_triggers:
                    used_triggers.add(k)
    
    if verbose and len(coincident_events) > 0:
        print(f"        Eventos coincidentes: {len(coincident_events)}")
    
    return coincident_events

def main():
    parser = argparse.ArgumentParser(description='Detector STA/LTA para archivos mseed de 3 canales')
    parser.add_argument('-I', type=str, required=True, 
                       help='Archivo mseed de entrada con 3 canales (en orden: 1º, 2º, 3º componente)')
    parser.add_argument('-O', type=str, required=True, help='Archivo de salida')
    parser.add_argument('-V', '--verbose', action='store_true', help='Información detallada')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, 
                       help='Tamaño de chunk en muestras')
    parser.add_argument('--hours', type=float, default=None,
                       help='Horas a procesar desde el inicio sincronizado (por defecto: todo el archivo)')
    
    # Parámetros específicos de STA/LTA
    parser.add_argument('--sta', type=float, default=sta_len,
                       help=f'Ventana STA en segundos (default: {sta_len})')
    parser.add_argument('--lta', type=float, default=lta_len,
                       help=f'Ventana LTA en segundos (default: {lta_len})')
    parser.add_argument('--thr-on', type=float, default=thr_on,
                       help=f'Umbral de activación (default: {thr_on})')
    parser.add_argument('--thr-off', type=float, default=thr_off,
                       help=f'Umbral de desactivación (default: {thr_off})')
    parser.add_argument('--min-dur', type=float, default=min_dur,
                       help=f'Duración mínima del evento en segundos (default: {min_dur})')
    parser.add_argument('--coincidence', type=int, default=1,
                       help='Número mínimo de canales que deben disparar (1-3, default: 1)')
    
    # Parámetros de filtrado
    parser.add_argument('--freq-min', type=float, default=freq_min,
                       help=f'Frecuencia mínima del filtro en Hz (default: {freq_min})')
    parser.add_argument('--freq-max', type=float, default=freq_max,
                       help=f'Frecuencia máxima del filtro en Hz (default: {freq_max})')
    parser.add_argument('--no-filter', action='store_true',
                       help='No aplicar filtro pasa banda')
    
    args = parser.parse_args()
    
    print("=== Detector STA/LTA - Entrada MSEED Directa ===")
    
    # Verificar archivo de entrada
    if not os.path.isfile(args.I):
        print(f"ERROR: No se encuentra el archivo {args.I}")
        return
    
    # Validar parámetros
    if args.coincidence < 1 or args.coincidence > 3:
        print("ERROR: --coincidence debe estar entre 1 y 3")
        return
    
    print(f"Archivo de entrada: {args.I}")
    print(f"Parámetros STA/LTA: STA={args.sta}s, LTA={args.lta}s")
    print(f"Umbrales: ON={args.thr_on}, OFF={args.thr_off}")
    print(f"Duración mínima: {args.min_dur}s")
    print(f"Coincidencia: {args.coincidence} canal(es)")
    
    if not args.no_filter:
        print(f"Filtro: {args.freq_min}-{args.freq_max} Hz")
    else:
        print("Filtro: DESHABILITADO")
    
    # Procesar archivo
    total_detections = 0
    
    with open(args.O, 'w') as ofile:
        try:
            # Cargar archivo mseed
            print("Cargando archivo mseed...")
            st = oc.read(args.I)
            
            # Verificar 3 canales
            if len(st) != 3:
                print(f"ERROR: Se esperaban 3 trazas, se encontraron {len(st)}")
                return
            
            print("Trazas encontradas (asumiendo orden: 1º, 2º, 3º componente):")
            for i, tr in enumerate(st):
                print(f"  {i+1}: {tr.stats.channel} - {tr.stats.sampling_rate} Hz")
            
            # Sincronizar
            print("Sincronizando trazas...")
            latest_start = np.max([x.stats.starttime for x in st])
            earliest_stop = np.min([x.stats.endtime for x in st])
            st.trim(latest_start, earliest_stop)
            
            # Limitar duración si se especifica
            if args.hours is not None:
                end_limit = latest_start + args.hours * 3600.0
                stop_time = min(earliest_stop, end_limit)
                st.trim(latest_start, stop_time)
                dur_h = float(st[0].stats.endtime - st[0].stats.starttime) / 3600.0
                print(f"Duración analizada: {dur_h:.2f} h")
            
            total_samples = len(st[0].data)
            duration_hours = total_samples / st[0].stats.sampling_rate / 3600
            
            print(f"Muestras totales: {total_samples:,}")
            print(f"Duración: {duration_hours:.1f} horas")
            print(f"Chunks necesarios: {(total_samples + args.chunk_size - 1) // args.chunk_size}")
            
            # Preprocesamiento
            print("Aplicando preprocesamiento...")
            st.detrend(type='linear')
            if not args.no_filter:
                st.filter(type='bandpass', freqmin=args.freq_min, freqmax=args.freq_max)
            
            dt = st[0].stats.delta
            net = st[0].stats.network
            sta = st[0].stats.station
            start_time = st[0].stats.starttime
            
            # Procesar por chunks
            print("Iniciando detección STA/LTA por chunks...")
            chunk_num = 0
            start_idx = 0
            
            while start_idx < total_samples:
                chunk_num += 1
                end_idx = min(start_idx + args.chunk_size, total_samples)
                
                if args.verbose:
                    print(f"  Chunk {chunk_num}: muestras {start_idx}-{end_idx}")
                
                # Extraer chunk
                chunk_data = []
                for trace in st:
                    chunk_data.append(trace.data[start_idx:end_idx])
                
                # Tiempo de inicio del chunk
                chunk_start_time = start_time + start_idx * dt
                
                # Procesar chunk
                chunk_detections = process_chunk(
                    chunk_data, chunk_start_time, dt, net, sta, ofile,
                    args.sta, args.lta, args.thr_on, args.thr_off, args.min_dur,
                    args.coincidence, args.verbose
                )
                
                total_detections += chunk_detections
                
                if args.verbose:
                    print(f"    Eventos detectados: {chunk_detections}")
                
                # Siguiente chunk con overlap
                start_idx += args.chunk_size - OVERLAP_SIZE
                if start_idx >= total_samples - OVERLAP_SIZE:
                    break
            
        except Exception as e:
            print(f"ERROR procesando archivo: {e}")
            return
    
    print(f"\n=== Detección completada ===")
    print(f"Total de eventos detectados: {total_detections}")
    print(f"Resultados en: {args.O}")

if __name__ == "__main__":
    main()