#!/usr/bin/env python3
"""
GPD con procesamiento por chunks usando modelo Keras moderno
Adaptado para gpd_final.keras
Uso:
# Uso básico
python gpd_chunked_keras.py -I estaciones.txt -O picks_detectados.txt -M gpd_final.keras

# Con modo verbose
time python gpd_chunked_keras.py -I estaciones.txt -O picks_detectados.txt -M gpd_final.keras -V  2>&1 | tee log.txt

# Ajustar probabilidad mínima
python gpd_chunked_keras.py -I estaciones.txt -O picks_detectados.txt --min-proba 0.90 -V

# Procesar sin filtrar
python gpd_chunked_keras.py -I estaciones.txt -O picks_detectados.txt --no-filter -V
"""

import numpy as np
import obspy.core as oc
import tensorflow as tf
from scipy.signal import detrend
import argparse
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Configuración
MIN_PROBA = 0.95
FREQ_MIN = 3.0
FREQ_MAX = 20.0
FILTER_DATA = True
DECIMATE_DATA = False
N_SHIFT = 10
BATCH_SIZE = 100

HALF_DUR = 2.00
ONLY_DT = 0.01
N_WIN = int(HALF_DUR/ONLY_DT)
N_FEAT = 2*N_WIN

# Tamaño de chunk (muestras por chunk)
CHUNK_SIZE = 100000  # ~16.7 minutos a 100 Hz
OVERLAP_SIZE = 8000  # Overlap para evitar perder eventos en los bordes

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """Función sliding window original"""
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

def process_chunk(model, chunk_data, chunk_start_time, dt, net, sta, output_file, verbose=False):
    """Procesar un chunk de datos y escribir picks"""
    picks_found = {'P': 0, 'S': 0}
    
    try:
        # Verificar longitud mínima
        min_length = min(len(chunk_data[0]), len(chunk_data[1]), len(chunk_data[2]))
        if min_length < N_FEAT:
            if verbose:
                print(f"    Chunk muy pequeño ({min_length} < {N_FEAT}), saltando")
            return picks_found
        
        # Crear ventanas deslizantes para el chunk
        tt = (np.arange(0, min_length - N_FEAT, N_SHIFT) + N_WIN) * dt
        
        sliding_N = sliding_window(chunk_data[0][:min_length], N_FEAT, stepsize=N_SHIFT)
        sliding_E = sliding_window(chunk_data[1][:min_length], N_FEAT, stepsize=N_SHIFT)
        sliding_Z = sliding_window(chunk_data[2][:min_length], N_FEAT, stepsize=N_SHIFT)
        
        # Verificar que todas las ventanas tienen el mismo tamaño
        min_windows = min(sliding_N.shape[0], sliding_E.shape[0], sliding_Z.shape[0])
        if min_windows == 0:
            return picks_found
            
        # IMPORTANTE: El modelo espera entrada de forma (batch, 400, 3)
        # Reorganizar los datos para el formato correcto
        tr_win = np.zeros((min_windows, N_FEAT, 3), dtype=np.float32)
        
        # Asignar canales en el orden correcto
        tr_win[:,:,0] = sliding_Z[:min_windows]  # Canal Z
        tr_win[:,:,1] = sliding_N[:min_windows]  # Canal N
        tr_win[:,:,2] = sliding_E[:min_windows]  # Canal E
        
        # Normalizar cada ventana individualmente
        for i in range(min_windows):
            max_val = np.max(np.abs(tr_win[i]))
            if max_val > 0:
                tr_win[i] = tr_win[i] / max_val
        
        tt = tt[:min_windows]
        
        # Predicción con el modelo Keras
        if verbose:
            print(f"    Procesando {min_windows} ventanas...")
        
        # El modelo devuelve probabilidades [P, S, Noise]
        predictions = model.predict(tr_win, verbose=0, batch_size=BATCH_SIZE)
        
        prob_P = predictions[:, 0]
        prob_S = predictions[:, 1]
        prob_N = predictions[:, 2]
        
        if verbose and min_windows > 0:
            print(f"    Probabilidades promedio: P={prob_P.mean():.3f}, S={prob_S.mean():.3f}, Noise={prob_N.mean():.3f}")
        
        # Detectar picks P
        from obspy.signal.trigger import trigger_onset
        trigs_P = trigger_onset(prob_P, MIN_PROBA, 0.1)
        
        for trig in trigs_P:
            if trig[1] <= trig[0]:
                continue
            # Encontrar el máximo en la ventana del trigger
            pick_idx = np.argmax(prob_P[trig[0]:trig[1]]) + trig[0]
            stamp_pick = chunk_start_time + tt[pick_idx]
            
            # Escribir pick con probabilidad
            output_file.write(f"{net} {sta} P {stamp_pick.isoformat()} {prob_P[pick_idx]:.3f}\n")
            picks_found['P'] += 1

        # Detectar picks S
        trigs_S = trigger_onset(prob_S, MIN_PROBA, 0.1)
        
        for trig in trigs_S:
            if trig[1] <= trig[0]:
                continue
            # Encontrar el máximo en la ventana del trigger
            pick_idx = np.argmax(prob_S[trig[0]:trig[1]]) + trig[0]
            stamp_pick = chunk_start_time + tt[pick_idx]
            
            # Escribir pick con probabilidad
            output_file.write(f"{net} {sta} S {stamp_pick.isoformat()} {prob_S[pick_idx]:.3f}\n")
            picks_found['S'] += 1
            
        # Limpiar memoria
        del tr_win, predictions, sliding_N, sliding_E, sliding_Z
        gc.collect()
        
    except Exception as e:
        print(f"ERROR procesando chunk: {e}")
        import traceback
        traceback.print_exc()
    
    return picks_found

def validate_model(model_path):
    """Validar que el modelo cargado es correcto"""
    try:
        model = tf.keras.models.load_model(model_path)
        
        # Verificar dimensiones
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        print(f"Modelo cargado: {model_path}")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
        print(f"  Parámetros: {model.count_params():,}")
        
        # Verificar predicción de prueba
        test_input = np.random.randn(1, 400, 3).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        
        assert test_output.shape == (1, 3), f"Output shape incorrecto: {test_output.shape}"
        assert np.allclose(test_output.sum(), 1.0, atol=0.1), "Las probabilidades no suman ~1.0"
        
        print(f"  Test output: P={test_output[0,0]:.3f}, S={test_output[0,1]:.3f}, Noise={test_output[0,2]:.3f}")
        print("✓ Modelo validado correctamente\n")
        
        return model
    except Exception as e:
        print(f"ERROR validando modelo: {e}")
        raise

def main():
    global MIN_PROBA, FILTER_DATA
    parser = argparse.ArgumentParser(description='GPD con procesamiento por chunks - Modelo Keras')
    parser.add_argument('-I', '--input', type=str, required=True, 
                       help='Archivo de entrada con lista de archivos')
    parser.add_argument('-O', '--output', type=str, required=True, 
                       help='Archivo de salida para picks')
    parser.add_argument('-M', '--model', type=str, default='gpd_final.keras',
                       help='Ruta al modelo Keras (default: gpd_final.keras)')
    parser.add_argument('-V', '--verbose', action='store_true', 
                       help='Modo verbose')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, 
                       help='Tamaño de chunk en muestras')
    parser.add_argument('--min-proba', type=float, default=MIN_PROBA,
                       help='Probabilidad mínima para detección')
    parser.add_argument('--no-filter', action='store_true',
                       help='Desactivar filtrado')
    
    args = parser.parse_args()
    
    # Actualizar configuración global
    MIN_PROBA = args.min_proba
    FILTER_DATA = not args.no_filter
    
    print("=== GPD Procesamiento por Chunks (Modelo Keras) ===")
    print(f"Configuración:")
    print(f"  Probabilidad mínima: {MIN_PROBA}")
    print(f"  Filtrado: {FILTER_DATA}")
    print(f"  Tamaño de chunk: {args.chunk_size:,} muestras")
    print(f"  Overlap: {OVERLAP_SIZE:,} muestras\n")
    
    # Leer archivo de entrada
    fdir = []
    with open(args.input) as f:
        for line in f:
            tmp = line.strip().split()
            if len(tmp) >= 3:
                fdir.append([tmp[0], tmp[1], tmp[2]])
    
    print(f"Número de estaciones: {len(fdir)}")
    
    # Cargar y validar modelo
    print("\nCargando modelo GPD...")
    model = validate_model(args.model)
    
    # Procesar cada estación
    total_picks = {'P': 0, 'S': 0}
    
    with open(args.output, 'w') as ofile:
        # Escribir encabezado
        ofile.write("# Network Station Phase Time Probability\n")
        
        for i, files in enumerate(fdir):
            print(f"\n--- Estación {i+1}/{len(fdir)} ---")
            print(f"  Archivos: {files}")
            
            # Verificar archivos
            if not all(os.path.isfile(f) for f in files):
                print("  ERROR: Archivos faltantes, saltando estación")
                continue
            
            try:
                # Cargar todas las trazas
                print("  Cargando trazas...")
                st = oc.Stream()
                
                # Intentar diferentes órdenes de componentes
                for j, file in enumerate(files):
                    tr = oc.read(file)[0]
                    # Intentar identificar componente por el nombre del archivo o canal
                    if 'Z' in file.upper() or (tr.stats.channel and 'Z' in tr.stats.channel.upper()):
                        st.insert(0, tr)  # Z primero
                    elif 'N' in file.upper() or (tr.stats.channel and 'N' in tr.stats.channel.upper()):
                        st.insert(1, tr)  # N segundo
                    else:
                        st.append(tr)  # E último
                
                # Verificar que tenemos 3 componentes
                if len(st) != 3:
                    print(f"  ERROR: Se esperaban 3 componentes, se encontraron {len(st)}")
                    continue
                
                # Sincronizar
                latest_start = max([x.stats.starttime for x in st])
                earliest_stop = min([x.stats.endtime for x in st])
                st.trim(latest_start, earliest_stop)
                
                # Verificar longitudes
                min_samples = min([len(tr.data) for tr in st])
                if min_samples < N_FEAT:
                    print(f"  ERROR: Trazas muy cortas ({min_samples} < {N_FEAT})")
                    continue
                
                total_samples = min_samples
                duration_hours = total_samples / st[0].stats.sampling_rate / 3600
                
                print(f"  Muestras: {total_samples:,}")
                print(f"  Duración: {duration_hours:.2f} horas")
                print(f"  Sampling rate: {st[0].stats.sampling_rate} Hz")
                
                # Preprocesamiento global
                st.detrend(type='linear')
                if FILTER_DATA:
                    print(f"  Aplicando filtro bandpass {FREQ_MIN}-{FREQ_MAX} Hz")
                    st.filter(type='bandpass', freqmin=FREQ_MIN, freqmax=FREQ_MAX)
                
                # Si la frecuencia de muestreo no es 100 Hz, interpolar
                if st[0].stats.sampling_rate != 100.0:
                    print(f"  Interpolando a 100 Hz...")
                    st.interpolate(100.0)
                
                dt = st[0].stats.delta
                net = st[0].stats.network or "XX"
                sta = st[0].stats.station or f"STA{i:03d}"
                start_time = st[0].stats.starttime
                
                # Procesar por chunks
                chunk_num = 0
                start_idx = 0
                station_picks = {'P': 0, 'S': 0}
                
                num_chunks = (total_samples + args.chunk_size - OVERLAP_SIZE - 1) // (args.chunk_size - OVERLAP_SIZE)
                print(f"  Procesando {num_chunks} chunks...")
                
                while start_idx < total_samples - N_FEAT:
                    chunk_num += 1
                    end_idx = min(start_idx + args.chunk_size, total_samples)
                    
                    if args.verbose:
                        progress = (start_idx / total_samples) * 100
                        print(f"    Chunk {chunk_num}/{num_chunks} ({progress:.1f}%): muestras {start_idx:,}-{end_idx:,}")
                    
                    # Extraer chunk
                    chunk_data = []
                    for trace in st:
                        chunk_data.append(trace.data[start_idx:end_idx].copy())
                    
                    # Tiempo de inicio del chunk
                    chunk_start_time = start_time + start_idx * dt
                    
                    # Procesar chunk
                    chunk_picks = process_chunk(model, chunk_data, chunk_start_time, 
                                              dt, net, sta, ofile, verbose=args.verbose)
                    
                    station_picks['P'] += chunk_picks['P']
                    station_picks['S'] += chunk_picks['S']
                    
                    if args.verbose and (chunk_picks['P'] > 0 or chunk_picks['S'] > 0):
                        print(f"      Picks encontrados: P={chunk_picks['P']}, S={chunk_picks['S']}")
                    
                    # Siguiente chunk con overlap
                    start_idx += args.chunk_size - OVERLAP_SIZE
                
                print(f"  Total estación: P={station_picks['P']}, S={station_picks['S']}")
                total_picks['P'] += station_picks['P']
                total_picks['S'] += station_picks['S']
                
            except Exception as e:
                print(f"  ERROR procesando estación: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
    
    print(f"\n=== Procesamiento completado ===")
    print(f"Total de picks detectados:")
    print(f"  Ondas P: {total_picks['P']}")
    print(f"  Ondas S: {total_picks['S']}")
    print(f"  Total: {total_picks['P'] + total_picks['S']}")
    print(f"Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main()