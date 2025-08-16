#!/usr/bin/env python
# Versión debug de gpd_predict.py con optimizaciones de memoria

import string
import time
import argparse as ap
import sys
import os
import gc  # Garbage collector

import numpy as np
import obspy.core as oc
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import losses
from keras.models import model_from_json
import tensorflow as tf
import matplotlib as mpl
import pylab as plt
mpl.rcParams['pdf.fonttype'] = 42

#####################
# Hyperparameters - OPTIMIZADOS PARA MEMORIA
min_proba = 0.95
freq_min = 3.0
freq_max = 20.0
filter_data = True
decimate_data = False
n_shift = 10
n_gpu = 0  # CAMBIADO: Usar CPU
#####################

# CAMBIADO: Batch size más pequeño para evitar out of memory
batch_size = 50  # Reducido de 3000 a 50

half_dur = 2.00
only_dt = 0.01
n_win = int(half_dur/only_dt)
n_feat = 2*n_win

def print_memory_usage():
    """Imprimir uso de memoria actual"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memoria usada: {memory_mb:.1f} MB")

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """Función sliding_window original"""
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

    if copy:
        return strided.copy()
    else:
        return strided

if __name__ == "__main__":
    parser = ap.ArgumentParser(
        prog='gpd_predict_debug.py',
        description='GPD con optimizaciones de memoria y debug')
    parser.add_argument('-I', type=str, required=True, help='Input file')
    parser.add_argument('-O', type=str, required=True, help='Output file')
    parser.add_argument('-P', default=True, action='store_false', help='Suppress plotting')
    parser.add_argument('-V', default=False, action='store_true', help='Verbose')
    parser.add_argument('--max-length', type=int, default=60000, 
                       help='Máximo número de muestras a procesar por archivo')
    
    args = parser.parse_args()
    plot = args.P

    print("=== GPD Predict Debug ===")
    print_memory_usage()

    # Leer archivo de entrada
    print(f"Leyendo archivo de entrada: {args.I}")
    fdir = []
    with open(args.I) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    
    nsta = len(fdir)
    print(f"Número de estaciones: {nsta}")

    # Verificar archivos de datos
    print("Verificando archivos de datos...")
    for i, files in enumerate(fdir):
        for j, filepath in enumerate(files):
            if not os.path.isfile(filepath):
                print(f"ERROR: Archivo no existe: {filepath}")
                sys.exit(1)
            else:
                # Información del archivo
                try:
                    tr = oc.read(filepath)[0]
                    print(f"  Estación {i+1}, Canal {j+1}: {len(tr.data)} muestras, "
                          f"SR: {tr.stats.sampling_rate} Hz")
                except Exception as e:
                    print(f"ERROR leyendo {filepath}: {e}")
                    sys.exit(1)

    # Cargar modelo
    print("Cargando modelo GPD...")
    print_memory_usage()
    
    try:
        json_file = open('model_pol.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, custom_objects={'tf':tf})
        model.load_weights("model_pol_best.hdf5")
        print("OK: Modelo cargado")
        print_memory_usage()
    except Exception as e:
        print(f"ERROR cargando modelo: {e}")
        sys.exit(1)

    # Archivo de salida
    ofile = open(args.O, 'w')
    print(f"Archivo de salida: {args.O}")

    # Procesar cada estación
    for i in range(nsta):
        print(f"\n--- Procesando estación {i+1}/{nsta} ---")
        print_memory_usage()
        
        # Cargar datos
        print("Cargando trazas sísmicas...")
        try:
            st = oc.Stream()
            st += oc.read(fdir[i][0])
            st += oc.read(fdir[i][1]) 
            st += oc.read(fdir[i][2])
            
            # Sincronizar trazas
            latest_start = np.max([x.stats.starttime for x in st])
            earliest_stop = np.min([x.stats.endtime for x in st])
            st.trim(latest_start, earliest_stop)
            
            print(f"Duración de datos: {st[0].stats.endtime - st[0].stats.starttime} s")
            print(f"Muestras por canal: {len(st[0].data)}")
            
        except Exception as e:
            print(f"ERROR cargando trazas: {e}")
            continue

        # Limitar longitud si es muy grande
        if len(st[0].data) > args.max_length:
            print(f"LIMITANDO: Reduciendo de {len(st[0].data)} a {args.max_length} muestras")
            for tr in st:
                tr.data = tr.data[:args.max_length]

        # Preprocesamiento
        print("Preprocesando datos...")
        st.detrend(type='linear')
        if filter_data:
            st.filter(type='bandpass', freqmin=freq_min, freqmax=freq_max)
        if decimate_data:
            st.interpolate(100.0)

        # Preparar variables
        sr = st[0].stats.sampling_rate
        dt = st[0].stats.delta
        net = st[0].stats.network
        sta = st[0].stats.station

        # Crear ventanas deslizantes
        print("Creando ventanas deslizantes...")
        print_memory_usage()
        
        try:
            tt = (np.arange(0, st[0].data.size, n_shift) + n_win) * dt
            
            sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)
            sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)
            sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)
            
            print(f"Ventanas creadas: {sliding_N.shape[0]}")
            print_memory_usage()
            
            # Apilar ventanas
            tr_win = np.zeros((sliding_N.shape[0], n_feat, 3), dtype=np.float32)
            tr_win[:,:,0] = sliding_N
            tr_win[:,:,1] = sliding_E
            tr_win[:,:,2] = sliding_Z
            
            # Normalizar
            tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
            tt = tt[:tr_win.shape[0]]
            
            print(f"Forma de datos para predicción: {tr_win.shape}")
            print_memory_usage()
            
        except Exception as e:
            print(f"ERROR en ventanas deslizantes: {e}")
            continue

        # Predicción
        print("Ejecutando predicción...")
        try:
            if args.V:
                ts = model.predict(tr_win, verbose=True, batch_size=batch_size)
            else:
                ts = model.predict(tr_win, verbose=False, batch_size=batch_size)
            
            print(f"Predicciones completadas: {ts.shape}")
            print_memory_usage()
            
        except Exception as e:
            print(f"ERROR en predicción: {e}")
            continue

        # Extraer probabilidades
        prob_P = ts[:,0]
        prob_S = ts[:,1]
        prob_N = ts[:,2]
        
        print(f"Probabilidades P - Max: {np.max(prob_P):.3f}, Mean: {np.mean(prob_P):.3f}")
        print(f"Probabilidades S - Max: {np.max(prob_S):.3f}, Mean: {np.mean(prob_S):.3f}")

        # Detectar picks P
        from obspy.signal.trigger import trigger_onset
        trigs = trigger_onset(prob_P, min_proba, 0.1)
        p_picks = []
        
        print(f"Triggers P encontrados: {len(trigs)}")
        for trig in trigs:
            if trig[1] == trig[0]:
                continue
            pick = np.argmax(ts[trig[0]:trig[1], 0]) + trig[0]
            stamp_pick = st[0].stats.starttime + tt[pick]
            p_picks.append(stamp_pick)
            ofile.write("%s %s P %s\n" % (net, sta, stamp_pick.isoformat()))

        # Detectar picks S
        trigs = trigger_onset(prob_S, min_proba, 0.1)
        s_picks = []
        
        print(f"Triggers S encontrados: {len(trigs)}")
        for trig in trigs:
            if trig[1] == trig[0]:
                continue
            pick = np.argmax(ts[trig[0]:trig[1], 1]) + trig[0]
            stamp_pick = st[0].stats.starttime + tt[pick]
            s_picks.append(stamp_pick)
            ofile.write("%s %s S %s\n" % (net, sta, stamp_pick.isoformat()))

        print(f"Picks detectados - P: {len(p_picks)}, S: {len(s_picks)}")

        # Limpiar memoria
        del tr_win, ts, sliding_N, sliding_E, sliding_Z
        gc.collect()
        print_memory_usage()

        # Generar gráfico si se solicita
        if plot and (len(p_picks) > 0 or len(s_picks) > 0):
            print("Generando gráfico...")
            try:
                fig = plt.figure(figsize=(12, 10))
                ax = []
                ax.append(fig.add_subplot(4,1,1))
                ax.append(fig.add_subplot(4,1,2,sharex=ax[0]))
                ax.append(fig.add_subplot(4,1,3,sharex=ax[0]))
                ax.append(fig.add_subplot(4,1,4,sharex=ax[0]))
                
                # Graficar trazas
                for j in range(3):
                    ax[j].plot(np.arange(st[j].data.size)*dt, st[j].data, 'k-', lw=0.5)
                    ax[j].set_ylabel(['N', 'E', 'Z'][j])
                
                # Graficar probabilidades (submuestreadas para visualización)
                step = max(1, len(tt)//1000)  # Máximo 1000 puntos
                ax[3].plot(tt[::step], prob_P[::step], 'r-', lw=1, label='P')
                ax[3].plot(tt[::step], prob_S[::step], 'b-', lw=1, label='S')
                ax[3].set_ylabel('Probabilidad')
                ax[3].set_xlabel('Tiempo (s)')
                ax[3].legend()
                
                # Marcar picks
                for p_pick in p_picks:
                    for j in range(4):
                        ax[j].axvline(p_pick-st[0].stats.starttime, color='red', lw=1, alpha=0.7)
                
                for s_pick in s_picks:
                    for j in range(4):
                        ax[j].axvline(s_pick-st[0].stats.starttime, color='blue', lw=1, alpha=0.7)
                
                plt.suptitle(f'Estación {net}.{sta} - P: {len(p_picks)}, S: {len(s_picks)} picks')
                plt.tight_layout()
                
                # Guardar en lugar de mostrar
                plot_name = f"gpd_plot_{net}_{sta}_{i+1}.png"
                plt.savefig(plot_name, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Gráfico guardado: {plot_name}")
                
            except Exception as e:
                print(f"ERROR generando gráfico: {e}")

    ofile.close()
    print(f"\n=== Procesamiento completado ===")
    print(f"Resultados guardados en: {args.O}")
    
    # Mostrar resumen del archivo de salida
    if os.path.exists(args.O):
        with open(args.O, 'r') as f:
            lines = f.readlines()
        p_count = len([l for l in lines if ' P ' in l])
        s_count = len([l for l in lines if ' S ' in l])
        print(f"Total de picks detectados - P: {p_count}, S: {s_count}")
        
        if len(lines) > 0:
            print("Primeros picks detectados:")
            for line in lines[:5]:
                print(f"  {line.strip()}")
        else:
            print("ADVERTENCIA: No se detectaron picks")
    
    print_memory_usage()