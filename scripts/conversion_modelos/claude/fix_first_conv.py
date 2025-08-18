#!/usr/bin/env python3
"""
Script para corregir los pesos de la primera capa Conv1D del modelo GPD
Ejecucion: python fix_first_conv.py - Elegir opción 1 o 2
"""

import tensorflow as tf
import h5py
import numpy as np

def fix_first_conv_layer(model_path='gpd_final.keras', weights_path='model_pol_best.hdf5', 
                         output_path='gpd_fixed.keras'):
    """
    Corrige los pesos de la primera capa convolucional
    """
    print("=== Corrigiendo primera capa Conv1D ===")
    
    # Cargar modelo
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo cargado: {model.count_params():,} parámetros")
    
    # Obtener el Sequential
    sequential = model.get_layer('sequential_1')
    
    # Acceder a la primera capa Conv1D
    conv1d_1 = sequential.get_layer('conv1d_1_1')
    print(f"Primera Conv1D shape actual: {conv1d_1.get_weights()[0].shape}")
    
    # Cargar los pesos correctos desde HDF5
    with h5py.File(weights_path, 'r') as f:
        conv_weights = f['model_weights']['sequential_1']['conv1d_1_1']
        
        # Los pesos están como [bias, kernel] en el HDF5
        kernel = conv_weights[list(conv_weights.keys())[1]][:]  # El kernel es el segundo
        bias = conv_weights[list(conv_weights.keys())[0]][:]    # El bias es el primero
        
        print(f"Kernel shape desde HDF5: {kernel.shape}")
        print(f"Bias shape desde HDF5: {bias.shape}")
        
        # Opción 1: Promediar los 3 canales para obtener 1
        if kernel.shape == (21, 3, 32):
            kernel_averaged = np.mean(kernel, axis=1, keepdims=True)
            print(f"Kernel promediado shape: {kernel_averaged.shape}")
            
            # Establecer los nuevos pesos
            conv1d_1.set_weights([kernel_averaged, bias])
            print("✓ Pesos de conv1d_1_1 corregidos (promediando canales)")
        
        # Opción 2: Usar solo el primer canal
        # kernel_single = kernel[:, 0:1, :]
        # conv1d_1.set_weights([kernel_single, bias])
    
    # Guardar modelo corregido
    model.save(output_path)
    print(f"✓ Modelo corregido guardado en: {output_path}")
    
    # Verificar
    print("\n=== Verificación ===")
    test_input = np.random.randn(1, 400, 3).astype(np.float32)
    test_input = test_input / np.max(np.abs(test_input))
    
    output = model.predict(test_input, verbose=0)
    print(f"Output shape: {output.shape}")
    print(f"Predicción: P={output[0,0]:.3f}, S={output[0,1]:.3f}, Noise={output[0,2]:.3f}")
    
    return model


def alternative_architecture_fix(json_path='model_pol.json', weights_path='model_pol_best.hdf5',
                                output_path='gpd_alternative.keras'):
    """
    Alternativa: Modificar la arquitectura para procesar los 3 canales juntos
    sin separarlos con Lambda
    """
    import json
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, BatchNormalization
    from tensorflow.keras.layers import Activation, Dropout, Flatten
    from tensorflow.keras.models import Model, Sequential
    
    print("=== Construyendo arquitectura alternativa ===")
    
    # Cargar configuración
    with open(json_path, 'r') as f:
        model_json = json.load(f)
    
    # Extraer configuración del Sequential
    if 'config' in model_json:
        layers = model_json['config']['layers']
    else:
        layers = model_json['model_config']['config']['layers']
    
    for layer in layers:
        if layer['class_name'] == 'Sequential':
            if isinstance(layer['config'], dict):
                seq_layers = layer['config']['layers']
            else:
                seq_layers = layer['config']
            break
    
    # Construir modelo SIN separar canales
    inputs = Input(shape=(400, 3), name='input')
    
    # Aplicar el Sequential directamente a la entrada de 3 canales
    x = inputs
    for layer_cfg in seq_layers:
        class_name = layer_cfg['class_name']
        config = layer_cfg['config'].copy()
        
        # Ajustar nombre
        config['name'] = f"{config['name']}_1"
        
        if class_name == 'Conv1D':
            # La primera Conv1D procesará 3 canales
            x = Conv1D(**config)(x)
        elif class_name == 'MaxPooling1D':
            x = MaxPooling1D(**config)(x)
        elif class_name == 'Dense':
            x = Dense(**config)(x)
        elif class_name == 'BatchNormalization':
            x = BatchNormalization(**config)(x)
        elif class_name == 'Activation':
            x = Activation(**config)(x)
        elif class_name == 'Dropout':
            x = Dropout(**config)(x)
        elif class_name == 'Flatten':
            x = Flatten(**config)(x)
    
    outputs = x
    model = Model(inputs=inputs, outputs=outputs, name='gpd_alternative')
    
    print(f"Modelo construido: {model.count_params():,} parámetros")
    
    # Cargar TODOS los pesos (ahora deberían coincidir)
    with h5py.File(weights_path, 'r') as f:
        weights_group = f['model_weights']['sequential_1']
        
        for layer in model.layers[1:]:  # Saltar input layer
            if layer.name in weights_group:
                layer_group = weights_group[layer.name]
                weights = []
                
                for key in sorted(layer_group.keys()):
                    weights.append(layer_group[key][:])
                
                # Invertir para Conv1D y Dense si es necesario
                if isinstance(layer, (Conv1D, Dense)) and len(weights) == 2:
                    weights = [weights[1], weights[0]]
                
                try:
                    layer.set_weights(weights)
                    print(f"✓ Pesos cargados para {layer.name}")
                except Exception as e:
                    print(f"⚠ Error en {layer.name}: {e}")
    
    # Guardar
    model.save(output_path)
    print(f"✓ Modelo alternativo guardado en: {output_path}")
    
    return model


if __name__ == "__main__":
    import sys
    
    print("Opciones de corrección:")
    print("1. Corregir primera capa (promediar canales)")
    print("2. Arquitectura alternativa (sin separar canales)")
    
    opcion = input("\nElegir opción (1 o 2): ")
    
    if opcion == "1":
        fix_first_conv_layer()
    elif opcion == "2":
        alternative_architecture_fix()
    else:
        print("Opción no válida")