#!/usr/bin/env python3
"""
Conversor unificado GPD: Extrae pesos del modelo original y los aplica a arquitectura recreada
EJECUTAR EN ENTORNO: gpd_py39 (Python 3.9 + TensorFlow 2.12)
"""

import argparse
import os
import sys
import json
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization

def check_environment():
    """Verificar entorno"""
    print("=== Verificación de entorno ===")
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    
    if not tf.__version__.startswith('2.'):
        print("ERROR: Se requiere TensorFlow 2.x")
        return False
        
    print("OK: Entorno moderno correcto")
    return True

def extract_weights_from_savedmodel(savedmodel_path):
    """Extraer pesos del SavedModel sin cargar el modelo completo"""
    print(f"\n=== Extrayendo pesos del SavedModel ===")
    
    try:
        # Cargar SavedModel
        saved_model = tf.saved_model.load(savedmodel_path)
        
        # Extraer variables
        variables = saved_model.variables
        weights_dict = {}
        
        print(f"Variables encontradas: {len(variables)}")
        
        for var in variables:
            name = var.name
            value = var.numpy()
            weights_dict[name] = value
            print(f"  {name}: {value.shape}")
        
        return weights_dict
        
    except Exception as e:
        print(f"ERROR extrayendo pesos: {e}")
        return None

def create_gpd_architecture():
    """Crear arquitectura GPD basada en análisis del modelo original"""
    print("\n=== Creando arquitectura GPD ===")
    
    # Arquitectura basada en el análisis del modelo original
    model = Sequential([
        # Primera capa convolucional
        Conv1D(32, 21, activation='relu', padding='same', input_shape=(400, 3), name='conv1d_1'),
        BatchNormalization(name='batch_normalization_1'),
        
        # Segunda capa convolucional
        Conv1D(32, 15, activation='relu', padding='same', name='conv1d_2'),
        BatchNormalization(name='batch_normalization_2'),
        
        # Tercera capa convolucional
        Conv1D(32, 11, activation='relu', padding='same', name='conv1d_3'),
        BatchNormalization(name='batch_normalization_3'),
        
        # Cuarta capa convolucional + pooling
        Conv1D(32, 9, activation='relu', padding='same', name='conv1d_4'),
        BatchNormalization(name='batch_normalization_4'),
        MaxPooling1D(2, name='max_pooling1d_1'),
        
        # Quinta capa convolucional
        Conv1D(32, 7, activation='relu', padding='same', name='conv1d_5'),
        BatchNormalization(name='batch_normalization_5'),
        
        # Sexta capa convolucional + pooling
        Conv1D(32, 5, activation='relu', padding='same', name='conv1d_6'),
        BatchNormalization(name='batch_normalization_6'),
        MaxPooling1D(2, name='max_pooling1d_2'),
        
        # Séptima capa convolucional
        Conv1D(32, 3, activation='relu', padding='same', name='conv1d_7'),
        BatchNormalization(name='batch_normalization_7'),
        
        # Octava capa convolucional + pooling
        Conv1D(32, 3, activation='relu', padding='same', name='conv1d_8'),
        BatchNormalization(name='batch_normalization_8'),
        MaxPooling1D(2, name='max_pooling1d_3'),
        
        # Aplanar
        Flatten(name='flatten_1'),
        
        # Capas densas
        Dense(512, activation='relu', name='dense_1'),
        Dropout(0.2, name='dropout_1'),
        Dense(512, activation='relu', name='dense_2'),
        Dropout(0.2, name='dropout_2'),
        
        # Salida
        Dense(3, activation='softmax', name='dense_3')
    ])
    
    print(f"Modelo creado con {len(model.layers)} capas")
    return model

def map_weights_smart(model, weights_dict):
    """Mapear pesos usando coincidencia inteligente de nombres"""
    print(f"\n=== Mapeando pesos inteligentemente ===")
    
    mapped_count = 0
    total_layers = 0
    
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            total_layers += 1
            layer_name = layer.name
            
            print(f"\nProcesando capa: {layer_name}")
            
            # Buscar pesos correspondientes
            layer_weights = []
            weight_names = []
            
            # Patrones de búsqueda para diferentes tipos de capas
            if 'conv1d' in layer_name:
                kernel_pattern = f"{layer_name}/kernel:0"
                bias_pattern = f"{layer_name}/bias:0"
                weight_names = [kernel_pattern, bias_pattern]
                
            elif 'batch_normalization' in layer_name:
                gamma_pattern = f"{layer_name}/gamma:0"
                beta_pattern = f"{layer_name}/beta:0"
                moving_mean_pattern = f"{layer_name}/moving_mean:0"
                moving_variance_pattern = f"{layer_name}/moving_variance:0"
                weight_names = [gamma_pattern, beta_pattern, moving_mean_pattern, moving_variance_pattern]
                
            elif 'dense' in layer_name:
                kernel_pattern = f"{layer_name}/kernel:0"
                bias_pattern = f"{layer_name}/bias:0"
                weight_names = [kernel_pattern, bias_pattern]
            
            # Buscar y mapear pesos
            found_weights = []
            for weight_name in weight_names:
                if weight_name in weights_dict:
                    weight_value = weights_dict[weight_name]
                    found_weights.append(weight_value)
                    print(f"  Encontrado: {weight_name} -> {weight_value.shape}")
                else:
                    print(f"  No encontrado: {weight_name}")
            
            # Aplicar pesos si se encontraron todos
            expected_weights = len(layer.get_weights())
            if len(found_weights) == expected_weights:
                try:
                    layer.set_weights(found_weights)
                    mapped_count += 1
                    print(f"  OK: Pesos aplicados a {layer_name}")
                except Exception as e:
                    print(f"  ERROR aplicando pesos a {layer_name}: {e}")
            else:
                print(f"  WARNING: Esperados {expected_weights} pesos, encontrados {len(found_weights)}")
    
    print(f"\n=== Resumen de mapeo ===")
    print(f"Capas con pesos: {total_layers}")
    print(f"Capas mapeadas exitosamente: {mapped_count}")
    print(f"Tasa de éxito: {mapped_count/total_layers*100:.1f}%" if total_layers > 0 else "0%")
    
    return mapped_count > 0

def test_reconstructed_model(model):
    """Probar modelo reconstruido"""
    print(f"\n=== Probando modelo reconstruido ===")
    
    try:
        # Información básica
        print(f"Entrada: {model.input_shape}")
        print(f"Salida: {model.output_shape}")
        print(f"Parámetros: {model.count_params():,}")
        
        # Prueba de predicción
        test_data = np.random.randn(5, 400, 3).astype(np.float32)
        predictions = model.predict(test_data, verbose=0)
        
        print(f"Predicciones: {predictions.shape}")
        
        # Verificar probabilidades válidas
        valid_probs = True
        for i, pred in enumerate(predictions):
            prob_sum = np.sum(pred)
            if not (0.99 <= prob_sum <= 1.01):
                print(f"WARNING: Probabilidades inválidas en muestra {i}: {prob_sum:.6f}")
                valid_probs = False
        
        if valid_probs:
            print("OK: Todas las predicciones son probabilidades válidas")
        
        # Mostrar ejemplos
        print("Ejemplos de predicciones:")
        for i in range(min(3, len(predictions))):
            p, s, n = predictions[i]
            print(f"  Muestra {i+1}: P={p:.3f}, S={s:.3f}, N={n:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Conversor unificado GPD')
    parser.add_argument('--savedmodel', required=True, help='Directorio del SavedModel')
    parser.add_argument('--output', required=True, help='Archivo .keras de salida')
    parser.add_argument('--test', action='store_true', help='Probar modelo reconstruido')
    
    args = parser.parse_args()
    
    print("=== Conversor Unificado GPD ===")
    
    # Verificar entorno
    if not check_environment():
        return False
    
    try:
        # Paso 1: Extraer pesos del SavedModel
        weights_dict = extract_weights_from_savedmodel(args.savedmodel)
        if not weights_dict:
            print("ERROR: No se pudieron extraer pesos")
            return False
        
        # Paso 2: Crear arquitectura GPD
        model = create_gpd_architecture()
        
        # Compilar modelo
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Paso 3: Mapear pesos
        if not map_weights_smart(model, weights_dict):
            print("ERROR: Mapeo de pesos falló")
            return False
        
        # Paso 4: Probar si se solicita
        if args.test:
            if not test_reconstructed_model(model):
                print("WARNING: Modelo no pasa todas las pruebas")
        
        # Paso 5: Guardar modelo
        print(f"\n=== Guardando modelo final ===")
        model.save(args.output, save_format='keras')
        
        if os.path.exists(args.output):
            size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"OK: Modelo guardado en {args.output} ({size_mb:.2f} MB)")
        
        # Verificación final
        print("\n=== Verificación final ===")
        loaded_model = tf.keras.models.load_model(args.output)
        test_input = np.random.randn(1, 400, 3)
        test_output = loaded_model.predict(test_input, verbose=0)
        print(f"Verificación: {test_output[0]}")
        
        print(f"\n=== Conversión completada exitosamente ===")
        print(f"Modelo GPD moderno disponible en: {args.output}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)