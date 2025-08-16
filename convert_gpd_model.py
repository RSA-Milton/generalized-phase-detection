#!/usr/bin/env python3
"""
Conversor del modelo GPD desde formato JSON+HDF5 legacy a formato .keras moderno
Maneja las capas Lambda problemáticas del modelo original
"""

import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D

def parse_lambda_function(lambda_str):
    """
    Parsea las funciones lambda del modelo GPD original
    Basado en el conocimiento de que el modelo usa funciones estándar
    """
    if "lambda x: tf.keras.activations.softmax" in lambda_str:
        return tf.keras.activations.softmax
    elif "lambda x: tf.abs" in lambda_str:
        return tf.abs
    elif "lambda x: tf.square" in lambda_str:
        return tf.square
    else:
        # Función genérica para casos no identificados
        print(f"Warning: Lambda no reconocida: {lambda_str}")
        return lambda x: x

def load_gpd_weights(hdf5_path):
    """Carga los pesos del archivo HDF5"""
    weights = {}
    with h5py.File(hdf5_path, 'r') as f:
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = obj[:]
        f.visititems(visit_func)
    return weights

# REMOVIDO: create_gpd_model_manual()
# La reconstrucción manual no es viable sin ingeniería reversa completa
# del JSON original y mapeo exacto de pesos

def convert_gpd_model(json_path, hdf5_path, output_path):
    """
    Convierte el modelo GPD desde JSON+HDF5 a formato .keras
    SOLO usando carga directa - sin reconstrucción manual defectuosa
    
    Args:
        json_path: Ruta al archivo model_pol.json
        hdf5_path: Ruta al archivo model_pol_best.hdf5
        output_path: Ruta de salida para el modelo .keras
    """
    
    print("=== Intentando carga directa del modelo GPD ===")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {tf.keras.__version__}")
    
    # Método ÚNICO: Carga directa con deserialización insegura
    # Si esto falla, no hay alternativa viable sin ingeniería reversa completa
    
    # Para TF 2.12 y anteriores - debería funcionar sin configuración especial
    if tf.__version__.startswith('2.12'):
        print("✓ TensorFlow 2.12 detectado - deserialización legacy disponible")
    else:
        print("⚠ Versión de TF diferente a 2.12 - puede requerir configuración especial")
        
        # Para Keras 3.x
        if hasattr(tf.keras.config, 'enable_unsafe_deserialization'):
            tf.keras.config.enable_unsafe_deserialization()
            print("✓ Deserialización insegura habilitada")
        
        # Para TF 2.13+
        if hasattr(tf.keras.utils, 'disable_interactive_logging'):
            tf.keras.utils.disable_interactive_logging()
    
    try:
        # Cargar JSON del modelo
        with open(json_path, 'r') as f:
            model_json = f.read()
            
        print("✓ JSON del modelo cargado")
        
        # Deserializar modelo desde JSON
        model = tf.keras.models.model_from_json(model_json, custom_objects={'tf': tf})
        print("✓ Arquitectura del modelo deserializada")
        
        # Cargar pesos
        model.load_weights(hdf5_path)
        print("✓ Pesos del modelo cargados")
        
        # Compilar modelo (requerido para guardar)
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        print("✓ Modelo compilado")
        
    except ValueError as e:
        if "Lambda" in str(e) or "lambda" in str(e):
            print(f"✗ ERROR: Deserialización de Lambda bloqueada")
            print(f"   Mensaje: {e}")
            print("\n=== SOLUCIONES RECOMENDADAS ===")
            print("1. Usar TensorFlow 2.12.0 exactamente")
            print("2. Usar entorno Docker con TF 2.12")
            print("3. Usar versión legacy de Keras")
            raise RuntimeError("Requiere entorno con deserialización legacy")
        else:
            print(f"✗ ERROR de deserialización: {e}")
            raise
            
    except Exception as e:
        print(f"✗ ERROR inesperado: {e}")
        raise
    
    # Guardar en formato moderno
    print(f"\n=== Guardando modelo convertido ===")
    print(f"Salida: {output_path}")
    
    try:
        model.save(output_path, save_format='keras')
        print("✓ Modelo guardado en formato .keras")
        
        # Verificar el modelo guardado
        loaded_model = tf.keras.models.load_model(output_path)
        print("✓ Modelo convertido verificado")
        
        # Mostrar resumen
        print(f"\n=== Resumen del modelo ===")
        loaded_model.summary()
        
        return model
        
    except Exception as e:
        print(f"✗ ERROR al guardar: {e}")
        raise

def test_converted_model(model_path, test_data_shape=(1, 400, 3)):
    """
    Prueba básica del modelo convertido
    """
    print(f"\n=== Probando modelo convertido ===")
    model = tf.keras.models.load_model(model_path)
    
    # Generar datos de prueba
    test_data = np.random.randn(*test_data_shape)
    
    # Predicción
    predictions = model.predict(test_data)
    print(f"Forma de salida: {predictions.shape}")
    print(f"Predicción ejemplo: {predictions[0]}")
    print(f"Suma probabilidades: {np.sum(predictions[0]):.6f}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convierte modelo GPD a formato moderno')
    parser.add_argument('--json', required=True, help='Ruta a model_pol.json')
    parser.add_argument('--hdf5', required=True, help='Ruta a model_pol_best.hdf5')
    parser.add_argument('--output', required=True, help='Ruta de salida (.keras)')
    parser.add_argument('--test', action='store_true', help='Probar modelo convertido')
    
    args = parser.parse_args()
    
    print("=== Conversor GPD ===")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {tf.keras.__version__}")
    
    try:
        model = convert_gpd_model(args.json, args.hdf5, args.output)
        
        if args.test:
            test_converted_model(args.output)
            
        print(f"\n✓ Conversión completada: {args.output}")
        
    except Exception as e:
        print(f"✗ Error en conversión: {e}")
        import traceback
        traceback.print_exc()