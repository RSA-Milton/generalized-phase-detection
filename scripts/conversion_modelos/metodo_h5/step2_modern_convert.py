#!/usr/bin/env python3
"""
Paso 2: Cargar SavedModel y convertir a formato .keras moderno
EJECUTAR EN ENTORNO: gpd_py39 (Python 3.9 + TensorFlow 2.12)
"""

import argparse
import os
import sys
import json
import numpy as np
import tensorflow as tf

def check_environment():
    """Verificar que estamos en el entorno moderno correcto"""
    print("=== Verificación de entorno ===")
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    
    # Verificar versiones críticas
    if not sys.version.startswith('3.9'):
        print("WARNING: Se recomienda Python 3.9")
        
    if not tf.__version__.startswith('2.'):
        print("ERROR: Se requiere TensorFlow 2.x")
        return False
        
    print("OK: Entorno moderno correcto")
    return True

def load_conversion_info(h5_path):
    """Cargar información del paso 1 desde el directorio del archivo H5"""
    h5_dir = os.path.dirname(h5_path) or '.'
    info_path = os.path.join(h5_dir, "conversion_info.json")
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        print("=== Información del paso 1 ===")
        print(f"Método: {info.get('method', 'unknown')}")
        print(f"Archivos fuente: {info['source_files']}")
        print(f"Parámetros del modelo: {info['model_info']['total_params']:,}")
        print(f"Entorno origen: TF {info['environment']['tensorflow_version']}")
        return info
    else:
        print("WARNING: No se encontró información del paso 1")
        return None

def load_h5_model(h5_path):
    """Cargar modelo H5 completo exportado desde el paso 1"""
    print(f"\n=== Cargando modelo H5 completo ===")
    print(f"Ruta: {h5_path}")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Archivo H5 no encontrado: {h5_path}")
    
    try:
        # Cargar modelo H5 completo
        print("Cargando modelo H5...")
        model = tf.keras.models.load_model(h5_path)
        print("OK: Modelo H5 cargado correctamente")
        
        return model
        
    except Exception as e:
        print(f"ERROR cargando modelo H5: {e}")
        raise

def convert_to_keras_model(h5_path):
    """Convertir modelo H5 legacy a modelo Keras moderno"""
    print(f"\n=== Convirtiendo modelo H5 a Keras moderno ===")
    
    try:
        # Cargar modelo H5 directamente
        print("Cargando modelo H5 como modelo Keras...")
        model = tf.keras.models.load_model(h5_path)
        print("OK: Modelo cargado como Keras nativo")
        
        return model
        
    except Exception as e:
        print(f"ERROR en carga: {e}")
        print("El modelo H5 puede tener incompatibilidades entre versiones de Keras")
        raise

def test_converted_model(model, original_info=None):
    """Probar modelo convertido"""
    print(f"\n=== Probando modelo convertido ===")
    
    try:
        # Verificar arquitectura
        print(f"Forma de entrada: {model.input_shape}")
        print(f"Forma de salida: {model.output_shape}")
        print(f"Total parámetros: {model.count_params():,}")
        
        # Comparar con información original si está disponible
        if original_info:
            orig_params = original_info['model_info']['total_params']
            curr_params = model.count_params()
            if orig_params != curr_params:
                print(f"WARNING: Parámetros diferentes. Original: {orig_params:,}, Actual: {curr_params:,}")
            else:
                print("OK: Número de parámetros coincide")
        
        # Prueba funcional
        print("Probando predicción...")
        test_data = np.random.randn(3, 400, 3).astype(np.float32)
        predictions = model.predict(test_data, verbose=0)
        
        print(f"Forma de predicciones: {predictions.shape}")
        
        # Verificar probabilidades
        all_valid = True
        for i, pred in enumerate(predictions):
            prob_sum = np.sum(pred)
            if not (0.99 <= prob_sum <= 1.01):
                print(f"WARNING: Probabilidades inválidas en muestra {i}: {prob_sum:.6f}")
                all_valid = False
        
        if all_valid:
            print("OK: Todas las predicciones son probabilidades válidas")
        
        # Mostrar ejemplo de predicción
        print(f"Ejemplo predicción [P, S, Ruido]: {predictions[0]}")
        
        return True
        
    except Exception as e:
        print(f"ERROR en prueba: {e}")
        return False

def save_keras_model(model, output_path):
    """Guardar modelo en formato .keras"""
    print(f"\n=== Guardando modelo .keras ===")
    print(f"Archivo de salida: {output_path}")
    
    try:
        # Guardar en formato .keras
        model.save(output_path, save_format='keras')
        print("OK: Modelo guardado en formato .keras")
        
        # Verificar archivo creado
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Tamaño del archivo: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"ERROR guardando modelo: {e}")
        return False

def verify_final_model(keras_path):
    """Verificación final del modelo .keras"""
    print(f"\n=== Verificación final ===")
    
    try:
        # Cargar modelo guardado
        print("Cargando modelo .keras para verificación...")
        final_model = tf.keras.models.load_model(keras_path)
        print("OK: Modelo .keras carga correctamente")
        
        # Prueba rápida
        test_data = np.random.randn(1, 400, 3).astype(np.float32)
        prediction = final_model.predict(test_data, verbose=0)
        print(f"Predicción de prueba: {prediction[0]}")
        print(f"Suma probabilidades: {np.sum(prediction[0]):.6f}")
        
        print("OK: Verificación final exitosa")
        return True
        
    except Exception as e:
        print(f"ERROR en verificación final: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Paso 2: Convertir H5 legacy a .keras moderno')
    parser.add_argument('--h5', required=True, help='Archivo H5 completo del paso 1')
    parser.add_argument('--output', required=True, help='Archivo de salida .keras')
    parser.add_argument('--test', action='store_true', help='Probar modelo convertido')
    
    args = parser.parse_args()
    
    print("=== Paso 2: Conversión H5 legacy a formato .keras moderno ===")
    
    # Verificar entorno
    if not check_environment():
        print("ERROR: Entorno incorrecto. Use: micromamba activate gpd_py39")
        return False
    
    try:
        # Cargar información del paso 1
        conversion_info = load_conversion_info(args.h5)
        
        # Convertir modelo H5 a modelo Keras moderno
        model = convert_to_keras_model(args.h5)
        
        if model is None:
            print("ERROR: No se pudo convertir modelo H5 a modelo Keras")
            return False
        
        # Probar si se solicita
        if args.test:
            if not test_converted_model(model, conversion_info):
                print("WARNING: Modelo no pasa todas las pruebas")
        
        # Guardar en formato .keras
        if not save_keras_model(model, args.output):
            print("ERROR: Falló el guardado")
            return False
        
        # Verificación final
        if not verify_final_model(args.output):
            print("ERROR: Falló la verificación final")
            return False
        
        print(f"\n=== Paso 2 completado exitosamente ===")
        print(f"Modelo .keras guardado en: {args.output}")
        print(f"El modelo está listo para usar en entornos modernos")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)