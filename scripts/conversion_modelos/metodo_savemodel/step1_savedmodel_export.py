#!/usr/bin/env python3
"""
Paso 1: Cargar modelo GPD legacy y exportar a SavedModel
EJECUTAR EN ENTORNO: gpd_python36 (Python 3.6 + TensorFlow 1.14)
"""

import argparse
import os
import sys
import json
import numpy as np

# Imports para TensorFlow 1.x
import tensorflow as tf
import keras
from keras.models import model_from_json

def check_environment():
    """Verificar que estamos en el entorno correcto"""
    print("=== Verificación de entorno ===")
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {keras.__version__}")
    
    # Verificar versiones críticas
    if not sys.version.startswith('3.6'):
        print("ERROR: Se requiere Python 3.6 exactamente")
        return False
        
    if not tf.__version__.startswith('1.'):
        print("ERROR: Se requiere TensorFlow 1.x")
        return False
        
    print("OK: Entorno legacy correcto")
    return True

def load_gpd_model(json_path, hdf5_path):
    """Cargar modelo GPD desde archivos legacy"""
    print("\n=== Cargando modelo GPD legacy ===")
    
    # Verificar archivos
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No encontrado: {json_path}")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"No encontrado: {hdf5_path}")
        
    print(f"Cargando arquitectura desde: {json_path}")
    
    try:
        # Cargar JSON del modelo
        with open(json_path, 'r') as f:
            model_json = f.read()
        print("OK: JSON del modelo cargado")
        
        # Deserializar modelo (aquí es donde fallaría en Python 3.9)
        print("Deserializando arquitectura del modelo...")
        model = model_from_json(model_json, custom_objects={'tf': tf})
        print("OK: Arquitectura deserializada")
        
        # Cargar pesos
        print(f"Cargando pesos desde: {hdf5_path}")
        model.load_weights(hdf5_path)
        print("OK: Pesos cargados")
        
        # Compilar modelo
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        print("OK: Modelo compilado")
        
        return model
        
    except Exception as e:
        print(f"ERROR cargando modelo: {e}")
        raise

def test_model_functionality(model):
    """Prueba básica del modelo cargado"""
    print("\n=== Probando funcionalidad del modelo ===")
    
    try:
        # Verificar arquitectura
        print(f"Forma de entrada: {model.input_shape}")
        print(f"Forma de salida: {model.output_shape}")
        print(f"Total parámetros: {model.count_params():,}")
        
        # Prueba con datos sintéticos
        print("Probando predicción con datos sintéticos...")
        test_data = np.random.randn(2, 400, 3).astype(np.float32)
        predictions = model.predict(test_data)
        
        print(f"Forma de predicciones: {predictions.shape}")
        print(f"Suma de probabilidades muestra 1: {np.sum(predictions[0]):.6f}")
        print(f"Suma de probabilidades muestra 2: {np.sum(predictions[1]):.6f}")
        
        # Verificar que son probabilidades válidas
        for i, pred in enumerate(predictions):
            if not (0.99 <= np.sum(pred) <= 1.01):
                print(f"WARNING: Probabilidades inválidas en muestra {i}")
                return False
                
        print("OK: Modelo funciona correctamente")
        return True
        
    except Exception as e:
        print(f"ERROR en prueba: {e}")
        return False

def export_to_savedmodel(model, output_path):
    """Exportar modelo a formato SavedModel (compatible con TF 2.x)"""
    print(f"\n=== Exportando a SavedModel ===")
    print(f"Directorio de salida: {output_path}")
    
    try:
        # Crear directorio si no existe
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Método mejorado: Usar model.save() de Keras en lugar de tf.saved_model.simple_save
        print("Intentando exportación con Keras model.save()...")
        
        try:
            # Compilar modelo antes de guardar (si no está compilado)
            if not hasattr(model, 'optimizer') or model.optimizer is None:
                model.compile(optimizer='adam', 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])
            
            # Guardar usando Keras (crea SavedModel con metadatos de Keras)
            model.save(output_path, save_format='tf')
            print("OK: Modelo exportado usando Keras model.save()")
            
        except Exception as e:
            print(f"Keras model.save() falló: {e}")
            print("Intentando método alternativo con tf.saved_model.simple_save...")
            
            # Fallback al método original
            # Obtener tensores de entrada y salida
            input_tensor = model.input
            output_tensor = model.output
            
            print(f"Tensor de entrada: {input_tensor}")
            print(f"Tensor de salida: {output_tensor}")
            
            # Crear sesión de TensorFlow
            sess = keras.backend.get_session()
            
            print("Exportando a SavedModel...")
            tf.saved_model.simple_save(
                sess,
                output_path,
                inputs={'input': input_tensor},
                outputs={'output': output_tensor}
            )
            print("OK: Modelo exportado con tf.saved_model.simple_save")
        
        # Verificar archivos creados
        files = os.listdir(output_path)
        print(f"Archivos creados: {files}")
        
        # Verificar estructura SavedModel
        savedmodel_pb = os.path.join(output_path, 'saved_model.pb')
        variables_dir = os.path.join(output_path, 'variables')
        keras_metadata = os.path.join(output_path, 'keras_metadata.pb')
        
        if os.path.exists(savedmodel_pb):
            size_mb = os.path.getsize(savedmodel_pb) / (1024 * 1024)
            print(f"saved_model.pb: {size_mb:.2f} MB")
        
        if os.path.exists(variables_dir):
            var_files = os.listdir(variables_dir)
            print(f"Variables: {var_files}")
        
        if os.path.exists(keras_metadata):
            print("OK: keras_metadata.pb encontrado (mejor compatibilidad)")
        else:
            print("WARNING: keras_metadata.pb no encontrado")
        
        return True
        
    except Exception as e:
        print(f"ERROR exportando: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_conversion_info(output_dir, json_path, hdf5_path, model):
    """Crear archivo de información para el paso 2"""
    info = {
        "conversion_step": 1,
        "method": "savedmodel",
        "source_files": {
            "json": json_path,
            "hdf5": hdf5_path
        },
        "model_info": {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "total_params": int(model.count_params()),
            "layers": len(model.layers)
        },
        "environment": {
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__
        },
        "export_path": output_dir,
        "next_step": "Use step2_savedmodel_convert.py in gpd_py39 environment"
    }
    
    info_path = os.path.join(output_dir, "conversion_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"INFO: Información de conversión guardada en {info_path}")

def test_savedmodel_export(output_path):
    """Verificar que el SavedModel se exportó correctamente"""
    print(f"\n=== Verificando SavedModel exportado ===")
    
    try:
        # Verificar estructura de archivos
        required_files = ['saved_model.pb']
        required_dirs = ['variables']
        
        for file in required_files:
            file_path = os.path.join(output_path, file)
            if os.path.exists(file_path):
                print(f"OK: {file} existe")
            else:
                print(f"ERROR: {file} no encontrado")
                return False
        
        for dir in required_dirs:
            dir_path = os.path.join(output_path, dir)
            if os.path.exists(dir_path):
                print(f"OK: directorio {dir} existe")
            else:
                print(f"ERROR: directorio {dir} no encontrado")
                return False
        
        # Intentar cargar SavedModel en TF 1.x para verificar
        print("Verificando carga de SavedModel...")
        try:
            # En TF 1.x usamos tf.saved_model.load_v2 si está disponible
            if hasattr(tf.saved_model, 'load'):
                loaded = tf.saved_model.load(tf.Session(), [tf.saved_model.tag_constants.SERVING], output_path)
                print("OK: SavedModel carga correctamente en TF 1.x")
            else:
                print("INFO: Verificación de carga no disponible en esta versión de TF")
        except Exception as e:
            print(f"WARNING: No se pudo verificar carga: {e}")
        
        print("OK: SavedModel exportado correctamente")
        return True
        
    except Exception as e:
        print(f"ERROR verificando SavedModel: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Paso 1: Exportar modelo GPD legacy a SavedModel')
    parser.add_argument('--json', required=True, help='Ruta a model_pol.json')
    parser.add_argument('--hdf5', required=True, help='Ruta a model_pol_best.hdf5')
    parser.add_argument('--output', default='gpd_savedmodel', help='Directorio de salida para SavedModel')
    parser.add_argument('--test', action='store_true', help='Probar modelo antes de exportar')
    parser.add_argument('--verify', action='store_true', help='Verificar SavedModel después de exportar')
    
    args = parser.parse_args()
    
    print("=== Paso 1: Exportación de modelo GPD legacy a SavedModel ===")
    
    # Verificar entorno
    if not check_environment():
        print("ERROR: Entorno incorrecto. Use: micromamba activate gpd_python36")
        return False
    
    try:
        # Cargar modelo
        model = load_gpd_model(args.json, args.hdf5)
        
        # Probar si se solicita
        if args.test:
            if not test_model_functionality(model):
                print("ERROR: Modelo no pasa las pruebas")
                return False
        
        # Exportar a SavedModel
        if not export_to_savedmodel(model, args.output):
            print("ERROR: Falló la exportación")
            return False
        
        # Verificar exportación si se solicita
        if args.verify:
            if not test_savedmodel_export(args.output):
                print("WARNING: Verificación de SavedModel falló")
        
        # Crear información para paso 2
        create_conversion_info(args.output, args.json, args.hdf5, model)
        
        print(f"\n=== Paso 1 completado exitosamente ===")
        print(f"SavedModel guardado en: {args.output}")
        print(f"Siguiente paso: Activar gpd_py39 y ejecutar step2_savedmodel_convert.py")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)