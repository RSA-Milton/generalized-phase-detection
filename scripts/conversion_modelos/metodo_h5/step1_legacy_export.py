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

def export_to_h5_complete(model, output_path):
    """Exportar modelo a formato H5 completo (arquitectura + pesos)"""
    print(f"\n=== Exportando a H5 completo ===")
    print(f"Archivo de salida: {output_path}")
    
    try:
        # Compilar modelo antes de guardar (requerido para .h5)
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        print("OK: Modelo compilado")
        
        # Guardar modelo completo en formato H5
        model.save(output_path)
        print("OK: Modelo exportado a H5 completo")
        
        # Verificar archivo creado
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Archivo creado: {output_path} ({size_mb:.2f} MB)")
        else:
            print("ERROR: Archivo no fue creado")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR exportando: {e}")
        return False

def create_conversion_info(output_path, json_path, hdf5_path, model):
    """Crear archivo de información para el paso 2"""
    # Extraer directorio y nombre base del archivo H5
    output_dir = os.path.dirname(output_path) or '.'
    
    info = {
        "conversion_step": 1,
        "method": "h5_complete",
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
        "export_path": output_path,
        "next_step": "Use step2_modern_convert.py in gpd_py39 environment with --h5 flag"
    }
    
    info_path = os.path.join(output_dir, "conversion_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"INFO: Información de conversión guardada en {info_path}")

def main():
    parser = argparse.ArgumentParser(description='Paso 1: Exportar modelo GPD legacy a H5 completo')
    parser.add_argument('--json', required=True, help='Ruta a model_pol.json')
    parser.add_argument('--hdf5', required=True, help='Ruta a model_pol_best.hdf5')
    parser.add_argument('--output', default='gpd_full_keras2.h5', help='Archivo H5 de salida')
    parser.add_argument('--test', action='store_true', help='Probar modelo antes de exportar')
    
    args = parser.parse_args()
    
    print("=== Paso 1: Exportación de modelo GPD legacy a H5 completo ===")
    
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
        
        # Exportar a H5 completo
        if not export_to_h5_complete(model, args.output):
            print("ERROR: Falló la exportación")
            return False
        
        # Crear información para paso 2
        create_conversion_info(args.output, args.json, args.hdf5, model)
        
        print(f"\n=== Paso 1 completado exitosamente ===")
        print(f"Modelo H5 completo guardado en: {args.output}")
        print(f"Siguiente paso: Activar gpd_py39 y ejecutar step2_modern_convert.py --h5 {args.output}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)