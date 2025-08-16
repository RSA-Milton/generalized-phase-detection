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

def load_conversion_info(savedmodel_dir):
    """Cargar información del paso 1"""
    info_path = os.path.join(savedmodel_dir, "conversion_info.json")
    
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

def inspect_savedmodel(savedmodel_path):
    """Inspeccionar estructura del SavedModel"""
    print(f"\n=== Inspeccionando SavedModel ===")
    print(f"Ruta: {savedmodel_path}")
    
    try:
        # Verificar estructura de archivos
        if not os.path.exists(savedmodel_path):
            raise FileNotFoundError(f"SavedModel no encontrado: {savedmodel_path}")
        
        savedmodel_pb = os.path.join(savedmodel_path, 'saved_model.pb')
        variables_dir = os.path.join(savedmodel_path, 'variables')
        
        if not os.path.exists(savedmodel_pb):
            raise FileNotFoundError("saved_model.pb no encontrado")
        
        if not os.path.exists(variables_dir):
            raise FileNotFoundError("directorio variables no encontrado")
        
        print(f"OK: Estructura SavedModel válida")
        
        # Listar signatures disponibles
        print("Inspeccionando signatures...")
        try:
            saved_model = tf.saved_model.load(savedmodel_path)
            
            if hasattr(saved_model, 'signatures'):
                signatures = saved_model.signatures
                print(f"Signatures disponibles: {list(signatures.keys())}")
                
                # Inspeccionar signature por defecto
                if 'serving_default' in signatures:
                    sig = signatures['serving_default']
                    print(f"Signature 'serving_default':")
                    print(f"  Entradas: {list(sig.structured_input_signature[1].keys())}")
                    print(f"  Salidas: {list(sig.structured_outputs.keys())}")
                    
                    # Obtener formas de tensores
                    for input_name, input_spec in sig.structured_input_signature[1].items():
                        print(f"  Entrada '{input_name}': {input_spec.shape} {input_spec.dtype}")
                    
                    for output_name, output_tensor in sig.structured_outputs.items():
                        print(f"  Salida '{output_name}': {output_tensor.shape}")
                else:
                    print("WARNING: No se encontró signature 'serving_default'")
            else:
                print("WARNING: SavedModel no tiene signatures")
            
            return saved_model
            
        except Exception as e:
            print(f"ERROR inspeccionando signatures: {e}")
            return None
        
    except Exception as e:
        print(f"ERROR inspeccionando SavedModel: {e}")
        raise

def convert_savedmodel_to_keras(savedmodel_path):
    """Convertir SavedModel a modelo Keras nativo"""
    print(f"\n=== Convirtiendo SavedModel a Keras ===")
    
    try:
        # Método 1: Carga directa como modelo Keras
        print("Intentando carga directa como modelo Keras...")
        model = tf.keras.models.load_model(savedmodel_path)
        print("OK: SavedModel cargado como modelo Keras")
        
        return model
        
    except Exception as e:
        print(f"Carga directa falló: {e}")
        
        # Método 2: Usar tf.keras.models.load_model con compile=False
        print("Intentando carga sin compilación...")
        try:
            model = tf.keras.models.load_model(savedmodel_path, compile=False)
            print("OK: SavedModel cargado sin compilación")
            return model
        except Exception as e2:
            print(f"Carga sin compilación falló: {e2}")
        
        # Método 3: Cargar SavedModel y crear modelo funcional
        print("Intentando conversión desde SavedModel con modelo funcional...")
        try:
            saved_model = tf.saved_model.load(savedmodel_path)
            
            if hasattr(saved_model, 'signatures') and 'serving_default' in saved_model.signatures:
                # Obtener función de inferencia
                inference_func = saved_model.signatures['serving_default']
                print("OK: Función de inferencia extraída")
                
                # Obtener especificaciones de entrada
                input_signature = inference_func.structured_input_signature[1]
                input_names = list(input_signature.keys())
                
                if len(input_names) == 1:
                    input_name = input_names[0]
                    input_spec = input_signature[input_name]
                    
                    print(f"Entrada detectada: {input_name} - {input_spec.shape}")
                    
                    # Método alternativo: Crear modelo usando tf.function
                    print("Creando modelo funcional personalizado...")
                    
                    class SavedModelWrapper(tf.keras.Model):
                        def __init__(self, saved_model_func, input_shape):
                            super().__init__()
                            self.saved_model_func = saved_model_func
                            self.input_spec = input_shape
                        
                        def call(self, inputs):
                            # Convertir inputs a dict si es necesario
                            if isinstance(inputs, tf.Tensor):
                                inputs_dict = {input_name: inputs}
                            else:
                                inputs_dict = inputs
                            
                            # Llamar función guardada
                            outputs = self.saved_model_func(**inputs_dict)
                            
                            # Extraer tensor de salida
                            if isinstance(outputs, dict):
                                output_names = list(outputs.keys())
                                return outputs[output_names[0]]
                            else:
                                return outputs
                        
                        def get_config(self):
                            return {"input_shape": self.input_spec}
                    
                    # Crear wrapper
                    wrapper_model = SavedModelWrapper(inference_func, input_spec.shape)
                    
                    # Construir el modelo con una llamada de ejemplo
                    example_input = tf.zeros([1] + list(input_spec.shape[1:]), dtype=input_spec.dtype)
                    _ = wrapper_model(example_input)
                    
                    print("OK: Modelo wrapper creado")
                    return wrapper_model
                    
                else:
                    print(f"ERROR: Múltiples entradas no soportadas: {input_names}")
                    return None
            else:
                print("ERROR: No se encontró signature válida")
                return None
                
        except Exception as e3:
            print(f"ERROR en conversión funcional: {e3}")
            import traceback
            traceback.print_exc()
            
            # Método 4: Último recurso - crear modelo Keras desde función
            print("Intentando último recurso: recrear modelo Keras desde función...")
            try:
                saved_model = tf.saved_model.load(savedmodel_path)
                inference_func = saved_model.signatures['serving_default']
                
                # Obtener especificaciones
                input_signature = inference_func.structured_input_signature[1]
                input_name = list(input_signature.keys())[0]
                input_spec = input_signature[input_name]
                
                print(f"Recreando modelo Keras con entrada: {input_spec.shape}")
                
                # Crear modelo Keras que wrappea la función
                @tf.function
                def inference_wrapper(inputs):
                    return inference_func(input=inputs)
                
                # Crear Input layer
                input_layer = tf.keras.Input(
                    shape=input_spec.shape[1:], 
                    dtype=input_spec.dtype,
                    name='input'
                )
                
                # Crear Lambda layer que usa la función
                lambda_layer = tf.keras.layers.Lambda(
                    lambda x: inference_wrapper(x)['output'],
                    name='savedmodel_inference'
                )
                
                outputs = lambda_layer(input_layer)
                
                # Crear modelo
                model = tf.keras.Model(inputs=input_layer, outputs=outputs, name='GPD_from_SavedModel')
                
                # Compilar modelo
                model.compile(optimizer='adam', 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])
                
                print("OK: Modelo Keras recreado con Lambda wrapper")
                return model
                
            except Exception as e4:
                print(f"ERROR en recreación Keras: {e4}")
                
                # Método 5: Función de inferencia simple (fallback final)
                print("Creando objeto de inferencia simple...")
                try:
                    saved_model = tf.saved_model.load(savedmodel_path)
                    inference_func = saved_model.signatures['serving_default']
                    
                    # Crear un callable simple mejorado
                    class InferenceOnly:
                        def __init__(self, func, input_shape, output_shape):
                            self.func = func
                            self.input_shape = input_shape
                            self.output_shape = output_shape
                            self.compiled = False  # Para compatibilidad
                        
                        def predict(self, inputs, **kwargs):
                            # Ignorar kwargs como verbose
                            if len(inputs.shape) == 2:
                                # Añadir dimensión de batch si falta
                                inputs = tf.expand_dims(inputs, 0)
                            
                            # Asegurar tipo float32
                            inputs = tf.cast(inputs, tf.float32)
                            
                            # Llamar función
                            outputs = self.func(input=inputs)
                            
                            # Extraer resultado
                            if isinstance(outputs, dict):
                                output_names = list(outputs.keys())
                                result = outputs[output_names[0]]
                            else:
                                result = outputs
                            
                            return result.numpy()
                        
                        def save(self, path, **kwargs):
                            print("ERROR: save() no disponible para InferenceOnly")
                            return False
                        
                        def count_params(self):
                            return 0  # No podemos contar parámetros
                    
                    # Obtener formas
                    input_signature = inference_func.structured_input_signature[1]
                    input_name = list(input_signature.keys())[0]
                    input_spec = input_signature[input_name]
                    
                    inference_only = InferenceOnly(inference_func, input_spec.shape, (None, 3))
                    print("OK: Objeto de inferencia simple creado")
                    print("WARNING: Funcionalidad muy limitada - conversión no completada")
                    
                    return inference_only
                    
                except Exception as e5:
                    print(f"ERROR final: {e5}")
                    return None

def test_converted_model(model, original_info=None):
    """Probar modelo convertido"""
    print(f"\n=== Probando modelo convertido ===")
    
    try:
        # Verificar arquitectura
        if hasattr(model, 'input_shape'):
            print(f"Forma de entrada: {model.input_shape}")
        else:
            print(f"Forma de entrada: {getattr(model, 'input_shape', 'No disponible')}")
            
        if hasattr(model, 'output_shape'):
            print(f"Forma de salida: {model.output_shape}")
        else:
            print(f"Forma de salida: {getattr(model, 'output_shape', 'No disponible')}")
        
        # Contar parámetros si es posible
        try:
            if hasattr(model, 'count_params'):
                total_params = model.count_params()
                print(f"Total parámetros: {total_params:,}")
                
                # Comparar con información original si está disponible
                if original_info:
                    orig_params = original_info['model_info']['total_params']
                    if orig_params != total_params:
                        print(f"WARNING: Parámetros diferentes. Original: {orig_params:,}, Actual: {total_params:,}")
                    else:
                        print("OK: Número de parámetros coincide")
            else:
                print("INFO: No se pudieron contar parámetros")
        except:
            print("INFO: Conteo de parámetros no disponible")
        
        # Prueba funcional
        print("Probando predicción...")
        test_data = np.random.randn(3, 400, 3).astype(np.float32)
        
        # Adaptar llamada según el tipo de modelo
        if hasattr(model, 'predict'):
            try:
                predictions = model.predict(test_data, verbose=0)
            except TypeError:
                # Fallback sin argumentos adicionales
                predictions = model.predict(test_data)
        else:
            print("ERROR: Modelo no tiene método predict()")
            return False
        
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
        import traceback
        traceback.print_exc()
        return False

def save_keras_model(model, output_path):
    """Guardar modelo en formato .keras"""
    print(f"\n=== Guardando modelo .keras ===")
    print(f"Archivo de salida: {output_path}")
    
    try:
        # Verificar si es un modelo Keras real
        if hasattr(model, 'save') and hasattr(model, 'layers'):
            print("Modelo Keras detectado, guardando...")
            
            # Compilar modelo antes de guardar si no está compilado
            if hasattr(model, 'compiled') and not model.compiled:
                print("Compilando modelo antes de guardar...")
                model.compile(optimizer='adam', 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])
            
            # Guardar en formato .keras
            model.save(output_path, save_format='keras')
            print("OK: Modelo guardado en formato .keras")
            
            # Verificar archivo creado
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"Tamaño del archivo: {size_mb:.2f} MB")
            
            return True
            
        else:
            print("ERROR: Objeto no es un modelo Keras válido")
            print(f"Tipo de objeto: {type(model)}")
            
            # Intentar guardar la función de inferencia como pickle
            print("Intentando guardar función de inferencia...")
            import pickle
            pickle_path = output_path.replace('.keras', '_inference.pkl')
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"WARNING: Función guardada como pickle en: {pickle_path}")
            print("Esto NO es un modelo .keras válido")
            return False
        
    except Exception as e:
        print(f"ERROR guardando modelo: {e}")
        import traceback
        traceback.print_exc()
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
    parser = argparse.ArgumentParser(description='Paso 2: Convertir SavedModel a .keras')
    parser.add_argument('--savedmodel', required=True, help='Directorio del SavedModel del paso 1')
    parser.add_argument('--output', required=True, help='Archivo de salida .keras')
    parser.add_argument('--test', action='store_true', help='Probar modelo convertido')
    parser.add_argument('--inspect', action='store_true', help='Inspeccionar SavedModel antes de convertir')
    
    args = parser.parse_args()
    
    print("=== Paso 2: Conversión SavedModel a formato .keras moderno ===")
    
    # Verificar entorno
    if not check_environment():
        print("ERROR: Entorno incorrecto. Use: micromamba activate gpd_py39")
        return False
    
    try:
        # Cargar información del paso 1
        conversion_info = load_conversion_info(args.savedmodel)
        
        # Inspeccionar SavedModel si se solicita
        if args.inspect:
            inspect_savedmodel(args.savedmodel)
        
        # Convertir SavedModel a modelo Keras
        model = convert_savedmodel_to_keras(args.savedmodel)
        
        if model is None:
            print("ERROR: No se pudo convertir SavedModel a modelo Keras")
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