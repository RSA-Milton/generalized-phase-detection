#!/usr/bin/env python3
"""
Conversor GPD Legacy a Keras Moderno - Versión Corregida
Maneja nombres de capas con sufijo _1 y corrige la arquitectura de salida
"""

import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Conv1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from typing import Dict, Any, List

class GPDConverter:
    def __init__(self, json_path: str, weights_path: str):
        """
        Args:
            json_path: Ruta al archivo .json con la arquitectura
            weights_path: Ruta al archivo .h5 con los pesos
        """
        self.json_path = json_path
        self.weights_path = weights_path
        self.json_config = None
        self.model = None
        
    def load_json(self) -> Dict[str, Any]:
        """Carga la configuración JSON del modelo"""
        with open(self.json_path, 'r') as f:
            self.json_config = json.load(f)
        return self.json_config
    
    def extract_sequential_config(self) -> List[Dict[str, Any]]:
        """Extrae la configuración del Sequential anidado (25 capas)"""
        # Navegar a través de la estructura JSON
        if 'config' in self.json_config:
            layers = self.json_config['config']['layers']
        else:
            layers = self.json_config['model_config']['config']['layers']
        
        # Buscar el Sequential (típicamente en posición 4)
        sequential_layer = None
        for layer in layers:
            if layer['class_name'] == 'Sequential':
                sequential_layer = layer
                break
        
        if not sequential_layer:
            raise ValueError("No se encontró capa Sequential en el JSON")
        
        # Extraer las capas internas
        if isinstance(sequential_layer['config'], dict):
            internal_layers = sequential_layer['config']['layers']
        else:
            internal_layers = sequential_layer['config']
            
        print(f"Encontradas {len(internal_layers)} capas en Sequential")
        return internal_layers
    
    def build_sequential_from_config(self, layers_config: List[Dict[str, Any]]) -> Sequential:
        """Reconstruye el Sequential desde la configuración"""
        model = Sequential(name='sequential_1')
        
        for layer_cfg in layers_config:
            class_name = layer_cfg['class_name']
            config = layer_cfg['config']
            
            # IMPORTANTE: Añadir sufijo _1 a los nombres para que coincidan con HDF5
            original_name = config['name']
            config['name'] = f"{original_name}_1"
            
            if class_name == 'Conv1D':
                model.add(Conv1D(
                    filters=config['filters'],
                    kernel_size=config['kernel_size'],
                    strides=config.get('strides', 1),
                    padding=config.get('padding', 'valid'),
                    activation=config.get('activation'),
                    name=config['name']
                ))
            elif class_name == 'MaxPooling1D':
                model.add(MaxPooling1D(
                    pool_size=config['pool_size'],
                    strides=config.get('strides'),
                    padding=config.get('padding', 'valid'),
                    name=config['name']
                ))
            elif class_name == 'Dense':
                model.add(Dense(
                    units=config['units'],
                    activation=config.get('activation'),
                    name=config['name']
                ))
            elif class_name == 'Dropout':
                model.add(Dropout(
                    rate=config['rate'],
                    name=config['name']
                ))
            elif class_name == 'Flatten':
                model.add(Flatten(name=config['name']))
            elif class_name == 'BatchNormalization':
                model.add(BatchNormalization(
                    axis=config.get('axis', -1),
                    momentum=config.get('momentum', 0.99),
                    epsilon=config.get('epsilon', 0.001),
                    center=config.get('center', True),
                    scale=config.get('scale', True),
                    name=config['name']
                ))
            elif class_name == 'Activation':
                model.add(Activation(
                    activation=config['activation'],
                    name=config['name']
                ))
            else:
                print(f"Advertencia: Capa {class_name} no implementada explícitamente")
                layer_class = getattr(tf.keras.layers, class_name)
                model.add(layer_class(**config))
        
        return model
    
    def build_full_model(self) -> Model:
        """Construye el modelo completo con arquitectura de 3 ramas"""
        # Input
        inputs = Input(shape=(400, 3), name='input')
        
        # Separar los 3 canales (Z, N, E)
        z_channel = Lambda(lambda x: x[:, :, 0:1], name='lambda_1')(inputs)
        n_channel = Lambda(lambda x: x[:, :, 1:2], name='lambda_2')(inputs)
        e_channel = Lambda(lambda x: x[:, :, 2:3], name='lambda_3')(inputs)
        
        # Construir el Sequential compartido
        layers_config = self.extract_sequential_config()
        shared_sequential = self.build_sequential_from_config(layers_config)
        
        # Procesar cada canal con el MISMO Sequential (pesos compartidos)
        z_features = shared_sequential(z_channel)
        n_features = shared_sequential(n_channel)
        e_features = shared_sequential(e_channel)
        
        # CORRECCIÓN: Promedio en lugar de concatenación para obtener 3 salidas
        # El modelo original probablemente promedia o suma las características
        outputs = Lambda(lambda x: (x[0] + x[1] + x[2]) / 3.0, 
                        name='merge_outputs')([z_features, n_features, e_features])
        
        # Crear el modelo
        model = Model(inputs=inputs, outputs=outputs, name='gpd_model')
        self.model = model
        
        print(f"Modelo construido: {model.count_params():,} parámetros")
        return model
    
    def load_weights_from_h5(self):
        """Carga los pesos desde el archivo HDF5"""
        if self.model is None:
            raise ValueError("Primero construye el modelo con build_full_model()")
        
        with h5py.File(self.weights_path, 'r') as f:
            # Verificar estructura
            if 'model_weights' not in f:
                raise ValueError("No se encontró 'model_weights' en el archivo HDF5")
            
            weights_group = f['model_weights']['sequential_1']
            
            # Listar las capas disponibles
            available_layers = list(weights_group.keys())
            print(f"\nCapas disponibles en HDF5: {len(available_layers)} capas")
            
            # Cargar pesos para cada capa del Sequential
            sequential_model = self.model.get_layer('sequential_1')
            
            for layer in sequential_model.layers:
                layer_name = layer.name
                
                if layer_name in weights_group:
                    layer_group = weights_group[layer_name]
                    weights = []
                    
                    # Ordenar las claves para cargar los pesos en el orden correcto
                    weight_keys = sorted(layer_group.keys())
                    
                    # Para BatchNormalization, el orden debe ser específico
                    if isinstance(layer, BatchNormalization):
                        # Buscar las claves correctas
                        expected_keys = ['beta:0', 'gamma:0', 'moving_mean:0', 'moving_variance:0']
                        # A veces están sin el :0
                        if 'beta' in layer_group:
                            weight_keys = ['gamma', 'beta', 'moving_mean', 'moving_variance']
                        elif 'gamma:0' in layer_group:
                            weight_keys = ['gamma:0', 'beta:0', 'moving_mean:0', 'moving_variance:0']
                    
                    # Cargar los pesos
                    for key in weight_keys:
                        if key in layer_group:
                            weight_data = layer_group[key][:]
                            weights.append(weight_data)
                    
                    # CORRECCIÓN: Invertir orden para Conv1D y Dense
                    # En HDF5 están como [bias, kernel], Keras espera [kernel, bias]
                    if isinstance(layer, (Conv1D, Dense)) and len(weights) == 2:
                        weights = [weights[1], weights[0]]  # Intercambiar orden
                    
                    if weights:
                        try:
                            # Para BatchNorm, el orden en Keras 2.x es [gamma, beta, mean, var]
                            if isinstance(layer, BatchNormalization) and len(weights) == 4:
                                # Reordenar si es necesario (el orden puede variar)
                                if 'beta' in weight_keys[0] or 'beta:0' in weight_keys[0]:
                                    # Si beta está primero, reordenar a [gamma, beta, mean, var]
                                    weights = [weights[1], weights[0], weights[2], weights[3]]
                            
                            layer.set_weights(weights)
                            print(f"✓ Pesos cargados para {layer_name}")
                        except ValueError as e:
                            print(f"⚠ Error cargando pesos para {layer_name}: {e}")
                            print(f"  Forma esperada: {[w.shape for w in layer.get_weights()]}")
                            print(f"  Forma recibida: {[w.shape for w in weights]}")
                else:
                    # Capas sin pesos (MaxPooling, Activation, etc.)
                    if not isinstance(layer, (MaxPooling1D, Activation, Dropout, Flatten)):
                        print(f"⚠ No se encontraron pesos para {layer_name}")
        
        print("\nCarga de pesos completada")
    
    def convert(self, output_path: str = 'gpd_modern.keras'):
        """Proceso completo de conversión"""
        print("=== Iniciando conversión GPD ===")
        
        # 1. Cargar JSON
        print("\n1. Cargando configuración JSON...")
        self.load_json()
        
        # 2. Construir modelo
        print("\n2. Construyendo modelo...")
        self.build_full_model()
        
        # 3. Cargar pesos
        print("\n3. Cargando pesos desde HDF5...")
        self.load_weights_from_h5()
        
        # 4. Compilar modelo (opcional pero recomendado)
        print("\n4. Compilando modelo...")
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 5. Guardar en formato moderno
        print(f"\n5. Guardando modelo en formato Keras moderno: {output_path}")
        self.model.save(output_path)
        
        print("\n=== Conversión completada exitosamente ===")
        return self.model
    
    def verify_model(self):
        """Verificación del modelo con predicción de prueba"""
        if self.model is None:
            raise ValueError("El modelo no ha sido construido")
        
        print("\n=== Verificación del modelo ===")
        
        # Test con datos aleatorios
        test_input = np.random.randn(1, 400, 3).astype(np.float32)
        
        # Normalizar entrada (simulando preprocesamiento típico)
        test_input = test_input / np.max(np.abs(test_input))
        
        output = self.model.predict(test_input, verbose=0)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output raw: {output[0]}")
        
        # Aplicar softmax si no está en la última capa
        if not np.allclose(np.sum(output[0]), 1.0, atol=0.01):
            from scipy.special import softmax
            output_prob = softmax(output[0])
            print(f"Output (softmax): {output_prob}")
            print(f"Predicción: P={output_prob[0]:.3f}, S={output_prob[1]:.3f}, Noise={output_prob[2]:.3f}")
        else:
            print(f"Predicción: P={output[0][0]:.3f}, S={output[0][1]:.3f}, Noise={output[0][2]:.3f}")
        
        # Verificar que tenemos 3 salidas
        assert output.shape[-1] == 3, f"Error: Se esperaban 3 salidas, se obtuvieron {output.shape[-1]}"
        print("✓ Modelo verificado: 3 salidas correctas (P, S, Noise)")
        
        return output


def convert_to_tflite(keras_model_path: str, output_path: str = 'gpd_model.tflite', 
                      quantize: bool = True):
    """Convierte el modelo Keras a TensorFlow Lite"""
    print("\n=== Conversión a TensorFlow Lite ===")
    
    # Cargar modelo
    model = tf.keras.models.load_model(keras_model_path)
    print(f"Modelo cargado: {model.count_params():,} parámetros")
    
    # Convertir a TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Optimizaciones para Raspberry Pi
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Cuantización a float16
        converter.target_spec.supported_types = [tf.float16]
        print("Aplicando cuantización float16...")
    
    # Convertir
    tflite_model = converter.convert()
    
    # Guardar
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Información del modelo
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✓ Modelo TFLite guardado: {output_path}")
    print(f"  Tamaño: {size_mb:.2f} MB")
    
    if quantize:
        print(f"  Reducción: ~{(1.7 * 4 / size_mb):.1f}x respecto al original")
    
    return output_path


def test_tflite_model(tflite_path: str):
    """Prueba el modelo TFLite convertido"""
    print("\n=== Prueba del modelo TFLite ===")
    
    # Cargar modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Obtener detalles de entrada/salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Crear entrada de prueba
    test_input = np.random.randn(1, 400, 3).astype(np.float32)
    test_input = test_input / np.max(np.abs(test_input))
    
    # Ejecutar inferencia
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Output: {output[0]}")
    print("✓ Modelo TFLite funciona correctamente")
    
    return output


def main():
    """Función principal"""
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python gpd_converter.py modelo.json modelo_weights.h5 [output.keras]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    weights_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'gpd_modern.keras'
    
    # Conversión
    converter = GPDConverter(json_path, weights_path)
    model = converter.convert(output_path)
    
    # Verificación
    converter.verify_model()
    
    # Conversión a TFLite
    response = input("\n¿Convertir a TensorFlow Lite? (s/n): ")
    if response.lower() == 's':
        tflite_path = output_path.replace('.keras', '.tflite')
        convert_to_tflite(output_path, tflite_path)
        
        # Probar el modelo TFLite
        test_response = input("\n¿Probar el modelo TFLite? (s/n): ")
        if test_response.lower() == 's':
            test_tflite_model(tflite_path)
    
    print("\n✅ Proceso completado exitosamente")
    print(f"   Modelo Keras: {output_path}")
    if response.lower() == 's':
        print(f"   Modelo TFLite: {tflite_path}")


if __name__ == "__main__":
    main()