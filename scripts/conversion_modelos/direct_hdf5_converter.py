#!/usr/bin/env python3
"""
Conversor directo GPD: Lee arquitectura del JSON (sin capas Lambda) y pesos del HDF5
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
from tensorflow.keras.layers import Dense, Dropout, Flatten
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

def analyze_original_json(json_path):
    """Analizar JSON original para entender la arquitectura (manejo robusto de estructuras)"""
    print(f"\n=== Analizando arquitectura desde JSON ===")
    
    with open(json_path, 'r') as f:
        model_config = json.load(f)
    
    print(f"Clase del modelo: {model_config['class_name']}")
    
    config = model_config.get('config', {})
    layers_data = config.get('layers', [])
    
    print(f"Número de capas principales: {len(layers_data)}")
    
    # Buscar la capa Sequential que contiene la arquitectura real
    sequential_layers = []
    
    for i, layer_data in enumerate(layers_data):
        print(f"\nCapa {i}: {layer_data.get('class_name')} - {layer_data.get('name')}")
        
        if layer_data.get('class_name') == 'Sequential':
            print(f"  ¡Encontrada capa Sequential con la arquitectura real!")
            
            # Extraer las subcapas del Sequential
            sequential_config = layer_data.get('config', [])
            if isinstance(sequential_config, list):
                print(f"  Subcapas en Sequential: {len(sequential_config)}")
                
                for j, sublayer in enumerate(sequential_config):
                    if isinstance(sublayer, dict):
                        sublayer_info = {
                            'index': j,
                            'class_name': sublayer.get('class_name', 'Unknown'),
                            'config': sublayer.get('config', {}),
                            'parent_layer': i
                        }
                        
                        # Extraer nombre de la configuración
                        sublayer_config = sublayer_info['config']
                        if isinstance(sublayer_config, dict):
                            sublayer_info['name'] = sublayer_config.get('name', f'layer_{j}')
                        else:
                            sublayer_info['name'] = f'layer_{j}'
                        
                        print(f"    {j}: {sublayer_info['class_name']} - {sublayer_info['name']}")
                        
                        # Extraer información específica por tipo de capa
                        if sublayer_info['class_name'] == 'Conv1D':
                            config = sublayer_info['config']
                            sublayer_info.update({
                                'filters': config.get('filters', 32),
                                'kernel_size': config.get('kernel_size', [3])[0] if isinstance(config.get('kernel_size'), list) else config.get('kernel_size', 3),
                                'activation': config.get('activation', 'linear'),
                                'padding': config.get('padding', 'valid')
                            })
                            print(f"      Conv1D: {sublayer_info['filters']} filtros, kernel={sublayer_info['kernel_size']}, act={sublayer_info['activation']}")
                            
                        elif sublayer_info['class_name'] == 'MaxPooling1D':
                            config = sublayer_info['config']
                            pool_size = config.get('pool_size', [2])
                            sublayer_info['pool_size'] = pool_size[0] if isinstance(pool_size, list) else pool_size
                            print(f"      MaxPooling1D: pool_size={sublayer_info['pool_size']}")
                            
                        elif sublayer_info['class_name'] == 'Dense':
                            config = sublayer_info['config']
                            sublayer_info.update({
                                'units': config.get('units', 1),
                                'activation': config.get('activation', 'linear')
                            })
                            print(f"      Dense: {sublayer_info['units']} unidades, act={sublayer_info['activation']}")
                            
                        elif sublayer_info['class_name'] == 'Dropout':
                            config = sublayer_info['config']
                            sublayer_info['rate'] = config.get('rate', 0.5)
                            print(f"      Dropout: rate={sublayer_info['rate']}")
                            
                        elif sublayer_info['class_name'] == 'BatchNormalization':
                            config = sublayer_info['config']
                            sublayer_info['axis'] = config.get('axis', -1)
                            print(f"      BatchNormalization: axis={sublayer_info['axis']}")
                            
                        elif sublayer_info['class_name'] == 'Flatten':
                            print(f"      Flatten")
                            
                        elif sublayer_info['class_name'] == 'Activation':
                            config = sublayer_info['config']
                            sublayer_info['activation'] = config.get('activation', 'linear')
                            print(f"      Activation: {sublayer_info['activation']}")
                        
                        sequential_layers.append(sublayer_info)
            break
    
    if not sequential_layers:
        print("ERROR: No se encontró la capa Sequential con la arquitectura")
        return []
    
    print(f"\n=== Arquitectura extraída ===")
    print(f"Total de capas en Sequential: {len(sequential_layers)}")
    for layer_info in sequential_layers:
        print(f"  {layer_info['index']}: {layer_info['class_name']} - {layer_info['name']}")
    
    return sequential_layers

def extract_weights_from_hdf5(hdf5_path):
    """Extraer todos los pesos del archivo HDF5 original"""
    print(f"\n=== Extrayendo pesos del HDF5 original ===")
    
    weights_data = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        def collect_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                weight_name = name
                weight_value = obj[:]
                weights_data[weight_name] = weight_value
                print(f"  {weight_name}: {weight_value.shape}")
        
        f.visititems(collect_weights)
    
    print(f"Total pesos extraídos: {len(weights_data)}")
    return weights_data

def create_gpd_model_from_analysis(architecture_info):
    """Crear modelo GPD basado en análisis del JSON (desde Sequential extraído)"""
    print(f"\n=== Creando modelo desde análisis ===")
    
    layers = []
    layer_names = []
    
    for layer_info in architecture_info:
        layer_class = layer_info['class_name']
        layer_name = layer_info['name']
        config = layer_info['config']
        
        # Saltar capas problemáticas o innecesarias
        if layer_class in ['Lambda', 'InputLayer']:
            print(f"  Saltando capa: {layer_class} - {layer_name}")
            continue
        
        print(f"  Añadiendo: {layer_class} - {layer_name}")
        layer_names.append(layer_name)
        
        # Crear capa según tipo
        if layer_class == 'Conv1D':
            if len(layers) == 0:  # Primera capa
                layer = Conv1D(
                    filters=layer_info.get('filters', 32),
                    kernel_size=layer_info.get('kernel_size', 3),
                    activation=layer_info.get('activation', 'linear'),
                    padding=layer_info.get('padding', 'valid'),
                    input_shape=(400, 3),
                    name=layer_name
                )
            else:
                layer = Conv1D(
                    filters=layer_info.get('filters', 32),
                    kernel_size=layer_info.get('kernel_size', 3),
                    activation=layer_info.get('activation', 'linear'),
                    padding=layer_info.get('padding', 'valid'),
                    name=layer_name
                )
            layers.append(layer)
            
        elif layer_class == 'MaxPooling1D':
            layer = MaxPooling1D(
                pool_size=layer_info.get('pool_size', 2),
                name=layer_name
            )
            layers.append(layer)
            
        elif layer_class == 'BatchNormalization':
            layer = BatchNormalization(name=layer_name)
            layers.append(layer)
            
        elif layer_class == 'Flatten':
            layer = Flatten(name=layer_name)
            layers.append(layer)
            
        elif layer_class == 'Dense':
            layer = Dense(
                units=layer_info.get('units', 1),
                activation=layer_info.get('activation', 'linear'),
                name=layer_name
            )
            layers.append(layer)
            
        elif layer_class == 'Dropout':
            layer = Dropout(rate=layer_info.get('rate', 0.5), name=layer_name)
            layers.append(layer)
            
        elif layer_class == 'Activation':
            # Las activaciones standalone se pueden saltar si ya están en las capas
            activation_type = layer_info.get('activation', 'linear')
            if activation_type not in ['linear', 'relu']:  # Solo añadir si es especial
                layer = tf.keras.layers.Activation(activation_type, name=layer_name)
                layers.append(layer)
            else:
                print(f"    Saltando Activation {activation_type} (ya integrada)")
                continue
        
        else:
            print(f"  WARNING: Tipo de capa no soportado: {layer_class}")
            continue
    
    if not layers:
        print("ERROR: No se crearon capas válidas")
        return None, []
    
    # Crear modelo secuencial
    model = Sequential(layers, name='GPD_Reconstructed')
    
    print(f"Modelo creado con {len(model.layers)} capas")
    
    # Mostrar resumen
    try:
        model.summary()
    except Exception as e:
        print(f"No se pudo mostrar summary: {e}")
    
    return model, layer_names

def map_weights_to_reconstructed_model(model, weights_data, layer_names):
    """Mapear pesos del HDF5 al modelo reconstruido"""
    print(f"\n=== Mapeando pesos al modelo reconstruido ===")
    
    mapped_layers = 0
    total_layers_with_weights = 0
    
    # Crear mapeo de nombres de capas
    weight_layer_mapping = {}
    
    # Analizar estructura de pesos en HDF5
    print("Analizando estructura de pesos...")
    layer_groups = {}
    for weight_name in weights_data.keys():
        # Extraer nombre de capa (formato: layer_name/weight_type)
        if '/' in weight_name:
            layer_part = weight_name.split('/')[0]
            weight_type = weight_name.split('/')[-1]
            
            if layer_part not in layer_groups:
                layer_groups[layer_part] = []
            layer_groups[layer_part].append(weight_name)
    
    print(f"Grupos de capas encontrados: {list(layer_groups.keys())}")
    
    # Mapear pesos por orden y tipo
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:
            total_layers_with_weights += 1
            layer_name = layer.name
            
            print(f"\nProcesando capa {i}: {layer_name} ({type(layer).__name__})")
            
            # Buscar pesos correspondientes
            found_weights = []
            
            # Buscar por nombre exacto primero
            if layer_name in layer_groups:
                weight_names = layer_groups[layer_name]
                print(f"  Encontrado grupo exacto: {weight_names}")
                
                # Ordenar pesos (kernel/weight primero, bias segundo, etc.)
                sorted_weights = []
                for pattern in ['kernel', 'weight', 'bias', 'gamma', 'beta', 'moving_mean', 'moving_variance']:
                    for wname in weight_names:
                        if pattern in wname:
                            sorted_weights.append(wname)
                
                # Cargar pesos en orden
                for wname in sorted_weights:
                    if wname in weights_data:
                        weight_value = weights_data[wname]
                        found_weights.append(weight_value)
                        print(f"    Cargado: {wname} -> {weight_value.shape}")
            
            # Si no encontramos por nombre exacto, buscar por patrón
            if not found_weights:
                # Buscar patrones alternativos según el tipo de capa
                layer_type = type(layer).__name__
                
                for group_name, weight_names in layer_groups.items():
                    if layer_type.lower() in group_name.lower() or any(
                        pattern in group_name.lower() 
                        for pattern in ['conv', 'dense', 'batch']
                    ):
                        print(f"  Probando grupo por patrón: {group_name}")
                        # Usar este grupo si no se ha usado
                        if group_name not in weight_layer_mapping.values():
                            weight_layer_mapping[layer_name] = group_name
                            
                            for wname in weight_names:
                                if wname in weights_data:
                                    weight_value = weights_data[wname]
                                    found_weights.append(weight_value)
                                    print(f"    Cargado por patrón: {wname} -> {weight_value.shape}")
                            break
            
            # Aplicar pesos si coinciden
            expected_weights = len(layer.get_weights())
            if len(found_weights) == expected_weights:
                try:
                    layer.set_weights(found_weights)
                    mapped_layers += 1
                    print(f"  OK: {expected_weights} pesos aplicados correctamente")
                except Exception as e:
                    print(f"  ERROR aplicando pesos: {e}")
                    # Mostrar formas esperadas vs encontradas
                    expected_shapes = [w.shape for w in layer.get_weights()]
                    found_shapes = [w.shape for w in found_weights]
                    print(f"    Esperado: {expected_shapes}")
                    print(f"    Encontrado: {found_shapes}")
            else:
                print(f"  WARNING: Esperados {expected_weights} pesos, encontrados {len(found_weights)}")
    
    print(f"\n=== Resumen del mapeo ===")
    print(f"Capas con pesos: {total_layers_with_weights}")
    print(f"Capas mapeadas: {mapped_layers}")
    print(f"Éxito: {mapped_layers/total_layers_with_weights*100:.1f}%" if total_layers_with_weights > 0 else "0%")
    
    return mapped_layers > (total_layers_with_weights * 0.8)  # Éxito si mapea >80%

def test_reconstructed_model(model):
    """Probar modelo reconstruido"""
    print(f"\n=== Probando modelo reconstruido ===")
    
    try:
        print(f"Entrada: {model.input_shape}")
        print(f"Salida: {model.output_shape}")
        print(f"Parámetros: {model.count_params():,}")
        
        # Prueba funcional
        test_data = np.random.randn(5, 400, 3).astype(np.float32)
        predictions = model.predict(test_data, verbose=0)
        
        print(f"Predicciones: {predictions.shape}")
        
        # Verificar probabilidades
        valid_count = 0
        for i, pred in enumerate(predictions):
            prob_sum = np.sum(pred)
            if 0.99 <= prob_sum <= 1.01:
                valid_count += 1
            else:
                print(f"  WARNING: Probabilidades inválidas en muestra {i}: {prob_sum:.6f}")
        
        print(f"Probabilidades válidas: {valid_count}/{len(predictions)}")
        
        # Mostrar ejemplos
        print("Ejemplos de predicciones:")
        for i in range(min(3, len(predictions))):
            p, s, n = predictions[i]
            print(f"  Muestra {i+1}: P={p:.3f}, S={s:.3f}, N={n:.3f}")
        
        return valid_count > 0
        
    except Exception as e:
        print(f"ERROR en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Conversor directo GPD desde JSON+HDF5')
    parser.add_argument('--json', required=True, help='Archivo model_pol.json')
    parser.add_argument('--hdf5', required=True, help='Archivo model_pol_best.hdf5')
    parser.add_argument('--output', required=True, help='Archivo .keras de salida')
    parser.add_argument('--test', action='store_true', help='Probar modelo reconstruido')
    
    args = parser.parse_args()
    
    print("=== Conversor Directo GPD (JSON + HDF5) ===")
    
    if not check_environment():
        return False
    
    try:
        # Paso 1: Analizar arquitectura del JSON
        architecture_info = analyze_original_json(args.json)
        
        # Paso 2: Extraer pesos del HDF5
        weights_data = extract_weights_from_hdf5(args.hdf5)
        
        # Paso 3: Crear modelo desde análisis
        model, layer_names = create_gpd_model_from_analysis(architecture_info)
        
        # Compilar modelo
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Paso 4: Mapear pesos
        mapping_success = map_weights_to_reconstructed_model(model, weights_data, layer_names)
        
        if not mapping_success:
            print("WARNING: Mapeo de pesos no fue completamente exitoso")
        
        # Paso 5: Probar si se solicita
        if args.test:
            if not test_reconstructed_model(model):
                print("WARNING: Modelo no pasa las pruebas")
        
        # Paso 6: Guardar modelo
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
        
        print(f"\n=== Conversión completada ===")
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