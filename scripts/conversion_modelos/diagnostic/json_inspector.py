#!/usr/bin/env python3
"""
Inspector para analizar la estructura del JSON del modelo GPD
"""

import json
import sys

def inspect_json_structure(json_path):
    """Inspeccionar estructura del JSON sin deserializar"""
    print(f"=== Inspeccionando {json_path} ===")
    
    with open(json_path, 'r') as f:
        model_config = json.load(f)
    
    print(f"Claves principales: {list(model_config.keys())}")
    print(f"Clase del modelo: {model_config.get('class_name')}")
    
    config = model_config.get('config', {})
    print(f"Claves de config: {list(config.keys())}")
    
    # Inspeccionar layers
    if 'layers' in config:
        layers = config['layers']
        print(f"\nTipo de 'layers': {type(layers)}")
        print(f"Número de elementos en layers: {len(layers)}")
        
        print("\n=== Análisis de capas ===")
        for i, layer in enumerate(layers):
            print(f"\nCapa {i}:")
            print(f"  Tipo: {type(layer)}")
            
            if isinstance(layer, dict):
                print(f"  Claves: {list(layer.keys())}")
                print(f"  class_name: {layer.get('class_name')}")
                
                layer_config = layer.get('config', {})
                print(f"  config tipo: {type(layer_config)}")
                
                if isinstance(layer_config, dict):
                    print(f"  config claves: {list(layer_config.keys())}")
                    print(f"  name: {layer_config.get('name', 'NO_NAME')}")
                elif isinstance(layer_config, list):
                    print(f"  config es lista con {len(layer_config)} elementos")
                    if len(layer_config) > 0:
                        print(f"  primer elemento config: {type(layer_config[0])}")
                        if isinstance(layer_config[0], dict):
                            print(f"  claves primer elemento: {list(layer_config[0].keys())}")
            elif isinstance(layer, list):
                print(f"  Es lista con {len(layer)} elementos")
                if len(layer) > 0:
                    print(f"  Primer elemento: {type(layer[0])}")
            else:
                print(f"  Valor directo: {layer}")
    
    # Buscar otras estructuras importantes
    if 'input_layers' in config:
        print(f"\ninput_layers: {config['input_layers']}")
    
    if 'output_layers' in config:
        print(f"output_layers: {config['output_layers']}")
    
    # Mostrar estructura completa (primeras capas)
    print(f"\n=== Estructura completa (primeras 3 capas) ===")
    if 'layers' in config and len(config['layers']) > 0:
        for i in range(min(3, len(config['layers']))):
            print(f"\nCapa {i} completa:")
            print(json.dumps(config['layers'][i], indent=2)[:500] + "...")

def main():
    if len(sys.argv) != 2:
        print("Uso: python json_inspector.py model_pol.json")
        return
    
    json_path = sys.argv[1]
    #inspect_json_structure(json_path)
    verify_architecture_hypothesis(json_path)

if __name__ == "__main__":
    main()