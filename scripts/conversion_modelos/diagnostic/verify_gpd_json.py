#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from typing import Dict, Any, List, Optional

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_model_config(model_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Soporta estructuras:
      - {"class_name":"Model","config":{...}}
      - {"model_config":{"class_name":"Model","config":{...}}}
    """
    if "config" in model_json and isinstance(model_json["config"], dict) and "layers" in model_json["config"]:
        return model_json["config"]
    mc = model_json.get("model_config", {})
    if isinstance(mc, dict) and "config" in mc:
        return mc["config"]
    raise ValueError("No se encontro 'config.layers' en el JSON. Revisa el archivo de entrada.")

def index_by_name(layers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {L["name"]: L for L in layers if "name" in L}

def is_seq_like(L: Dict[str, Any]) -> bool:
    """
    Devuelve True si la capa parece un Sequential real o un sub-modelo con capas internas.
    """
    if L.get("class_name") == "Sequential":
        return True
    if L.get("class_name") == "Model" and isinstance(L.get("config"), dict) and "layers" in L["config"]:
        return True
    return False

def get_inbound_sources(layer: Dict[str, Any]) -> List[str]:
    """
    Extrae los nombres de capas que alimentan a 'layer' segun inbound_nodes.
    Keras serializa inbound_nodes como lista de nodos, cada nodo es lista de entradas,
    cada entrada es [layer_name, node_index, tensor_index, kwargs_dict].
    """
    sources = []
    inb = layer.get("inbound_nodes", [])
    for node in inb:
        for entry in node:
            if isinstance(entry, list) and entry:
                sources.append(entry[0])
    return sources

def count_concat_inputs(layer: Dict[str, Any]) -> int:
    inb = layer.get("inbound_nodes", [])
    if not inb:
        return 0
    # Numero de entradas del primer nodo
    return len(inb[0])

def get_internal_layers_block(layer):
    """
    Devuelve la lista de capas internas si 'layer' es un Sequential o un Model anidado.
    Maneja ambos formatos de Keras:
      - Sequential con config=dict -> config['layers']
      - Sequential con config=list -> config (lista directa)
      - Model con config=dict -> config['layers']
    """
    cls = layer.get("class_name")
    cfg = layer.get("config")
    if cls == "Sequential":
        if isinstance(cfg, dict):
            return cfg.get("layers", [])
        elif isinstance(cfg, list):
            return cfg
        else:
            return []
    elif cls == "Model":
        if isinstance(cfg, dict):
            return cfg.get("layers", [])
        else:
            return []
    return []


def find_first(layers: List[Dict[str, Any]], pred) -> Optional[Dict[str, Any]]:
    for L in layers:
        if pred(L):
            return L
    return None

def verify_architecture_hypothesis(model_json: Dict[str, Any]) -> bool:
    cfg = get_model_config(model_json)
    layers = cfg["layers"]
    by_name = index_by_name(layers)

    # localizar secuencia compartida y concatenate
    seq_layer = find_first(layers, is_seq_like)
    if not seq_layer:
        print("No se encontro un bloque Sequential/Model anidado.")
        return False

    concat_layer = find_first(layers, lambda L: L.get("class_name") == "Concatenate")
    if not concat_layer:
        print("No se encontro una capa Concatenate.")
        return False

    # 1) entradas a Sequential
    seq_sources = get_inbound_sources(seq_layer)
    n_seq = len(seq_sources)
    print(f"Conexiones al Sequential: {n_seq}")
    if seq_sources:
        print("Entradas al Sequential desde:", seq_sources)

    # Comprobar que las fuentes del Sequential son Lambda
    lambda_flags = [by_name.get(s, {}).get("class_name") == "Lambda" for s in seq_sources]
    if seq_sources:
        print("¿Fuentes del Sequential son Lambda?:", lambda_flags)

    # 2) entradas a Concatenate
    n_concat = count_concat_inputs(concat_layer)
    concat_sources = get_inbound_sources(concat_layer)
    print(f"Inputs a Concatenate: {n_concat}")
    if concat_sources:
        print("Entradas a Concatenate desde:", concat_sources)

    # 3) (opcional) tamano del bloque interno del Sequential si existe
    internal_layers = get_internal_layers_block(seq_layer)
    if internal_layers:
        print(f"Capas internas del bloque compartido: {len(internal_layers)} (esperado ~25)")

    # Validacion principal: 3 entradas al Sequential y 3 al Concatenate
    ok_seq = (n_seq == 3)
    ok_concat = (n_concat == 3)
    ok_lambda = all(lambda_flags) if lambda_flags else False

    result = ok_seq and ok_concat and ok_lambda
    print(f"Hipotesis valida: {result}")
    return result

def main():
    if len(sys.argv) != 2:
        print("Uso: python verify_gpd_json.py /ruta/al/modelo.json")
        sys.exit(2)
    path = sys.argv[1]
    try:
        mj = load_json(path)
        ok = verify_architecture_hypothesis(mj)
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
