#!/usr/bin/env python3
import json, sys

def get_model_config(obj):
    return obj.get("config") or obj.get("model_config", {}).get("config")

def is_seq_like(L):
    return L.get("class_name") == "Sequential" or (
        L.get("class_name") == "Model" and isinstance(L.get("config"), dict) and "layers" in L["config"]
    )

def main():
    if len(sys.argv) != 3:
        print("Uso: python extract_sequential.py legacy_model.json gpd_feature_extractor.json")
        sys.exit(2)

    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "r", encoding="utf-8") as f:
        mj = json.load(f)

    cfg = get_model_config(mj)
    layers = cfg["layers"]
    seq_layer = next(L for L in layers if is_seq_like(L))  # toma el primer bloque secuencial
    # Guardamos el bloque tal cual (incluye 'class_name', 'name' y 'config')
    with open(dst, "w", encoding="utf-8") as w:
        json.dump(seq_layer, w, indent=2)
    print("Guardado:", dst, "| nombre:", seq_layer.get("name"))

if __name__ == "__main__":
    main()
