#!/usr/bin/env python3
import json, tensorflow as tf

LEGACY_WEIGHTS = "model_pol_best.hdf5"       # pon aqui tu .h5 con pesos
SEQ_JSON_PATH  = "gpd_feature_extractor.json"
TFLITE_OUT     = "gpd_lite_dynamic.tflite"
H5_OUT         = "gpd_lite_tf2.h5"
SAVEDMODEL_DIR = "gpd_lite_savedmodel"

# 1) Cargar el bloque Sequential desde JSON
with open(SEQ_JSON_PATH, "r", encoding="utf-8") as f:
    seq_json = json.load(f)

# tf.keras puede deserializar un Sequential/Model desde JSON
feature_extractor = tf.keras.models.model_from_json(json.dumps(seq_json))
print("Feature extractor name:", feature_extractor.name)

# 2) Armar el grafo tri-rama con slicing nativo (sin Lambda legacy)
inp = tf.keras.Input(shape=(400, 3), name="input_400x3")
z = inp[:, :, 0:1]   # canal Z
n = inp[:, :, 1:2]   # canal N
e = inp[:, :, 2:3]   # canal E

# Reutiliza la MISMA instancia (comparticion de pesos)
yz = feature_extractor(z)
yn = feature_extractor(n)
ye = feature_extractor(e)

# 3) Concatenacion final (usa el mismo nombre si quieres trazabilidad)
out = tf.keras.layers.Concatenate(name="activation_7")([yz, yn, ye])

model = tf.keras.Model(inputs=inp, outputs=out, name="gpd_lite_tf2")
model.summary()

# 4) Cargar pesos legacy por nombre (mapea capas internas del Sequential)
#    skip_mismatch=True para omitir capas no encontradas (deberian ser 0 si nombres coinciden)
model.load_weights(LEGACY_WEIGHTS, by_name=True, skip_mismatch=True)
print("Pesos cargados (by_name=True, skip_mismatch=True).")

# 5) Guardados en formatos modernos
model.save(H5_OUT)
model.save(SAVEDMODEL_DIR)
print("Guardado:", H5_OUT, "y", SAVEDMODEL_DIR)

# 6) Conversion TFLite (dynamic range; rapido y compatible)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range
tflite_model = conv.convert()
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)
print("TFLite listo:", TFLITE_OUT)
