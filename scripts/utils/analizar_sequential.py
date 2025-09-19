# analizar_sequential.py
# Entorno: Python 3.9, tensorflow 2.12, keras 2.12, numpy<2, h5py 3.8

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential as KSequential

# --- cargar el modelo ---
# Nota: si hay Lambda layers con funciones personalizadas
# y falla la carga, intenta: load_model(..., compile=False, custom_objects={})
model = load_model('./models/gpd_v2.keras', compile=False)

# --- localizar el bloque siames sequential_1 ---
try:
    seq = model.get_layer('sequential_1')
except ValueError:
    # fallback: primer submodelo Sequential encontrado en el grafo
    seq = None
    for l in model.layers:
        if isinstance(l, KSequential):
            seq = l
            break
if seq is None:
    raise RuntimeError("No se encontro 'sequential_1' ni un submodelo Sequential.")

# --- resumen del submodelo ---
print("\n=== Resumen de sequential_1 ===")
seq.summary()

# --- helpers ---
def act_name(layer):
    a = getattr(layer, 'activation', None)
    try:
        return a.__name__ if a else None
    except Exception:
        return str(a)

def out_shape(layer):
    try:
        return layer.output_shape
    except Exception:
        try:
            return layer.compute_output_shape(layer.input_shape)
        except Exception:
            return None

# --- detalle de capas ---
print("\n=== Detalle por capa ===")
total_params = 0
for i, l in enumerate(seq.layers):
    print(f"{i:02d} | {l.name:25s} | {l.__class__.__name__:15s} | "
          f"out={out_shape(l)} | "
          f"filters={getattr(l,'filters',None)} "
          f"kernel={getattr(l,'kernel_size',None)} "
          f"stride={getattr(l,'strides',None)} "
          f"units={getattr(l,'units',None)} "
          f"activation={act_name(l)} "
          f"params={l.count_params()}")
    total_params += l.count_params()
print(f"\nTotal params (submodelo): {total_params}")

# --- formas de los pesos ---
print("\n=== Pesos por capa ===")
for l in seq.layers:
    for w in l.weights:
        print(f"{l.name:25s} -> {w.name:40s} {tuple(w.shape)}")

# --- guardar el submodelo para pruebas aisladas ---
try:
    seq.save('gpd_v2_sequential_1.keras')  # TF 2.12 guarda arquitectura+pesos
    print("\nSubmodelo guardado en gpd_v2_sequential_1.keras")
except Exception as e:
    print("\nAviso al guardar submodelo:", e)

# --- exportar diagrama PNG del submodelo ---
# Requiere: pip install pydot graphviz  y tener 'dot' en el PATH del sistema
try:
    tf.keras.utils.plot_model(seq, to_file='sequential_1.png',
                              show_shapes=True, expand_nested=True, dpi=200)
    print("Diagrama exportado a sequential_1.png")
except Exception as e:
    print("Aviso al exportar diagrama:", e)

# --- opcional: comprobar usos del bloque compartido en el grafo principal ---
try:
    print("\nVeces que el bloque es usado (inbound nodes):", len(seq._inbound_nodes))
except Exception:
    pass
