import os
import shutil
import pandas as pd

# Rutas de origen y destino
SRC_DIR = "/home/rsa/projects/gpd/data/dataset/test/"
DST_DIR = "/home/rsa/projects/gpd/data/dataset/test_100/"
CSV_FILE = "/home/rsa/projects/gpd/data/dataset/dataset_estratificado_1000.csv"

# Crear directorio destino si no existe
os.makedirs(DST_DIR, exist_ok=True)

# Leer archivo CSV con la lista de archivos seleccionados
df = pd.read_csv(CSV_FILE)

# Iterar sobre la columna "mseed" y copiar archivos
not_found = []
copiados = 0

for fname in df["mseed"].unique():
    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DST_DIR, fname)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)  # copia preservando metadata
        copiados += 1
    else:
        not_found.append(fname)

print(f"Archivos copiados: {copiados}")
if not_found:
    print("\nArchivos no encontrados en el directorio origen:")
    for nf in not_found:
        print(" -", nf)
else:
    print("\nTodos los archivos fueron copiados correctamente.")
