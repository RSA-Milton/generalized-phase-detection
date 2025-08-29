# Utilizar script de inferencia legacy
1. Activar entorno 
micromamba activate gpd_python36 # PC Ubuntu
2. Ejecutar el script de inferencia legacy 
time python inferencia_legacy.py -I /home/rsa/data/legacy/archivo.mseed -O /home/rsa/data/out/archivo.out -V --hours 4

# Utilizar script de inferencia keras
1. 1. Activar entorno 
micromamba activate gpd_py39
2. Ejecutar el script de inferencia keras
time python inferencia_keras.py -I /home/rsa/data/TENG/TENG_20250827_040000.mseed -O archivo.out -V