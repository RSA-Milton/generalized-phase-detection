# Utilizar script de inferencia legacy
1. Activar entorno 
micromamba activate gpd_python36 # PC Ubuntu
micromamba activate gpd_py36
2. Ejecutar el script de inferencia legacy 
time python inferencia_legacy.py -I /home/rsa/data/legacy/archivo_3ch.mseed -O /home/rsa/data/out/eventos.out -V --hours 4

# Utilizar script de inferencia keras
1. Activar entorno 
micromamba activate gpd_py39
2. Ejecutar el script de inferencia keras
time python inferencia_keras.py -I /home/rsa/data/TENG/archivo_3ch.mseed -O /home/rsa/data/out/eventos.out -V

# Utilizar script de deteccion STA/LTA
1. Activar entorno 
micromamba activate gpd_py39
2. Ejecutar el script 
basico:
time python stalta_detector.py -I /home/rsa/data/TENG/archivo_3ch.mseed -O /home/rsa/data/out/eventos.out -V
conservador:
python stalta_detector.py -I /home/rsa/data/TENG/archivo_3ch.mseed -O /home/rsa/data/out/eventos.out --coincidence 2 -V