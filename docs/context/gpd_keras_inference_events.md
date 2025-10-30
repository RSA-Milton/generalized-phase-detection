# GPD Keras Inference Events - Documentación Técnica

## Información General

**Archivo:** `scripts/inference/gpd_keras_inference_events.py`

**Propósito:** Script de evaluación de eventos sísmicos usando el modelo GPD (Generalized Phase Detection). Procesa archivos MSEED preprocesados y detecta automáticamente las fases sísmicas P y S mediante redes neuronales profundas.

**Autor/Proyecto:** Sistema de detección automática de fases sísmicas RSA-Milton

**Dependencias principales:**
- TensorFlow/Keras (modelo de deep learning)
- ObsPy (procesamiento de datos sismológicos)
- NumPy (operaciones numéricas y ventanas deslizantes)
- Pandas (manejo de datasets y resultados)

---

## Arquitectura del Sistema

### Flujo de Procesamiento General

```
Dataset CSV (metadata + referencias)
         ↓
Archivos MSEED preprocesados (3 componentes)
         ↓
Modelo GPD Keras (.keras o .hdf5)
         ↓
Detección de fases P y S
         ↓
CSV de resultados + análisis estadístico
```

### Configuración del Sistema

**Parámetros del modelo GPD:**
```python
DEFAULT_MIN_PROBA_P = 0.55   # Umbral de probabilidad para fase P
DEFAULT_MIN_PROBA_S = 0.85   # Umbral de probabilidad para fase S (más estricto)
n_shift = 10                  # Paso entre ventanas (0.1 segundos @ 100Hz)
batch_size = 100              # Tamaño de lote para predicción
half_dur = 2.00               # Medio tamaño de ventana en segundos
only_dt = 0.01                # Intervalo temporal esperado (100Hz)
n_win = 200                   # Número de muestras de medio tamaño
n_feat = 400                  # Tamaño total de ventana en muestras (4 segundos)
```

**Rutas por defecto:**
- Datos MSEED: `data/processed/mseed_events/test_1000/`
- Dataset entrada: `data/processed/datasets/test/dataset_estratificado_1000_with_snr.csv`
- Resultados salida: `results/gpd_keras/resultados_evaluacion_1000_agente_labr.csv`
- Modelo por defecto: definido en `config.py`

---

## Funciones Principales

### 1. `sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True)`

**Propósito:** Crea ventanas deslizantes eficientes sobre series temporales usando NumPy stride tricks.

**Parámetros:**
- `data`: Array de datos de entrada
- `size`: Tamaño de cada ventana
- `stepsize`: Paso entre ventanas consecutivas
- `axis`: Eje sobre el cual aplicar la ventana
- `copy`: Si crear copia o vista (default: True por seguridad)

**Retorna:** Array con forma `(num_ventanas, size)` conteniendo todas las ventanas.

**Implementación técnica:**
- Usa `np.lib.stride_tricks.as_strided` para eficiencia de memoria
- No copia datos innecesariamente (a menos que `copy=True`)
- Validaciones de seguridad para evitar accesos fuera de rango

**Ejemplo:**
```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
windows = sliding_window(data, size=4, stepsize=2)
# Resultado:
# [[1, 2, 3, 4],
#  [3, 4, 5, 6],
#  [5, 6, 7, 8],
#  [7, 8, 9, 10]]
```

---

### 2. `process_event_file(model, mseed_file, min_proba_P, min_proba_S, verbose=False)`

**Propósito:** Función nuclear que procesa un archivo de evento sísmico individual y detecta las fases P y S.

**Parámetros:**
- `model`: Modelo Keras/TensorFlow cargado en memoria
- `mseed_file`: Ruta completa al archivo MSEED preprocesado
- `min_proba_P`: Umbral de probabilidad para activar detección de fase P (típicamente 0.55)
- `min_proba_S`: Umbral de probabilidad para activar detección de fase S (típicamente 0.85)
- `verbose`: Flag para imprimir información de depuración

**Retorna:** Diccionario con estructura:
```python
{
    'num_p': int,        # Número total de fases P detectadas
    'num_s': int,        # Número total de fases S detectadas
    't_p': str,          # Tiempo ISO del mejor pick P (o None)
    't_s': str,          # Tiempo ISO del mejor pick S (o None)
    'prob_p': float,     # Probabilidad del mejor pick P (o None)
    'prob_s': float      # Probabilidad del mejor pick S (o None)
}
```

#### Análisis Detallado Paso a Paso

##### **PASO 1: Inicialización y Carga de Datos** (líneas 81-99)

```python
result = {
    'num_p': 0, 'num_s': 0,
    't_p': None, 't_s': None,
    'prob_p': None, 'prob_s': None
}

st = oc.read(mseed_file)  # Lee archivo MSEED con ObsPy
```

**Conceptos clave:**
- **MSEED (Mini-SEED):** Formato estándar de intercambio de datos sismológicos
- **Stream (st):** Colección de trazas sísmicas (típicamente 3 componentes)
- **Trazas esperadas:** Norte (N), Este (E), Vertical (Z)

**Validación crítica:**
```python
if len(st) != 3:
    if verbose:
        print(f"    ERROR: {len(st)} trazas encontradas, se esperaban 3")
    return result  # Retorna resultado vacío sin crashear
```

**Extracción de metadata temporal:**
```python
dt = st[0].stats.delta              # Intervalo de muestreo (ej: 0.01s = 100Hz)
start_time = st[0].stats.starttime  # Tiempo UTC absoluto de inicio
```

**Significado de `dt`:**
- Define la resolución temporal de los datos
- `dt = 0.01s` → 100 muestras por segundo (100Hz)
- `dt = 0.02s` → 50 muestras por segundo (50Hz)
- Crucial para convertir índices de array → tiempo real

##### **PASO 2: Creación de Ventanas Deslizantes** (líneas 101-112)

**¿Por qué ventanas deslizantes?**

El modelo GPD fue entrenado con ventanas fijas de **4 segundos** (400 muestras @ 100Hz). Para analizar una traza completa de 30-60 segundos, necesitamos:

1. Dividir la señal en ventanas de 4 segundos
2. Superponer ventanas para no perder información en los bordes
3. Evaluar cada ventana independientemente

**Visualización:**
```
Traza completa de 30 segundos (3000 muestras @ 100Hz):
[0____400____800____1200____1600____2000____2400____2800____3000]
 |--w1--|
      |--w2--|
           |--w3--|
                |--w4--|
                     ...
                                         |--w295--|
                                              |--w296--|
```

Cada ventana tiene 400 muestras, con paso de 10 muestras (0.1 segundos).

**Cálculo del vector de tiempos:**
```python
data_length = len(st[0].data)  # Ej: 3000 muestras
tt = (np.arange(0, data_length, n_shift) + n_win) * dt
```

Desglose:
- `np.arange(0, 3000, 10)` → `[0, 10, 20, 30, ..., 2990]` (índices de inicio)
- `+ n_win` (+ 200) → `[200, 210, 220, ..., 3190]` (índices del centro de ventana)
- `* dt` (× 0.01) → `[2.00, 2.10, 2.20, ..., 31.90]` (tiempos en segundos)

**Resultado:** `tt` contiene el tiempo del **punto central** de cada ventana.

**Generación de ventanas por componente:**
```python
sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)  # Norte
sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)  # Este
sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)  # Vertical
```

Forma de cada matriz: `(num_ventanas, 400)`
- Para 3000 muestras: `num_ventanas = floor((3000 - 400) / 10) + 1 = 261`

**Protección contra inconsistencias:**
```python
min_windows = min(sliding_N.shape[0], sliding_E.shape[0], sliding_Z.shape[0])
if min_windows == 0:
    return result  # Datos insuficientes
```

Garantiza que todas las componentes tengan ventanas válidas.

##### **PASO 3: Apilamiento y Normalización** (líneas 114-124)

**Apilamiento de componentes:**
```python
tr_win = np.zeros((min_windows, n_feat, 3), dtype=np.float32)
tr_win[:,:,0] = sliding_N[:min_windows]  # Canal 0: Norte
tr_win[:,:,1] = sliding_E[:min_windows]  # Canal 1: Este
tr_win[:,:,2] = sliding_Z[:min_windows]  # Canal 2: Vertical
```

**Forma final:** `(num_ventanas, 400_muestras, 3_canales)`

**Ejemplo con números reales:**
```
Para un evento de 30s @ 100Hz:
tr_win.shape = (261, 400, 3)
- 261 ventanas temporales
- 400 muestras por ventana (4 segundos)
- 3 canales (N, E, Z)
```

**Normalización crítica:**
```python
max_vals = np.max(np.abs(tr_win), axis=1, keepdims=True) + 1e-9
tr_win = tr_win / max_vals
```

**Desglose matemático:**
1. `np.abs(tr_win)`: valores absolutos → `(261, 400, 3)`
2. `np.max(..., axis=1)`: máximo por ventana (a través de las 400 muestras) → `(261, 3)`
3. `keepdims=True`: mantiene dimensión → `(261, 1, 3)`
4. `+ 1e-9`: epsilon para evitar división por cero
5. División por broadcasting: cada muestra se divide por el máximo de su ventana/canal

**¿Por qué normalizar?**
- El modelo debe ser **invariante a la amplitud**
- Un sismo Mw 3.0 a 100 km tiene la misma forma de onda que uno Mw 5.0 a 300 km
- Solo importa la **forma** de la señal, no su amplitud absoluta
- Cada ventana se normaliza independientemente (importante para cambios de amplitud en la traza)

**Recorte del vector de tiempos:**
```python
tt = tt[:min_windows]  # Asegura sincronización con ventanas
```

##### **PASO 4: Predicción con el Modelo Neural** (líneas 126-130)

```python
ts = model.predict(tr_win, verbose=False, batch_size=batch_size)

prob_P = ts[:,0]  # Probabilidades de fase P
prob_S = ts[:,1]  # Probabilidades de fase S
```

**Arquitectura del modelo (inferida):**
- **Input:** `(batch, 400, 3)` - ventanas temporales multicanal
- **Output:** `(batch, 2)` - dos probabilidades por ventana
  - `[:,0]`: probabilidad de que exista una llegada de onda P
  - `[:,1]`: probabilidad de que exista una llegada de onda S

**Ejemplo de salida:**
```python
# Para 261 ventanas:
ts.shape = (261, 2)

# Ventana 100 (centrada en t=12.0s):
ts[100] = [0.92, 0.15]  # 92% prob P, 15% prob S → Llegada P detectada

# Ventana 200 (centrada en t=22.0s):
ts[200] = [0.23, 0.88]  # 23% prob P, 88% prob S → Llegada S detectada
```

**Procesamiento por lotes:**
- `batch_size=100`: procesa 100 ventanas simultáneamente
- Acelera inferencia en GPU
- Para 261 ventanas: 3 lotes (100 + 100 + 61)

##### **PASO 5: Detección de Fases P** (líneas 132-143)

**Uso de `trigger_onset` (ObsPy):**
```python
from obspy.signal.trigger import trigger_onset
trigs_p = trigger_onset(prob_P, min_proba_P, 0.1)
```

**Parámetros:**
- `prob_P`: serie temporal de probabilidades `[0.2, 0.3, 0.6, 0.7, 0.8, 0.6, 0.5, 0.3, ...]`
- `min_proba_P=0.55`: umbral ON (activación del trigger)
- `0.1`: umbral OFF (desactivación del trigger)

**Funcionamiento (histéresis):**
```
prob_P:  0.2  0.3  0.6  0.7  0.8  0.6  0.5  0.3  0.2  0.6  0.7  0.1
índice:   0    1    2    3    4    5    6    7    8    9   10   11
                   ↑_____ON (>0.55)_____↑
                                              OFF (<0.1)
Trigger 1: (2, 9)  ← inicio en índice 2, fin en índice 9
```

**Retorno:**
```python
trigs_p = [(2, 9), (45, 52), (120, 128)]  # Lista de tuplas (inicio, fin)
```

Cada tupla representa una **región de detección** (trigger).

**Extracción del pick óptimo dentro de cada trigger:**
```python
p_detections = []
for trig in trigs_p:
    if trig[1] == trig[0]:
        continue  # Ignora triggers de un solo punto (spikes)

    # Busca el máximo de probabilidad dentro de la región
    pick_idx = np.argmax(ts[trig[0]:trig[1], 0]) + trig[0]

    # Convierte índice → tiempo UTC absoluto
    pick_time = start_time + tt[pick_idx]

    # Extrae probabilidad correspondiente
    pick_prob = ts[pick_idx, 0]

    # Almacena detección
    p_detections.append((pick_time, pick_prob, pick_idx))
```

**Ejemplo numérico:**
```
Trigger: (2, 9)
ts[2:9, 0] = [0.60, 0.70, 0.80, 0.60, 0.50, 0.30, 0.20]
               idx 0   1    2    3    4    5    6
                        ↑
                      máximo = 0.80 en posición 2 (local)

pick_idx = 2 + 2 = 4 (global)
pick_time = start_time + tt[4] = 2023-10-15T12:34:00.000Z + 2.40s
                                = 2023-10-15T12:34:02.400Z
pick_prob = 0.80
```

**Resultado:**
```python
p_detections = [
    (UTCDateTime('2023-10-15T12:34:02.400Z'), 0.80, 4),
    (UTCDateTime('2023-10-15T12:34:07.100Z'), 0.67, 45),
    (UTCDateTime('2023-10-15T12:34:14.800Z'), 0.92, 120)
]
```

##### **PASO 6: Detección de Fases S** (líneas 145-155)

```python
trigs_s = trigger_onset(prob_S, min_proba_S, 0.1)

s_detections = []
for trig in trigs_s:
    if trig[1] == trig[0]:
        continue
    pick_idx = np.argmax(ts[trig[0]:trig[1], 1]) + trig[0]  # Canal 1 (S)
    pick_time = start_time + tt[pick_idx]
    pick_prob = ts[pick_idx, 1]
    s_detections.append((pick_time, pick_prob, pick_idx))
```

**Diferencias con fase P:**
1. **Umbral más alto:** `min_proba_S=0.85` vs `min_proba_P=0.55`
   - Las ondas S son más difíciles de detectar (menor amplitud, más dispersión)
   - Umbral más estricto reduce falsos positivos

2. **Canal diferente:** `ts[:, 1]` en lugar de `ts[:, 0]`

3. **Lógica idéntica:** mismo algoritmo de trigger + búsqueda de máximo

**Relación física P-S:**
- Onda P llega primero (velocidad ~6 km/s en corteza)
- Onda S llega después (velocidad ~3.5 km/s)
- Diferencia típica: 3-10 segundos para eventos locales (< 300 km)

##### **PASO 7: Selección del Mejor Pick** (líneas 157-173)

**Para fase P:**
```python
result['num_p'] = len(p_detections)

if result['num_p'] > 0:
    # Ordenar por probabilidad (mayor a menor)
    p_detections.sort(key=lambda x: x[1], reverse=True)

    # Seleccionar el de mayor probabilidad
    best_p = p_detections[0]

    # Guardar en formato ISO
    result['t_p'] = best_p[0].isoformat()
    result['prob_p'] = float(best_p[1])
```

**Ejemplo con múltiples detecciones:**
```python
# ANTES del sort:
p_detections = [
    (UTCDateTime('...T12:34:02.400Z'), 0.80, 4),   # Segunda mejor
    (UTCDateTime('...T12:34:07.100Z'), 0.67, 45),  # Peor
    (UTCDateTime('...T12:34:14.800Z'), 0.92, 120)  # MEJOR
]

# DESPUÉS del sort:
p_detections = [
    (UTCDateTime('...T12:34:14.800Z'), 0.92, 120),  # ← seleccionada
    (UTCDateTime('...T12:34:02.400Z'), 0.80, 4),
    (UTCDateTime('...T12:34:07.100Z'), 0.67, 45)
]

# Resultado final:
result['t_p'] = '2023-10-15T12:34:14.800000Z'
result['prob_p'] = 0.92
```

**Para fase S:** Lógica idéntica con `s_detections`.

**Criterios de selección:**
| Criterio | ¿Se usa? | Razón |
|----------|----------|-------|
| Primera detección temporal | ❌ NO | Puede ser ruido o precursor |
| Mayor amplitud de señal | ❌ NO | No evaluada en esta función |
| Mayor probabilidad del modelo | ✅ SÍ | El modelo ha aprendido a discriminar verdaderas llegadas |

**Justificación del criterio:**
- El modelo GPD fue entrenado con miles de eventos etiquetados manualmente
- Aprendió patrones sutiles de forma de onda que caracterizan verdaderas llegadas P/S
- La probabilidad más alta refleja la mayor confianza del modelo
- Funciona mejor que heurísticas simples (amplitud, primera llegada, etc.)

##### **PASO 8: Limpieza de Memoria** (líneas 175-177)

```python
del tr_win, ts, sliding_N, sliding_E, sliding_Z
gc.collect()
```

**¿Por qué es necesario?**

**Cálculo de memoria para un evento típico:**
```python
# Traza de 30s @ 100Hz:
data_length = 3000 muestras

# Ventanas deslizantes (3 componentes):
sliding_N: (261, 400) × 4 bytes (float32) = 418 KB
sliding_E: (261, 400) × 4 bytes = 418 KB
sliding_Z: (261, 400) × 4 bytes = 418 KB

# Array apilado:
tr_win: (261, 400, 3) × 4 bytes = 1.25 MB

# Predicciones:
ts: (261, 2) × 4 bytes = 2 KB

# TOTAL por evento: ~2.5 MB
```

**Impacto en procesamiento masivo:**
```
Procesando 1000 eventos sin limpieza:
- 1000 eventos × 2.5 MB = 2.5 GB de RAM
- Python no libera automáticamente (garbage collector perezoso)
- Riesgo de OOM (Out Of Memory) en sistemas con RAM limitada

Con limpieza explícita:
- Solo ~10-50 MB en memoria (1-2 eventos activos)
- Procesamiento estable de datasets ilimitados
```

**Funciones de limpieza:**
- `del variable`: elimina referencia en el namespace actual
- `gc.collect()`: fuerza recolección de basura inmediata
- Combinados: liberación garantizada antes del siguiente evento

##### **PASO 9: Manejo de Errores** (líneas 179-183)

```python
except Exception as e:
    if verbose:
        print(f"    ERROR procesando {mseed_file}: {e}")

return result  # Siempre retorna algo, nunca crashea
```

**Filosofía de diseño:**
- **Fail-safe:** un evento corrupto no detiene el procesamiento completo
- **Retorno consistente:** siempre devuelve diccionario con estructura esperada
- **Logging opcional:** solo muestra errores si `verbose=True`

**Errores comunes capturados:**
- Archivo MSEED corrupto o formato inválido
- Número incorrecto de trazas (≠ 3)
- Inconsistencias de muestreo entre componentes
- Problemas de memoria en eventos muy largos
- Errores del modelo (input shape incorrecto)

---

### 3. `main()`

**Propósito:** Función principal que orquesta el flujo completo de evaluación.

#### Subsecciones Principales

##### **A. Parsing de Argumentos** (líneas 186-226)

Argumentos CLI disponibles:

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--mseed-dir` | `data/processed/mseed_events/test_1000/` | Directorio con archivos MSEED |
| `--csv-input` | `dataset_estratificado_1000_with_snr.csv` | Dataset de referencia |
| `--csv-output` | `resultados_evaluacion_1000_agente_labr.csv` | Archivo de salida |
| `--stations` | `['LABR']` | Estaciones a procesar |
| `--min-proba-p` | `0.55` | Umbral de probabilidad P |
| `--min-proba-s` | `0.85` | Umbral de probabilidad S |
| `--model-path` | Default del config | Nombre del modelo a usar |
| `-V, --verbose` | `False` | Modo detallado |

**Ejemplos de uso:**
```bash
# Evaluación básica
python gpd_keras_inference_events.py -V

# Umbrales personalizados
python gpd_keras_inference_events.py --min-proba-p 0.60 --min-proba-s 0.90 -V

# Múltiples estaciones
python gpd_keras_inference_events.py --stations CHAI LABR CUSH PORT -V

# Modelo específico
python gpd_keras_inference_events.py --model-path gpd_v2.keras -V
```

##### **B. Carga del Modelo** (líneas 228-277)

```python
# Resolución de ruta del modelo
if args.model_path is None:
    model_path = config.get_default_model_path()
else:
    model_path = config.get_models_dir() / args.model_path

# Carga con compilación deshabilitada (solo inferencia)
model = load_model(model_path, compile=False)
```

**Formatos soportados:**
- `.keras`: formato nativo de Keras 3.x
- `.hdf5` / `.h5`: formato legacy de Keras 2.x

**Verificación de integridad:**
```python
print(f"OK: Modelo cargado - input_shape={model.input_shape}, output_shape={model.output_shape}")
# Salida esperada: input_shape=(None, 400, 3), output_shape=(None, 2)
```

##### **C. Carga y Filtrado del Dataset** (líneas 279-295)

```python
df_ref = pd.read_csv(args.csv_input)
df_filtered = df_ref[df_ref['Estacion'].isin(args.stations)]
```

**Estructura esperada del CSV de entrada:**
```
Estacion,mseed,SNR_P,SNR_S,...
LABR,evento_001_LABR.mseed,15.3,8.7,...
LABR,evento_002_LABR.mseed,22.1,12.4,...
CHAI,evento_001_CHAI.mseed,18.9,10.2,...
```

**Filtrado dinámico:**
- Permite procesar subconjuntos de estaciones sin modificar el dataset
- Útil para evaluación paralela en múltiples máquinas

##### **D. Loop de Procesamiento** (líneas 297-336)

```python
results = []

for idx, row in df_filtered.iterrows():
    estacion = row['Estacion']
    mseed_name = row['mseed']
    mseed_path = os.path.join(args.mseed_dir, mseed_name)

    # Verificar existencia del archivo
    if not os.path.isfile(mseed_path):
        continue

    # Procesar evento
    detection = process_event_file(model, mseed_path,
                                   args.min_proba_p, args.min_proba_s,
                                   args.verbose)

    # Preparar fila de resultado
    result_row = {
        'Estacion': estacion,
        'mseed': mseed_name,
        'Num-P': detection['num_p'],
        'Num-S': detection['num_s'],
        'T-P': detection['t_p'] if detection['t_p'] else 'NA',
        'T-S': detection['t_s'] if detection['t_s'] else 'NA',
        'Pond T-P': detection['prob_p'] if detection['prob_p'] else 'NA',
        'Pond T-S': detection['prob_s'] if detection['prob_s'] else 'NA'
    }

    results.append(result_row)
```

**Estructura del CSV de salida:**
```
Estacion,mseed,Num-P,Num-S,T-P,T-S,Pond T-P,Pond T-S
LABR,evento_001_LABR.mseed,1,1,2023-10-15T12:34:05.120Z,2023-10-15T12:34:09.870Z,0.876,0.923
LABR,evento_002_LABR.mseed,2,0,2023-10-15T14:22:11.340Z,NA,0.654,NA
```

##### **E. Análisis Estadístico Completo** (líneas 350-496)

El script genera un **reporte exhaustivo** con múltiples métricas:

**1. Estadísticas por estación:**
```
LABR  : Total=200, Zero-P=15 (7.5%), Multi-P=45 (22.5%), Valid-P=185 (92.5%), ...
CHAI  : Total=200, Zero-P=22 (11.0%), Multi-P=38 (19.0%), Valid-P=178 (89.0%), ...
```

**2. Estadísticas globales:**
- Eventos sin detecciones (Num-P = 0, Num-S = 0)
- Eventos con exactamente 1 detección
- Eventos con múltiples detecciones
- Eventos con datos utilizables (T-P ≠ 'NA')

**3. Calidad de detección:**
```
Eventos con P detectada que también tienen S: 850/920 (92.4%)
Eventos con S detectada que también tienen P: 850/880 (96.6%)
```

Métrica clave: eventos con **ambas fases** son los más útiles para localización.

**4. Distribución de probabilidades:**
```
Probabilidades P seleccionadas: N=920, media=0.742, min=0.551, max=0.987
Probabilidades S seleccionadas: N=880, media=0.901, min=0.852, max=0.998
```

Análisis de confianza del modelo en las detecciones seleccionadas.

**5. Resumen de utilidad:**
```
Eventos sin detecciones: 45 (4.5%)
Eventos con una fase utilizable: 115 (11.5%)
Eventos con ambas fases utilizables: 840 (84.0%)
Total eventos utilizables: 955 (95.5%)
```

**Nota sobre código legacy:**
Las líneas 414-445 contienen código para analizar "correcciones P-S" que **no está implementado** en el procesamiento actual. Parece ser código residual de una versión anterior que aplicaba correcciones temporales cuando S era detectada antes que P.

---

## Estructura de Datos

### Dataset de Entrada (CSV)

**Columnas requeridas:**
- `Estacion`: Código de estación (ej: LABR, CHAI, CUSH)
- `mseed`: Nombre del archivo MSEED correspondiente

**Columnas opcionales:**
- `SNR_P`, `SNR_S`: Relación señal/ruido de fases manuales
- `Magnitud`, `Distancia`, `Profundidad`: Parámetros del evento
- `Pick_P_manual`, `Pick_S_manual`: Tiempos de referencia

### Archivos MSEED

**Formato esperado:**
- 3 trazas (componentes N, E, Z)
- Misma frecuencia de muestreo (típicamente 100 Hz)
- Preprocesados (filtrado, detrend, normalización base)
- Duración típica: 30-60 segundos
- Fase P centrada aproximadamente en t=10-15s

**Metadata importante:**
- `stats.delta`: intervalo de muestreo (0.01s para 100Hz)
- `stats.starttime`: tiempo UTC absoluto de inicio
- `stats.station`: código de estación
- `stats.channel`: canal (HHN, HHE, HHZ o similar)

### CSV de Resultados

**Columnas generadas:**
```
Estacion       str    Código de estación
mseed          str    Nombre del archivo procesado
Num-P          int    Número total de fases P detectadas
Num-S          int    Número total de fases S detectadas
T-P            str    Tiempo ISO del mejor pick P (o 'NA')
T-S            str    Tiempo ISO del mejor pick S (o 'NA')
Pond T-P       float  Probabilidad del pick P (o 'NA')
Pond T-S       float  Probabilidad del pick S (o 'NA')
```

**Interpretación de resultados:**

| Caso | Num-P | Num-S | Interpretación |
|------|-------|-------|----------------|
| Ideal | 1 | 1 | Detección limpia de ambas fases |
| Bueno | 2-3 | 1-2 | Múltiples triggers, mejor pick seleccionado |
| Parcial | 1 | 0 | Solo fase P detectada (evento pequeño o ruidoso) |
| Problemático | 0 | 0 | Sin detecciones (ruido o señal muy débil) |
| Ruidoso | >5 | >5 | Múltiples falsos positivos (señal muy ruidosa) |

---

## Consideraciones Técnicas

### Rendimiento y Escalabilidad

**Recursos por evento:**
- Tiempo de procesamiento: ~0.1-0.5 segundos (CPU) / ~0.01-0.05 segundos (GPU)
- Memoria pico: ~2.5 MB por evento
- I/O disco: lectura de ~50-200 KB (archivo MSEED comprimido)

**Escalabilidad:**
- 1000 eventos: ~2-8 minutos (CPU) / ~10-50 segundos (GPU)
- 10000 eventos: ~20-80 minutos (CPU) / ~2-8 minutos (GPU)
- Cuello de botella: I/O de disco para datasets grandes

**Optimizaciones posibles:**
1. Paralelización por estación (múltiples procesos)
2. Carga de datos en batch (reducir I/O)
3. Usar GPU para inferencia (speedup ~10-50x)
4. Cacheo de modelo en memoria compartida (multiprocessing)

### Limitaciones Conocidas

**1. Parámetros hardcodeados:**
- `n_shift`, `batch_size`, `half_dur` no son configurables vía CLI
- Cambiarlos requiere modificar el código fuente

**2. Asunciones sobre datos:**
- Frecuencia de muestreo fija (100 Hz)
- Exactamente 3 componentes
- Preprocesamiento previo (no aplica filtros ni correcciones)

**3. Código legacy:**
- Referencias a columna `'Corregido'` que no se crea (líneas 414-445)
- Variable `correcciones_aplicadas` no definida
- Estas secciones siempre reportarán "información no disponible"

**4. Manejo de errores:**
- Archivos faltantes se saltan silenciosamente (si no verbose)
- No hay reintentos en caso de errores transitorios
- Eventos con errores no se registran en el CSV de salida

### Validación de Resultados

**Métricas clave para evaluar calidad:**

1. **Tasa de detección:**
   - `Valid-P / Total_events`: debe ser > 85% para dataset de calidad
   - `Valid-S / Total_events`: debe ser > 75% (S es más difícil)

2. **Tasa de ambas fases:**
   - `Valid_both / Total_events`: debe ser > 70% para localización efectiva

3. **Distribución de probabilidades:**
   - Media de `Pond T-P`: debe ser > 0.7
   - Media de `Pond T-S`: debe ser > 0.85
   - Si las medias son bajas, considerar bajar umbrales

4. **Tasa de múltiples detecciones:**
   - `Multi-P / Total_events`: idealmente < 30%
   - Si es > 50%, indica señales muy ruidosas o umbrales muy bajos

**Banderas rojas:**
- `Zero-P > 20%`: dataset de baja calidad o modelo inadecuado
- `Multi-P > 50%`: señales muy ruidosas, considerar aumentar umbrales
- `Prob_P_media < 0.6`: detecciones de baja confianza, revisar umbrales

---

## Integración con el Pipeline

### Paso anterior: Preprocesamiento

**Script relacionado:** `scripts/preprocessing/preprocess_mseed_events.py`

Debe generar:
- Archivos MSEED con 3 componentes (N, E, Z)
- Frecuencia de muestreo normalizada (100 Hz)
- Aplicar filtros pasa-banda
- Detrend y remoción de respuesta instrumental
- Duración fija (típicamente 30-60s)
- P centrada aproximadamente en el medio

### Paso posterior: Evaluación de Precisión

**Script relacionado:** `scripts/evaluation/evaluate_picks.py` (hipotético)

Usaría el CSV generado para:
- Comparar `T-P` vs `Pick_P_manual`
- Calcular errores temporales (diferencias en segundos)
- Generar curvas de precisión-recall
- Estadísticas de error por estación, magnitud, distancia

---

## Troubleshooting

### Problemas Comunes

**1. ERROR: Modelo no encontrado**
```
Solución:
- Verificar que el modelo existe en el directorio de modelos
- Usar --model-path con el nombre correcto del archivo
- Listar modelos disponibles: ls <ruta_modelos>/*.keras
```

**2. ERROR: X trazas encontradas, se esperaban 3**
```
Causa: Archivo MSEED corrupto o preprocesamiento incorrecto
Solución:
- Revisar pipeline de preprocesamiento
- Verificar archivo específico con ObsPy: obspy.read(archivo)
- Excluir archivo del dataset si está corrupto
```

**3. Muy pocas detecciones (Zero-P > 50%)**
```
Causa: Umbrales muy altos o modelo inadecuado
Solución:
- Reducir umbrales: --min-proba-p 0.40 --min-proba-s 0.70
- Verificar que el modelo es compatible con los datos
- Revisar preprocesamiento (filtros, normalización)
```

**4. Demasiadas detecciones (Multi-P > 70%)**
```
Causa: Señales muy ruidosas o umbrales muy bajos
Solución:
- Aumentar umbrales: --min-proba-p 0.65 --min-proba-s 0.90
- Mejorar preprocesamiento (filtros más agresivos)
- Revisar calidad de datos en origen
```

**5. Out of Memory**
```
Causa: Procesamiento de demasiados eventos simultáneos
Solución:
- La limpieza con gc.collect() ya está implementada
- Reducir batch_size en el código (línea 40)
- Procesar dataset en lotes más pequeños
```

---

## Referencias

**Bibliotecas utilizadas:**
- [ObsPy](https://docs.obspy.org/): Procesamiento sismológico
- [TensorFlow/Keras](https://www.tensorflow.org/): Deep learning
- [NumPy](https://numpy.org/): Computación numérica
- [Pandas](https://pandas.pydata.org/): Análisis de datos

**Método GPD:**
- Ross, Z. E., et al. (2018). "Generalized Seismic Phase Detection with Deep Learning". Bulletin of the Seismological Society of America.

**Conceptos sismológicos:**
- **Fase P (Primary):** Onda compresional, primera en llegar
- **Fase S (Secondary):** Onda de corte, llega después de P
- **Pick:** Tiempo de llegada de una fase sísmica
- **SNR (Signal-to-Noise Ratio):** Relación señal/ruido

---

## Historial de Versiones

**Última actualización:** 2025-10-30

**Cambios recientes:**
- Documentación completa del sistema
- Análisis detallado de `process_event_file()`
- Identificación de código legacy no funcional
- Ejemplos numéricos y visualizaciones

**Mantenedores:** Proyecto RSA-Milton / Generalized Phase Detection
