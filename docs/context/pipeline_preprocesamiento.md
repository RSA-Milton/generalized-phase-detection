# Pipeline de Preprocesamiento: GPD con 1 Componente (Z)

## Resumen Ejecutivo

Este documento describe el pipeline para transformar archivos miniSEED de componente vertical (Z) a vectores listos para entrenamiento de GPD.

---

## 1. Especificaciones del Dataset

### 1.1 Dataset Original (Entrada)

**Características generales:**
- **Total de archivos:** 22,191
- **Formato:** miniSEED (.mseed)
- **Distribución de clases:** Balanceado (33.33% cada clase)
  - P (onda primaria): ~7,397 archivos
  - S (onda secundaria): ~7,397 archivos
  - Ruido (noise): ~7,397 archivos

**Características temporales:**
- **Tasa de muestreo:** 64 Hz
- **Duración:** ~6 segundos
- **Número de muestras:** 384-385 samples por traza
- **Variación:** ±1 muestra

**Características espaciales:**
- **Componentes:** 1 (solo Z - vertical)

**Nomenclatura de archivos:**
- **P:** `<event>_<sampleidx>_P.mseed`
- **S:** `<event>_<sampleidx>_S.mseed`
- **Ruido:** `noise_<sampleidx>.mseed`

### 1.2 Dataset Objetivo (Salida para GPD)

**Características generales:**
- **Total de muestras:** 22,191 (preservado)
- **Formato:** NumPy arrays (.npy) o HDF5 (.h5)
- **Distribución de clases:** Balanceado (igual que entrada)

**Características temporales:**
- **Tasa de muestreo:** 100 Hz
- **Duración:** 4 segundos (exactos)
- **Número de muestras:** 400 samples por traza

**Características espaciales:**
- **Shape por muestra:** (400,)
  - Dimensión 0: Tiempo (400 pasos)
  - 1 canal (Z)

**Características de señal:**
- **Normalización:** Max-abs aplicada
- **Detrend:** DC offset y tendencia lineal removidos
- **Filtrado:** Pasa-banda 1-30 Hz
- **Alineación temporal:** Recorte central (ventana centrada)

**Metadatos asociados:**
- **Labels:** Array 1D con clases codificadas (0: P, 1: S, 2: Noise)
- **Tipo de dato:** float32
- **Etiquetas:** int8

---

## 2. Pipeline de Procesamiento

### 2.1 Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: archivo.mseed (64 Hz, ~6s, 384-385 samples, 1 canal Z) │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 1: Lectura con ObsPy                                      │
│  - Cargar archivo mseed                                         │
│  - Extraer traza única (Trace)                                  │
│  - Verificar: sampling_rate == 64 Hz                            │
│  - Verificar: 380 <= npts <= 390                                │
│  - Inferir clase del nombre de archivo                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 2: Detrend / Demean                                       │
│  - trace.detrend('demean')  # Remover offset DC                 │
│  - trace.detrend('linear')  # Remover tendencia lineal          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 3: Filtrado Pasa-Banda 1-30 Hz                            │
│  - Butterworth 4 polos, zero-phase                              │
│  - freqmin=1.0, freqmax=30.0 Hz                                 │
│  - Nota: 30 Hz < Nyquist(32 Hz) del muestreo original          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 4: Remuestreo (64 Hz → 100 Hz)                            │
│  - Método: linear o lanczos                                     │
│  - Output: ~600 muestras @ 100 Hz (6 segundos)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 5: Normalización Max-Abs                                  │
│  - maxabs = max(|data|)                                         │
│  - data_norm = data / max(maxabs, 1e-10)                        │
│  - Resultado: rango típico [-1, 1]                              │
│  - Se normaliza ANTES del recorte                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 6: Recorte Temporal (6s → 4s, centrado)                   │
│  - Input: ~600 muestras                                         │
│  - Output: 400 muestras                                         │
│  - start = (total - 400) // 2  # ≈100                           │
│  - end = start + 400            # ≈500                          │
│  - data_crop = data_norm[start:end]                             │
│  - Elimina ~1s al inicio y ~1s al final                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 7: Validaciones por Muestra                               │
│  - len(data_crop) == 400                                        │
│  - No contiene NaN                                              │
│  - No contiene Inf                                              │
│  - max(|data_crop|) <= 1.0 + margen numérico                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 8: Etiquetado                                             │
│  - P → 0                                                        │
│  - S → 1                                                        │
│  - Noise → 2                                                    │
│  - Guardar nombre de archivo original como metadata             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 9: Salida por Muestra                                     │
│  - Vector: shape (400,), dtype float32                          │
│  - Label: tipo int8                                             │
│  - Metadata: filename, class_name                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: array (100 Hz, 4s, shape=(400,))                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Detalles de Cada Paso

### PASO 1: Lectura con ObsPy

**Objetivo:** Cargar archivo mseed y extraer traza única

**Acciones:**
```python
from obspy import read

# Cargar archivo mseed
stream = read('archivo.mseed')
trace = stream[0]  # Extraer traza única

# Verificar propiedades
sampling_rate = trace.stats.sampling_rate  # Debe ser 64 Hz
npts = trace.stats.npts  # Debe ser ~384-385

# Inferir clase del nombre de archivo
if filepath.endswith('_P.mseed'):
    class_name = "P"
elif filepath.endswith('_S.mseed'):
    class_name = "S"
elif filepath.startswith('noise_'):
    class_name = "Noise"
```

**Validaciones:**
- ✓ `sampling_rate == 64.0` Hz
- ✓ `380 <= npts <= 390`

---

### PASO 2: Detrend / Demean

**Objetivo:** Quitar offset DC y tendencia lineal lenta

**Método:**
```python
trace.detrend('demean')  # Remover offset DC
trace.detrend('linear')  # Remover tendencia lineal
```

**Justificación:**
- Evita que el filtro y la normalización se contaminen con pendientes largas

---

### PASO 3: Filtrado Pasa-Banda 1-30 Hz

**Objetivo:** Limpiar ruido fuera del rango de interés

**Método:**
```python
trace.filter('bandpass', 
             freqmin=1.0, 
             freqmax=30.0, 
             corners=4, 
             zerophase=True)
```

**Justificación técnica:**
- Muestreo original: 64 Hz → Nyquist = 32 Hz
- Por eso el corte alto debe ser menor que 32 Hz
- Se usa 30 Hz como límite superior seguro

**Efecto:**
- Elimina ruido de muy baja frecuencia (<1 Hz)
- Elimina ruido de muy alta frecuencia (>30 Hz)

---

### PASO 4: Remuestreo (64 Hz → 100 Hz)

**Objetivo:** Cambiar tasa de muestreo a 100 Hz para compatibilidad con GPD

**Método recomendado:**
```python
trace.interpolate(100.0, method='linear')
# Alternativa: method='lanczos'
```

**Resultado esperado:**
- ~600 muestras (≈6 s a 100 Hz)
- `trace.stats.sampling_rate == 100.0`

---

### PASO 5: Normalización Max-Abs

**Objetivo:** Escalar amplitudes al rango [-1, 1]

**Método:**
```python
x = trace.data.astype(np.float32)
maxabs = np.max(np.abs(x))
x = x / max(maxabs, 1e-10)
```

**Justificación:**
- Esta normalización (peak normalization) conserva la forma impulsiva de la llegada
- No infla ruido débil
- Es consistente con el flujo típico de inferencia de GPD

**Importante:**
- Se normaliza ANTES del recorte para que el escalado use toda la energía disponible

---

### PASO 6: Recorte Temporal (6s → 4s, centrado)

**Objetivo:** Reducir de ~600 muestras a 400 muestras (4 segundos)

**Estrategia de recorte:**
```python
total = len(x)  # ~600
keep = 400
start = (total - keep) // 2  # ≈100
end = start + keep            # ≈500
x = x[start:end]
```

**Efecto:**
- Elimina ~1 segundo al inicio
- Elimina ~1 segundo al final
- Produce longitud fija de 400 muestras

**Nota:**
- Recorte puramente geométrico (ventana central)
- No se garantiza que el pick esté exactamente centrado (a menos que se incorpore metadata de pick)

---

### PASO 7: Validaciones por Muestra

**Verificaciones:**
```python
assert len(data_crop) == 400
assert not np.isnan(data_crop).any()
assert not np.isinf(data_crop).any()
assert np.max(np.abs(data_crop)) <= 1.0 + 1e-6  # Margen numérico
```

**Interpretación:**
- Tras normalización max-abs, el rango típico es [-1, 1]
- Valores mayores indican error de flujo

---

### PASO 8: Etiquetado

**Mapeo de clases:**
- P → 0
- S → 1
- Noise → 2

**Implementación:**
```python
class_mapping = {'P': 0, 'S': 1, 'Noise': 2}
label = class_mapping[class_name]
```

**Metadata opcional:**
- `filename`: Nombre del archivo original
- `class_name`: Clase como string ("P" | "S" | "Noise")

---

### PASO 9: Salida por Muestra

**Estructura final:**
```python
# Vector de datos
data_out = x  # Shape (400,), dtype float32

# Etiqueta
label_out = label  # Tipo int8

# Metadata (dict)
metadata = {
    'filename': filepath,
    'class_name': class_name
}
```

---

## 4. Particionamiento del Dataset

**Estrategia sugerida:**
- **Train:** 70% (~15,533 muestras)
- **Val:** 15% (~3,329 muestras)
- **Test:** 15% (~3,329 muestras)

**Justificación:**
- Mantener las 3 clases balanceadas en cada split
- Se recomienda estratificar por clase para no desbalancear

---

## 5. Resumen del Pipeline en Pseudocódigo

```python
trace = read(filepath)[0]              # 1. leer

trace.detrend('demean')               # 2a quitar DC
trace.detrend('linear')               # 2b quitar tendencia

trace.filter('bandpass',              # 3  filtrar 1-30 Hz
             freqmin=1.0,
             freqmax=30.0,
             corners=4,
             zerophase=True)

trace.interpolate(100.0,              # 4  remuestrear a 100 Hz
                  method='linear')

x = trace.data.astype(np.float32)
maxabs = np.max(np.abs(x))            # 5  normalizar peak
x = x / max(maxabs, 1e-10)

total = len(x)                        # 6  recorte central a 4 s
keep = 400
start = (total - keep)//2
end = start + keep
x = x[start:end]

assert len(x) == 400                  # 7  validar salida
label = class_from_filename(filepath) # 8  mapear a {P:0,S:1,Noise:2}
```

---

## 6. Notas Importantes

### Diferencias Clave con Pipeline de 3 Componentes

- **Filtro:** 30 Hz (NO 45 Hz) porque el origen es 64 Hz → Nyquist=32 Hz
- **Componentes:** 1 sola componente (Z). No se apilan N/E
- **Normalización:** Max-abs (NO z-score). Se hace antes del recorte
- **Recorte:** Puramente geométrico (ventana central)
- **Shape final:** (400,) NO (400, 3)

### Consideraciones

- El recorte es ventana central fija
- No se garantiza que el pick esté exactamente centrado (a menos que se incorpore metadata de pick)
- El resultado final debe ser siempre (400,) @100 Hz, float32, con etiqueta asociada

---

## 7. Resumen de Transformaciones

```
INPUT:  archivo.mseed
        - 64 Hz
        - ~6 segundos (384-385 samples)
        - 1 componente (Z)
        - Amplitudes arbitrarias

        ↓ [Detrend: demean + linear]
        
        - Media ≈ 0
        - Sin drift lineal

        ↓ [Filtrado 1-30 Hz]
        
        - Solo frecuencias útiles sísmicas
        - Butterworth 4° orden, zero-phase

        ↓ [Remuestreo 64→100 Hz]
        
        - 100 Hz
        - ~6 segundos (600 samples)

        ↓ [Normalización max-abs]
        
        - Rango [-1, 1]
        - Conserva forma impulsiva

        ↓ [Recorte central 6s→4s]
        
        - 100 Hz
        - 4 segundos (400 samples)
        - Ventana central

OUTPUT: array
        - Shape: (400,)
        - 100 Hz, 4 segundos
        - Normalizado, filtrado, centrado
        - dtype: float32
        - Label: int8 (0/1/2)
```

---

**FIN DEL DOCUMENTO**

Para Claude Code: Este documento describe el pipeline específico para datos de 1 componente (Z). Las diferencias principales con el pipeline de 3 componentes son el filtro a 30 Hz (no 45 Hz), normalización max-abs (no z-score), y shape final (400,) en lugar de (400,3).