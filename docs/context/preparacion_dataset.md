# Contexto: Preparacion del Dataset Preprocesado para Reentrenamiento de GPD

## Estado actual

El dataset ha sido procesado exitosamente siguiendo el pipeline de preprocesamiento aprobado para la componente Z. El proceso incluyo:

* Lectura de 22,191 archivos miniSEED a 64 Hz (~6 segundos cada uno).
* Filtrado pasa banda 1–30 Hz, detrend y remuestreo a 100 Hz.
* Normalizacion por valor maximo absoluto (max-abs) aplicada antes del recorte.
* Recorte central de 6 s → 4 s (400 muestras).
* Etiquetado automatico a partir del nombre del archivo.

**Resultados obtenidos:**

* Total procesado: 22,189 muestras (solo 2 fallidas).
* Shape final: X (22189, 400), y (22189,).
* Tipos: float32 e int8.
* Distribucion de clases:

  * P: 7720 (34.79%)
  * S: 6753 (30.43%)
  * Noise: 7716 (34.77%)
* Rango de amplitud: [-1, 1]
* Media global: ~0.000005
* Desviacion estandar global: ~0.26

**Validaciones visuales realizadas:**

* Muestras por clase: señales centradas, rango [-1,1], sin clipping.
* Histogramas de amplitud: distribuciones diferenciadas y simetricas.
* Verificacion de padding: sin tramos planos, recorte correcto.

➡️ **Conclusion:** el dataset esta limpio, fisicamente coherente y listo para iniciar el proceso de division y registro estadistico previo al reentrenamiento.

---

## Objetivo de los siguientes pasos

Se requiere preparar el dataset para el entrenamiento del modelo GPD. Esto implica tres tareas:

1. **Generar la particion train/val/test estratificada por clase (70/15/15).**
2. **Calcular y validar las metricas de balance por clase y por particion.**
3. **Registrar las estadisticas globales y por particion en un archivo JSON para trazabilidad.**

---

## Paso 1: Split estratificado (70/15/15)

**Objetivo:** asegurar que las tres clases (P, S, Noise) esten representadas proporcionalmente en los conjuntos de entrenamiento, validacion y prueba.

**Requisitos:**

* Usar division estratificada basada en los labels (`y`).
* Semilla fija (`random_state=42`) para reproducibilidad.
* Ratios: 70% entrenamiento, 15% validacion, 15% prueba.

**Pseudocodigo:**

```
El dataset esta en el directorio /home/rsa/projects/gpd/data/processed/datasets/processed-dataset/
Incluye los archivos: X.npy, y.npy y metadata.csv 

# Dividir 70% entrenamiento, 30% restante para validacion+prueba
(X_train, y_train), (X_temp, y_temp) = split(X, y, test_size=0.3, stratify=y)

# Del 30% restante, dividir mitad y mitad
(X_val, y_val), (X_test, y_test) = split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

Guardar cada subset en el directorio /home/rsa/projects/gpd/data/processed/datasets/retrain-set/
```

**Resultado esperado:**

* X_train, y_train: ~15532 muestras.
* X_val, y_val: ~3328 muestras.
* X_test, y_test: ~3329 muestras.
* Distribucion de clases similar a la global en los tres splits.

---

## Paso 2: Metricas de validacion del balance

**Objetivo:** confirmar que las proporciones de clases se mantienen en cada subconjunto.

**Validaciones requeridas:**

* Conteo de muestras por clase en cada subset.
* Porcentajes relativos por clase (deben coincidir con la distribucion global ±1%).
* Medida opcional de divergencia (KL o Jensen-Shannon) para cuantificar la diferencia entre distribuciones.

**Pseudocodigo:**

```
Para cada subset en [ALL, TRAIN, VAL, TEST]:
    Calcular conteos de clases {0:P, 1:S, 2:Noise}
    Calcular porcentajes (count / total)

Comparar distribuciones entre subsets y con el total.
Si las diferencias exceden ±2%, revisar el random_state o el metodo de estratificacion.
```

**Indicadores esperados:**

* Diferencias menores al 1% entre subconjuntos.
* Divergencia KL ~ 0.000–0.005.

---

## Paso 3: Registro de estadisticas globales (dataset_stats.json)

**Objetivo:** conservar trazabilidad numerica del dataset preprocesado para futuras comparaciones o auditorias.

**Metricas requeridas:**

* Para cada subset (ALL, TRAIN, VAL, TEST):

  * `mean`, `std`, `min`, `max` de las señales.
  * Conteos de muestras por clase (balance).
* Adicionalmente, opcional por clase dentro de cada subset.

**Formato de salida:** archivo JSON ubicado junto al dataset, por ejemplo:

```
retrain-set/
├── X_train.npy
├── y_train.npy
├── X_val.npy
├── y_val.npy
├── X_test.npy
├── y_test.npy
└── dataset_stats.json
```

**Estructura del archivo:**

```
{
  "created_at": "2025-10-29T21:30:00Z",
  "dataset_path": "/home/rsa/projects/gpd/data/processed/datasets/retrain-set",
  "sizes": {"train": 15532, "val": 3328, "test": 3329},
  "balance": {
      "train": {"P": 34.7, "S": 30.5, "Noise": 34.8},
      "val":   {"P": 34.8, "S": 30.3, "Noise": 34.9},
      "test":  {"P": 34.8, "S": 30.4, "Noise": 34.8}
  },
  "stats_global": {
      "train": {"mean": 0.00001, "std": 0.263, "min": -1.0, "max": 1.0},
      "val":   {"mean": 0.00001, "std": 0.261, "min": -1.0, "max": 1.0},
      "test":  {"mean": 0.00001, "std": 0.262, "min": -1.0, "max": 1.0}
  },
  "stats_per_class": {
      "train": {"0": {...}, "1": {...}, "2": {...}},
      "val": {...},
      "test": {...}
  }
}
```

**Pseudocodigo:**

```
Para cada subset:
    Calcular mean, std, min, max
    Guardar estadisticas globales
    Repetir por clase usando indices de y == clase

Escribir todo en JSON con timestamp actual y ruta del dataset.
```

**Uso futuro:**

* Permite validar que nuevas versiones del dataset mantengan consistencia estadistica.
* Facilita comparar rendimiento del modelo segun versiones de datos.

---

## Resultado esperado tras completar los tres pasos

* Dataset dividido estratificadamente (train/val/test) con proporciones 70/15/15.
* Balance de clases confirmado y documentado.
* Archivo `dataset_stats.json` con estadisticas globales y por clase.
* Dataset listo para ser cargado en TensorFlow/Keras para el reentrenamiento de GPD.
