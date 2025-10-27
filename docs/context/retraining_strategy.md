# Contexto: Reentrenamiento de GPD con Datos Limitados

## Información del Proyecto

### Modelo Base
- **Nombre:** GPD (Generalized Phase Detection)
- **Estado:** Modelo preentrenado
- **Arquitectura:** Red convolucional para detección de fases sísmicas

### Entorno Técnico
- **Python:** 3.9
- **Framework:** TensorFlow-Keras 2.12
- **Datos disponibles:** ~1000 muestras locales
- **Objetivo:** Reentrenar con pocos datos sin romper compatibilidad

### Formato de Entrada
- **Shape esperado:** (400, 3) 
  - 400 pasos temporales
  - 3 componentes (típicamente Z, N, E)
- **Tasa de muestreo objetivo:** 100 Hz
- **Ventana temporal:** 4 segundos

---

## 1. Alineación de Datos al Contrato del Modelo

### 1.1 Forma y Tasa de Muestreo
- **Input del modelo:** Ventanas de (400, 3)
- **Si datos originales están a 250 Hz o 64 Hz:** Resamplear a 100 Hz
- **Razón:** Mantener compatibilidad sin modificar arquitectura ni pesos frontales

### 1.2 Estrategia de Ventaneo
- **Técnica:** Ventanas deslizantes
- **Posición crítica:** Centrar la información relevante (P, S o ruido) en el tramo central
- **Lógica GPD:** Clasifica el centro de la ventana (no los extremos)

### 1.3 Sistema de Etiquetado
**Tres clases exclusivas:**
1. **P (onda primaria)**
2. **S (onda secundaria)**  
3. **Ruido**

**Criterios de etiquetado:**
- **P/S positivo:** Si el pick cae dentro del margen central (±0.2–0.3 s según sampling)
- **Ruido:** Ventanas de eventos que NO contienen el centro de P ni de S
- **Ventana central crítica:** Los picks deben estar en el centro para clasificación correcta

---

## 2. Curación y Partición del Dataset

### 2.1 Balanceo por Clase
**Problema:** Con pocos datos, el ruido suele dominar

**Soluciones:**
- **Muestreo estratificado:** Mismo número de P, S, Ruido por época
- **Class weights:** Peso mayor a clases P y S en la función de pérdida

### 2.2 Partición Sin Fuga (CRÍTICO)
- **Principio:** Separar por estación o por evento
- **Prohibido:** Mezclar ventanas del mismo sismo en train/val/test
- **Propuesta:** 70/15/15 (train/val/test)
- **Alternativa:** K-fold (5-fold) si los datos son muy escasos

### 2.3 Limpieza de Etiquetas
- **Importancia:** 1000 muestras NO toleran ruido de anotación
- **Acción:** Revisión manual de picks dudosos (outliers)
- **Prioridad:** Calidad sobre cantidad

---

## 3. Estrategia de Transfer Learning (NÚCLEO)

### Filosofía General
- Ajustar lo mínimo posible
- Preservar el "saber general" del modelo preentrenado
- Avanzar gradualmente desde capas superficiales a profundas

### FASE A: Solo Cabeza Densa (Primera Iteración)

**Configuración:**
- **Congelar:** TODAS las capas convolucionales y BatchNorm
- **Entrenar:** Solo capas densas finales

**Hiperparámetros:**
- Learning Rate: 1e-3 → 3e-4
- Early Stopping: Paciencia 5–8 épocas
- Métrica: Validation F1 (macro o por clase)
- Checkpointing: Guardar mejor modelo

**Criterio de éxito:** Val F1 deja de mejorar

### FASE B: Descongelar Último Bloque Conv (Refinamiento)

**Configuración:**
- **Descongelar:** Solo el bloque conv más profundo (Conv1D_4 + BN + ReLU)
- **Mantener congelado:** Resto de capas

**Hiperparámetros:**
- Learning Rate: 1e-4 → 1e-5 (MUY bajo)
- Épocas: 3–10 (pocas)
- Early Stopping: Activo

**Objetivo:** Adaptar features de alto nivel al dominio local sin sobreajustar

### FASE C: Descongelar Penúltimo Bloque (OPCIONAL)

**Condiciones para aplicar:**
- Val F1 sigue mejorando en Fase B
- NO hay señales de sobreajuste

**Configuración:**
- Learning Rate: ≤1e-5 (extremadamente bajo)

**Criterio de abandono:** Si empeora, volver al mejor checkpoint de Fase B

### Tratamiento Especial de BatchNorm

**Con pocos datos:**
- **Preferible:** Congelar BN (trainable=False)
- **Razón:** Evitar corrupción de estadísticos con batch pequeños

**Alternativa - Recalibración de BN:**
- Correr datos locales en modo entrenamiento
- NO actualizar pesos
- Solo refrescar medias/varianzas de BN

---

## 4. Regularización y Configuración

### 4.1 Componentes Obligatorios
- **Early Stopping:** Imprescindible
- **Model Checkpoint:** Guardar mejor modelo por val_F1
- **Logging:** TensorBoard para trazabilidad
- **Semillas fijas:** Reproducibilidad

### 4.2 Técnicas de Regularización
- **Weight Decay (L2):** Moderado en capas densas (p. ej., 1e-4)
- **Label Smoothing:** Leve (0.05) para manejar etiquetas ruidosas
- **Dropout:** Si ya existe en el modelo, mantenerlo activo

### 4.3 Configuración de Entrenamiento
- **Batch size:** 16–64 (según RAM disponible)
- **Optimizador:** Adam (recomendado para transfer learning)
- **Scheduler LR:** ReduceLROnPlateau opcional

---

## 5. Data Augmentation Específica Sísmica

### Filosofía
- **Intensidad:** Baja (no romper la distribución)
- **Objetivo:** Regularización efectiva pero conservadora

### 5.1 Técnicas Recomendadas

#### Escalado de Amplitud
- **Rango:** 0.7–1.3 (aleatorio)
- **Aplicación:** Por ventana

#### Ruido Aditivo
- **SNR objetivo:** 15–30 dB
- **Tipo:** Ruido gaussiano leve

#### Jitter Temporal
- **Desplazamiento:** ±5–15 muestras
- **CRÍTICO:** Mantener etiqueta centrada
- **Implementación:** Shift circular o padding

#### Filtros Leves
- **Tipo:** Pasa-banda dentro del rango útil
- **Condición:** Solo si replica preprocesamiento real
- **Precaución:** No aplicar filtros arbitrarios

### 5.2 Hard-Negative Mining

**Estrategia avanzada:**
1. Correr modelo sobre tramos largos de ruido
2. Identificar falsos positivos "convencidos" (alta confianza)
3. Añadirlos como negativos extra al dataset
4. Re-entrenar para robustez

**Beneficio:** Mejora discriminación de ruido

---

## 6. Función de Pérdida y Métricas

### 6.1 Función de Pérdida
- **Principal:** Categorical Cross-Entropy
- **Con class weights:** Si hay desbalance significativo
- **Fórmula weights:** `inverse_freq` o `sqrt(inverse_freq)`

### 6.2 Métricas de Evaluación (Soporte a Decisión)

#### Durante Entrenamiento
- **F1 por clase:** P, S, Ruido (separados)
- **Macro-F1:** Promedio no ponderado
- **PR-AUC por clase:** Más informativo con desbalance

#### Análisis Post-Entrenamiento
- **Matriz de confusión:** P vs S vs Ruido
- **Precision-Recall curves:** Por cada clase
- **Calibration plots:** Para ajuste de umbrales

### 6.3 Calibración de Umbrales

**Principio:** NO asumir umbral 0.5 universal

**Estrategia:**
1. Para P y S: Elegir umbral que maximice F1
2. Considerar objetivo operativo:
   - Mayor precision en P si se usa para alerta temprana
   - Mayor recall si se busca no perder eventos
3. Umbral independiente por clase

---

## 7. Validación Robusta (Anti-Overfitting)

### 7.1 Estrategias de Validación

#### Validación por Evento/Estación
- Evaluar en eventos nunca vistos
- Evaluar en estaciones nunca vistas
- **Objetivo:** Medir generalización real

#### Validación en Bloques Continuos
- **Qué:** Segmentos largos de ruido real (no solo ventanas cortas)
- **Por qué:** Detectar falsos positivos en operación continua
- **Duración:** Horas de registro continuo

### 7.2 Reporte de Varianza
- **K-fold CV:** Reportar media ± std de F1/PR-AUC
- **Bootstrapping:** Si K-fold no es viable
- **Distribuciones:** Boxplots de métricas por fold

### 7.3 Análisis de Errores
- Inspeccionar manualmente falsos positivos/negativos
- Identificar patrones sistemáticos
- Iterar en preprocesamiento o augmentation

---

## 8. Pipeline de Implementación Recomendado

### Paso 1: Preparación de Datos
```
1. Cargar señales originales (250 Hz)
2. Resamplear a 100 Hz
3. Normalizar (per-component z-score o global)
4. Generar ventanas centradas en picks
5. Generar ventanas de ruido (balanceadas)
6. Particionar sin fuga (por evento/estación)
```

### Paso 2: Setup del Modelo
```
1. Cargar pesos preentrenados de GPD
2. Congelar todas las capas excepto densas finales
3. Compilar con CCE + class_weights + LR inicial
4. Configurar callbacks (EarlyStopping, ModelCheckpoint, TensorBoard)
```

### Paso 3: Entrenamiento Fase A
```
1. Entrenar solo cabeza densa
2. Monitorear val_F1_macro
3. Guardar mejor checkpoint
4. Evaluar en test set
```

### Paso 4: Entrenamiento Fase B (si es necesario)
```
1. Cargar mejor checkpoint de Fase A
2. Descongelar último bloque conv
3. Reducir LR (1e-4)
4. Re-entrenar 3-10 épocas
5. Evaluar mejora
```

### Paso 5: Evaluación Final
```
1. Calibrar umbrales por clase
2. Matriz de confusión en test
3. F1/Precision/Recall por clase
4. Validación en datos continuos
5. Análisis de falsos positivos/negativos
```

---

## 9. Consideraciones Especiales para TF-Keras 2.12

### Compatibilidad
- Usar `tf.keras` (no `keras` standalone)
- BatchNorm: `model.trainable = False` congela BN correctamente
- Callbacks: `ModelCheckpoint(save_best_only=True, monitor='val_f1_macro')`

### Métricas Personalizadas
- Implementar F1 como métrica custom si no existe
- Usar `tf.keras.metrics.F1Score` (disponible en TF 2.12+)

### Congelamiento de Capas
```python
# Correcto en TF-Keras 2.12
for layer in model.layers[:-N]:  # Congela excepto últimas N
    layer.trainable = False
model.compile(...)  # IMPORTANTE: re-compilar después
```

---

## 10. Checklist de Implementación

### Pre-Entrenamiento
- [ ] Datos resampled a 100 Hz
- [ ] Ventanas de (400, 3) generadas
- [ ] Picks centrados en ventanas (±0.2-0.3s)
- [ ] Etiquetas P/S/Ruido balanceadas
- [ ] Partición sin fuga implementada (70/15/15)
- [ ] Dataset limpio (picks revisados manualmente)

### Configuración del Modelo
- [ ] Pesos GPD cargados correctamente
- [ ] Capas convolucionales congeladas (Fase A)
- [ ] Learning rate configurado (1e-3 inicial)
- [ ] Class weights calculados
- [ ] Early stopping configurado (paciencia 5-8)
- [ ] ModelCheckpoint guardando mejor val_F1

### Entrenamiento
- [ ] Logging activo (TensorBoard)
- [ ] Semillas fijas (reproducibilidad)
- [ ] Batch size apropiado (16-64)
- [ ] Validación por evento/estación
- [ ] Métricas monitoreadas: F1, PR-AUC por clase

### Post-Entrenamiento
- [ ] Umbrales calibrados por clase
- [ ] Matriz de confusión analizada
- [ ] Falsos positivos inspeccionados
- [ ] Validación en bloques continuos
- [ ] Modelo final guardado

---

## 11. Métricas de Éxito

### Mínimos Aceptables
- **F1 para P:** ≥0.85
- **F1 para S:** ≥0.80
- **F1 para Ruido:** ≥0.90
- **Macro-F1:** ≥0.85

### Indicadores de Sobreajuste
- Val loss >> Train loss
- Val F1 << Train F1 (gap >10%)
- Métricas empeoran en test respecto a validation

### Señales de Éxito
- Gap train/val <5% en F1
- Generalización a estaciones/eventos no vistos
- Pocos falsos positivos en ruido continuo

---

## 12. Troubleshooting Común

### Problema: Sobreajuste Inmediato
- **Solución:** Reducir LR, aumentar L2, más data augmentation
- **Alternativa:** Congelar más capas

### Problema: No Converge
- **Solución:** Aumentar LR, revisar normalización de datos
- **Alternativa:** Descongelar menos capas primero

### Problema: Desbalance de Clases No Resuelto
- **Solución:** Aumentar class weights, hard-negative mining
- **Alternativa:** Muestreo estratificado más agresivo

### Problema: Ruido Domina Predicciones
- **Solución:** Aumentar peso de P/S, calibrar umbral de ruido más alto
- **Alternativa:** Hard-negative mining focalizado

---

## Referencias y Recursos

### Papers Clave
- GPD original (Ross et al., 2018): "Generalized Seismic Phase Detection with Deep Learning"
- Transfer learning sísmico: Revisar literatura sobre domain adaptation

### Datasets Benchmark
- STEAD (para comparación de performance)
- Datos locales específicos de tu región

### Herramientas Útiles
- ObsPy: Preprocesamiento de datos sísmicos
- TensorBoard: Monitoreo de entrenamiento
- Scikit-learn: Métricas y validación

---

## Notas Finales

Este documento sirve como guía completa para implementar el reentrenamiento de GPD con datos limitados. La estrategia prioriza:

1. **Conservación del conocimiento preentrenado** (transfer learning gradual)
2. **Prevención de sobreajuste** (regularización robusta)
3. **Validación rigurosa** (sin fuga de datos, varianza reportada)
4. **Practicidad** (compatible con TF-Keras 2.12, Python 3.9)

Para Claude Code: Usa este contexto para generar código modular, bien documentado y siguiendo las fases descritas. Prioriza claridad y reproducibilidad.