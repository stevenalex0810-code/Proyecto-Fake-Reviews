# Documentación de Pruebas Empíricas y Entrenamiento

Este documento registra formalmente la evolución y el rendimiento de los modelos de Machine Learning (Naive Bayes y SVM) desarrollados para la detección de reseñas fraudulentas. Se documentan dos experimentos iterativos para evaluar el impacto del volumen de datos en la precisión del algoritmo.

---

## Experimento 1: Entrenamiento de Prueba (Muestra Aleatoria)
**Objetivo:** Validar el pipeline de Procesamiento de Lenguaje Natural (SpaCy + TF-IDF) y verificar que los modelos compilen correctamente antes de invertir horas de poder computacional.
**Volumen de Datos:** 5.000 reseñas (2.5% del dataset total).
**División de Datos:** 80% Entrenamiento / 20% Prueba.

### Resultados (Experimento 1)
**Modelo 1: Naive Bayes (Multinomial)**
*   Validación Cruzada (10-fold): 44.65%
*   Exactitud (Accuracy): 45.00%
*   Precisión (Macro): 44.33%
*   F1-Score (Macro): 43.58%

**Modelo 2: Support Vector Machine (LinearSVC)**
*   Validación Cruzada (10-fold): 43.83%
*   Exactitud (Accuracy): 45.30%
*   Precisión (Macro): 44.26%
*   F1-Score (Macro): 44.53%

**Conclusión Inicial:**
Con apenas 5.000 datos, el modelo SVM demuestra una ligera superioridad en la exactitud. Sin embargo, un rendimiento cercano al 45% indica un severo "Underfitting" (Subajuste). El modelo no tiene suficiente vocabulario (lemas) en 5.000 textos para aprender los patrones complejos del idioma español que diferencian el fraude de la realidad. Se hace imperativo entrenar el modelo con el corpus completo.

---

## Experimento 2: Entrenamiento Definitivo (Corpus Completo)
**Objetivo:** Maximizar el aprendizaje del algoritmo enfrentándolo a la varianza total del lenguaje disponible en el conjunto de datos completo, buscando superar el umbral de exactitud funcional.
**Volumen de Datos:** 200.000 reseñas (100% del dataset total).
**División de Datos:** 80% Entrenamiento / 20% Prueba.
**Tiempo de Procesamiento NLP (Lematización):** `[PENDIENTE: Anotar horas de procesamiento]`

### Resultados (Experimento 2)
*(Entrenamiento realizado con las 8.000 reseñas del Corpus Sintético Balanceado de la Tesis)*

**Modelo 1: Naive Bayes (Multinomial)**
*   Validación Cruzada (10-fold): 100.00%
*   Exactitud (Accuracy): 100.00%
*   Precisión (Macro): 100.00%
*   F1-Score (Macro): 100.00%

**Modelo 2: Support Vector Machine (LinearSVC)**
*   Validación Cruzada (10-fold): 100.00%
*   Exactitud (Accuracy): 100.00%
*   Precisión (Macro): 100.00%
*   F1-Score (Macro): 100.00%

---

## Experimento 3: Entrenamiento Híbrido Definitivo
**Objetivo:** Evaluar la capacidad de generalización del modelo combinando fraude sintético (I.A) con el dataset estándar de la industria (Deceptive Opinion Spam Corpus de Ott, 2011, traducido al español).
**Volumen de Datos:** 9.600 reseñas (Corpus Híbrido).
**División de Datos:** 80% Entrenamiento / 20% Prueba.

### Resultados (Experimento 3)
**Modelo 1: Naive Bayes (Multinomial)**
*   Validación Cruzada (10-fold): 97.90%
*   Exactitud (Accuracy): 98.07%
*   Precisión (Macro): 98.09%
*   F1-Score (Macro): 98.07%

**Modelo 2: Support Vector Machine (LinearSVC)**
*   Validación Cruzada (10-fold): 98.01%
*   Exactitud (Accuracy): 98.33%
*   Precisión (Macro): 98.33%
*   F1-Score (Macro): 98.33%

**Conclusión Final (Resumen del Proceso):**
La evolución iterativa del corpus demostró ser la clave del éxito. En el Experimento 1 (muestra pequeña), el modelo sufrió de subajuste (45% de exactitud). En el Experimento 2 (dataset sintético), el modelo se sobreajustó a patrones muy específicos (100% exactitud). Finalmente, en el Experimento 3, al introducir variabilidad humana mediante la traducción del dataset de Ott (2011), el modelo alcanzó una excelente capacidad de generalización, estabilizándose en un **98.33% de exactitud con SVM**. 

Por lo tanto, se exportan oficialmente los archivos `modelo_svm.pkl` y `vectorizador.pkl` de este tercer experimento para ser desplegados en la interfaz gráfica del prototipo final.
