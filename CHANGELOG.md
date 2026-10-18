# Cambios

## 2026-10-17 — preparación para cierre académico y publicación

- Se actualizó el README para reflejar las fases completadas, documentar la descarga manual del dataset, fijar Python 3.12 como mínimo compatible y describir resultados, límites y comandos del entorno virtual.
- Se añadió `fuentes/entrena.ipynb` como notebook complementario de análisis, reproducción temporal y consulta de MLflow; el notebook anterior se conserva.
- Se amplió `.gitignore` para excluir el CSV raíz y la caché local de PowerShell, y permitir capturas seleccionadas en `evidencias/mlflow/`.
- El informe Word/PDF y las capturas de MLflow no se añadieron porque no están disponibles en el proyecto; tampoco se inicializó Git.

## 2026-10-17 — integración MLflow (Fase 4)

- Se instaló MLflow 3.17.1 únicamente en `.venv`, manteniendo las versiones de pandas, NumPy, scikit-learn y matplotlib.
- Se añadió `fuentes/mlflow_tracking.py` para volver a ejecutar y registrar por separado ambos modelos bajo los protocolos original y agrupado.
- Se añadió una prueba de integración para verificar parámetros, métricas, etiquetas y artefactos en un backend SQLite temporal.
- Se documentaron el comando de registro, la interfaz local, el orden de ejecución y la captura de evidencias.

## 2026-10-17 — auditoría de Fase 3

- Se identificó y documentó la fuente de Kaggle, su atribución y licencia; la fecha local de descarga y la identidad byte a byte permanecen sin verificar.
- Se verificó la consistencia de métricas, folds, matrices y reportes, y se reprodujeron los resultados originales en una carpeta aparte.
- Se midió el solapamiento de textos duplicados y se ejecutó una evaluación adicional agrupada por texto, sin reemplazar resultados previos.
- Se añadieron comprobaciones unitarias para grupos con etiquetas conflictivas.

## 2026-10-17

- Se creó la estructura inicial de carpetas del proyecto.
- Se añadió la preparación reproducible del dataset, el informe de calidad y el manifiesto SHA-256.
- Se generaron gráficas SVG de clases, faltantes, longitud de textos y duplicados sin dependencias adicionales.
- Se documentó la conservación de duplicados para evaluación posterior.
- Se añadieron pruebas unitarias de las reglas de preparación.
- Se creó el entorno virtual y se fijaron dependencias de entrenamiento compatibles con Python 3.13.3.
- Se añadieron pipeline TF-IDF, búsquedas acotadas para regresión logística y Random Forest, evaluación, resultados y gráficas de comparación.
- Se añadieron pruebas pequeñas para pipeline, búsqueda y métricas; MLflow permanece pendiente de la Fase 4.
- Se ejecutaron ambos GridSearchCV: F1 macro CV/prueba de 0.7438/0.7457 para regresión logística y 0.6879/0.6658 para Random Forest; tiempos finales de 22.284 s y 18.915 s.
- Se documentó el solapamiento de 71 textos distintos entre entrenamiento y prueba como posible fuente de optimismo por duplicados.
