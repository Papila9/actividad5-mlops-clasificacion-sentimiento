# Guía breve de seguimiento con MLflow

## Registrar ejecuciones

Desde la raíz del proyecto, ejecuta:

```powershell
.\.venv\Scripts\python.exe fuentes\mlflow_tracking.py
```

El script reproduce las evaluaciones con el dataset limpio y agrega cuatro runs al experimento `Actividad5_Clasificacion_Sentimiento`: regresión logística y Random Forest bajo `evaluacion_original`, y los dos modelos bajo `evaluacion_agrupada`. Cada escenario se mantiene separado. La evaluación agrupada usa partición y validación cruzada `StratifiedGroupKFold` por texto exacto y comprueba cero solapamientos.

## Abrir la interfaz

En otra terminal, desde la raíz del proyecto:

```powershell
.\.venv\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Visita <http://127.0.0.1:5000>. Selecciona el experimento y compara las cuatro ejecuciones. Los archivos `mlflow.db` y `mlartifacts/` son el almacén local del seguimiento.

## Evidencias sugeridas

1. Captura la página del experimento donde se vean los cuatro runs y sus etiquetas de evaluación.
2. Captura la tabla de runs con `f1_macro`, `accuracy`, `roc_auc_ovr_macro` y `cv_f1_macro_mean`/`cv_f1_macro_std`.
3. Abre un run de cada modelo y escenario; captura sus parámetros y los artefactos `comparacion_modelos.csv`, `confusion_matrix_*.csv` y `classification_report_*.json`.
4. Guarda las capturas manualmente en `evidencias/mlflow/` si se necesitan para la entrega. No incluyen los textos originales; los artefactos de resultados contienen métricas agregadas.

El tiempo se registra como `grid_search_fit_duration_seconds`: mide el `GridSearchCV.fit` completo, incluida la selección por CV y el refit final con todo el subconjunto de entrenamiento. No representa solo una llamada a `fit` del estimador final.
