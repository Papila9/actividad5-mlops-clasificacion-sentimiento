# Actividad 5: clasificación de sentimiento y seguimiento de experimentos

Proyecto académico de clasificación supervisada de comentarios sobre aerolíneas con TF-IDF, Regresión Logística y Random Forest. Incluye preparación y versionamiento de datos, evaluación reproducible, auditoría de duplicados y seguimiento local con MLflow. No es un sistema listo para producción.

## Dataset y procedencia

Se utiliza **Twitter US Airline Sentiment**, disponible en [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data). La página atribuye el origen a la biblioteca Data for Everyone de Crowdflower, acredita a Figure Eight y un colaborador, y muestra la licencia **CC BY-NC-SA 4.0**. Consultada el 17 de octubre de 2026. La copia local tiene el nombre, tamaño y esquema esperados, pero no hay registro de descarga ni hash del editor; por eso no se afirma una autenticación byte a byte de su procedencia.

Para obtener el archivo, descarga `Tweets.csv` desde la página citada y colócalo en `datos/datos_ini/Tweets.csv`. El CSV no se incluye en Git: contiene texto de usuarios y la fuente aplica una licencia con condiciones de atribución, uso no comercial y compartir bajo la misma licencia.

El CSV original tiene 14,640 filas y 15 campos: `tweet_id` (identificador), `airline_sentiment` (etiqueta objetivo), `airline_sentiment_confidence` (confianza de la anotación), `negativereason` (motivo negativo), `negativereason_confidence` (confianza del motivo), `airline` (aerolínea), `airline_sentiment_gold` y `negativereason_gold` (etiquetas de referencia suplementarias), `name` (nombre de cuenta), `retweet_count` (conteo de retuits), `text` (comentario), `tweet_coord` (coordenadas), `tweet_created` (fecha del tuit), `tweet_location` (ubicación declarada) y `user_timezone` (zona horaria declarada). El modelo usa únicamente `text` como entrada y `airline_sentiment` como objetivo; los demás campos se descartan en la preparación.

## Requisitos y configuración

Las dependencias fijadas en `requirements.txt` se probaron con Python **3.13.3**. El mínimo compatible con el conjunto actual de dependencias es Python 3.12 (NumPy 2.5.4 requiere Python 3.12 o posterior).

Desde PowerShell, en la raíz del proyecto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Las instrucciones de instalación se verificaron en `.venv` con Python 3.13.3; `pip check` no reportó dependencias rotas. Para abrir el notebook se requiere un editor con soporte Jupyter y un kernel compatible con `.venv`. Este entorno no tiene Jupyter, `ipykernel`, `nbformat` ni `python-docx`; por ello, el notebook nuevo solo se validó como JSON y mediante compilación estática de sus celdas, no se ejecutó completo.

## Estructura

```text
datos/datos_ini/       CSV original descargado localmente (no versionado)
datos/datos_limp/      CSV limpio, informe de calidad y manifiesto SHA-256
fuentes/datos_prep.py  carga, validación, limpieza y versionamiento
fuentes/train.py       pipelines TF-IDF, GridSearchCV, métricas y gráficas
fuentes/auditoria_fase3.py auditoría de resultados y evaluación agrupada
fuentes/mlflow_tracking.py reproducción y registro de cuatro ejecuciones
fuentes/entrena.ipynb  notebook complementario de análisis y reproducción
tests/                 pruebas unitarias
reportes/              resultados tabulares y guías
evidencias/            gráficas generadas y capturas académicas seleccionadas
mlflow.db, mlartifacts/ almacenamiento local de MLflow (no versionado)
```

El notebook anterior `Comparacion_HuggingFace_Sentimientos_VF.ipynb` se conserva en la raíz como material histórico; el notebook complementario nuevo está en `fuentes/entrena.ipynb`.

## Preparación y ejecución

Después de descargar el CSV en `datos/datos_ini/Tweets.csv`, ejecuta las fases en orden:

```powershell
.\.venv\Scripts\python.exe fuentes\datos_prep.py
.\.venv\Scripts\python.exe fuentes\train.py
.\.venv\Scripts\python.exe fuentes\auditoria_fase3.py
.\.venv\Scripts\python.exe fuentes\mlflow_tracking.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La preparación conserva el CSV original, produce una tabla de dos columnas y escribe el hash y las reglas en `datos/datos_limp/manifiesto.json`. Los resultados documentados corresponden a las salidas validadas bajo `reportes/entrenamiento/` y `reportes/auditoria_fase3/`.

**Aviso de sobrescritura:** `train.py` usa `reportes/entrenamiento/` y `evidencias/` por defecto; una nueva ejecución reemplaza los archivos con esos nombres. Antes de repetirla, guarda una copia o especifica `--output-dir` y `--evidence-dir` nuevos. La auditoría escribe en sus rutas predeterminadas. `mlflow_tracking.py` agrega cuatro runs nuevos cada vez que se ejecuta; no borra runs anteriores.

### Notebook

Abre `fuentes/entrena.ipynb` desde el editor con kernel `.venv`. El notebook muestra la calidad, las métricas validadas y los runs de MLflow; su celda opcional de reproducción vuelve a entrenar en una carpeta temporal y compara con los resultados archivados. No se han generado resultados de notebook porque en el entorno actual no están instalados Jupyter ni `nbclient`; el JSON y el código de sus celdas se validaron estáticamente.

La carpeta `evidencias/mlflow/` contiene las cinco capturas usadas por el generador.

### Pruebas

Ejecuta las pruebas usando explícitamente el Python virtual:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La última ejecución de la suite completó **7 pruebas con resultado OK**. 

## Entrenamiento y resultados

El entrenamiento hace una partición estratificada 80/20 (semilla 42), selecciona hiperparámetros con GridSearchCV y cinco folds estratificados, y usa F1 macro para seleccionar. TF-IDF se ajusta dentro del pipeline y solo con datos de entrenamiento. La prueba se reserva para la evaluación final. Se informan accuracy, precisión macro, recall macro, F1 macro, F1 ponderado y ROC-AUC OVR.

| Escenario | Modelo | Accuracy | F1 macro prueba | ROC-AUC OVR | F1 macro CV (media ± DE) |
|---|---|---:|---:|---:|---:|
| Partición original estratificada | Regresión Logística | 0.7982 | 0.7457 | 0.9110 | 0.7438 ± 0.0099 |
| Partición original estratificada | Random Forest | 0.7278 | 0.6658 | 0.8640 | 0.6879 ± 0.0061 |
| Evaluación agrupada por texto | Regresión Logística | 0.7927 | 0.7437 | 0.9102 | 0.7434 ± 0.0079 |
| Evaluación agrupada por texto | Random Forest | 0.7452 | 0.6787 | 0.8696 | 0.6820 ± 0.0169 |

La evaluación agrupada es un escenario adicional: el holdout y la validación cruzada separan grupos de textos idénticos y no sustituyen ni promedian las métricas originales. En el split aleatorio se encontraron 71 textos compartidos entre entrenamiento y prueba, 11 con etiquetas conflictivas entre esos conjuntos. Hay 25 grupos de texto con etiquetas conflictivas en el dataset. La evaluación agrupada no tuvo solapamiento de textos entre sus subconjuntos. No se realizó una prueba de significancia estadística.

Las tablas, métricas detalladas, folds, matrices y reportes por clase están en las carpetas de `reportes/`; las gráficas de análisis y de evaluación están en `evidencias/`. El conjunto contiene desbalance entre clases y etiquetas inconsistentes para textos repetidos; las métricas representan estos protocolos y no garantizan rendimiento en datos futuros.

## MLflow local

El experimento `Actividad5_Clasificacion_Sentimiento` contiene cuatro runs verificados: cada uno de los dos modelos bajo `evaluacion_original` y `evaluacion_agrupada`. Las ejecuciones registran parámetros, versiones, métricas de prueba y CV, tiempos del GridSearchCV, tablas, matrices, gráficas y reportes agregados. No registran textos ni serializan modelos.

Inicia el servidor desde la raíz del proyecto:

```powershell
.\.venv\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abre <http://127.0.0.1:5000> y selecciona el experimento. El servidor local se probó y respondió HTTP 200. `mlflow.db`, `mlartifacts/` y `mlruns/` se excluyen de Git. Consulta `reportes/entrenamiento/guia_mlflow.md` para obtener capturas reales de la interfaz; cinco capturas reales de la interfaz est?n incluidas en `evidencias/mlflow/`.

## Versionamiento, evidencias y limitaciones

Las figuras de análisis y cinco capturas reales de MLflow están incluidas bajo `evidencias/`. El informe pdf está en `reportes/Actividad 5_Gestión de Proyectos de Inteligencia artificial.docx`.

La selección de Regresión Logística se basa descriptivamente en el mayor F1 macro de la partición original y resultados similares en la evaluación agrupada; no se ha demostrado significancia estadística. El proyecto es académico, no se evaluó en producción y la evaluación usa una sola semilla. No se ha publicado en GitHub ni se han creado commits.
