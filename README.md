# Actividad 5: Clasificación de sentimiento y seguimiento de experimentos

Proyecto académico de clasificación supervisada de comentarios sobre aerolíneas mediante TF-IDF, Regresión Logística y Random Forest. Incluye preparación y versionamiento de datos, evaluación reproducible, auditoría de duplicados y seguimiento local de experimentos con MLflow.

El proyecto tiene fines académicos y no constituye un sistema listo para producción.

## 1. Dataset y procedencia

Se utiliza el conjunto de datos **Twitter US Airline Sentiment**, disponible en [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data). La página atribuye su procedencia a la biblioteca Data for Everyone de Crowdflower, acredita a Figure Eight y a un colaborador, y muestra la licencia **CC BY-NC-SA 4.0**. La página se consultó el 17 de octubre de 2026.

La copia local presenta el nombre, tamaño y esquema esperados; sin embargo, no se dispone de un registro de descarga ni de un hash publicado por el editor que permita autenticarla byte a byte.

Para obtener los datos, descarga `Tweets.csv` desde la página indicada y guárdalo en:

`datos/datos_ini/Tweets.csv`

El archivo CSV original no se incluye en Git. Además de contener texto de usuarios, la fuente señala condiciones de atribución, uso no comercial y distribución bajo la misma licencia. Antes de redistribuir los datos, deben revisarse las condiciones aplicables.

El archivo original contiene 14,640 filas y 15 campos:

- `tweet_id`: identificador del tuit.
- `airline_sentiment`: etiqueta objetivo del análisis de sentimiento.
- `airline_sentiment_confidence`: confianza de la anotación.
- `negativereason`: motivo asociado al sentimiento negativo.
- `negativereason_confidence`: confianza de la anotación del motivo.
- `airline`: aerolínea mencionada.
- `airline_sentiment_gold` y `negativereason_gold`: etiquetas suplementarias de referencia.
- `name`: nombre de la cuenta.
- `retweet_count`: número de retuits.
- `text`: contenido textual del comentario.
- `tweet_coord`: coordenadas asociadas al tuit.
- `tweet_created`: fecha de creación.
- `tweet_location`: ubicación declarada.
- `user_timezone`: zona horaria declarada.

Para el modelado se utiliza únicamente `text` como variable de entrada y `airline_sentiment` como variable objetivo. Los demás campos se descartan durante la preparación para evitar incorporar información que no forma parte de la configuración definida para este experimento.

## 2. Requisitos y configuración

El proyecto se probó con Python 3.13.3 y las dependencias especificadas en `requirements.txt`. La instalación se verificó dentro del entorno virtual `.venv` y `pip check` no reportó dependencias incompatibles.

Desde PowerShell, en la raíz del proyecto, ejecuta los siguientes comandos por separado:

**Crear el entorno virtual**

```powershell
py -3.13 -m venv .venv
```

**Instalar las dependencias**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para trabajar con el notebook se requiere un editor compatible con Jupyter y un kernel asociado al entorno virtual `.venv`.

El notebook complementario `fuentes/entrena.ipynb` se validó mediante la revisión de su estructura JSON y la compilación estática de sus celdas. No se ejecutó de principio a fin en el entorno utilizado para esa validación, por lo que no se presentan resultados nuevos generados por el notebook.

## 3. Estructura del proyecto

```text
datos/
├── datos_ini/                 CSV original descargado localmente
└── datos_limp/                CSV limpio, informe de calidad y manifiesto SHA-256

fuentes/
├── datos_prep.py              Preparación, validación y limpieza
├── train.py                   Entrenamiento, ajuste y evaluación
├── auditoria_fase3.py         Auditoría de duplicados y evaluación agrupada
├── mlflow_tracking.py         Registro de experimentos en MLflow
├── genera_informe.py          Generación del informe técnico
└── entrena.ipynb              Notebook complementario

tests/                         Pruebas unitarias
reportes/                      Resultados tabulares, informes y guías
evidencias/                    Gráficas y capturas seleccionadas
mlflow.db                      Base local de MLflow, no versionada
mlartifacts/                   Artefactos locales de MLflow, no versionados
```

El notebook anterior `Comparacion_HuggingFace_Sentimientos_VF.ipynb` se conserva en la raíz como material histórico. El notebook complementario de esta actividad se encuentra en `fuentes/entrena.ipynb`.

El CSV original debe descargarse localmente y no se incluye en el repositorio. Los archivos de almacenamiento local de MLflow también se excluyen del control de versiones.

## 4. Preparación y ejecución

Después de descargar `Tweets.csv` y colocarlo en `datos/datos_ini/Tweets.csv`, ejecuta las fases en el siguiente orden, desde la raíz del proyecto.

### 4.1. Preparación de los datos

```powershell
.\.venv\Scripts\python.exe fuentes\datos_prep.py
```

El proceso carga y valida los datos, aplica las reglas de limpieza y genera el conjunto preparado, el informe de calidad y el manifiesto con el hash SHA-256 y las reglas aplicadas.

### 4.2. Entrenamiento y evaluación

```powershell
.\.venv\Scripts\python.exe fuentes\train.py
```

Este script entrena y evalúa los modelos definidos en el proyecto, realiza la búsqueda de hiperparámetros y genera los resultados y las gráficas correspondientes.

### 4.3. Auditoría y evaluación agrupada

```powershell
.\.venv\Scripts\python.exe fuentes\auditoria_fase3.py
```

Esta fase analiza los textos repetidos y permite comparar los resultados de la partición original con los de la evaluación agrupada por texto.

### 4.4. Seguimiento de experimentos

```powershell
.\.venv\Scripts\python.exe fuentes\mlflow_tracking.py
```

El script registra las ejecuciones de los modelos en MLflow y almacena los parámetros, las métricas y los artefactos configurados para cada experimento.

### 4.5. Pruebas unitarias

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La última ejecución documentada de la suite completó **7 pruebas con resultado OK**.

### Precaución sobre los archivos de salida

El script `train.py` utiliza `reportes/entrenamiento/` y `evidencias/` como rutas de salida predeterminadas. Una nueva ejecución puede reemplazar archivos con los mismos nombres. Antes de repetir el entrenamiento, conserva una copia de los resultados que quieras mantener.

La auditoría también escribe en sus rutas predeterminadas. Por su parte, `mlflow_tracking.py` agrega nuevas ejecuciones al experimento y no elimina automáticamente las anteriores.

## 5. Notebook complementario

El archivo `fuentes/entrena.ipynb` reúne información sobre la calidad de los datos, las métricas documentadas y las ejecuciones de MLflow. Incluye una celda opcional de reproducción que vuelve a entrenar los modelos en una carpeta temporal y compara los resultados con los archivos archivados.

El notebook no se ejecutó completamente en el entorno utilizado para su validación, porque no estaban instalados Jupyter, `ipykernel`, `nbformat` ni `nbclient`. La validación se limitó a la estructura JSON y la compilación estática de las celdas. Por lo tanto, su contenido debe considerarse complementario a los resultados generados y archivados mediante los scripts.

## 6. Entrenamiento y resultados

El entrenamiento utiliza una partición estratificada de 80 % para entrenamiento y 20 % para prueba, con semilla 42. Los hiperparámetros se seleccionan mediante `GridSearchCV` con cinco folds estratificados y F1 macro como métrica de selección.

La transformación TF-IDF se ajusta dentro del pipeline y únicamente con los datos de entrenamiento de cada partición. El conjunto de prueba se reserva para la evaluación final. Las métricas reportadas son accuracy, precisión macro, recall macro, F1 macro, F1 ponderado y ROC-AUC One-vs-Rest (OVR).

### 6.1. Comparación de modelos

| Escenario | Modelo | Accuracy | F1 macro de prueba | ROC-AUC OVR | F1 macro CV (media ± DE) |
|---|---|---:|---:|---:|---:|
| Partición original estratificada | Regresión Logística | 0.7982 | 0.7457 | 0.9110 | 0.7438 ± 0.0099 |
| Partición original estratificada | Random Forest | 0.7278 | 0.6658 | 0.8640 | 0.6879 ± 0.0061 |
| Evaluación agrupada por texto | Regresión Logística | 0.7927 | 0.7437 | 0.9102 | 0.7434 ± 0.0079 |
| Evaluación agrupada por texto | Random Forest | 0.7452 | 0.6787 | 0.8696 | 0.6820 ± 0.0169 |

### 6.2. Interpretación de los resultados

En los dos escenarios evaluados, Regresión Logística obtuvo un F1 macro superior al de Random Forest. En la partición original, su F1 macro de prueba fue de 0.7457, frente a 0.6658 para Random Forest. En la evaluación agrupada por texto, los valores fueron 0.7437 y 0.6787, respectivamente.

Estos resultados respaldan una preferencia descriptiva por Regresión Logística dentro de los experimentos realizados. No se efectuó una prueba de significancia estadística, por lo que no puede afirmarse que las diferencias sean estadísticamente significativas ni que los resultados se mantengan necesariamente en datos futuros.

### 6.3. Auditoría de duplicados

En la partición aleatoria original se identificaron 71 textos compartidos entre entrenamiento y prueba, de los cuales 11 presentaban etiquetas conflictivas entre ambos conjuntos. En el dataset completo se identificaron 25 grupos de textos idénticos con etiquetas conflictivas.

La evaluación agrupada constituye un escenario adicional: los textos idénticos se mantienen dentro de un mismo subconjunto para evitar que aparezcan simultáneamente en entrenamiento y prueba. En esta evaluación no hubo solapamiento de textos entre los subconjuntos.

Los resultados de la evaluación agrupada no sustituyen ni promedian los de la partición original; ambos escenarios se reportan por separado para analizar la sensibilidad de los resultados al tratamiento de los textos repetidos.

### 6.4. Limitaciones de los datos y la evaluación

El conjunto presenta desbalance entre las clases de sentimiento y etiquetas inconsistentes para determinados textos repetidos. Estas características pueden afectar las métricas y la interpretación del rendimiento.

Además, la evaluación se realizó con los datos y protocolos descritos en este documento y una semilla de partición definida. No se realizó una prueba de significancia estadística ni una validación en producción. En consecuencia, las métricas no garantizan el mismo desempeño ante datos nuevos.

Las métricas detalladas, matrices de confusión, reportes por clase, tablas y resultados de validación cruzada se encuentran en las carpetas correspondientes de `reportes/`. Las gráficas de análisis y evaluación están en `evidencias/`.

## 7. Seguimiento de experimentos con MLflow

El experimento `Actividad5_Clasificacion_Sentimiento` contiene cuatro ejecuciones documentadas, correspondientes a dos modelos —Regresión Logística y Random Forest— en dos escenarios: `evaluacion_original` y `evaluacion_agrupada`.

Las ejecuciones registran parámetros, versiones de dependencias, métricas de prueba y validación cruzada, tiempos de búsqueda de hiperparámetros y los artefactos configurados, como tablas, matrices, gráficas y reportes agregados. Según la implementación documentada, no se registran los textos originales ni se serializan los modelos.

Para iniciar el servidor local de MLflow, ejecuta desde la raíz del proyecto:

```powershell
.\.venv\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Mantén abierta la terminal y abre la siguiente dirección en el navegador:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

Selecciona el experimento `Actividad5_Clasificacion_Sentimiento` para consultar las ejecuciones y sus resultados.

El servidor local se probó y respondió con HTTP 200. Esta comprobación confirma la respuesta del servicio local, pero no implica que exista un despliegue del modelo en producción.

Los archivos `mlflow.db`, `mlartifacts/` y `mlruns/` se excluyen del control de versiones. La guía `reportes/entrenamiento/guia_mlflow.md` documenta el acceso a la interfaz y las capturas seleccionadas se encuentran en `evidencias/mlflow/`.

## 8. Versionamiento y evidencias

El proyecto organiza el código, las pruebas, los reportes y las evidencias para facilitar la revisión de los resultados. Las gráficas y capturas seleccionadas de MLflow se conservan bajo `evidencias/`.

El CSV original, el entorno virtual y los archivos de almacenamiento local de MLflow no se incluyen en Git. El manifiesto de datos conserva el hash SHA-256 del archivo preparado y documenta las reglas de procesamiento aplicadas.

El informe técnico debe consultarse en la ruta y con la extensión del archivo realmente generado dentro de `reportes/`. Antes de publicar una versión final, verifica que el nombre y el formato del documento coincidan con los archivos disponibles.

## 9. Uso de herramientas de inteligencia artificial

Durante el desarrollo de esta actividad se utilizaron herramientas de inteligencia artificial como apoyo a tareas de programación, revisión, organización y documentación. La declaración académica correspondiente debe identificar las herramientas efectivamente utilizadas y describir su contribución de acuerdo con las instrucciones del curso.

El uso de estas herramientas no sustituye la necesidad de comprender las decisiones metodológicas, revisar el código y verificar los resultados. La interpretación presentada en este README debe poder relacionarse con los experimentos y evidencias disponibles en el repositorio.

## 10. Conclusiones

En los escenarios evaluados, Regresión Logística obtuvo mejores resultados descriptivos que Random Forest en F1 macro. La evaluación agrupada por texto permitió examinar el efecto de los textos repetidos y evitar su solapamiento entre entrenamiento y prueba.

El seguimiento con MLflow facilita la consulta de los parámetros, métricas y artefactos registrados en las ejecuciones. La preparación de los datos, las pruebas y la documentación contribuyen a la reproducibilidad del análisis dentro del entorno descrito.

No obstante, el desbalance de clases, las etiquetas conflictivas, la ausencia de una prueba de significancia estadística y la falta de validación en producción limitan las conclusiones que pueden extraerse. Los resultados deben interpretarse como evidencia de un experimento académico y no como una garantía de desempeño en un entorno real.
