| Modelo | accuracy | precision_macro | recall_macro | f1_macro | f1_weighted | roc_auc_ovr_macro |
|---|---|---|---|---|---|---|
| logistic_regression | 0.7982 | 0.7444 | 0.7504 | 0.7457 | 0.8013 | 0.9110 |
| random_forest | 0.7278 | 0.6606 | 0.6742 | 0.6658 | 0.7330 | 0.8640 |

## Auditoría de trazabilidad y fuente

La fuente identificada es [Twitter US Airline Sentiment en Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data), consultada el 17 de octubre de 2026. La cuenta del URL es Crowdflower; la página indica como fuente original Crowdflower's Data for Everyone library y acredita a Figure Eight y un colaborador. Muestra licencia CC BY-NC-SA 4.0. El archivo local coincide por nombre, tamaño aproximado de 3.42 MB y esquema; no hay hash del editor ni registro de descarga que autentique su procedencia byte a byte.

Las métricas JSON/CSV, los cinco folds seleccionados, las matrices y los reportes por clase concuerdan. Ambas matrices suman 2,928 filas de prueba y sus conteos por clase coinciden con los soportes de los reportes. La reproducción aislada de ambos GridSearchCV reprodujo parámetros, folds y métricas de prueba. No se guardaron objetos de modelo serializados.

En el split aleatorio original, 71 textos idénticos aparecieron en train y test: 60 grupos tenían una sola etiqueta y 11 grupos tenían etiquetas distintas entre los conjuntos. Hay 25 grupos conflictivos en todo el dataset. Este cruce puede elevar el resultado del split original.

### Evaluación adicional agrupada por texto idéntico

Escenario distinto: primer fold de `StratifiedGroupKFold` como holdout, y validación interna de cinco folds con agrupamiento por texto exacto. Train y test contienen 11,712 y 2,928 filas, respectivamente; el solapamiento de textos fue cero. Estas métricas complementan y no sustituyen las originales.

| Modelo | F1 macro CV (media ± DE) | F1 macro prueba | Accuracy | Precision macro | Recall macro | F1 ponderado | ROC-AUC OVR | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Regresión Logística | 0.7434 ± 0.0079 | 0.7437 | 0.7927 | 0.7350 | 0.7544 | 0.7962 | 0.9102 | 27.620 s |
| Random Forest | 0.6820 ± 0.0169 | 0.6787 | 0.7452 | 0.6738 | 0.6842 | 0.7474 | 0.8696 | 22.278 s |

Los parámetros, folds individuales, matrices y reportes adicionales están en `reportes/auditoria_fase3/evaluacion_agrupada/`. La reproducción del split original y las comprobaciones están en `reportes/auditoria_fase3/`.
