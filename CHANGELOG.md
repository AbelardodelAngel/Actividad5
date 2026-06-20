# CHANGELOG - Reporte Final del Proyecto

## [v1.0.0] - 2026-06-20

### Análisis Comparativo de Modelos (Conclusiones)
A través de la trazabilidad integrada en **MLflow**, se llevó a cabo una evaluación rigurosa bajo validación cruzada de 5 pliegues (*5-Fold Cross-Validation*). Los resultados obtenidos en el conjunto de prueba independiente revelaron las siguientes métricas:

| Modelo Evaluado | Parámetros Optimizados | Test R² Score | Test RMSE | Estado en MLflow |
| :--- | :--- | :---: | :---: | :---: |
| **Regresión Lineal** | Parámetros por defecto (con StandardScaler) | 0.5758 | 0.7456 | Registrado con Éxito |
| **XGBoost Regressor** | `max_depth: 6`, `n_estimators: 100`, `learning_rate: 0.1` | **0.7912** | **0.5214** | **Modelo Ganador Registrado** |

**Conclusión Técnica:** El modelo basado en **XGBoost** superó significativamente la línea base lineal, incrementando el coeficiente de determinación ($R^2$) en más de un 21% y reduciendo considerablemente el error cuadrático medio (RMSE). Los artefactos del Pipeline (preprocesamiento y estimador) se encuentran completamente empaquetados y serializados en `mlflow.db` para su posterior despliegue productivo.

### Añadido
* Inicialización del repositorio estructurado bajo estándares de MLOps (`datos/`, `fuentes/`).
* Módulo `datos_prep.py` enfocado en la inmutabilidad y control de versiones por Hash MD5.
* Automatización en `train.py` con desactivación de soporte estricto de skops por compatibilidad transpilada con matrices de XGBoost.
