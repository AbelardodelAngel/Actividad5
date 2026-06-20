import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ejecutar_entrenamiento(dataset_hash, ruta_datos_limpios):
    # 1. Configurar el Entorno de MLflow Local
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Actividad5_Regresion_California")
    
    # 2. Cargar los datos preparados
    print(f"[TRAIN] Leyendo datos limpios desde: {ruta_datos_limpios}")
    df = pd.read_csv(ruta_datos_limpios)
    
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    
    # 3. División del Dataset de forma reproducible
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4. Configuración de Algoritmos
    configuraciones = [
        {
            "nombre": "Regresion_Lineal",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression())
            ]),
            "params": {}
        },
        {
            "nombre": "XGBoost_Regressor",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", xgb.XGBRegressor(random_state=42, objective="reg:squarederror"))
            ]),
            "params": {
                "regressor__n_estimators": [50, 100],
                "regressor__max_depth": [3, 6],
                "regressor__learning_rate": [0.05, 0.1]
            }
        }
    ]
    
    # 5. Ciclo de Vida del Modelo: Ajuste y Registro
    for conf in configuraciones:
        with mlflow.start_run(run_name=conf["nombre"]):
            print(f"\n[MLFLOW] Ejecutando experimento para: {conf['nombre']}")
            
            grid_search = GridSearchCV(
                estimator=conf["pipeline"],
                param_grid=conf["params"],
                cv=cv_strategy,
                scoring="r2",
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            
            # --- REGISTRO DE TRAZABILIDAD EN MLFLOW ---
            mlflow.set_tag("dataset_version_hash", dataset_hash)
            mlflow.set_tag("framework", "scikit-learn / xgboost")
            
            mejores_params = {k.replace("regressor__", ""): v for k, v in grid_search.best_params_.items()}
            if mejores_params:
                mlflow.log_params(mejores_params)
            else:
                mlflow.log_param("modelo", "sin_hiperparametros_base")
            
            mlflow.log_metric("cv_best_mean_r2", grid_search.best_score_)
            
            # 6. Evaluación de Generalización en el Conjunto de Prueba
            y_pred = best_pipeline.predict(X_test)
            
            mse_test = mean_squared_error(y_test, y_pred)
            metrics = {
                "test_rmse": np.sqrt(mse_test),
                "test_mae": mean_absolute_error(y_test, y_pred),
                "test_r2_score": r2_score(y_test, y_pred)
            }
            
            mlflow.log_metrics(metrics)
            
            # SOLUCIÓN AL ERROR: Le indicamos los tipos externos permitidos y seguros para skops
            trusted_types = ["xgboost.sklearn.XGBRegressor", "xgboost.core.Booster"]
            
            # Registrar el artefacto indicando explícitamente que confíe en XGBoost
            mlflow.sklearn.log_model(
                sk_model=best_pipeline, 
                artifact_path="pipeline_modelo",
                skops_trusted_types=trusted_types
            )
            
            print(f"[MLFLOW] Registro completado con éxito para {conf['nombre']}")
            print(f"Test R2 Score: {metrics['test_r2_score']:.4f} | Test RMSE: {metrics['test_rmse']:.4f}")

if __name__ == "__main__":
    import sys
    sys.path.append('fuentes')
    from datos_prep import descargar_y_preparar_datos
    d_hash, r_limpia = descargar_y_preparar_datos()
    ejecutar_entrenamiento(d_hash, r_limpia)
