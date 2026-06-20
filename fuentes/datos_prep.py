import pandas as pd
import hashlib
import os
from sklearn.datasets import fetch_california_housing

def descargar_y_preparar_datos():
    """
    Carga el dataset público nativo de California Housing, lo almacena en datos_ini,
    genera un Hash MD5 para control de versiones y guarda la versión limpia en datos_limp.
    """
    # Asegurar la existencia de las carpetas del proyecto
    os.makedirs("datos/datos_ini", exist_ok=True)
    os.makedirs("datos/datos_limp", exist_ok=True)
    
    # 1. Fase datos_ini: Carga directa desde scikit-learn y conversión a DataFrame
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    
    ruta_ini = "datos/datos_ini/california_raw.csv"
    df.to_csv(ruta_ini, index=False)
    
    # CONTROL DE VERSIONES: Calculamos el hash MD5 del contenido
    dataset_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    print(f"[DATA MANAGEMENT] Dataset público cargado desde sklearn.")
    print(f"[DATA MANAGEMENT] Hash MD5 de versión: {dataset_hash}")
    
    # 2. Fase datos_limp: Limpieza básica (eliminación de duplicados o valores nulos si los hubiera)
    df_clean = df.drop_duplicates()
    
    # Guardar en la carpeta de datos limpios
    ruta_limp = "datos/datos_limp/california_clean.csv"
    df_clean.to_csv(ruta_limp, index=False)
    print(f"[DATA MANAGEMENT] Dataset limpio guardado en {ruta_limp}")
    
    return dataset_hash, ruta_limp
