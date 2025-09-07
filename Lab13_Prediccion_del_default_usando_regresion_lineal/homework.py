#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import pickle
import gzip
import os
import json

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import median_absolute_error, r2_score, mean_squared_error, mean_absolute_error


# ================================ #
# Paso 1: Cargar y limpiar datos
# ================================ #
def preparar_datos(df):
    df = df.copy()
    df['Age'] = 2021 - df['Year']
    df.drop(columns=['Year', 'Car_Name'], inplace=True)
    return df


# ================================ #
# Paso 2: Dividir los datos
# ================================ #
def dividir_datos(train_df, test_df):
    X_train = train_df.drop(columns="Present_Price")
    y_train = train_df["Present_Price"]
    X_test = test_df.drop(columns="Present_Price")
    y_test = test_df["Present_Price"]
    return X_train, y_train, X_test, y_test


# ============================== #
# Paso 3: Crear pipeline
# ============================== #
def crear_pipeline(x_train):
    categoricas = ['Fuel_Type', 'Selling_type', 'Transmission']
    numericas = [col for col in x_train.columns if col not in categoricas]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("num", MinMaxScaler(), numericas),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("regressor", LinearRegression()),
    ])
    return pipeline


# ========================================== #
# Paso 4: Optimización de hiperparámetros
# ========================================== #
def optimizar_hyperparametros(pipeline):
    param_grid = {
        'feature_selection__k': range(1, 25),
        'regressor__fit_intercept': [True, False],
        'regressor__positive': [True, False]
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    return grid_search


# ============================ #
# Paso 5: Calcular métricas
# ============================ #
def calcular_metricas(modelo, x, y, dataset_name):
    y_pred = modelo.predict(x)
    metricas = {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": float(r2_score(y, y_pred)),
        "mse": float(mean_squared_error(y, y_pred)),
        "mad": float(median_absolute_error(y, y_pred))
    }
    return metricas


# ================================ #
# Paso 6: Guardar el modelo
# ================================ #
def guardar_comprimido(modelo, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump(modelo, f)


# ================================ #
# Paso 7: Guardar métricas
# ================================ #
def guardar_jsonl(registros, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in registros:
            f.write(json.dumps(item) + '\n')


# ============================ #
# EJECUCIÓN PRINCIPAL DEL LAB
# ============================ #
if __name__ == "__main__":
    
    print("Cargando datos...")
    train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    print(train_df.columns)

    print("Datos cargados. Limpiando...")
    train_df = preparar_datos(train_df)
    test_df = preparar_datos(test_df)

    X_train, y_train, X_test, y_test = dividir_datos(train_df, test_df)

    print("Datos preparados. Creando pipeline...")
    pipeline = crear_pipeline(X_train)

    print("Ajustando modelo ...")
    modelo_ajustado = optimizar_hyperparametros(pipeline)
    modelo_ajustado.fit(X_train, y_train)

    print("Modelo entrenado. Guardando modelo...")
    guardar_comprimido(modelo_ajustado, 'files/models/model.pkl.gz')

    print("Calculando métricas...")
    metricas_train = calcular_metricas(modelo_ajustado, X_train, y_train, "train")
    metricas_test = calcular_metricas(modelo_ajustado, X_test, y_test, "test")

    guardar_jsonl(
        [metricas_train, metricas_test],
        'files/output/metrics.json'
    )

    print("Todo listo")