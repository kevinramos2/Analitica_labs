# pylint: disable=line-too-long
"""
Escriba el codigo que ejecute la accion solicitada.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def pregunta_01():
    """
    El archivo `files//shipping-data.csv` contiene información sobre los envios
    de productos de una empresa. Cree un dashboard estático en HTML que
    permita visualizar los siguientes campos:

    * `Warehouse_block`

    * `Mode_of_Shipment`

    * `Customer_rating`

    * `Weight_in_gms`

    El dashboard generado debe ser similar a este:

    https://github.com/jdvelasq/LAB_matplotlib_dashboard/blob/main/shipping-dashboard-example.png

    Para ello, siga las instrucciones dadas en el siguiente video:

    https://youtu.be/AgbWALiAGVo

    Tenga en cuenta los siguientes cambios respecto al video:

    * El archivo de datos se encuentra en la carpeta `data`.

    * Todos los archivos debe ser creados en la carpeta `docs`.

    * Su código debe crear la carpeta `docs` si no existe.

    """
    # Crear carpeta docs si no existe
    os.makedirs("docs", exist_ok=True)

    # Leer archivo CSV desde la carpeta 'files'
    df = pd.read_csv("files/input/shipping-data.csv")

    # Gráfico 1: Warehouse_block
    plt.figure()
    df["Warehouse_block"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Warehouse Block")
    plt.xlabel("Block")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig("docs/shipping_per_warehouse.png")
    plt.close()

    # Gráfico 2: Mode_of_Shipment
    plt.figure()
    df["Mode_of_Shipment"].value_counts().plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Mode of Shipment")
    plt.xlabel("Modo")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig("docs/mode_of_shipment.png")
    plt.close()

    # Gráfico 3: Promedio de Customer_rating por bloque o general
    plt.figure()
    df.groupby("Warehouse_block")["Customer_rating"].mean().plot(kind="bar", color="salmon", edgecolor="black")
    plt.title("Average Customer Rating by Warehouse")
    plt.xlabel("Block")
    plt.ylabel("Promedio Rating")
    plt.tight_layout()
    plt.savefig("docs/average_customer_rating.png")
    plt.close()

    # Gráfico 4: Weight_in_gms (histograma)
    plt.figure()
    df["Weight_in_gms"].plot(kind="hist", bins=20, color="mediumpurple", edgecolor="black")
    plt.title("Weight Distribution (grams)")
    plt.xlabel("Peso (g)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig("docs/weight_distribution.png")
    plt.close()

    # Crear HTML del dashboard
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Shipping Dashboard</title>
</head>
<body>
    <h1>Shipping Dashboard</h1>

    <h2>Warehouse Block</h2>
    <img src="shipping_per_warehouse.png" width="500">

    <h2>Mode of Shipment</h2>
    <img src="mode_of_shipment.png" width="500">

    <h2>Average Customer Rating by Warehouse</h2>
    <img src="average_customer_rating.png" width="500">

    <h2>Weight Distribution</h2>
    <img src="weight_distribution.png" width="500">

</body>
</html>
""")
