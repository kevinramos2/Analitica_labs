"""
Escriba el codigo que ejecute la accion solicitada.
"""


import os
import zipfile
import pandas as pd
import glob


def clean_campaign_data():
    """
    En esta tarea se le pide que limpie los datos de una campaña de
    marketing realizada por un banco, la cual tiene como fin la
    recolección de datos de clientes para ofrecerls un préstamo.

    La información recolectada se encuentra en la carpeta
    files/input/ en varios archivos csv.zip comprimidos para ahorrar
    espacio en disco.

    Usted debe procesar directamente los archivos comprimidos (sin
    descomprimirlos). Se desea partir la data en tres archivos csv
    (sin comprimir): client.csv, campaign.csv y economics.csv.
    Cada archivo debe tener las columnas indicadas.

    Los tres archivos generados se almacenarán en la carpeta files/output/.

    client.csv:
    - client_id
    - age
    - job: se debe cambiar el "." por "" y el "-" por "_"
    - marital
    - education: se debe cambiar "." por "_" y "unknown" por pd.NA
    - credit_default: convertir a "yes" a 1 y cualquier otro valor a 0
    - mortage: convertir a "yes" a 1 y cualquier otro valor a 0

    campaign.csv:
    - client_id
    - number_contacts
    - contact_duration
    - previous_campaing_contacts
    - previous_outcome: cmabiar "success" por 1, y cualquier otro valor a 0
    - campaign_outcome: cambiar "yes" por 1 y cualquier otro valor a 0
    - last_contact_day: crear un valor con el formato "YYYY-MM-DD",
      combinando los campos "day" y "month" con el año 2022.

    economics.csv:
    - client_id
    - const_price_idx
    - eurobor_three_months
    """
    carpeta_entrada = "files/input"
    archivos_comprimidos = glob.glob(f"{carpeta_entrada}/*.zip")

    lista_dfs = []


    for archivo_zip in archivos_comprimidos:
        with zipfile.ZipFile(archivo_zip, 'r') as archivo_zip_ref:
            nombre_csv = archivo_zip_ref.namelist()[0]
            
            with archivo_zip_ref.open(nombre_csv) as archivo_csv:
                df_actual = pd.read_csv(archivo_csv, index_col=None)
                lista_dfs.append(df_actual)

    df_total = pd.concat(lista_dfs, ignore_index=True)

    # --- DataFrame clientes ---
    columnas_clientes = ["client_id", "age", "job", "marital", "education", "credit_default", "mortgage"]
    df_clientes = df_total[columnas_clientes].copy()

    # Limpieza columna job
    df_clientes["job"] = df_clientes["job"].str.replace(".", "", regex=False)
    df_clientes["job"] = df_clientes["job"].str.replace("-", "_", regex=False)

    # Limpieza columna education
    df_clientes["education"] = df_clientes["education"].str.replace(".", "_", regex=False)
    df_clientes["education"] = df_clientes["education"].replace("unknown", pd.NA)

    # Transformación credit_default y mortgage
    df_clientes["credit_default"] = df_clientes["credit_default"].apply(lambda val: 1 if str(val).lower() == "yes" else 0)
    df_clientes["mortgage"] = df_clientes["mortgage"].apply(lambda val: 1 if str(val).lower() == "yes" else 0)

    # --- DataFrame campaña ---
    columnas_campania = ["client_id", "number_contacts", "contact_duration", "previous_campaign_contacts", "previous_outcome", "campaign_outcome"]

    meses_map = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12"
    }

    df_campania = df_total[columnas_campania].copy()

    df_campania["previous_outcome"] = df_campania["previous_outcome"].apply(lambda val: 1 if str(val).lower() == "success" else 0)
    df_campania["campaign_outcome"] = df_campania["campaign_outcome"].apply(lambda val: 1 if str(val).lower() == "yes" else 0)

    df_campania["last_contact_date"] = "2022-" + df_total["month"].str.lower().map(meses_map) + "-" + df_total["day"].astype(str).str.zfill(2)

    # --- DataFrame economía ---
    columnas_economia = ["client_id", "cons_price_idx", "euribor_three_months"]
    df_economia = df_total[columnas_economia].copy()

    os.makedirs("files/output", exist_ok=True)
    df_clientes.to_csv("files/output/client.csv", index=False)
    df_campania.to_csv("files/output/campaign.csv", index=False)
    df_economia.to_csv("files/output/economics.csv", index=False)




if __name__ == "__main__":
    clean_campaign_data()
    
