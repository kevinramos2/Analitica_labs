"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""

# pylint: disable=import-outside-toplevel
import pandas as pd
import re

def pregunta_01():
    """
    Construya y retorne un dataframe de Pandas a partir del archivo
    'files/input/clusters_report.txt'. Los requierimientos son los siguientes:

    - El dataframe tiene la misma estructura que el archivo original.
    - Los nombres de las columnas deben ser en minusculas, reemplazando los
      espacios por guiones bajos.
    - Las palabras clave deben estar separadas por coma y con un solo
      espacio entre palabra y palabra.


    """

    #Abrimos el archivo
    with open("files/input/clusters_report.txt", "r") as archivo:
        lineas = archivo.readlines()

    #Limpiamos las lineas (se quitan lineas vacias y espacios finales)
    lineas = [linea.rstrip() for linea in lineas if linea.strip()]
    info = []
    lineaActual = None
    palabraClaveActual = ""

    #Expresión regular para capturar cluster, cantidad, porcentaje y el resto
    patron = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+,\d+)\s+(.*)")

    #Se busca desde dónde empiezan los datos (ignorando encabezados)
    for i, linea in enumerate(lineas):
        if patron.match(linea):
            #Si se encuentra una linea que cuadre con el patron, empezamos a tomar los datos útiles desde ahí
            infoLinea = lineas[i:]
            break

    #Se recorre cada línea obtenida anteriormente
    for linea in infoLinea:
        match = patron.match(linea)
        if match:
            if lineaActual is not None:
                palabrasClave = re.sub(r"\s+", " ", palabraClaveActual).strip().rstrip(".")
                palabrasClave = palabrasClave.lstrip('%').strip() 
                lineaActual.append(palabrasClave)
                info.append(lineaActual)

            #Se extraen los datos de la nueva linea usando el patrón
            cluster = int(match.group(1))
            cantidad = int(match.group(2))
            porcentaje = float(match.group(3).replace(",", "."))
            palabraClaveActual = match.group(4)
            lineaActual = [cluster, cantidad, porcentaje]
        else:
            #Si la linea no coincide con el patrón, es porque es la continuación de las palabras claves
            palabraClaveActual += " " + linea.strip()

    #Después del bucle, guardamos manualmente la última línea
    if lineaActual is not None:
        palabrasClave = re.sub(r"\s+", " ", palabraClaveActual).strip().rstrip(".")
        palabrasClave = palabrasClave.lstrip('%').strip()
        lineaActual.append(palabrasClave)
        info.append(lineaActual)

    #Se crea el df 
    df = pd.DataFrame(info, columns=[
        "cluster",
        "cantidad_de_palabras_clave",
        "porcentaje_de_palabras_clave",
        "principales_palabras_clave"
    ])

    return df


if __name__ == "__main__":
    print(pregunta_01())