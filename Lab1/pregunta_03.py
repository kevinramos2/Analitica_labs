"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en el archivo data.csv. En este laboratorio
solo puede utilizar las funciones y librerias basicas de python. No puede
utilizar pandas, numpy o scipy.
"""


def pregunta_03():
    """
    Retorne la suma de la columna 2 por cada letra de la primera columna como
    una lista de tuplas (letra, suma) ordendas alfabeticamente.

    Rta/
    [('A', 53), ('B', 36), ('C', 27), ('D', 31), ('E', 67)]

    """
    sumas = {}

    with open("files/input/data.csv","r") as archivo:
        for renglon in archivo:
            div = renglon.strip().split("\t")
            letra = div[0]
            valor = int(div[1])

            if letra in sumas:
                sumas[letra] += valor
            else:
                sumas[letra] = valor
                
    resultado3 = sorted(sumas.items())
    return resultado3

if __name__ == "__main__":
    print(pregunta_03())