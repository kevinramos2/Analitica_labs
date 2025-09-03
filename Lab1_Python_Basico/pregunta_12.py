"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en el archivo data.csv. En este laboratorio
solo puede utilizar las funciones y librerias basicas de python. No puede
utilizar pandas, numpy o scipy.
"""


def pregunta_12():
    """
    Genere un diccionario que contengan como clave la columna 1 y como valor
    la suma de los valores de la columna 5 sobre todo el archivo.

    Rta/
    {'A': 177, 'B': 187, 'C': 114, 'D': 136, 'E': 324}

    """

    sumaLetra = {}

    with open("files/input/data.csv", "r") as archivo:
        for renglon in archivo:
            div = renglon.strip().split("\t")
            letra = div[0]
            columna5 = div[4].split(",")

            suma = 0
            for i in columna5:
                claveValor = i.split(":")
                valor = int(claveValor[1])
                suma += valor

            if letra in sumaLetra:
                sumaLetra[letra] += suma
            else:
                sumaLetra[letra] = suma

    #Ordenar alfabéticamente
    resultado12 = {}
    for clave in sorted(sumaLetra.keys()):
        resultado12[clave] = sumaLetra[clave]

    return resultado12

if __name__ == "__main__":
    print(pregunta_12())