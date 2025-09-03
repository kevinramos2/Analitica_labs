"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en el archivo data.csv. En este laboratorio
solo puede utilizar las funciones y librerias basicas de python. No puede
utilizar pandas, numpy o scipy.
"""


def pregunta_05():
    """
    Retorne una lista de tuplas con el valor maximo y minimo de la columna 2
    por cada letra de la columa 1.

    Rta/
    [('A', 9, 2), ('B', 9, 1), ('C', 9, 0), ('D', 8, 3), ('E', 9, 1)]

    """

    valores = {}

    with open("files/input/data.csv", "r") as archivo:
        for renglon in archivo:
            div = renglon.strip().split("\t")
            letra = div[0]
            valor = int(div[1])

            if letra in valores:
                valores[letra].append(valor)
            else:
                valores[letra] = [valor]

    resultado5 = []

    for letra in sorted(valores.keys()):
        maximo = max(valores[letra])
        minimo = min(valores[letra])
        resultado5.append((letra, maximo,minimo))
        
    return resultado5

if __name__ == "__main__":
    print(pregunta_05())
