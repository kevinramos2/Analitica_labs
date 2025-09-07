"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en el archivo data.csv. En este laboratorio
solo puede utilizar las funciones y librerias basicas de python. No puede
utilizar pandas, numpy o scipy.
"""


def pregunta_09():
    """
    Retorne un diccionario que contenga la cantidad de registros en que
    aparece cada clave de la columna 5.

    Rta/
    {'aaa': 13,
     'bbb': 16,
     'ccc': 23,
     'ddd': 23,
     'eee': 15,
     'fff': 20,
     'ggg': 13,
     'hhh': 16,
     'iii': 18,
     'jjj': 18}}

    """

    conteoClaves = {}

    with open("files/input/data.csv","r") as archivo:
        for renglon in archivo:
            div = renglon.strip().split("\t")
            clavesValores = div[4].split(",")
            
            #Para contar una vez por registro se usa set
            claveSet = set()
            for cv in clavesValores:
                clave, _ = cv.split(":")
                claveSet.add(clave)

            #Se cuentan las apariciones de la clave
            for clave in claveSet:
                if clave in conteoClaves:
                    conteoClaves[clave] += 1
                else:
                    conteoClaves[clave] = 1

    #Se ordena el diccionario alfabéticamente
    resultado9 = {}
    for clave in sorted(conteoClaves.keys()):
        resultado9[clave] = conteoClaves[clave]

    return resultado9

if __name__ == "__main__":
    print(pregunta_09())