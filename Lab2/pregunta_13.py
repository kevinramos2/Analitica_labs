"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en los archivos `tbl0.tsv`, `tbl1.tsv` y 
`tbl2.tsv`. En este laboratorio solo puede utilizar las funciones y 
librerias de pandas para resolver las preguntas.
"""
import pandas as pd

def pregunta_13():
    """
    Si la columna `c0` es la clave en los archivos `tbl0.tsv` y `tbl2.tsv`,
    compute la suma de `tbl2.c5b` por cada valor en `tbl0.c1`.

    Rta/
    c1
    A    146
    B    134
    C     81
    D    112
    E    275
    Name: c5b, dtype: int64
    """

    df0 = pd.read_csv('files/input/tbl0.tsv', sep='\t')
    df2 = pd.read_csv('files/input/tbl2.tsv', sep='\t')
    
    # Hacer el merge usando la clave c0
    merged = pd.merge(df0, df2, on="c0")
    
    # Agrupar por c1 y sumar c5b
    result = merged.groupby("c1")["c5b"].sum()
    
    return result

if __name__ == "__main__":
    print(pregunta_13())