# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:42:33 2024

@author: Ignacio Calvo
"""

# Este archivo contiene una funcion para la lectura de parametros: lector_parametros
# una para la lectura de las matrices: lector_matrices
# y una para leer el numero de nodos: lector_nodos

import os
import re
from scipy.io import hb_read

# -----------------------------------------------------------------------------
# Función para leer los parametros de cada problema
# -----------------------------------------------------------------------------

def lector_parametros(nombre_archivo_txt, ruta_archivo = None):

    if ruta_archivo is None:
        ruta_archivo = os.getcwd()  # ruta de la carpeta actual
    
    # Construir la ruta completa al archivo
    ruta_completa = os.path.join(ruta_archivo, nombre_archivo_txt)
    
    casos = []
    inicio_lista = False
    
    try:
    
        with open(ruta_completa, 'r') as file:
            for linea in file:
                linea = linea.strip()  # Eliminar espacios en blanco alrededor de la línea
                
                if linea.startswith("1\t["):  # Verificar el inicio de la lista de casos
                    inicio_lista = True
                    
                if inicio_lista:
                    partes = linea.split('\t', 1)  # Dividir la línea en el tabulador
                    
                    if len(partes) == 2:
                        _, parametros = partes
                        parametros = eval(parametros.strip())  # Convertir a lista
        
                        # Filtrar los valores, excluyendo las etiquetas
                        valores = [parametros[i+1] for i in range(0, len(parametros), 2)]
                        casos.append(valores)
                    else: break
                
    except FileNotFoundError:
        print(f"El archivo {nombre_archivo_txt} no se encuentra en {ruta_archivo}.")
         
    return casos

# -----------------------------------------------------------------------------
# Función para leer las matrices asociadas al problema
# -----------------------------------------------------------------------------

def lector_matrices(formulacion, ruta_carpetas = None):

    if ruta_carpetas is None:
        ruta_carpetas = os.getcwd()  # ruta de la carpeta actual
    
    As = {}
    Bs = {}
    Cs = {}
    Ms = {}
    Hs = {}
    Ls = {}
    
    carpetas = [f for f in os.scandir(ruta_carpetas) if f.is_dir()]
    # carpetas es un iterador de objetos os.DirEntry
    
    # Función para extraer el número de la carpeta y poder ordenarlas
    def extraer_numero(carpeta):
        match = re.search(r'(\d+)', carpeta.name)
        return int(match.group(1)) if match else 0
    
    # Ordenar las carpetas por el número extraído del nombre
    carpetas.sort(key = extraer_numero)  
    # carpeta es un objeto os.DirEntry. El atributo path nos da su direccion
    
    if formulacion == 'standard' :
        
        for carpeta in carpetas:
        
            Ms[carpeta.name] = hb_read(os.path.join(carpeta.path, 'K.hwb'))
            Hs[carpeta.name] = hb_read(os.path.join(carpeta.path, 'N.hwb'))
            
        return Ms, Hs
    
    elif formulacion == 'mixed' :
        
        for carpeta in carpetas:

            As[carpeta.name] = hb_read(os.path.join(carpeta.path, 'A.hwb'))
            Bs[carpeta.name] = hb_read(os.path.join(carpeta.path, 'B.hwb'))
            Cs[carpeta.name] = hb_read(os.path.join(carpeta.path, 'C.hwb'))
            Hs[carpeta.name] = hb_read(os.path.join(carpeta.path, 'PN.hwb'))
            Ls[carpeta.name] = hb_read(os.path.join(carpeta.path, 'DN.hwb'))
        
        return As, Bs, Cs, Hs, Ls
    
    else: 
        
        return "No se reconoce la formulación. Esta debe ser 'standard' o 'mixed'"
        

# -----------------------------------------------------------------------------
# Función para leer numero de nodos
# -----------------------------------------------------------------------------
                 

def lector_unknowns(ruta_carpetas = None):
        
    if ruta_carpetas is None:
        ruta_carpetas = os.getcwd()  # ruta de la carpeta actual

    unknowns = []
    name_log = 'iris.log'
    
    carpetas = [f for f in os.scandir(ruta_carpetas) if f.is_dir()]
    # carpetas es un iterador de objetos os.DirEntry
    
    # Función para extraer el número de la carpeta y poder ordenarlas
    def extraer_numero(carpeta):
        match = re.search(r'(\d+)', carpeta.name)
        return int(match.group(1)) if match else 0
    
    # Ordenar las carpetas por el número extraído del nombre
    carpetas.sort(key = extraer_numero)  
    # carpeta es un objeto os.DirEntry. El atributo path nos da su direccion
    
    for carpeta in carpetas:
           
        ruta_completa_log = os.path.join(carpeta.path, name_log) 
        
        try:
        
            with open(ruta_completa_log, 'r') as f:

                for linea in f:
                    
                    # Buscamos una línea que contenga "Number of nodes"
                    if "Number of unknowns" in linea:
                        
                        # Usamos una expresión regular para extraer el número
                        match = re.search(r'Number of unknowns\s*:\s*(\d+)', linea)
                        if match:
                            unknowns.append(int(match.group(1)))
                            break
        
        except FileNotFoundError:
            print(f"El archivo {name_log} no se encuentra en {ruta_carpetas}.")
        
    return unknowns