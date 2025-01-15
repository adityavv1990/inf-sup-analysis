# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:05:16 2024

@author: Ignacio Calvo Ramón-Borja
"""
 
# Este archivo es para ejecutar la función y obtener las gráficas

import sys
import os
import numpy as np
import time

start_time = time.time()
ruta_principal = '/home/aditya/Documents/locking/simulations/clamped-beam/infsup-analysis'
# Carpeta en la que estan las funciones que voy a utilizar
sys.path.append(os.path.join(ruta_principal))

from lector_casos import lector_parametros, lector_unknowns, lector_matrices
from inf_sup import mixed_infsup, primal_infsup, mixed_infsup_C
import matplotlib.pyplot as plt

# Ruta donde se encuentran los problemas varios
ruta = os.path.join(ruta_principal, "mixed-p1p0-KPP-dependent-on-h")

# Lectura de los datos y parámetros
casos = lector_parametros('clamped.txt', ruta_archivo = ruta)
num_ecs = lector_unknowns(ruta_carpetas = ruta)

print("Number of divisions: ", casos,)
print("Number of equations: ", num_ecs)
print("\n\n")

# Lectura las matrices (formulacion mixed)
As, Bs, Cs, Hs, Ls = lector_matrices('mixed', ruta_carpetas = ruta)
#Ms, Hs = lector_matrices('standard', ruta_carpetas = ruta)


# Pasamos las matrices a formato lista
matsB = [mat for mat in Bs.values()]
matsH = [mat for mat in Hs.values()]
matsL = [mat for mat in Ls.values()]
matsC = [mat for mat in Cs.values()]
matsA = [mat for mat in As.values()]


#matsM = [mat for mat in Ms.values()]
#matsH = [mat for mat in Hs.values()]

# Loop with the regularization parameter: 
epsArray = [0.0]

for eps in epsArray:

    print("\n\n")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("Regularisation parameter eps = ", eps)
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("\n\n")


    # Mediante un bucle aplicamos el algoritmo numérico
    betaMin = []
    betaMax = []

    count = 0
    #for M, H in zip(matsM, matsH):
    for B, H, L, C, A in zip(matsB, matsH, matsL, matsC, matsA):
        print("----------------------------------------------------------")
        print("Solving the eigenvalues for N = ", casos[count],flush=True)
        #beta.append(primal_infsup (M, H, eps))
        eigenValues = mixed_infsup_C(B, H, C)
        #eigenValues = primal_infsup(M, H, eps)
        minEigenValue = eigenValues[0]
        maxEigenValue = eigenValues[1]
        betaMin.append(minEigenValue)
        betaMax.append(maxEigenValue)
        print("Maximum EigenValue = ", maxEigenValue, flush=True)
        print("Minimum EigenValue = ", minEigenValue, flush=True)
        count+=1
        print("----------------------------------------------------------")
        print("\n\n")    
    
    # Guardo en un archivo de texto
    np.savetxt(ruta + "/beta_h_mineig_fromC_" + str(eps) + ".txt", np.vstack((num_ecs, betaMin)))
    np.savetxt(ruta + "/beta_h_maxeig_fromC_" + str(eps) + ".txt", np.vstack((num_ecs, betaMax)))
    end_time = time.time()
    elapsed_time = end_time-start_time
    print("----------------------------------------------------------")
    print(f"Total time of execution: {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------")
    print("\n\n")