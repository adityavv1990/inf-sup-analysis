# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:05:16 2024

@author: Aditya Vasudevan, modified code of Ignacio Calvo Ramón-Borja
"""
 
# This code analysis the stability of a mixed finite element formulation
# it takes as input, matrices A, B, C that are the matrices from the stifness:
#
#           | A     B.T |
#     K  =  |           |
#           | B      C  |    
#
#
#  where A is an n by n matrix
#        B is m by n
#        C is m by m

#  The code also reads the matrices H and L that are gram Matrices, where
#        H is n by n     (gram matrix corresponding to the primal variable)
#        L is m by m     (gram matrix corresponding to the dual variable) 

#  There are three important functions in the file inf_sup.py
#       - mixed_infsup: that computes the eigenvalues of the matrix B Hinv B.T
#       - mixed_infsup_C: that computes the eigenvalues of the matrix B.T Cinv B
#       - primal_infsup: that computes the eigenvalues of the matrix A
#       - primal_infsup_onKerB: that computes the eigenvalues of the matrix A but only for vectors in the kernel of B


import sys
import os
import numpy as np
import time

from lector_casos import lector_parametros, lector_unknowns, lector_matrices
from inf_sup import mixed_infsup, primal_infsup, mixed_infsup_C, primal_infsup_onKerB, mixed_infsup_C2
import matplotlib.pyplot as plt
import subprocess


start_time = time.time()
ruta_principal = '/home/aditya/Documents/locking/simulations/clamped-beam/infsup-analysis'
# Carpeta en la que estan las funciones que voy a utilizar
sys.path.append(os.path.join(ruta_principal))


# Ruta donde se encuentran los problemas varios
ruta = os.path.join(ruta_principal, "mixed-p1p1c")

print("Reading from the directory:           ", ruta)

# Lectura de los datos y parámetros
casos = lector_parametros('clamped.txt', ruta_archivo = ruta)
num_ecs = lector_unknowns(ruta_carpetas = ruta)

print("Number of divisions: ", casos)
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
    betaMinFromH = []
    betaMaxFromH = []
    betaMinFromC = []
    betaMaxFromC = []
    betaMinFromC2 = []
    betaMaxFromC2 = []
    alphaMinFromA = []
    alphaMaxFromA = []
    alphaMaxFromAonkerB = []
    alphaMinFromAonkerB = []

    count = 0
    #for M, H in zip(matsM, matsH):
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        print("----------------------------------------------------------")
        print("Solving the eigenvalues for N = ", casos[count],flush=True)
        print("----------------------------------------------------------")
        print("\n\n")
        #beta.append(primal_infsup (M, H, eps))
        eigenValuesFromH = mixed_infsup(B ,H, L)
        minEigenValue = eigenValuesFromH[0]
        maxEigenValue = eigenValuesFromH[1]
        betaMinFromH.append(minEigenValue)
        betaMaxFromH.append(maxEigenValue)
        print("Maximum EigenValue from B Hinv B.T= ", maxEigenValue, flush=True)
        print("Minimum EigenValue from B Hinv B.T= ", minEigenValue, flush=True)
        print("\n\n")
        

        eigenValuesFromC = mixed_infsup_C(B, H, C)
        minEigenValue = eigenValuesFromC[0]
        maxEigenValue = eigenValuesFromC[1]
        betaMinFromC.append(minEigenValue)
        betaMaxFromC.append(maxEigenValue)
        print("Maximum EigenValue from B.T Cinv B = ", maxEigenValue, flush=True)
        print("Minimum EigenValue from B.T Cinv B = ", minEigenValue, flush=True)
        print("\n\n")


        eigenValuesFromC2 = mixed_infsup_C2(B, H, C)
        minEigenValue = eigenValuesFromC2[0]
        maxEigenValue = eigenValuesFromC2[1]
        betaMinFromC2.append(minEigenValue)
        betaMaxFromC2.append(maxEigenValue)
        print("Maximum EigenValue from B Hinv B.T = ", maxEigenValue, flush=True)
        print("Minimum EigenValue from B Hinv B.T = ", minEigenValue, flush=True)
        print("\n\n")

        #eigenValues = primal_infsup(M, H, eps)
        
        eigenValuesFromA = primal_infsup(A, H)

        minEigenValue = eigenValuesFromA[0]
        maxEigenValue = eigenValuesFromA[1]
        alphaMinFromA.append(minEigenValue)
        alphaMaxFromA.append(maxEigenValue)
        print("Maximum EigenValue from A = ", maxEigenValue, flush=True)
        print("Minimum EigenValue from A = ", minEigenValue, flush=True)
        print("\n\n")

        eigenValuesFromAonkerB = primal_infsup_onKerB(A, H, B)

        minEigenValue = eigenValuesFromAonkerB[0]
        maxEigenValue = eigenValuesFromAonkerB[1]
        alphaMinFromAonkerB.append(minEigenValue)
        alphaMaxFromAonkerB.append(maxEigenValue)
        print("Maximum EigenValue from A on kerB = ", maxEigenValue, flush=True)
        print("Minimum EigenValue from A on kerB = ", minEigenValue, flush=True)
        print("\n\n")


        count+=1

    
    # Guardo en un archivo de texto

    np.savetxt(ruta + "/beta_h_mineig_fromH.txt", np.vstack((num_ecs, betaMinFromH)))
    np.savetxt(ruta + "/beta_h_maxeig_fromH.txt", np.vstack((num_ecs, betaMaxFromH)))

    np.savetxt(ruta + "/beta_h_mineig_fromC.txt", np.vstack((num_ecs, betaMinFromC)))
    np.savetxt(ruta + "/beta_h_maxeig_fromC.txt", np.vstack((num_ecs, betaMaxFromC)))

    
    np.savetxt(ruta + "/beta_h_mineig_fromC2.txt", np.vstack((num_ecs, betaMinFromC2)))
    np.savetxt(ruta + "/beta_h_maxeig_fromC2.txt", np.vstack((num_ecs, betaMaxFromC2)))

    np.savetxt(ruta + "/alpha_h_mineig_fromA.txt", np.vstack((num_ecs, alphaMinFromA)))
    np.savetxt(ruta + "/alpha_h_maxeig_fromA.txt", np.vstack((num_ecs, alphaMaxFromA)))
    
    np.savetxt(ruta + "/alpha_h_mineig_fromAOnKerB.txt", np.vstack((num_ecs, alphaMinFromAonkerB)))
    np.savetxt(ruta + "/alpha_h_maxeig_fromAOnKerB.txt", np.vstack((num_ecs, alphaMaxFromAonkerB)))


    end_time = time.time()
    elapsed_time = end_time-start_time
    print("----------------------------------------------------------")
    print(f"Total time of execution: {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------")
    print("\n\n")

    command = "cp output.txt " + ruta
    result = subprocess.run(command, shell=True, text=True, capture_output=True)