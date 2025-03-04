# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:05:16 2024

@author: Aditya Vasudevan, modified code of Ignacio Calvo Ramón-Borja
"""
 
# This code analysis the stability of a mixed finite element formulation
# it takes as input, matrices A, B, C that are the matrices from the stiffness:
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
#       - mixed_infsup: that computes the eigenvalues of the generalized problem: B H^{-1} B.T w = \lambda L w 
#       - mixed_infsup_C: that computes the eigenvalues of the generalized problem B.T C^{-1} B w = \lambda H w
#       - primal_infsup: that computes the eigenvalues of the generalized problem: A x = \lambda H x
#       - primal_infsup_onKerB: that computes the eigenvalues of the matrix A but only for vectors in the kernel of B (this is not accurate)


import sys
import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import time

from lector_casos import lector_parametros, lector_unknowns, lector_matrices
from inf_sup import mixed_infsup, primal_infsup, mixed_infsup_C, primal_infsup_onKerB, mixed_infsup_C2, mixed_infsup_gamma
from inf_sup import evaluateNullSpaceOfMatrix, checkSingularityOfAKK, is_symmetric_sparse, checkpositiveDefiniteness
import matplotlib.pyplot as plt
import subprocess



#################################################################
# Set flags for the evaluation of different eigenvalue problems

evalBetaFromH = False
evalBetaFromC = False
evalBetaFromC2 = False
evalGammaFromC = False
evalAlphaFromA = False
evalAlphaFromAonKerB = False
evalNullSpaceBt = False
evalSingularityOfAKK = False
checkSymmetryOfMatrix = False
checkPostiveDefiniteness = False

readMatrices = 'mixed'
#readMatrices = 'standard'

print("evalBetafromH            is      : ", evalBetaFromH)
print("evalBetafromC            is      : ", evalBetaFromC)
print("evalBetafromC2           is      : ", evalBetaFromC2)
print("evalGammaFromC           is      : ", evalGammaFromC)
print("evalAlphaFromA           is      : ", evalAlphaFromA)
print("evalAlphaFromAonKerB     is      : ", evalAlphaFromAonKerB)
print("evalNullSpaceBt          is      : ", evalNullSpaceBt)
print("evalSingularityOfAKK     is      : ", evalSingularityOfAKK)
print("checkSymmetryOfMatrix    is      : ", checkSymmetryOfMatrix)
print("checkPostiveDefiniteness is      : ", checkPostiveDefiniteness)
print("reading matrices for forumlation : ", readMatrices)

#################################################################

start_time = time.time()
ruta_principal = '/media/DATOS/aditya/Simulations/locking/'
#ruta_principal = '/media/DATOS/aditya/Simulations/locking/'
# Carpeta en la que estan las funciones que voy a utilizar
sys.path.append(os.path.join(ruta_principal))


# Ruta donde se encuentran los problemas varios
ruta = os.path.join(ruta_principal, "3D_stokes/6405_stokes_p2p0_p0_at_v0")

print("Reading from the directory:           ", ruta)

# Lectura de los datos y parámetros
casos = lector_parametros('stokes.txt', ruta_archivo = ruta)
num_ecs = lector_unknowns(ruta_carpetas = ruta)

print("Number of divisions: ", casos)
print("Number of equations: ", num_ecs)
print("\n\n")

# Lectura las matrices (formulacion mixed)
if readMatrices == 'mixed':
    As, Bs, Cs, Hs, Ls = lector_matrices('mixed', ruta_carpetas = ruta)
    # Pasamos las matrices a formato lista
    matsB = [mat for mat in Bs.values()]
    matsH = [mat for mat in Hs.values()]
    matsL = [mat for mat in Ls.values()]
    matsC = [mat for mat in Cs.values()]
    matsA = [mat for mat in As.values()]
elif readMatrices == 'standard':
    Ms, Hs = lector_matrices('standard', ruta_carpetas = ruta)
    matsA = [mat for mat in Ms.values()]
    matsH = [mat for mat in Hs.values()]




if (evalBetaFromH and readMatrices == 'mixed'):

    t1 = time.time()
        
    print("----------------------------------------------------------")
    print("Solving the eigenvalue problem: B H^{-1}B.T x = \lambda L X")
    print("----------------------------------------------------------")
    print("\n\n")
    
    count = 0

    filenameMin = "beta_h_mineig_fromH.txt"
    filenameMax = "beta_h_maxeig_fromH.txt"
    
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
    
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        eigenValuesFromH = mixed_infsup(B ,H, L)
        minEigenValue = eigenValuesFromH[0]
        maxEigenValue = eigenValuesFromH[1]
        print("Maximum EigenValue = ", maxEigenValue, flush=True)
        print("Minimum EigenValue = ", minEigenValue, flush=True)
        print("\n\n")

        with open(filenameMin, 'a') as f:
            f.write(f"{casos[count]} {minEigenValue}\n")
        with open(filenameMax, 'a') as f:
            f.write(f"{casos[count]} {maxEigenValue}\n")
        
        count += 1

    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to solve  B H^{-1}B.T x = \lambda L : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")
   


 
if (evalBetaFromC and readMatrices == 'mixed'): 

    t1 = time.time()
    
    print("----------------------------------------------------------")
    print("Solving the eigenvalue problem: B.T C^{-1}B x = \lambda H X")
    print("----------------------------------------------------------")
    print("\n\n")

    filenameMin = "beta_h_mineig_fromC.txt"
    filenameMax = "beta_h_maxeig_fromC.txt"
    
    count = 0
    
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        if (checkSymmetryOfMatrix):
            flag = is_symmetric_sparse(C)
            print("The matrix C is symmetric!" if flag else "The matrix C is unsymmetric!", flush=True)
        
        eigenValuesFromC = mixed_infsup_C(B, H, C)
        minEigenValue = eigenValuesFromC[0]
        maxEigenValue = eigenValuesFromC[1]
        print("Maximum EigenValue = ", maxEigenValue, flush=True)
        print("Minimum EigenValue = ", minEigenValue, flush=True)
        print("\n\n")

        with open(filenameMin, 'a') as f:
            f.write(f"{casos[count]} {minEigenValue}\n")
        with open(filenameMax, 'a') as f:
            f.write(f"{casos[count]} {maxEigenValue}\n")

        count += 1

    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to solve  B.T C^{-1}B x = \lambda H x : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")
    



if (evalGammaFromC and readMatrices == 'mixed'): 

    t1 = time.time()
    
    print("----------------------------------------------------------")
    print("Solving the eigenvalue problem: -C x = \lambda L X")
    print("----------------------------------------------------------")
    print("\n\n")

    filenameMin = "gamma_h_mineig_fromC.txt"
    filenameMax = "gamma_h_maxeig_fromC.txt"
    count = 0
    
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        if (checkSymmetryOfMatrix):
            flag = is_symmetric_sparse(C)
            print("The matrix C is symmetric!" if flag else "The matrix C is unsymmetric!", flush=True)
        
        eigenValuesFromC = mixed_infsup_gamma(-C,L)
        minEigenValue = eigenValuesFromC[0]
        maxEigenValue = eigenValuesFromC[1]
        print("Maximum EigenValue = ", maxEigenValue, flush=True)
        print("Minimum EigenValue = ", minEigenValue, flush=True)
        print("\n\n")

        with open(filenameMin, 'a') as f:
            f.write(f"{casos[count]} {minEigenValue}\n")
        with open(filenameMax, 'a') as f:
            f.write(f"{casos[count]} {maxEigenValue}\n")
        
        count += 1
    
    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to solve  -C x = \lambda L x : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")




if (evalBetaFromC2 and readMatrices == 'mixed'):
    
    t1 = time.time()
    print("----------------------------------------------------------")
    print("Solving the eigenvalue problem: B H^{-1}B.T x = \lambda C X")
    print("----------------------------------------------------------")
    print("\n\n")
    
    filenameMin = "beta_h_mineig_fromC2.txt"
    filenameMax = "beta_h_maxeig_fromC2.txt"
    count = 0
    
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        if (checkSymmetryOfMatrix):
            flag = is_symmetric_sparse(C)
            print("The matrix C is symmetric!" if flag else "The matrix C is unsymmetric!", flush=True)
        
        eigenValuesFromC2 = mixed_infsup_C2(B, H, C)
        minEigenValue = eigenValuesFromC2[0]
        maxEigenValue = eigenValuesFromC2[1]
        print("Maximum EigenValue = ", maxEigenValue, flush=True)
        print("Minimum EigenValue = ", minEigenValue, flush=True)
        print("\n\n")

        with open(filenameMin, 'a') as f:
            f.write(f"{casos[count]} {minEigenValue}\n")
        with open(filenameMax, 'a') as f:
            f.write(f"{casos[count]} {maxEigenValue}\n")
        count += 1

    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to solve  B H^{-1}B.T x = \lambda C X : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")



if (evalAlphaFromA):
    
    t1 = time.time()
    print("----------------------------------------------------------")
    print("Solving the eigenvalue problem:          A x = \lambda H x")
    print("----------------------------------------------------------")
    print("\n\n")
    
    filenameMin = "alpha_h_mineig_fromA.txt"
    filenameMax = "alpha_h_maxeig_fromA.txt"
    count = 0

    for A, H in zip(matsA, matsH):
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        if (checkSymmetryOfMatrix):
            flag = is_symmetric_sparse(A)
            print("The matrix A is symmetric!" if flag else "The matrix A is unsymmetric!", flush=True)

        eigenValuesFromA = primal_infsup(A, H)

        minEigenValue = eigenValuesFromA[0]
        maxEigenValue = eigenValuesFromA[1]
        print("Maximum EigenValue = ", maxEigenValue, flush=True)
        print("Minimum EigenValue A = ", minEigenValue, flush=True)
        print("\n\n")

        with open(filenameMin, 'a') as f:
            f.write(f"{casos[count]} {minEigenValue}\n")
        with open(filenameMax, 'a') as f:
            f.write(f"{casos[count]} {maxEigenValue}\n")

        count += 1
    
    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to solve  A x = \lambda H x : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")




if (evalAlphaFromAonKerB and readMatrices == 'mixed'):
    
    t1 = time.time()
    print("----------------------------------------------------------")
    print("Solving the eigenvalue problem:        P A x = \lambda H x")
    print("----------------------------------------------------------")
    print("\n\n")

    filenameMin = "alpha_h_mineig_fromAOnKerB.txt"
    filenameMax = "alpha_h_maxeig_fromAOnKerB.txt"
    count = 0
   
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        eigenValuesFromAonkerB = primal_infsup_onKerB(A, H, B)

        minEigenValue = eigenValuesFromAonkerB[0]
        maxEigenValue = eigenValuesFromAonkerB[1]
        print("Maximum EigenValue = ", maxEigenValue, flush=True)
        print("Minimum EigenValue = ", minEigenValue, flush=True)
        print("\n\n")

        with open(filenameMin, 'a') as f:
            f.write(f"{casos[count]} {minEigenValue}\n")
        with open(filenameMax, 'a') as f:
            f.write(f"{casos[count]} {maxEigenValue}\n")
        count+=1
    
    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to solve P A x = \lambda H x : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")




if (evalNullSpaceBt and readMatrices == 'mixed'):
    
    t1 = time.time()
    print("----------------------------------------------------------")
    print("Performing SVD of B.T to evaluate its null space          ")
    print("----------------------------------------------------------")
    print("\n\n")
    count = 0

   
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        (m, n) = B.shape

        nullSpaceDim = evaluateNullSpaceOfMatrix(B.T)

        print("Total dimensions of B.T = ", m, flush=True)
        print("Dimension of null space = ", nullSpaceDim, flush=True)
        print("\n\n")
        count+=1
    
    
    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to calculate null space of B.T : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")



if (evalSingularityOfAKK and readMatrices == "mixed"):

    t1 = time.time()
    print("----------------------------------------------------------")
    print("Checking if the operator A on ker(B) is non-singular      ")
    print("----------------------------------------------------------")
    print("\n\n")
    count = 0

   
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        (m, n) = B.shape

        checkSingularityOfAKK(A, B)
        print("\n\n")
        count+=1
    
    
    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to check for singularity of AKK : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")



if (checkSymmetryOfMatrix and readMatrices == "mixed"):

    t1 = time.time()
    print("----------------------------------------------------------")
    print("Checking if either of the matrices A or C are symmetric:  ")
    print("----------------------------------------------------------")
    print("\n\n")
    count = 0

   
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")

        flag = is_symmetric_sparse(A)
        print("The matrix A is positive definite!" if flag else "The matrix A is not positive definite!", flush=True)
        flag = is_symmetric_sparse(C)
        print("The matrix C is symmetric!" if flag else "The matrix C is unsymmetric!", flush=True)
        
        print("\n\n")
        count+=1
    
    
    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to check for singularity of AKK : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")




if (checkPostiveDefiniteness and readMatrices == "mixed"):

    t1 = time.time()
    print("-----------------------------------------------------------------------------")
    print("Checking if either of the matrices A or C are positive or negative definte:  ")
    print("-----------------------------------------------------------------------------")
    print("\n\n")
    count = 0

   
    for A, B, C, H, L in zip(matsA, matsB, matsC, matsH, matsL):
        print("----------")
        print(" N = ", casos[count],flush=True)
        print("----------")
        
        print("Checking positive definiteness of A :------>")
        checkpositiveDefiniteness(A)

        print("Checking positive definiteness of -C :------>")
        checkpositiveDefiniteness(-C)
        
        print("\n\n")
        count+=1
    
    
    t2 = time.time()
    elapsed_time = t2-t1
    print("----------------------------------------------------------------------")
    print(f"Time to check for singularity of AKK : {elapsed_time:.6f} seconds", flush=True)
    print("----------------------------------------------------------------------")
    print("\n\n")





end_time = time.time()
elapsed_time = end_time-start_time
print("----------------------------------------------------------")
print(f"Total time of execution: {elapsed_time:.6f} seconds", flush=True)
print("----------------------------------------------------------")
print("\n\n")

command = "cp output.txt " + ruta
result = subprocess.run(command, shell=True, text=True, capture_output=True)
