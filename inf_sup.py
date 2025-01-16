# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:34:40 2024

@author: Ignacio Calvo Ramón-Borja
"""

# Este archivo contiene la función mixed_infsup
import numpy as np
from scipy.sparse.linalg import eigsh, spsolve, LinearOperator, ArpackNoConvergence, eigs, lobpcg, inv, splu, norm, svds
from scipy.linalg import sqrtm, matrix_balance
from scipy.sparse import csc_matrix
import time, math, sys
from scipy.linalg import eigh, inv
from cvxpy.atoms.affine.wraps import psd_wrap



def is_positive_definite_sparse(matrix):
    try:
        # Ensure the matrix is in CSC format
        if not isinstance(matrix, csc_matrix):
            matrix = csc_matrix(matrix)
        
        # Perform Cholesky decomposition
        _ = splu(matrix)
        return True
    except Exception as e:
        # Any error during decomposition indicates the matrix is not positive definite
        return False




def checkpositiveDefiniteness(matM, matH):
    
    if (is_positive_definite_sparse(matM)):
        print("M is positive definite")
    else:
        print("M is not positive definite")
    
    if (is_positive_definite_sparse(matH)):
        print("H is positive definite")
    else:
        print("H is not positive definite")



def symmetrizeMatrix(matrix):

    matrix = matrix + matrix.T

    return matrix



def operador_HnegHalf_M_HnegHalf(sqrtH, M):
      
    # Obtenemos los tamaños de las matrices B y H
    n, n = M.shape
    assert M.shape == (n, n), "M ha de ser una matriz cuadrada"
    assert sqrtH.shape == (n, n), "H ha de ser una matriz cuadrada"

    # Función para definir nuestra operacion matvec H
    def matvec(x):
            
    # next we calculate its inverse of sqrtH
        sqrtHminushalfX = spsolve(sqrtH, x)
        MsqrtHminusHalfX = M @ sqrtHminushalfX
        Mtilde = spsolve(sqrtH, MsqrtHminusHalfX)
        return Mtilde
        
    # Es necesario proporcionar el tamaño del operador lineal
    shape = (n, n)
    #MtM = M.T @ M
    
    return LinearOperator(shape, matvec=matvec)




def mixed_infsup(matB, matH, matL):
    """
    Calcula el valor de la constante inf-sup de la matriz B de una
    discretización dada. Lo hace a partir de la resolución del problema
    de autovalor generalizado B * H^(-1) * B^T * x = lambda * L * x. 
    
    Nos ayuda en el estudio de la estabilidad de una formulación.

    Parameters
    ----------
    matB : scipy.sparse matrix
        Matriz asociada a la forma bilineal B. De dimensiones (m, n)
    matH : scipy.sparse matrix
        Matriz de norma primal. Es simétrica y definida positiva, de
        dimensiones (n, n)
    matL : scipy.sparse matrix
        Matriz de norma dual. Es simétrica y definida positiva, de
        dimensiones (m, m)

    Returns
    -------
    float
        La raíz cuadrada del menor autovalor del problema de autovalor
        generalizado B * H^(-1) * B^T * x = lambda * L * x.
        
        Si el cálculo no converge, devuelve un mensaje de error de convergencia 
        ('Error de convergencia').

    """
    
    matB = matB.astype(np.float64)
    matH = matH.astype(np.float64)
    matL = matL.astype(np.float64)

    m,n = matB.shape

    def operador_BHinvBt(B, H):
      
        # Obtenemos los tamaños de las matrices B y H
        m, n = B.shape
        assert H.shape == (n, n), "H ha de ser una matriz cuadrada de dimension compatible con B"
        
        # Función para definir nuestra operacion matvec B*H^(-1)*B^T*x
        def matvec(x):                         
            # Primero B^T * x
            Btx = B.T @ x
            # Ahora resolvemos H^(-1) * B^T * x
            HinvBtx = spsolve(H, Btx)
            # B * H^(-1) * B^T * x
            return B @ HinvBtx
        
        # Es necesario proporcionar el tamaño del operador lineal
        shape = (m, m)
    
        return LinearOperator(shape, matvec=matvec)
    
    operator = operador_BHinvBt(matB, matH)
    
    try:
        # Calculamos el menor autovalor
        eigValues, _ = eigsh(A = operator, k = m-1, M = matL, which = 'SA', tol = 1e-5)
        eigValueMax, _ = eigsh(A = operator, k = 1, M = matL, which = 'LA', tol = 1e-5)

        eigValues = np.append(eigValues, eigValueMax)

        rank = (abs(eigValues) > 1e-5).sum()

        print("Eigenvalues = ", eigValues, flush=True)
        print("MAX Eigenvalue = ", eigValueMax, flush=True)
        print("shape of eigenvalues = ", eigValues.shape, flush=True)
        print("m = ", m, flush=True)
        print("rank of the matrix B H^-1 B.T= ", rank, flush=True)

        mineigenValue = eigValues[m-rank]
        maxeigenValue = eigValues[-1]
        eigenValues = [mineigenValue, maxeigenValue]
        return eigenValues 
        
    except ArpackNoConvergence:
        
        return "Error de convergencia"
    
 
def mixed_infsup_C(matB, matH, matC):
    """
    Calcula el valor de la constante inf-sup de la matriz B de una
    discretización dada. Lo hace a partir de la resolución del problema
    de autovalor generalizado B^T * C^(-1) * B * x = lambda * H * x. 
    
    Nos ayuda en el estudio de la estabilidad de una formulación.

    Parameters
    ----------
    matB : scipy.sparse matrix
        Matriz asociada a la forma bilineal B. De dimensiones (m, n)
    matH : scipy.sparse matrix
        Matriz de norma primal. Es simétrica y definida positiva, de
        dimensiones (n, n)
    matC : scipy.sparse matrix
        Matriz asociada a la forma bilineal C. De dimensiones (m, m)

    Returns
    -------
    float
        La raíz cuadrada del menor autovalor del problema de autovalor
        generalizado B * H^(-1) * B^T * x = lambda * L * x.
        
        Si el cálculo no converge, devuelve un mensaje de error de convergencia 
        ('Error de convergencia').

    """
    
    matB = matB.astype(np.float64)
    matH = matH.astype(np.float64)
    matC = matC.astype(np.float64)

    m,n = matB.shape

    def operador_BtCinvB(B, C):
      
        # Obtenemos los tamaños de las matrices B y C
        m, n = B.shape
        assert C.shape == (m, m), "C ha de ser una matriz cuadrada de dimension compatible con B"
        
        # Función para definir nuestra operacion matvec B*H^(-1)*B^T*x
        def matvec(x):                         
            # Primero B^T * x
            Bx = B @ x
            # Ahora resolvemos H^(-1) * B^T * x
            CinvBx = spsolve(-C, Bx)
            # B * H^(-1) * B^T * x
            return B.T @ CinvBx
        
        # Es necesario proporcionar el tamaño del operador lineal
        shape = (n, n)
    
        return LinearOperator(shape, matvec=matvec)
    
    operator = operador_BtCinvB(matB, matC)
    
    try:
        # Calculamos el menor autovalor
        eigValues, _ = eigsh(A = operator, k = n-1, M = matH, which = 'SA', tol = 1e-5)
        eigValuesMax, _ = eigsh(A = operator, k = 1, M = matH, which = 'LA', tol = 1e-5)

        eigValues = np.append(eigValues, eigValuesMax)

        rank = (abs(eigValues) > 1e-1).sum()

        print("Eigenvalues = ", eigValues, flush=True)
        print("shape of eigenvalues = ", eigValues.shape, flush=True)
        print("n = ", n, flush=True)
        print("rank of the matrix B.T C^-1 B= ", rank, flush=True)

        mineigenValue = eigValues[n-rank]
        maxeigenValue = eigValues[-1]
        eigenValues = [mineigenValue, maxeigenValue]
        return eigenValues 
        
    except ArpackNoConvergence:
        
        return "Error de convergencia"








def primal_infsup(matM, matH, eps = 0.0):
    """
    Calcula el valor de la constante inf-sup de la matriz M de una
    discretización dada. Lo hace a partir de la resolución del problema
    de autovalor generalizado B * H^(-1) * B^T * x = lambda * L * x. 
    
    Nos ayuda en el estudio de la estabilidad de una formulación.

    Parameters
    ----------
    matB : scipy.sparse matrix
        Matriz asociada a la forma bilineal B. De dimensiones (m, n)
    matH : scipy.sparse matrix
        Matriz de norma primal. Es simétrica y definida positiva, de
        dimensiones (n, n)
    matL : scipy.sparse matrix
        Matriz de norma dual. Es simétrica y definida positiva, de
        dimensiones (m, m)

    Returns
    -------
    float
        La raíz cuadrada del menor autovalor del problema de autovalor
        generalizado B * H^(-1) * B^T * x = lambda * L * x.
        
        Si el cálculo no converge, devuelve un mensaje de error de convergencia 
        ('Error de convergencia').

    """
    
    matM = matM.astype(np.float64)
    matH = matH.astype(np.float64)

    N, N = matM.shape

    ## We will verify if H and M are positive definite or not
    checkpositiveDefiniteness(matM, matH)
    matM = symmetrizeMatrix(matM)
    matH = symmetrizeMatrix(matH)
    
    scale_M = norm(matM)
    scale_H = norm(matH)

    print("scaling of matrix M = ", scale_M)
    print("scaling of matrix H = ", scale_H)

    matM = matM/scale_M
    matH = matH/scale_H

    # convert H from sparse to dense and take square root and convert it back
    #Hdense = matH.toarray()
    
    
    #start_time = time.time()
    #sqrtHdense = sqrtm(Hdense)
    #end_time = time.time()
    #elapsed_time = end_time-start_time
    #print(f"Elapsed time in calculating sqrt(H): {elapsed_time:.6f} seconds", flush=True)


    #sqrtH = csc_matrix(sqrtHdense)


    #operator = operador_HnegHalf_M_HnegHalf(sqrtH, matM)
    
    allValues = np.ndarray([], dtype=float)
    eigenValues = []

    # try:
    #     # Calculamos el menor autovalor
    #     start_time = time.time()
    #     print("Evaluating the eigenvalue using linear operator-------------->",flush=True)        
    #     #MtM = matM.T @ matM
    #     mineig1, _ = eigsh(A = operator, k = 1, which = 'SM', tol = 1e-5)
    #     print("Minimum-eigenvalue from operator = ", mineig1, flush=True)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time in linear operator block: {elapsed_time:.6f} seconds", flush=True)
        
    # except ArpackNoConvergence:    
    #     print("Error de convergencia")
    #     mineig1 = np.array([0.0])
    
    try: 
        start_time = time.time()
        print("Evaluating the eigenvalue direct matrix evaluation-------------->", flush=True)        
        #Hminushalf = inv(sqrtHdense)
        #HminushalfSparse = csc_matrix(Hminushalf)        
        #Mtilde = HminushalfSparse @ matM @ HminushalfSparse
        #MtildeDense = Mtilde.toarray()
        #MtildeBalanced, permscale = matrix_balance(MtildeDense)
        #print(permscale, flush=True)
        #MtildeBalanceSparse = csc_matrix(MtildeBalanced)
        #mineig2, _ = eigsh(MtildeBalanceSparse, k = 1, which = 'SM', tol = 1e-3, maxiter=N)
        Mdense = matM.toarray()
        Hdense = matH.toarray()
        #mineig2, _ = eigsh(A =  matM, k = 1, M = matH, which = 'SM', tol = 1e-3, maxiter=N*200)
        #cond_number_M = np.linalg.cond(Mdense)
        #cond_number_H = np.linalg.cond(Hdense)

        regularized_M = Mdense + eps * np.eye(Mdense.shape[0])
        regularized_H = Hdense + eps * np.eye(Hdense.shape[0])

        #rankM = np.linalg.matrix_rank(regularized_M)
        #rankH = np.linalg.matrix_rank(regularized_H)
        #print("shape of matrix M = ", regularized_M.shape, flush=True)
        #print("Rank of regularized matrix M = ", rankM, flush=True)
        #print("shape of regularized matrix H = ", regularized_H.shape, flush=True)
        #print("Rank of regularized matrix H = ", rankH, flush=True) 


        regularized_M_sparse = csc_matrix(regularized_M)
        regularized_H_sparse = csc_matrix(regularized_H)

        #mineig2 = eigh(regularized_M, regularized_H, eigvals_only=True, subset_by_index = [0,0], driver='gvx')
        allValues, _ = eigsh(A =  regularized_M_sparse, k = N-1, M = regularized_H_sparse, which = 'SA', tol = 1e-3, maxiter=N*200)
        eigMaxValue, _ = eigsh(A =  regularized_M_sparse, k = 1, M = regularized_H_sparse, which = 'LA', tol = 1e-3, maxiter=N*200)
        allValues = np.append(allValues, eigMaxValue)

        allValues = allValues * scale_M/scale_H

        rankM = (abs(allValues) > 1e-3).sum()

        print("Eigenvalues = ", allValues, flush=True)
        print("shape of eigenvalues = ", allValues.shape, flush=True)
        print("n = ", N, flush=True)
        print("rank of the matrix A = ", rankM, flush=True)


        minEigenValue = allValues[N-rankM]
        maxEigenValue = allValues[-1]
        eigenValues = [minEigenValue, maxEigenValue]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in balanced matrix block: {elapsed_time:.6f} seconds", flush=True)

    except ArpackNoConvergence:
        print("Error de convergencia")
        eigenValues = [0.0, 0.0]

    return eigenValues




def primal_infsup_onKerB(matM, matH, matB, eps = 0.0):
    """
    Calcula el valor de la constante inf-sup de la matriz M de una
    discretización dada. Lo hace a partir de la resolución del problema
    de autovalor generalizado B * H^(-1) * B^T * x = lambda * L * x. 
    
    Nos ayuda en el estudio de la estabilidad de una formulación.

    Parameters
    ----------
    matB : scipy.sparse matrix
        Matriz asociada a la forma bilineal B. De dimensiones (m, n)
    matH : scipy.sparse matrix
        Matriz de norma primal. Es simétrica y definida positiva, de
        dimensiones (n, n)
    matL : scipy.sparse matrix
        Matriz de norma dual. Es simétrica y definida positiva, de
        dimensiones (m, m)

    Returns
    -------
    float
        La raíz cuadrada del menor autovalor del problema de autovalor
        generalizado B * H^(-1) * B^T * x = lambda * L * x.
        
        Si el cálculo no converge, devuelve un mensaje de error de convergencia 
        ('Error de convergencia').

    """
    
    matM = matM.astype(np.float64)
    matH = matH.astype(np.float64)
    matB = matB.astype(np.float64)

    n, n = matM.shape

    ## We will verify if H and M are positive definite or not
    checkpositiveDefiniteness(matM, matH)
    matM = symmetrizeMatrix(matM)
    matH = symmetrizeMatrix(matH)
    
    ## We will next evaluate the kernel of the B matrix
    m , n = matB.shape
    print("The shape of B = ", matB.shape, flush=True)


    print("Calculating the null space of B", flush=True)
    u , s, vt = svds(matB, k = m-1, tol = 1e-5, which = 'SM')
    
    rankB = (s > 1e-5).sum()
    print("The rank of B = ", rankB, flush=True)
    nullspaceindices = np.where(s < 1e-5)
    nullSpace = vt[nullspaceindices].T

    print("The null space of B = ", nullSpace, flush=True)
    print("The shape of the null space of B = ", nullSpace.shape, flush=True)

    # Projection matrix
    temp = nullSpace.T @ nullSpace
    tempInv = np.linalg.inv(temp)
    print("tempInv = ", tempInv, flush=True)
    P = nullSpace @ tempInv @ nullSpace.T
    print("P = ", P, flush=True)

    
    # convert H from sparse to dense and take square root and convert it back
    #Hdense = matH.toarray()
    
    
    #start_time = time.time()
    #sqrtHdense = sqrtm(Hdense)
    #end_time = time.time()
    #elapsed_time = end_time-start_time
    #print(f"Elapsed time in calculating sqrt(H): {elapsed_time:.6f} seconds", flush=True)


    #sqrtH = csc_matrix(sqrtHdense)


    #operator = operador_HnegHalf_M_HnegHalf(sqrtH, matM)
    
    allValues = np.ndarray([], dtype=float)
    eigenValues = []

    # try:
    #     # Calculamos el menor autovalor
    #     start_time = time.time()
    #     print("Evaluating the eigenvalue using linear operator-------------->",flush=True)        
    #     #MtM = matM.T @ matM
    #     mineig1, _ = eigsh(A = operator, k = 1, which = 'SM', tol = 1e-5)
    #     print("Minimum-eigenvalue from operator = ", mineig1, flush=True)
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Elapsed time in linear operator block: {elapsed_time:.6f} seconds", flush=True)
        
    # except ArpackNoConvergence:    
    #     print("Error de convergencia")
    #     mineig1 = np.array([0.0])
    
    try: 
        start_time = time.time()
        print("Evaluating the eigenvalue direct matrix evaluation-------------->", flush=True)        
        #Hminushalf = inv(sqrtHdense)
        #HminushalfSparse = csc_matrix(Hminushalf)        
        #Mtilde = HminushalfSparse @ matM @ HminushalfSparse
        #MtildeDense = Mtilde.toarray()
        #MtildeBalanced, permscale = matrix_balance(MtildeDense)
        #print(permscale, flush=True)
        #MtildeBalanceSparse = csc_matrix(MtildeBalanced)
        #mineig2, _ = eigsh(MtildeBalanceSparse, k = 1, which = 'SM', tol = 1e-3, maxiter=N)
        Mdense = matM.toarray()
        Hdense = matH.toarray()

        PMP = P @ Mdense @ P
        PHP = P @ Hdense @ P

        print("PMP = ", PMP, flush=True)
        print("PHP = ", PHP, flush=True)

        #mineig2, _ = eigsh(A =  matM, k = 1, M = matH, which = 'SM', tol = 1e-3, maxiter=N*200)
        #cond_number_M = np.linalg.cond(Mdense)
        #cond_number_H = np.linalg.cond(Hdense)


        #rankM = np.linalg.matrix_rank(regularized_M)
        #rankH = np.linalg.matrix_rank(regularized_H)
        #print("shape of matrix M = ", regularized_M.shape, flush=True)
        #print("Rank of regularized matrix M = ", rankM, flush=True)
        #print("shape of regularized matrix H = ", regularized_H.shape, flush=True)
        #print("Rank of regularized matrix H = ", rankH, flush=True) 


        PMP_sparse = csc_matrix(PMP)
        PHP_sparse = csc_matrix(PHP)

        PMP_sparse = psd_wrap(PMP_sparse)
        PHP_sparse = psd_wrap(PHP_sparse)

        #mineig2 = eigh(regularized_M, regularized_H, eigvals_only=True, subset_by_index = [0,0], driver='gvx')
        allValues, _ = eigsh(A =  PMP_sparse, k = n-1, M = PHP_sparse, which = 'SA', tol = 1e-3, maxiter=n*200)
        eigMaxValue, _ = eigsh(A =  PMP_sparse, k = 1, M = PHP_sparse, which = 'LA', tol = 1e-3, maxiter=n*200)
        allValues = np.append(allValues, eigMaxValue)

        rankM = (abs(allValues) > 1e-3).sum()

        print("Eigenvalues = ", allValues, flush=True)
        print("shape of eigenvalues = ", allValues.shape, flush=True)
        print("n = ", n, flush=True)
        print("rank of the matrix A = ", rankM, flush=True)


        minEigenValue = allValues[n-rankM]
        maxEigenValue = allValues[-1]
        eigenValues = [minEigenValue, maxEigenValue]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in balanced matrix block: {elapsed_time:.6f} seconds", flush=True)

    except ArpackNoConvergence:
        print("Error de convergencia")
        eigenValues = [0.0, 0.0]

    return eigenValues    