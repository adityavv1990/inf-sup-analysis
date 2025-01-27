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




def evaluateNullSpaceBt(matB):

    # null space of the B.T-matrix
    m,n = matB.shape
    minDim = min(m,n)
    print("   The shape of B = ", matB.shape, flush=True)
    print("   Performing SVD of B.T to evaluate ker(B.T)", flush=True)
    u , s, vt = svds(matB.T, k = minDim-1, which = 'SM', tol = 1e-5)
    dimKernel = (abs(s) < 1e-6).sum()
    print("   The eigenvalues of B.T = ", s, flush=True)
    print("   The number of zero eigenvalues of B.T = ", dimKernel, flush=True)
    print("   The dimension of the kernel of B.T = ", dimKernel, flush=True)

    return dimKernel



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

    start_time = time.time()
    
    matB = matB.astype(np.float64)
    matH = matH.astype(np.float64)
    matL = matL.astype(np.float64)

    m,n = matB.shape
    minDim = min(m,n)
    print("   The shape of B = ", matB.shape, flush=True)
    print("   The shape of H = ", matH.shape, flush=True)
    print("   The shape of L = ", matL.shape, flush=True)

    #dimKernel = evaluateNullSpaceBt(matB)

    Bdense = matB.toarray()
    Hdense = matH.toarray()
    Ldense = matL.toarray()

    Hinv = np.linalg.inv(Hdense)
    BHinvBT = Bdense @ Hinv @ Bdense.T

    # We will evaluate the rank of the matrix B*H^(-1)*B^T
    rank = np.linalg.matrix_rank(BHinvBT)
    m,m = BHinvBT.shape

    print("Dimension of the square matrix B*H^(-1)*B^T = ", m, flush=True)
    print("Rank of B*H^(-1)*B^T = ", rank, flush=True)



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
        delta = 1e-5
        eigValues, _ = eigsh(A = operator, k = m-1, M = matL, which = 'SA', tol=1e-5)
        eigValueMax, _ = eigsh(A = operator, k = 1, M = matL, which = 'LA', tol=1e-5)

        eigValues = np.append(eigValues, eigValueMax)

        
        print("tolerance is             :", delta)
        print("Eigenvalues of B Hinv B.T= ", eigValues, flush=True)
        print("Number of zero eigenvalues of B Hinv B.T = ", m-rank, flush=True)
        print("rank of the matrix B H^-1 B.T= ", rank, flush=True)

        mineigenValue = eigValues[m-rank] # the first non-zero eigenvalue
        maxeigenValue = eigValues[-1]
        eigenValues = [mineigenValue, maxeigenValue]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in computing eigenvalues of B Hinv B.T {elapsed_time:.6f} seconds", flush=True)
 
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

    start_time = time.time()
    
    matB = matB.astype(np.float64)
    matH = matH.astype(np.float64)
    matC = matC.astype(np.float64)

    scale_C = norm(matC)
    matC = matC/scale_C
    m,n = matB.shape
    print("The shape of B = ", matB.shape, flush=True)
    print("The shape of H = ", matH.shape, flush=True)
    print("The shape of C = ", matC.shape, flush=True)

    scale_C = norm(matC)
    matC = matC/scale_C

    print("scaling of matrix C = ", scale_C)

    delta = 1e-6
    u , s, vt = svds(matC, k = m-1, which = 'SM')
    umax , smax, vtmax = svds(matC, k = 1, which = 'LM')

    s = np.append(s, smax)
    dimKernel = (abs(s) < delta).sum()
    print("   The eigenvalues of C = ", s, flush=True)
    print("   The number of zero eigenvalues of C = ", dimKernel, flush=True)
    print("   The dimension of the kernel of C = ", dimKernel, flush=True)
    

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
        eigValues, _ = eigsh(A = operator, k = n-1, M = matH, which = 'SA')
        eigValuesMax, _ = eigsh(A = operator, k = 1, M = matH, which = 'LA')

        eigValues = eigValues / scale_C

        eigValues = np.append(eigValues, eigValuesMax)

        rank = (abs(eigValues) > 1e-10).sum()

        print("Eigenvalues of B.T Cinv B = ", eigValues, flush=True)
        print("rank of the matrix B.T Cinv B= ", rank, flush=True)
        print("Number of zero eigenvalues of B.T Cinv B = ", n-rank, flush=True)

        mineigenValue = eigValues[n-rank]
        maxeigenValue = eigValues[-1]
        eigenValues = [mineigenValue, maxeigenValue]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in computing eigenvalues of B Cinv B.T {elapsed_time:.6f} seconds", flush=True)

        return eigenValues 
        
    except ArpackNoConvergence:
        
        return "Error de convergencia"




def mixed_infsup_C2(matB, matH, matC):
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

    start_time = time.time()
    
    matB = matB.astype(np.float64)
    matH = matH.astype(np.float64)
    matC = matC.astype(np.float64)

    scale_C = norm(matC)
    matC = matC/scale_C
    m,n = matB.shape
    print("The shape of B = ", matB.shape, flush=True)
    print("The shape of H = ", matH.shape, flush=True)
    print("The shape of C = ", matC.shape, flush=True)

    scale_C = norm(matC)
    matC = matC/scale_C

    print("scaling of matrix C = ", scale_C)

    u , s, vt = svds(matC, k = m-1, which = 'SM')
    umax , smax, vtmax = svds(matC, k = 1, which = 'LM')

    s = np.append(s, smax)
    dimKernel = (abs(s) < 1e-6).sum()
    print("   The eigenvalues of C = ", s, flush=True)
    print("   The number of zero eigenvalues of C = ", dimKernel, flush=True)
    print("   The dimension of the kernel of C = ", dimKernel, flush=True)
    

    def operador_BHinvBt(B, H):
      
        # Obtenemos los tamaños de las matrices B y C
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
        eigValues, _ = eigsh(A = operator, k = m-1, M = matC, which = 'SA')
        eigValuesMax, _ = eigsh(A = operator, k = 1, M = matC, which = 'LA')

        eigValues = eigValues * scale_C

        eigValues = np.append(eigValues, eigValuesMax)

        rank = (abs(eigValues) > 1e-10).sum()

        print("Eigenvalues of B Hinv B.T = \lambda C x", eigValues, flush=True)
        print("rank of the matrix B Hinv B.T = ", rank, flush=True)
        print("Number of zero eigenvalues of B Hinv B.T = ", n-rank, flush=True)

        mineigenValue = eigValues[m-rank]
        maxeigenValue = eigValues[-1]
        eigenValues = [mineigenValue, maxeigenValue]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in computing eigenvalues of B.T Hinv B = \lambda C x {elapsed_time:.6f} seconds", flush=True)

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

    start_time = time.time()
    
    matM = matM.astype(np.float64)
    matH = matH.astype(np.float64)

    n, n = matM.shape
    print("The shape of A = ", matM.shape, flush=True)
    print("The shape of H = ", matH.shape, flush=True)

    ## We will verify if H and M are positive definite or not
    checkpositiveDefiniteness(matM, matH)
    matM = symmetrizeMatrix(matM)
    matH = symmetrizeMatrix(matH)
    
    scale_M = norm(matM)
    scale_H = norm(matH)

    print("scaling of matrix A = ", scale_M)
    print("scaling of matrix H = ", scale_H)

    matM = matM/scale_M
    matH = matH/scale_H
    
    allValues = np.ndarray([], dtype=float)
    eigenValues = []
    
    try: 

        #Mdense = matM.toarray()
        #Hdense = matH.toarray()
        
        #regularized_M = Mdense + eps * np.eye(Mdense.shape[0])
        #regularized_H = Hdense + eps * np.eye(Hdense.shape[0])

        
        #regularized_M_sparse = csc_matrix(regularized_M)
        #regularized_H_sparse = csc_matrix(regularized_H)


        allValues, _ = eigsh(A =  matM, k = math.ceil(n/2.0), M = matH, which = 'SA', tol = 1e-10,  maxiter=n*200)
        eigMaxValue, _ = eigsh(A = matM, k = 1, M = matH, which = 'LA', tol = 1e-10,  maxiter=n*200)
        allValues = np.append(allValues, eigMaxValue)

        allValues = allValues * scale_M/scale_H

        rankM = (abs(allValues) < 1e-10).sum()

        print("Eigenvalues of A = ", allValues, flush=True)
        print("rank of the matrix A = ", rankM, flush=True)
        print("Number of zero eigenvalues of A = ", n-rankM, flush=True)

        minEigenValue = allValues[rankM]
        maxEigenValue = allValues[-1]
        eigenValues = [minEigenValue, maxEigenValue]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in computing eigenvalues of A: {elapsed_time:.6f} seconds", flush=True)

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

    start_time = time.time()
    
    matM = matM.astype(np.float64)
    matH = matH.astype(np.float64)
    matB = matB.astype(np.float64)

    n, n = matM.shape
    m , n = matB.shape
    minDim = min(m,n)

    print("The shape of M = ", matM.shape, flush=True)
    print("The shape of H = ", matH.shape, flush=True)
    print("The shape of B = ", matB.shape, flush=True)

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

    ## We will next evaluate the kernel of the B matrix

    print("Calculating the null space of B", flush=True)
    u , s, vt = svds(matB, k = minDim-1,  which = 'SM')

    nullspaceindices = np.where(s < 1e-6)
    nullSpace = vt[nullspaceindices].T

    print("The eigenvalues of B = ", s, flush=True)
    print("The shape of the null space of B = ", nullSpace.shape, flush=True)

    # Projection matrix
    temp = nullSpace.T @ nullSpace
    tempInv = np.linalg.inv(temp)
    P = nullSpace @ tempInv @ nullSpace.T

        
    allValues = np.ndarray([], dtype=float)
    eigenValues = []

    try: 

        Mdense = matM.toarray()
        Hdense = matH.toarray()

        PMP = P @ Mdense
        PHP =  Hdense

        PMP_sparse = csc_matrix(PMP)
        PHP_sparse = csc_matrix(PHP)

        #mineig2 = eigh(regularized_M, regularized_H, eigvals_only=True, subset_by_index = [0,0], driver='gvx')
        allValues, _ = eigsh(A =  PMP_sparse, k = n-1, ncv = n * 100, M = PHP_sparse, which = 'SA',  maxiter=n*200)
        eigMaxValue, _ = eigsh(A =  PMP_sparse, k = 1, ncv = n * 100, M = PHP_sparse, which = 'LA',  maxiter=n*200)

        allValues = np.append(allValues, eigMaxValue)

        allValues = allValues * scale_M/scale_H

        rankM = (abs(allValues) > 1e-10).sum()


        print("Eigenvalues of P * A * X= \lambda H * x ", allValues, flush=True)
        print("rank of the matrix P * A = ", rankM, flush=True)
        print("number of zero eigenvalues of P * A = ", n - rankM, flush=True)

        minEigenValue = allValues[n-rankM]
        maxEigenValue = allValues[-1]
        eigenValues = [minEigenValue, maxEigenValue]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in computing eigenvalues of P * A : {elapsed_time:.6f} seconds", flush=True)

    except ArpackNoConvergence:
        print("Error de convergencia")
        eigenValues = [0.0, 0.0]
