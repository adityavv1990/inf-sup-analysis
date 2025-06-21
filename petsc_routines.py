# -*- coding: utf-8 -*-
"""
Routines for numerical inf-sup analysis using petsc and slepc for
the eigenvalue solution
"""

# Este archivo contiene la función mixed_infsup

from petsc4py import PETSc
from slepc4py import SLEPc


import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import norm
import scipy.sparse as sp
import time, math, sys


# from scipy.sparse.linalg import eigsh, spsolve, LinearOperator, ArpackNoConvergence, eigs, lobpcg, inv, splu, norm, svds
# from scipy.linalg import sqrtm, matrix_balance
# from scipy.sparse import csc_matrix
# from scipy.linalg import eigh, inv, eig

def scipy_to_petsc(mat):
    mat = mat.tocsr()
    petsc_mat = PETSc.Mat().createAIJ(size=mat.shape, csr=(mat.indptr, mat.indices, mat.data))
    petsc_mat.assemble()
    return petsc_mat




def mixed_infsup(matB, matH, matA, matL):
    """
    Computes the value of the inf-sup constant for the matrix B of a
    given discretization. It does so by solving the generalized eigenvalue
    problem B * H^(-1) * B^T * x = lambda * L * x.

    The eigensolver used here is from the PETSc library. 

    Parameters
    ----------
    matB : scipy.sparse matrix
        Matrix associated with the bilinear form B. Dimensions (m, n)
    matH : scipy.sparse matrix
        Primal norm matrix. It is symmetric and positive definite, with
        dimensions (n, n)
    matL : scipy.sparse matrix
        Dual norm matrix. It is symmetric and positive definite, with
        dimensions (m, m)
    matA : scipy.sparse matrix
        Matrix associated with the bilinear form A. Dimensions (n, n)

    Returns
    -------
    float
        The smallest eigenvalue of the generalized
        eigenvalue problem B * H^(-1) * B^T * x = lambda * L * x. It's
        square root is the inf-sup constant.

        If the computation does not converge, returns a convergence error
        message ('Error de convergencia').
    """

    start_time = time.time()
    
    matB = matB.astype(np.float64)
    matH = matH.astype(np.float64)
    matL = matL.astype(np.float64)
    matA = matA.astype(np.float64)

    m,n = matB.shape
    minDim = min(m,n)
    print("   The shape of B = ", matB.shape, flush=True)
    print("   The shape of H = ", matH.shape, flush=True)
    print("   The shape of L = ", matL.shape, flush=True)

    B = scipy_to_petsc(matB)
    H = scipy_to_petsc(matH)
    L = scipy_to_petsc(matL)

    # Create KSP (linear solver) for H
    ksp = PETSc.KSP().create()
    ksp.setOperators(H)
    ksp.setType('preonly')  # Direct solve
    ksp.getPC().setType('lu')  # LU factorization
    ksp.setFromOptions()

    # Define shell matrix A = B H^{-1} B^T
    def mult_A(shell_mat, x, y):
        # Step 1: z = B^T x
        z = PETSc.Vec().createSeq(n)
        B.multTranspose(x, z)

        # Step 2: solve H y = z → y_temp
        y_temp = PETSc.Vec().createSeq(n)
        ksp.solve(z, y_temp)

        # Step 3: y = B y_temp
        B.mult(y_temp, y)

    # Create MatShell for A
    A_shell = PETSc.Mat().createPython([m, m], comm=PETSc.COMM_WORLD)
    A_shell.setPythonContext(type('ShellContext', (), {'mult': mult_A}))
    A_shell.setUp()

    # Solve generalized eigenproblem A x = lambda L x
    E = SLEPc.EPS().create()
    E.setOperators(A_shell, L)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)      # Default eigensolver type
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)  # Get smallest real part
    E.setDimensions(nev=m)
    E.setFromOptions()
    E.solve()
    
    # check convergence
    if E.getConverged() <= 0 or E.getConvergedReason() < 0:
        print("ERROR: Eigensolver did not converged", flush=True)
        sys.exit(1)
    
    # Output results
    nconv = E.getConverged()
    print(f"Number of converged eigenpairs: {nconv}")
    xr, xi = A_shell.getVecs()

    minValue = 1e20
    maxValue = -1e20

    for i in range(nconv):
        eigval = E.getEigenpair(i, xr, xi)
        print(f"Eigenvalue {i}: {eigval.real:.6f} + {eigval.imag:.6f}j")
        if (eigval.real < minValue and eigval.real > 1e-12):
            minValue = eigval.real
        if eigval.real > maxValue:
            maxValue = eigval.real

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time in computing eigenvalues of B Hinv B.T {elapsed_time:.6f} seconds", flush=True)

    return np.array([minValue, maxValue])




def mixed_infsup_C(matB, matH, matC):
    """
    Computes the value of the inf-sup constant for the matrix B of a
    given discretization. It does so by solving the generalized eigenvalue
    problem B.T * C^(-1) * B * x = lambda * H * x.
    
    ***
    Eq. 13 in D. Chapelle and K. J. Bathe, The inf-sup test,
    Computers and Structures, Vol. 47, No. 4/5, pp. 537-545. 1993
    ****
    
    The eigensolver used here is from the PETSc library. 

    Parameters
    ----------
    matB : scipy.sparse matrix
        Matrix associated with the bilinear form B. Dimensions (m, n)
    matH : scipy.sparse matrix
        Primal norm matrix. It is symmetric and positive definite, with
        dimensions (n, n)
    matC : scipy.sparse matrix
        Matrix associated with the bilinear form C. Dimensions (n, n)

    Returns
    -------
    float
        The smallest eigenvalue of the generalized
        eigenvalue problem B.T * C^(-1) * B * x = lambda * H * x. It's 
        square root is the inf-sup constant.

        If the computation does not converge, returns a convergence error
        message ('Error de convergencia').
    """

    start_time = time.time()
    
    matB = matB.astype(np.float64)
    matH = matH.astype(np.float64)
    matC = matC.astype(np.float64)

    scale_C = norm(matC)
    matC = matC / scale_C  # Normalize C to avoid numerical issues

    m,n = matB.shape
    print("The shape of B = ", matB.shape, flush=True)
    print("The shape of H = ", matH.shape, flush=True)
    print("The shape of C = ", matC.shape, flush=True)

    B = scipy_to_petsc(matB)
    H = scipy_to_petsc(matH)
    C = scipy_to_petsc(matC)

    # Create KSP (linear solver) for C
    ksp = PETSc.KSP().create()
    ksp.setOperators(-C)
    ksp.setType('preonly')  # Direct solve
    ksp.getPC().setType('lu')  # LU factorization
    ksp.setFromOptions()
    # ksp.setType('cg')  # Or 'gmres' if C is not symmetric positive definite
    # pc = ksp.getPC()
    # pc.setType('hypre')  # Or 'ilu' if available
    # ksp.setTolerances(rtol=1e-5)
    # ksp.setFromOptions()

    # Define shell matrix A = B.T C^{-1} B
    def mult_C(shell_mat, x, y):
        # Step 1: z = B^T x
        # z = PETSc.Vec().createSeq(m)
        # B.mult(x, z)

        # # Step 2: solve H y = z → y_temp
        # y_temp = PETSc.Vec().createSeq(m)
        # ksp.solve(z, y_temp)

        # # Step 3: y = B y_temp
        # B.multTranspose(y_temp, y)
        y.set(0.0)
        xtemp = B.getVecLeft()
        B.mult(x, xtemp)         # xtemp = B * x

        ytemp = C.getVecLeft()
        ksp.solve(xtemp, ytemp)  # ytemp = (−C)^−1 * B * x
        B.multTranspose(ytemp, y)  # y = B^T * (−C)^−1 * B * x

    # Create MatShell for A
    A_shell = PETSc.Mat().createPython([n, n], comm=PETSc.COMM_WORLD)
    A_shell.setPythonContext(type('ShellContext', (), {'mult': mult_C}))
    A_shell.setUp()

    # Solve generalized eigenproblem A x = lambda L x
    E = SLEPc.EPS().create()
    E.setOperators(A_shell, H)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)      # Default eigensolver type
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)  # Get smallest real part
    E.setDimensions(nev=n)
    E.setFromOptions()
    E.setTolerances(tol=1e-8)
    E.solve()
    
    # check convergence
    if E.getConverged() <= 0 or E.getConvergedReason() < 0:
        print("ERROR: Eigensolver did not converged", flush=True)
        sys.exit(1)
    
    # Output results
    nconv = E.getConverged()
    print(f"Number of converged eigenpairs: {nconv}")
    xr, xi = A_shell.getVecs()

    minValue = 1e20
    maxValue = -1e20

    for i in range(nconv):
        eigval = E.getEigenpair(i, xr, xi)
        eigval = eigval / scale_C  # Rescale eigenvalue
        print(f"Eigenvalue {i}: {eigval.real} + {eigval.imag}j")
        if (abs(eigval.real) < minValue and abs(eigval.real) > 1e-4):
            minValue = eigval.real
        if eigval.real > maxValue:
            maxValue = eigval.real

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time in computing eigenvalues of B.T Cinv B {elapsed_time:.6f} seconds", flush=True)

    return np.array([minValue, maxValue])




def mixed_infsup_stabilized_U(matA, matB, matC, matH):

    """
    Calculates the minimum and maximum eigenvalue of the generalized eigenvalue problem
    (A + B^T * C^(-1) * B ) x = lambda * H * x.
    Parameters
    ----------
    matA : scipy.sparse matrix
        Matrix associated with the bilinear form A. Dimensions (n, n)
    matB : scipy.sparse matrix
        Matrix associated with the bilinear form B. Dimensions (m, n)
    matC : scipy.sparse matrix
        Matrix associated with the bilinear form C. Dimensions (m, m)
    matH : scipy.sparse matrix
        Primal norm matrix. It is symmetric and positive definite, with
        dimensions (n, n)

    Returns
    -------
    float [2]
        Returns the minimum non-zero and maximum eigenvalue of the generalized eigenvalue
        problem (A + B^T * C^(-1) * B ) x = lambda * H * x.
        
        If the calculation does not converge, it returns a convergence error message 
        ('Convergence error').

    """


    start_time = time.time()
    
    matA = matA.astype(np.float64)
    matB = matB.astype(np.float64)
    matC = matC.astype(np.float64)
    matH = matH.astype(np.float64)

    m,n = matB.shape
    print("The shape of A = ", matA.shape, flush=True)
    print("The shape of B = ", matB.shape, flush=True)
    print("The shape of C = ", matC.shape, flush=True)
    print("The shape of H = ", matH.shape, flush=True)

    A = scipy_to_petsc(matA)
    B = scipy_to_petsc(matB)
    C = scipy_to_petsc(matC)
    H = scipy_to_petsc(matH)

    # Create KSP (linear solver) for C

    ksp = PETSc.KSP().create()
    ksp.setOperators(-C)
    # ksp.setType('preonly')  # Direct solve
    # ksp.getPC().setType('lu')  # LU factorization
    # ksp.setFromOptions()
    ksp.setType('cg')  # Or 'gmres' if C is not symmetric positive definite
    pc = ksp.getPC()
    pc.setType('jacobi')  # Or 'ilu' if available
    ksp.setTolerances(rtol=1e-5)
    ksp.setFromOptions()


    # Define shell matrix A = A + B^T C^{-1} B
    def mult_shell(shell_mat, x, y):

        # Compute B * x
        temp1 = B.createVecLeft()
        B.mult(x, temp1)

        # Solve C * z = B * x
        temp2 = temp1.duplicate()
        ksp.solve(temp1, temp2)

        # Compute B^T * z
        temp3 = x.duplicate()
        B.multTranspose(temp2, temp3)

        # Compute A * x
        temp4 = x.duplicate()
        A.mult(x, temp4)

        y.array = temp3.array + temp4.array
    
    # Create the shell matrix
    shell_mat = PETSc.Mat().createPython([n, n], comm=PETSc.COMM_WORLD)
    shell_mat.setPythonContext(type('ShellContext', (), {'mult': mult_shell}))
    shell_mat.setUp()

    # Set up SLEPc eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(shell_mat, H)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)      # Default eigensolver type
    E.setDimensions(nev=n)
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)  # Get smallest real part
    E.setTolerances(1e-9)
    E.setFromOptions()
    E.solve()

    # check convergence
    if E.getConverged() <= 0 or E.getConvergedReason() < 0:
        print("ERROR: Eigensolver did not converged", flush=True)
        sys.exit(1)
    
    # Output results
    nconv = E.getConverged()
    print(f"Number of converged eigenpairs: {nconv}")
    xr, xi = shell_mat.getVecs()

    minValue = 1e20
    maxValue = -1e20

    for i in range(nconv):
        eigval = E.getEigenpair(i, xr, xi)
        print(f"Eigenvalue {i}: {eigval.real} + {eigval.imag}j")
        if (abs(eigval.real) < minValue and abs(eigval.real) > 1e-4):
            minValue = eigval.real
        if eigval.real > maxValue:
            maxValue = eigval.real

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time in computing eigenvalues of (A + B.T Cinv B) x = \lambda H x {elapsed_time:.6f} seconds", flush=True)

    return np.array([minValue, maxValue])




def mixed_infsup_stabilized_P(matA, matB, matC, matL):

    """
    Calculates the minimum and maximum eigenvalue of the generalized eigenvalue problem
    (B A^{-1} B^T + C) x = lambda * L * x..
    Parameters
    ----------
    matA : scipy.sparse matrix
        Matrix associated with the bilinear form A. Dimensions (n, n)
    matB : scipy.sparse matrix
        Matrix associated with the bilinear form B. Dimensions (m, n)
    matC : scipy.sparse matrix
        Matrix associated with the bilinear form C. Dimensions (m, m)
    matL : scipy.sparse matrix
        Dual norm matrix. It is symmetric and positive definite, with
        dimensions (m, m)

    Returns
    -------
    float [2]
        Returns the minimum non-zero and maximum eigenvalue of the generalized eigenvalue
        problem (B A^{-1} B^T + C ) x = lambda * L * x.
        
        If the calculation does not converge, it returns a convergence error message 
        ('Convergence error').

    """


    start_time = time.time()
    
    matA = matA.astype(np.float64)
    matB = matB.astype(np.float64)
    matC = matC.astype(np.float64)
    matL = matL.astype(np.float64)

    m,n = matB.shape
    print("The shape of A = ", matA.shape, flush=True)
    print("The shape of B = ", matB.shape, flush=True)
    print("The shape of C = ", matC.shape, flush=True)
    print("The shape of L = ", matL.shape, flush=True)

    A = scipy_to_petsc(matA)
    B = scipy_to_petsc(matB)
    C = scipy_to_petsc(matC)
    L = scipy_to_petsc(matL)

    # Create KSP (linear solver) for C

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('preonly')  # Direct solve
    ksp.getPC().setType('lu')  # LU factorization
    ksp.setFromOptions()

    # Define shell matrix A = A + B^T C^{-1} B
    def mult_shell(shell_mat, x, y):

        # Compute B * x
        temp1 = B.createVecLeft()
        B.multTranspose(x, temp1)

        # Solve C * z = B * x
        temp2 = temp1.duplicate()
        ksp.solve(temp1, temp2)

        # Compute B^T * z
        temp3 = x.duplicate()
        B.mult(temp2, temp3)

        # Compute A * x
        temp4 = x.duplicate()
        C.mult(x, temp4)
        temp4.scale(-1.0)

        y.array = temp3.array + temp4.array
    
    # Create the shell matrix
    shell_mat = PETSc.Mat().createPython([m, m], comm=PETSc.COMM_WORLD)
    shell_mat.setPythonContext(type('ShellContext', (), {'mult': mult_shell}))
    shell_mat.setUp()

    # Set up SLEPc eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(shell_mat, L)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)      # Default eigensolver type
    E.setDimensions(nev=m)
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)  # Get smallest real part
    E.setTolerances(1e-9)
    E.setFromOptions()
    E.solve()

    # check convergence
    if E.getConverged() <= 0 or E.getConvergedReason() < 0:
        print("ERROR: Eigensolver did not converged", flush=True)
        sys.exit(1)
    
    # Output results
    nconv = E.getConverged()
    print(f"Number of converged eigenpairs: {nconv}")
    xr, xi = shell_mat.getVecs()

    minValue = 1e20
    maxValue = -1e20

    for i in range(nconv):
        eigval = E.getEigenpair(i, xr, xi)
        #print(f"Eigenvalue {i}: {eigval.real} + {eigval.imag}j")
        if (abs(eigval.real) < minValue and abs(eigval.real) > 1e-4):
            minValue = eigval.real
        if eigval.real > maxValue:
            maxValue = eigval.real

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time in computing eigenvalues of  (B A^-1 B^T + C ) x = lambda * L * x {elapsed_time:.6f} seconds", flush=True)

    return np.array([minValue, maxValue])

