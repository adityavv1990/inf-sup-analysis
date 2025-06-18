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

    The eigensolver user here is from the PETSc library. 

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
        The square root of the smallest eigenvalue of the generalized
        eigenvalue problem B * H^(-1) * B^T * x = lambda * L * x.

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