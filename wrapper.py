# -*- coding: utf-8 -*-

from config import eigenSolver

if eigenSolver == 'petsc':
    from petsc4py import PETSc
    from petsc4py import PETSc

    from petsc_routines import mixed_infsup_petsc
else:

    from scipy_routines import mixed_infsup_scipy


def mixed_infsup(matB, matH, matA, matL):
    """
    Computes the value of the inf-sup constant for the matrix B of a
    given discretization. It does so by solving the generalized eigenvalue
    problem B * H^(-1) * B^T * x = lambda * L * x.

    The eigensolver used here is determined by the configuration setting.

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
    float or str
        The square root of the smallest eigenvalue of the generalized
        eigenvalue problem B * H^(-1) * B^T * x = lambda * L * x.
        If the computation does not converge, returns a convergence error
        message ('Error de convergencia').
    """

    if eigenSolver == 'petsc':
        return mixed_infsup_petsc(matB, matH, matA, matL)
    else:
        return mixed_infsup_scipy(matB, matH, matA, matL)
    
