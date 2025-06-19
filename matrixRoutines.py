# -*- coding: utf-8 -*-

def is_symmetric_sparse(mat):
    """
    Check if a sparse matrix is symmetric.

    Parameters
    ----------
    mat : scipy.sparse matrix
        The matrix to check.

    Returns
    -------
    bool
        True if the matrix is symmetric, False otherwise.
    """
    return (mat != mat.T).nnz == 0