import cython
import numpy as np
cimport numpy as np
from numpy import int, double
from numpy.core import intc
from fipy.tools.clapack cimport dgesv_, dgtsvx_

@cython.boundscheck(False)
cpdef np.ndarray dgesv(
        np.ndarray[double, ndim=2] A,
        np.ndarray[double, ndim=2] B):
    cdef int N = A.shape[0]
    cdef int NRHS = B.shape[0]
    cdef int info = 0     
    cdef np.ndarray ipiv = np.zeros(N, intc)
    
    dgesv_(&N, &NRHS, <double *> A.data, &N, <int *> ipiv.data, <double *> B.data, &N, &info)
    return B

def general_solve(A,B):
    A = A.copy()
    B = B.copy()
    return dgesv(A,B)
    
@cython.boundscheck(False)
cpdef np.ndarray dgtsvx( np.ndarray[double, ndim=1] dl,
                        np.ndarray[double, ndim=1] d,
                        np.ndarray[double, ndim=1] du,
                        np.ndarray[double, ndim=2] b):
    """ Tridiagonal matrix solver from LAPACK.
    Solves the following matrix equation to find x.
    
        Ax = b
    
    Inputs: 
        (N is the number of rows or columns of the (square) matrix)
        (NRHS is the number of columns on the right hand side)
    dl : double precision array of length N-1 : lower diagonal
    d  : double precision array of length N   : diagonal
    du : double precision array of length N-1 : upper diagonal
    b  : double precision 2D array, size (NRHS,N) : right side of eqn.
    
    Returns:
    x  : double precision 2D array, size (NRHS,N) : solution
    
    """
    cdef:
        char fact  = 'N'
        char trans = 'N'
        int n      = d.shape[0]
        int nrhs   = b.shape[0]
        int ldb  = n
        int ldx  = n
        int info = 0
        double rcond = 0.0        
        np.ndarray dlf   = np.zeros(n-1, double)
        np.ndarray df    = np.zeros(n, double)
        np.ndarray duf   = np.zeros(n-1, double)
        np.ndarray du2   = np.zeros(n-2, double)
        np.ndarray ipiv  = np.zeros(n, intc)
        np.ndarray x     = np.zeros((nrhs,n), double)
        np.ndarray ferr  = np.zeros(nrhs, double)
        np.ndarray berr  = np.zeros(nrhs, double)
        np.ndarray work  = np.zeros(3*n, double)
        np.ndarray iwork = np.zeros(n, intc)
    
    dgtsvx_(&fact, 
            &trans, 
            &n, 
            &nrhs, 
            <double *> dl.data, 
            <double *> d.data,
            <double *> du.data, 
            <double *> dlf.data, 
            <double *> df.data,
            <double *> duf.data, 
            <double *> du2.data, 
            <int *> ipiv.data, 
            <double *> b.data,
            &ldb,
            <double *> x.data,
            &ldx,
            &rcond,
            <double *> ferr.data,
            <double *> berr.data,
            <double *> work.data,
            <int *> iwork.data,
            &info)
    
    if info != 0:
        raise ValueError("DGTSVX failed: info={0}".format(info))

    return x.transpose()


cpdef tridiagonal_solve(np.ndarray[double, ndim=1] lower_diag,
        np.ndarray[double, ndim=1] diag,
        np.ndarray[double, ndim=1] upper_diag,
        np.ndarray[double, ndim=2] right_hand_side):
    dl = lower_diag.copy()
    d  = diag.copy()
    du = upper_diag.copy()
    b  = right_hand_side.copy()
    return dgtsvx(dl,d,du,b)

