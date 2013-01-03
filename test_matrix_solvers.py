from __future__ import division
import numpy as np
from scipy.linalg import solve
import pyximport
pyximport.install(setup_args = {'options' :
                                {'build_ext' :
                                 {'libraries' : 'lapack',
                                  'include_dirs' : np.get_include(),
                                  }}})

import matrix_solvers
#import matrix_solvers_memoryview as matrix_solvers

dl = np.ones(4, dtype=np.float64)-2.0
d  = np.ones(5, dtype=np.float64)+2.0
du = np.ones(4, dtype=np.float64)-2.0
A = np.diag(d) + \
    np.diag(du, k=1) + \
    np.diag(dl, k=-1)
B = np.ones(5, dtype=np.float64)

def test_assemble_matrix():
    A = np.diag(d) + \
        np.diag(du, k=1) + \
        np.diag(dl, k=-1)
    B = np.ones(5, dtype=np.float64)

def test_scipy():
    A = np.diag(d) + \
        np.diag(du, k=1) + \
        np.diag(dl, k=-1)
    BB = np.array(B)
    X = solve(A,BB)
    #print "scipy  :", X

def test_lapack_general():
    BB = np.array((B,))
    BB = matrix_solvers.general_solve(A,BB)
    #print "dgesv  :", BB[0]

def test_lapack_dgesv():
    A = np.diag(d) + \
        np.diag(du, k=1) + \
        np.diag(dl, k=-1)
    BB = np.array((B,))
    BB = matrix_solvers.dgesv(A,BB)
    #print "dgesv  :", BB[0]

def test_lapack_tridiagonal():
    BB2 = np.array((B,))
    BB2 = matrix_solvers.tridiagonal_solve(dl,d,du,BB2)
    #print "dgtsvx :", BB[0]

def test_lapack_dgtsvx():
    BB2 = np.array((B,))
    BB2 = matrix_solvers.dgtsvx(dl,d,du,BB2)
    #print "dgtsvx :", BB[0]

if __name__ == '__main__':
    n = 8
    dl = np.ones(n-1, dtype=np.float64)-2.0
    d  = np.ones(n, dtype=np.float64)+2.0
    du = np.ones(n-1, dtype=np.float64)-2.0
    A = np.diag(d) + \
        np.diag(du, k=1) + \
        np.diag(dl, k=-1)
    B = np.arange(n, dtype=np.float64)
    BB = np.array((B,B)).transpose()
    X = solve(A,BB)
    print 'scipy :', X
    
    L = matrix_solvers.dgtsvx(dl,d,du,BB.transpose())
    print 'dgtsvx:', L



