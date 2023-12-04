import numpy as np

def gauss_iter_solve(a,b,x0,tol=1e-8,alg=None):

    """Solve a linear system using Gauss-Seidel elimination.

    Parameters
    ----------
    A : numpy.ndarray, shape=(M,M)
        The coefficient matrix
    b : numpy.ndarray, shape=(M,)
        The dependent variable values
    x0 : numpy.ndarray, shape=(M,)
        Initial guess of the solution
    tol : float
        Stopping criteria for convergence
    alg : string, 'seidel' or 'gauss'
        Method of iteration

    Returns
    -------
    numpy.ndarray, shape=(M,)
        The solution vector
    """
    # ensures that the coef matrix and rhs vector are ndarrays
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    # make sure that the coef matrix is square
    m = len(a)
    ndim = len(a.shape)
    if not x0:
        x0 = (b/np.trace(a))
    else:
        x0 = np.array(x0, dtype=float)
    if ndim != 2:
        raise ValueError(f"A has {ndim} dimensions"
                         + ", should be 2")
    if a.shape[1] != m:
        raise ValueError(f"A has {m} rows and {a.shape[1]} cols"
                         + ", should be square")
    # make sure the rhs vector is a 1d array
    ndimb = len(b.shape)
    if ndimb != 1:
        raise ValueError(f"b has {ndimb} dimensions"
                         + ", should be 1D")
    # make sure number of rhs values matches number of eqns
    if len(b) != m:
        raise ValueError(f"A has {m} rows, b has {len(b)} values"
                         + ", dimensions incompatible")
    if alg.lower().strip() not in ['seidel', 'jacobi']:
        raise ValueError("Unrecognized iteration algorithm, choose either 'seidel' or 'jacobi")
    #-------------------------------------------------------------------------------------
    # perform gauss-seidel based on selection

    eps_a = np.ones(np.shape(x0)) * 100
    xk = x0

    while (any(eps_a) > tol):
        if alg == 'seidel':
            xn = 
        else:
            pass
