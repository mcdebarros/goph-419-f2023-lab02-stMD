import numpy as np

def gauss_iter_solve(a,b,x0=None,tol=1e-8,alg='seidel'):

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
    if not x0:
        x = np.zeros_like(b)
    else:
        x = np.array(x0,dtype=float)
        if x.shape != b.shape:
            raise ValueError() #FIX THIS!!!
    if alg.strip().lower() not in ('seidel','jacobi'):
        raise ValueError("Unrecognized iteration algorithm, choose either 'seidel' or 'jacobi'")
    #-------------------------------------------------------------------------------------
    # perform gauss-seidel based on selection

    eps_a = 2 * tol
    count = 0
    a_diag = np.diag(1.0/np.diag(a))
    b_star = a_diag @ b
    a_star = a_diag @ a
    a_s = a_star - np.eye(m)
    max_iter = 100

    if alg.strip().lower() == 'jacobi':
        while eps_a > tol and count < max_iter:
            xo = x.copy()
            count += 1
            x = b_star - a_s @ x
            dx = x - xo
            eps_a = np.linalg.norm(dx) / np.linalg.norm(x)
    else:
        while eps_a > tol and count < max_iter:
            xo = x.copy()
            count += 1
            for i,a_row in enumerate(a_s):
                x[i] = b_star[i] - np.dot(a_row,x)
            dx = x - xo
            eps_a = np.linalg.norm(dx) / np.linalg.norm(x)

    if count >= max_iter:
        raise RuntimeWarning() #FIX THIS!!!
    return x

def spline_function(xd,yd,order=3):

    #add to dcstring for unique values in xd

    k_sort = np.argsort(xd)
    xd = np.array([xd[k] for k in k_sort])
    yd = np.array([yd[k] for k in k_sort])

    if order not in (1,2,3):
        raise ValueError(f"Chosen order of {order} not supported.")
    
    N = len(xd)
    a = yd[:-1]
    dx = np.diff(xd)
    dy = np.diff(yd)
    f1 = dy/dx

    if order == 1:

        b = f1
        def s1(x):

            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            return a[k] + b[k] * (x - xd[k])
        
        return s1
    elif order == 2:
        
        A0 = np.hstack([np.diag(dx[:-1]),np.zeros((N-2,1))])
        A1 = np.hstack([np.zeros((N-2,1)),np.diag(dx[:-1])])
    else:
        pass



            

                
            
            
                
