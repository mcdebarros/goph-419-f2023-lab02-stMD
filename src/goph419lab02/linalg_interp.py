# Import external packages

import numpy as np

# Define functions

def gauss_iter_solve(a,b,x0=None,tol=1e-8,alg='seidel'):

    """Solve a linear system using Gauss-Seidel elimination.

    Parameters
    ----------
    a : numpy.ndarray, shape=(M,M)
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

    # Check that coefficient matrix and RHS vectors are ndarrays

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    # Check that coefficient matrix is square

    m = len(a)
    ndim = len(a.shape)
    if ndim != 2:
        raise ValueError(f"A has {ndim} dimensions"
                         + ", should be 2")
    if a.shape[1] != m:
        raise ValueError(f"A has {m} rows and {a.shape[1]} cols"
                         + ", should be square")
    
    # Check that RHS vector is a 1D array

    ndimb = len(b.shape)
    if ndimb != 1:
        raise ValueError(f"b has {ndimb} dimensions"
                         + ", should be 1D")
    
    # Check that number of RHS values equals number of equations in a

    if len(b) != m:
        raise ValueError(f"A has {m} rows, b has {len(b)} values"
                         + ", dimensions incompatible")
    
    # Initialize an inital guess of zeros if no intial guess provided and check that x and b have the same shape

    if not x0:
        x = np.zeros_like(b)
    else:
        x = np.array(x0,dtype=float)
        if x.shape != b.shape:
            raise ValueError(f"X has shape {x.shape}, b has shape {b.shape}"
                             + ", should be same length") 
    
    # Check that alg is one of 'seidel' or 'jacobi'

    if alg.strip().lower() not in ('seidel','jacobi'):
        raise ValueError("Unrecognized iteration algorithm"
                         + ", choose either 'seidel' or 'jacobi'")
    
    #-------------------------------------------------------------------------------------

    # Initialize error and iteration counter

    eps_a = 2 * tol
    count = 0
    max_iter = 100

    # Normalize coefficient matrix and b vector

    a_diag = np.diag(1.0/np.diag(a))
    b_star = a_diag @ b
    a_star = a_diag @ a
    a_s = a_star - np.eye(m)

    # Perform Gauss-Seidel iteration based on alg

    if alg.strip().lower() == 'jacobi':

        # Perform Gauss-Seidel method until convergence or maximum iterations

        while eps_a > tol and count < max_iter:

            xo = x.copy() # Copy new x as old x
            count += 1 # Increase iteration counter
            x = b_star - a_s @ x # Compute new x guess
            dx = x - xo # Calculate difference between old and new guess
            eps_a = np.linalg.norm(dx) / np.linalg.norm(x) # Relative error update

    else:

        # Perform Jacobi method until convergence or maximum iterations

        while eps_a > tol and count < max_iter:

            xo = x.copy() # Copy new x as old x
            count += 1 # Increaste iteration counter

            # Update each element in the x vector

            for i,a_row in enumerate(a_s):
                x[i] = b_star[i] - np.dot(a_row,x)

            dx = x - xo # Difference between old and new guesses
            eps_a = np.linalg.norm(dx) / np.linalg.norm(x) # Relative error update

    # Warn user if maximum iterations reached before convergence

    if count >= max_iter:
        raise RuntimeWarning(f"")
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
        A1 = np.hstack([np.zeros((N-2,1)),np.diag(dx[1:])])
        A = np.vstack([np.zeros((1,N-1)),(A0+A1)])
        B = np.zeros((N-1,))
        B[1:] = np.diff(f1)
        A[0,:2] = [1,-1]

        c = np.linalg.solve(A,B)
        b = f1 - c * dx

        def s2(x):

            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2
        return s2
    else:
        
        A = np.zeros((N,N))
        A[1:-1,:-2] += np.diag(dx[:-1])
        A[1:-1,1:-1] += np.diag(2 * (dx[:-1] + dx[1:]))
        A[1:-1,2:] += np.diag(dx[1:])
        A[0,:3] = [-dx[1],dx[0]+dx[1],-dx[0]]
        A[-1,-3:] = [-dx[-1],dx[-1]+dx[-2],-dx[-2]]

        B = np.zeros((N,))
        B[1:-1] = 3 * np.diff(f1)

        c = gauss_iter_solve(A,B)

        d = np.diff(c) / (3 * dx)

        b = f1 - c[:-1] * dx - d * dx ** 2

        def s3(x):

            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2 + d[k] * (x - xd[k]) ** 3
        return s3





            

                
            
            
                
