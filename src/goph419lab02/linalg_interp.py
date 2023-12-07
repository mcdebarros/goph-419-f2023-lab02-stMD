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
        raise RuntimeWarning(f"No convergence after {max_iter} iterations"
                             + ", returning last updated x vector")
    
    # Return calculated x vector

    return x

def spline_function(xd,yd,order=3):

    """Design n order functions for spline interpolation.

    Parameters
    ----------
    xd : numpy.ndarray, shape=(n,1)
        vector of unique x values in ascending order
    yd : numpy.ndarray, shape=(n,1)
        vector of y values
    order : int, 1 or 2 or 3
        order of interpolating function

    Returns
    -------
    s{order} : function
        Interpolating function
    """

    # Check that xd is sorted in ascending order and reorder if necessary

    k_sort = np.argsort(xd) # Array of coefficients for sorted xd vector
    xd = np.array([xd[k] for k in k_sort]) # Sort xd
    yd = np.array([yd[k] for k in k_sort]) # Sort yd

    # Check that xd and yd have same shape

    if (xd.shape != yd.shape):
        raise ValueError(f"xd has length {len(xd)}, yd has length {len(yd)}"
                         + ", should be same")
    
    # Check that order is 1 or 2 or 3

    if order not in (1,2,3):
        raise ValueError(f"Chosen order of {order} not supported.")
    
    # Initialize first order differences, f1 vector, a vector
    
    N = len(xd) # Number of data points
    a = yd[:-1] # Trimmed yd vector
    dx = np.diff(xd) # Vector of xd differences
    dy = np.diff(yd) # Vector of yd differences
    f1 = dy/dx # First order difference

    # Calculate interpolation function based on order

    if order == 1:

        # Map first order differences to b

        b = f1

        # Define first order function

        def s1(x):

            # Assign spline location based on value in x

            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            # Return interpolated value at x
            
            return a[k] + b[k] * (x - xd[k])
        
        # Return first order function
        
        return s1
    
    elif order == 2:

        # Design second order coefficient matrix
        
        A0 = np.hstack([np.diag(dx[:-1]),np.zeros((N-2,1))]) # Stack 0 vector to left edge of right-side lower portion
        A1 = np.hstack([np.zeros((N-2,1)),np.diag(dx[1:])]) # Stack 0 vector to right edge of left-side lower portion
        A = np.vstack([np.zeros((1,N-1)),(A0+A1)]) # Stack 0 row above combined A0 and A1 matrices
        B = np.zeros((N-1,)) # Initialize b vector
        B[1:] = np.diff(f1) # Second order differences
        A[0,:2] = [1,-1] # Add constant entries to first row

        # Solve for c coefficients w/ numpy.linalg (not sufficiently diagonally dominant for Gauss Seidel) and compute b vector

        c = np.linalg.solve(A,B)
        b = f1 - c * dx

        # Define second order spline function

        def s2(x):

            # Assign spline location based on value in x

            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            # Return interpolated values at x
            
            return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2
        
        # Return second order function

        return s2
    
    else:

        # Design second order coefficient matrix
        
        A = np.zeros((N,N)) # Initialize N by N 0 matrix
        A[1:-1,:-2] += np.diag(dx[:-1]) # Add lower off-diagonal entries, exclude first and last rows
        A[1:-1,1:-1] += np.diag(2 * (dx[:-1] + dx[1:])) # Add main diagonal entries, exclude first and last rows
        A[1:-1,2:] += np.diag(dx[1:]) # Add upper off-diagonal entries, exclude first and last rows
        A[0,:3] = [-dx[1],dx[0]+dx[1],-dx[0]] # Add first row entries
        A[-1,-3:] = [-dx[-1],dx[-1]+dx[-2],-dx[-2]] # Add last row entries

        # Design B vector

        B = np.zeros((N,))
        B[1:-1] = 3 * np.diff(f1)

        # Use gauss_iter_solve() function to compute c coefficients

        c = gauss_iter_solve(A,B)

        # Compute d coefficients

        d = np.diff(c) / (3 * dx)

        # Third order difference

        b = f1 - c[:-1] * dx - d * dx ** 2

        # Define third order spline function

        def s3(x):

            # Assign spline location based on value in x

            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            # Return interpolated values at x
            
            return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2 + d[k] * (x - xd[k]) ** 3
        
        # Return third order function

        return s3
    
def build_sys(xd,yd):

    N = len(xd) # Number of data points
    dx = np.diff(xd) # Vector of xd differences
    dy = np.diff(yd) # Vector of yd differences
    f1 = dy/dx # First order difference

    A = np.zeros((N,N)) # Initialize N by N 0 matrix
    A[1:-1,:-2] += np.diag(dx[:-1]) # Add lower off-diagonal entries, exclude first and last rows
    A[1:-1,1:-1] += np.diag(2 * (dx[:-1] + dx[1:])) # Add main diagonal entries, exclude first and last rows
    A[1:-1,2:] += np.diag(dx[1:]) # Add upper off-diagonal entries, exclude first and last rows
    A[0,:3] = [-dx[1],dx[0]+dx[1],-dx[0]] # Add first row entries
    A[-1,-3:] = [-dx[-1],dx[-1]+dx[-2],-dx[-2]] # Add last row entries

    # Design B vector

    B = np.zeros((N,))
    B[1:-1] = 3 * np.diff(f1)

    return A,B

def check_dom(A):

    for i in range(len(A)):

        on_d = abs(A[i,i])
        off_d = 0
        
        for j in range(len(A)):

            off_d += abs(A[i,j])

        off_d -= on_d

        if off_d > on_d:

            return False

    return True

def norm_sys(A,B):

    m = len(B)
    a_diag = np.diag(1.0/np.diag(A))
    b_star = a_diag @ B
    a_star = a_diag @ A
    a_s = a_star - np.eye(m)

    return a_s,b_star

def jacobi_only(a,b):

    count = 0
    tol = 0.5e-15
    x = np.zeros(np.shape(b))
    max_iter = 100
    eps_a = 2 * tol

    while eps_a > tol and count < max_iter:

        xo = x.copy() # Copy new x as old x
        count += 1 # Increase iteration counter
        x = b - a @ x # Compute new x guess
        dx = x - xo # Calculate difference between old and new guess
        eps_a = np.linalg.norm(dx) / np.linalg.norm(x) # Relative error update

    if count >= max_iter:

        print(f"No convergence after {max_iter} iterations. Returning last guess.")

    else:

        print("System solved!")

    return(x)

def check_soln(x,xa):
    
    for i in range(len(xa)):
        pe = (x[i] - xa[i]) / xa[i]
        if pe >= 0.001:
            return False
    
    return True

def solve_spline_coefs(xd,yd,c):

    a = yd[:-1] # Trimmed yd vector
    dx = np.diff(xd) # Vector of xd differences
    dy = np.diff(yd) # Vector of yd differences
    f1 = dy/dx # First order difference
    d = np.diff(c) / (3 * dx)
    b = f1 - c[:-1] * dx - d * dx ** 2

    return a,b,d

def gen_spline_func(a,b,c,d,x,xd):

    def spline_func(x):

        # Assign spline location based on value in x

        k = (0 if x <=  xd[0]
                else len(a) - 1 if x >= xd[-1]
            else np.nonzero(xd<x)[0][-1])
        
        # Return interpolated values at x
        
        return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2 + d[k] * (x - xd[k]) ** 3
    
    # Return third order function

    return spline_func




    







            

                
            
            
                
