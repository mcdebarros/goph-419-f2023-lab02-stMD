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
    if all(x0) == all(np.zeros(np.shape(b))):
        x0 = np.zeros(np.shape(b))
    else:
        x0 = (b/np.trace(a))
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
    if alg not in ('seidel','jacobi'):
        raise ValueError("Unrecognized iteration algorithm, choose either 'seidel' or 'jacobi")
    #-------------------------------------------------------------------------------------
    # perform gauss-seidel based on selection

    eps_a = np.array(np.ones(np.shape(b))) * 100
    xn = x0.copy()
    dx = np.array(np.zeros(np.shape(b)))
    count = 0
    a_diag = np.zeros(np.shape(a))
    ade = np.diag(a)
    for i in range (0,np.size(a[0,:])):
        a_diag[i,i] = ade[i]
    a_star = np.matmul(np.linalg.inv(a_diag),a)
    ass = a_star - np.identity(np.size(a[0,:]))
    b_star = np.matmul(np.linalg.inv(a_diag),b)

    if alg == 'jacobi': #jacobi
        while any(eps_a > tol):
            xo = xn.copy()
            count += 1
            print("Iteration: " + f"{count}")
            for i in range(0,len(b)):
                xn = b_star - (ass @ xo)
                dx = xn - xo
                eps_a = np.abs(dx/xo)
    else:
        while any(eps_a) > tol:
            xo = xn.copy()
            count += 1
            print("Iteration: " + f"{count}")
            for i in range (0,len(b)):
                sig = np.dot(a_star[i,:i],xn[:i]) + np.dot(a_star[i,i+1:],xo[i+1:])
                xn[i] = (b_star[i] - sig) / a_star[i,i]
                dx[i] = xn[i] - xo[i]
                eps_a[i] = np.abs(dx[i] / xo[i])
    return xn

def spline_function(xd,yd,order):
    pass

a = np.array([[9,1,0,0],
              [0,12,7,0],
              [0,0,8,8],
              [0,0,0,7]])

b = np.transpose(np.array([1,2,3,4]))
x = np.transpose(np.array([[5,2,1,3]]))

jaco = gauss_iter_solve(a,b,x,alg='jacobi')
seid = gauss_iter_solve(a,b,x,alg='seidel')

print(jaco)

act = np.linalg.solve(a,b)

print(seid)
print(act)
            

                
            
            
                
