import numpy as np
import matplotlib.pyplot as plt

from goph419lab02.linalg_interp import gauss_iter_solve,spline_function

def test_gs_solve():

    a = np.array([[9,1,0,0],
                [0,12,7,0],
                [0,0,8,7],
                [0,0,0,7]])

    b = np.array([1,2,3,4])

    jaco = gauss_iter_solve(a,b,alg='jacobi')
    seid = gauss_iter_solve(a,b)
    exp = np.linalg.solve(a,b)

    print(f"Expected: \n{exp}")
    print(f"Seidel: \n{seid}")
    print(f"Jacobi: \n{jaco}")

def test_lin_spline():

    xd = np.linspace(-5,5,10)
    yd = 12 + 55 * xd
    s1 = spline_function(xd,yd,order=1)
    xp = np.linspace(-8,8,100)
    yp_exp = 12 + 55 * xp
    yp_act = np.array([s1(x) for x in xp])
    eps_t = np.linalg.norm(yp_exp - yp_act) / np.linalg.norm(yp_exp)

    plt.figure()
    plt.plot(xd,yd,'rx',label='data')
    plt.plot(xp,yp_act,'--k',label='s1')
    plt.legend()
    plt.text(1,12,f"eps_t = {eps_t}")
    plt.savefig('figures/test_lin_spline.png')

def test_quad_spline():
    
    xd = np.linspace(-5,5,5)
    yd = 12 + 55 * xd + 21 * xd **2
    s2 = spline_function(xd,yd,order=2)
    xp = np.linspace(-8,8,100)
    yp_exp = 12 + 55 * xp + 21 * xp **2
    yp_act = np.array([s2(x) for x in xp])
    eps_t = np.linalg.norm(yp_exp - yp_act) / np.linalg.norm(yp_exp)

    plt.figure()
    plt.plot(xd,yd,'rx',label='data')
    plt.plot(xp,yp_act,'--k',label='s2')
    plt.legend()
    plt.text(1,12,f"eps_t = {eps_t}")
    plt.savefig('figures/test_quad_spline.png')

def test_cube_spline():
    
    xd = np.linspace(-5,5,10)
    yd = 12 + 55 * xd + 21 * xd **2 - 4 * xd ** 3
    s3 = spline_function(xd,yd)
    xp = np.linspace(-8,8,100)
    yp_exp = 12 + 55 * xp + 21 * xp **2 - 4 * xp ** 3
    yp_act = np.array([s3(x) for x in xp])
    eps_t = np.linalg.norm(yp_exp - yp_act) / np.linalg.norm(yp_exp)

    plt.figure()
    plt.plot(xd,yd,'rx',label='data')
    plt.plot(xp,yp_act,'--k',label='s3')
    plt.legend()
    plt.text(1,12,f"eps_t = {eps_t}")
    plt.savefig('figures/test_cube_spline.png')




if __name__ == "__main__":

    test_gs_solve()
    test_lin_spline()
    test_quad_spline()
    test_cube_spline()
