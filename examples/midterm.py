import numpy as np
import matplotlib.pyplot as plt

from goph419lab02.linalg_interp import spline_function as spf
from goph419lab02.linalg_interp import gauss_iter_solve as gis
from goph419lab02.linalg_interp import build_sys as bs
from goph419lab02.linalg_interp import check_dom as cd
from goph419lab02.linalg_interp import norm_sys as ns
from goph419lab02.linalg_interp import jacobi_only as jo
from goph419lab02.linalg_interp import check_soln as cs
from goph419lab02.linalg_interp import solve_spline_coefs as ssc
from goph419lab02.linalg_interp import gen_spline_func as gsf

def main():

    year_check = 2015.25
    print("Loading data...")
    carbon = np.loadtxt(".\\examples\\MKCO2.txt")
    year = carbon[:,0]
    co2 = carbon[:,1]
    print("Loaded!")
    print("Building system of equations...")
    A,B = bs(year,co2)
    print("System built!")
    print("Checking diagonal dominance...")
    d_dom = cd(A)
    if d_dom == True:
        print("System is diagonally dominant!")
    else:
        print("System is not diagonally dominant!")
        print("Convergence is not guaranteed!")
    print("Normalizing system...")
    a,b = ns(A,B)
    print("System normalized!")
    print("Attempting to solve system...")
    x = jo(a,b)
    print("Checking solution accuracy...")
    xa = np.linalg.solve(a,b)
    acc = cs(x,xa)
    if acc == True:
        print("Solution accurate to within 00.1%")
        c = x
    else:
        print("Solution not accurate. Check data and try again.")
        exit()
    print("Solving for spline ccoefficients...")
    a,b,d = ssc(year,co2,c)
    print("Coefficients obtained!")
    print("Generating spline function...")
    spline = gsf(a,b,c,d,year_check,year)
    print("Spline function obtained")
    print("Calculating CO2 in March 2015...")
    mar = spline(year_check)
    print(f"CO2 in March of 2015: {mar} micromol/mol")
    print("Interpolating for all data...")
    cubic = spf(year,co2)
    x_vec = np.linspace(year[0],year[-1],100)
    c_plot = np.array([cubic(yr) for yr in x_vec])
    plt.subplot(1,1,1)
    plt.plot(x_vec,c_plot,'--b',label="Interpolated splines")
    plt.plot(year,co2,'xr',label="Raw Data")
    plt.plot(year_check,mar,'xg',label="CO2 in March 2015")
    plt.ylabel("CO2 Concentration [micromol/mol]")
    plt.xlabel("Year")
    plt.legend()
    plt.savefig(".\\figures\\co2_splines.png")
    plt.show()

if __name__ == "__main__":
    main()

