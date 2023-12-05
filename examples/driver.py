# Import external packages

import numpy as np
import matplotlib.pyplot as plt

# Import spline interpolator

from goph419lab02.linalg_interp import spline_function

# Main body of program

def main():

    # Load data from file

    water = np.loadtxt("C:\\Users\\mcdeb\\GOPH419\\goph-419-f2023-lab02-stMD\\examples\\wdvt.txt")
    air = np.loadtxt("C:\\Users\\mcdeb\\GOPH419\\goph-419-f2023-lab02-stMD\\examples\\advt.txt")

    # Retrieve spline functions for water data

    s1_water = spline_function(water[:,0],water[:,1],order=1)
    s2_water = spline_function(water[:,0],water[:,1],order=2)
    s3_water = spline_function(water[:,0],water[:,1])

    # Initialize vector of x coordinates for plotting water data

    T_water = np.linspace(water[0,0],water[-1,0],100)

    # Perform spline interpolation for water data

    rho_1_water = np.array([s1_water(T) for T in T_water])
    rho_2_water = np.array([s2_water(T) for T in T_water])
    rho_3_water = np.array([s3_water(T) for T in T_water])

    # Retrieve spline functions for air data

    s1_air = spline_function(air[:,0],air[:,1],order=1)
    s2_air = spline_function(air[:,0],air[:,1],order=2)
    s3_air = spline_function(air[:,0],air[:,1])

    # Initialize vector of x coordinates for plotting air data

    T_air = np.linspace(air[0,0],air[-1,0],100)

    # Perform spline interpolation for air data

    rho_1_air = np.array([s1_air(T) for T in T_air])
    rho_2_air = np.array([s2_air(T) for T in T_air])
    rho_3_air = np.array([s3_air(T) for T in T_air])

    # Set figure dimensions

    plt.figure(figsize=(10,8))

    # Subplots (1,3,5) plot water data and interpolated splines for orders (1,2,3)

    plt.subplot(3,2,1)
    plt.plot(water[:,0],water[:,1],'xb',label="Data")
    plt.plot(T_water,rho_1_water,'--r',label="Linear")
    plt.legend()
    plt.ylabel("Density [kg/m^3]")
    plt.title("Water")

    plt.subplot(3,2,3)
    plt.plot(water[:,0],water[:,1],'xb',label="Data")
    plt.plot(T_water,rho_2_water,'--r',label="Quadratic")
    plt.legend()
    plt.ylabel("Density [kg/m^3]")

    plt.subplot(3,2,5)
    plt.plot(water[:,0],water[:,1],'xb',label="Data")
    plt.plot(T_water,rho_3_water,'--r',label="Cubic")
    plt.legend()
    plt.ylabel("Density [kg/m^3]")
    plt.xlabel("Temperature [deg C]")

    # Subplots (2,4,6) plot air data and interpolated splines for orders (1,2,3)

    plt.subplot(3,2,2)
    plt.plot(air[:,0],air[:,1],'xb',label="Data")
    plt.plot(T_air,rho_1_air,'--r',label="Linear")
    plt.legend()
    plt.ylabel("Density [kg/m^3]")
    plt.title("Air")

    plt.subplot(3,2,4)
    plt.plot(air[:,0],air[:,1],'xb',label="Data")
    plt.plot(T_air,rho_2_air,'--r',label="Quadratic")
    plt.legend()
    plt.ylabel("Density [kg/m^3]")

    plt.subplot(3,2,6)
    plt.plot(air[:,0],air[:,1],'xb',label="Data")
    plt.plot(T_air,rho_3_air,'--r',label="Cubic")
    plt.legend()
    plt.ylabel("Density [kg/m^3]")
    plt.xlabel("Temperature [deg C]")

    # Save figure to local directory

    plt.savefig("figures/air_water_density.png")

# Run program

if __name__ == "__main__":
    main()