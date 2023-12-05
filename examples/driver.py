import numpy as np

def main():

    water = np.sort(np.loadtxt("C:\\Users\\mcdeb\\GOPH419\\goph-419-f2023-lab02-stMD\\examples\\wdvt.txt"))
    air = np.sort(np.loadtxt("C:\\Users\\mcdeb\\GOPH419\\goph-419-f2023-lab02-stMD\\examples\\advt.txt"))


    print(np.shape(water))
    print(np.shape(air))

main()