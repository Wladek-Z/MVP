from ising import Ising
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def write_task4(filepath):
    """write average magnetisation and susceptibility data to file.
       filepath: {str} file path of data file to be written"""
    thermal_energies = np.arange(1, 3.1, 0.1)   # List of thermal energies to probe
    
    data = []                                   # Initialise empty data array
    for kBT in thermal_energies:
        I = Ising(50, kBT, "G")
        I.run(10000)
        M, M2 = I.avg_M()
        X = I.susceptibility(M, M2)
        data.append([kBT, M, X])
    
    """write thermal energy, average total magnetisation and susceptibility data to file"""
    df = pd.DataFrame(data, columns=['kBT', '<M>', 'chi'])
    df.to_csv(filepath, mode='a', index=False, header=False)


if __name__ == "__main__":
    filepath = "./task4.txt"
    write_task4(filepath)