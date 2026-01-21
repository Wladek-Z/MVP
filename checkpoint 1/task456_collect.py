from ising import Ising
import pandas as pd
import numpy as np

def write_task4(filepath):
    """write average magnetisation and susceptibility data to file.
       filepath: {str} file path of data file to be written"""
    thermal_energies = np.arange(1, 3, 0.1)           # List of thermal energies to probe
    thermal_energies = np.round(thermal_energies, 1)  # Round to 1 decimal place
    
    data = []                                         # Initialise empty data array
    for kBT in thermal_energies:                      # Loop over thermal energies
        I = Ising(50, kBT, 'G')
        I.run(10000)                                  # Run simulation for 10000 sweeps    

        M, M2 = I.avg_M()
        chi = I.susceptibility(M, M2)
        data.append([kBT, M, chi])                    # Append data to array 

        print(f"kBT = {kBT}   M = {M}   chi = {chi}") # Print progress to terminal
    
    """write thermal energy, average total magnetisation and susceptibility data to file"""
    df = pd.DataFrame(data, columns=['kBT', 'M', 'chi'])
    df.to_csv(filepath, mode='a', index=False, header=True)


if __name__ == "__main__":
    filepath = "./task4_3.txt"
    write_task4(filepath)