from ising import Ising
import pandas as pd
import numpy as np

def collect_task4(filepath):
    """compute and write average magnetisation and susceptibility data to file.
       filepath: {str} file path of data file to be written"""
    thermal_energies = np.arange(3, 0.9, -0.1)        # List of thermal energies to probe
    thermal_energies = np.round(thermal_energies, 1)  # Round to 1 decimal place
    
    data = []                                         # Initialise empty data array
    I = Ising(50, None, 'G')                          # Initialise Ising model class instance 

    for kBT in thermal_energies:                      # Loop over thermal energies, starting from hot state
        """reuse Ising instance to retain spin lattice between runs"""
        I.kBT = kBT                                   # Set thermal energy

        I.E = np.empty(0)                             # Reset energy and magnetisation arrays
        I.M = np.empty(0)

        I.run(10000)                                  # Run simulation for 10000 sweeps    

        M, M2 = I.avg_M(I.M)                          # Calculate average magnetisation, magnetisation squared
        chi = I.susceptibility(M, M2)                 # Calculate susceptibility
        data.append([kBT, np.abs(M), chi])            # Append data to array 

        print(f"kBT = {kBT}   M = {np.abs(M)}   chi = {chi}") 
    
    """write thermal energy, average total magnetisation and susceptibility data to file"""
    df = pd.DataFrame(data, columns=['kBT', 'M', 'chi'])
    df.to_csv(filepath, mode='a', index=False, header=True)

def collect_task56(filepath, dynamics):
    """compute and write average energy and heat capacity data to file.
       filepath: {str} file path of data file to be written
       dynamics: {str} dynamics type, 'G' for Glauber, 'K' for Kawasaki"""
    while dynamics not in {'G', 'K'}:                       # Validate user input
            dynamics = input("Please enter 'G' or 'K' ")

    thermal_energies = np.arange(3, 0.9, -0.1)              # List of thermal energies to probe
    thermal_energies = np.round(thermal_energies, 1)        # Round to 1 decimal place
    
    data = []                                               # Initialise empty data array
    I = Ising(50, None, dynamics)                           # Initialise Ising model class instance 

    for kBT in thermal_energies:                            # Loop over thermal energies
        """reuse Ising instance to retain spin lattice between runs"""
        I.kBT = kBT                                         # Set thermal energy

        I.E = np.empty(0)                                   # Reset energy and magnetisation arrays
        I.M = np.empty(0)

        I.run(10000)                                        # Run simulation for 10000 sweeps    

        E, E2 = I.avg_E(I.E)
        C = I.heat_capacity(E, E2)
        sigma = I.jackknife(I.E, len(I.E), C, I.L, I.kBT)
        data.append([kBT, E, C, sigma])                     # Append data to array 

        print(f"kBT = {kBT}   E = {E}   C = {C}   sigma = {sigma}")     
    
    """write thermal energy, average total energy, heat capacity, and jackknife error data to file"""
    df = pd.DataFrame(data, columns=['kBT', 'E', 'C', 'sigma'])
    df.to_csv(filepath, mode='a', index=False, header=True)


if __name__ == "__main__":
    filepath = "./task4.txt"
    collect_task4(filepath)