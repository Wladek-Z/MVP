import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def task3(filename):
    """
    Obtain the phase diagram of the SIRS model system in the pS_I-pR_S plane, with constant pI_R. 
    Plot the average fraction of infected sites as a colour map.
     
    Arguments:
        filename: Name of file containing data on average fraction of infected sites for varying pS_I and pR_S
    """
    data = pd.read_csv(filename)
    pS_I = data['pS_I'].values
    pR_S = data['pR_S'].values
    I_frac = data['I_frac'].values
    # Reshape data for plotting
    size = len(set(pS_I))
    pS_I = pS_I.reshape(size, size)
    pR_S = pR_S.reshape(size, size)
    I_frac = I_frac.reshape(size, size)
    # Plot colour map of average fraction of infected sites for varying pS_I and pR_S
    plt.imshow(I_frac, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
    plt.colorbar(label='Average Fraction of Infected Sites')
    plt.xlabel(r'$p_{S \rightarrow I}$')
    plt.ylabel(r'$p_{R \rightarrow S}$')
    plt.title(r'Phase Diagram of SIRS Model ($p_{I \rightarrow R} = 0.5$)')
    plt.show()

if __name__ == "__main__":
    filename = "task3_1.txt"
    task3(filename)