import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def task3(filename):
    """
    Obtain the phase diagram of the SIRS model system in the pS_I-pR_S plane, with constant pI_R. 
    Plot the average fraction of infected sites as a colour map.
     
    Arguments:
        filename: filepath to the task3 data
    """
    # Read in data
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

def task4(filename):
    """
    Plot the variance in fraction of infected sites against pS_I, including error bars.

    Arguments:
        filename: filepath to task4 data
    """
    # Read in data
    data = pd.read_csv(filename)
    pS_I = data['pS_I'].values()
    I_var = data['I_var'].values()
    I_err = data['I_err'].values()
    # Plot data
    plt.errorbar(pS_I, I_var, yerr=I_err, fmt='.', color='deepskyblue', capsize=5, ecolor='crimson')
    plt.title(r'Phase Diagram of SIRS Model ($p_{I \rightarrow R} = p_{R \rightarrow S} = 0.5$)')
    plt.xlabel(r'$p_{S \rightarrow I}$')
    plt.ylabel(r'$(\langle I^2 \rangle - \langle I \rangle^2)/N')
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=None, help='Filepath to the data for the relevant task (default: None)')
    parser.add_argument('-t', '--task', type=int, choices=[3, 4], default=None, help='Select which task number to perform the analysis for (default: None)')
    args = parser.parse_args()

    if args.task == 3:
        task3(args.filename)
    elif args.task == 4:
        task4(args.filename)