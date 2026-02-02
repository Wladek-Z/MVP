import pandas as pd
from matplotlib import pyplot as plt

def plot_task4(filepath):
    """plot average magnetisation and susceptibility data from file.
       filepath: {str} file path of data file to be plotted. Must be task 4 data"""
    df = pd.read_csv(filepath)                           # Read data from file
    
    plt.figure(figsize=(8, 8))                           # Initialise figure
    
    plt.subplot(2, 1, 1)                                 # Plot average magnetisation
    plt.plot(df['kBT'], df['M'], 'o-', color='blueviolet')
    plt.xlabel(r'Thermal Energy, $k_B T$ [$J$]')
    plt.ylabel(r'Average Magnetisation, $|\langle M \rangle|$')
    plt.title('Average Magnetisation vs Thermal Energy (Glauber Dynamics)')

    plt.subplot(2, 1, 2)                                 # Plot susceptibility
    plt.plot(df['kBT'], df['chi'], 'o-', color='orangered')
    plt.xlabel(r'Thermal Energy, $k_B T$ [$J$]')
    plt.ylabel(r'Susceptibility, $\chi$')
    plt.title('Susceptibility vs Thermal Energy (Glauber Dynamics)')
    
    plt.tight_layout()
    plt.show()

def plot_task56(filepath, dynamics='Unknown'):
    """plot average magnetisation and susceptibility data from file.
       filepath: {str} file path of data file to be plotted. Must be task 5 or 6 data"""
    if "task5" in filepath:                              # Determine dynamics type from filename
        dynamics = 'Glauber'
    elif "task6" in filepath:
        dynamics = 'Kawasaki'

    df = pd.read_csv(filepath)                           # Read data from file
    
    plt.figure(figsize=(8, 8))                           # Initialise figure
    
    plt.subplot(2, 1, 1)                                 # Plot average magnetisation
    plt.plot(df['kBT'], df['E'], 'o-', color='forestgreen')
    plt.xlabel(r'Thermal Energy, $k_B T$ [$J$]')
    plt.ylabel(r'Average Energy, $\langle E \rangle$ [$J$]')
    plt.title(f'Average Energy vs Thermal Energy ({dynamics} Dynamics)')

    plt.subplot(2, 1, 2)                                 # Plot heat capacity
    plt.plot(df['kBT'], df['C'], '-', color='deepskyblue')
    plt.errorbar(df['kBT'], df['C'], yerr=df['sigma'], fmt='.', color='deepskyblue', capsize=5, ecolor='crimson')
    plt.xlabel(r'Thermal Energy, $k_B T$ [$J$]')
    plt.ylabel(r'Heat Capacity, $C$ [$k_B$]')
    plt.title(f'Heat Capacity vs Thermal Energy ({dynamics} Dynamics)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filepath = "./task5.txt"
    plot_task56(filepath)