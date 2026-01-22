import pandas as pd
from matplotlib import pyplot as plt

def plot_task4(filepath):
    """plot average magnetisation and susceptibility data from file.
       filepath: {str} file path of data file to be plotted"""
    df = pd.read_csv(filepath)                           # Read data from file
    
    plt.figure(figsize=(8, 8))                           # Initialise figure
    
    plt.subplot(2, 1, 1)                                 # Plot average magnetisation
    plt.plot(df['kBT'], df['M'], 'o-', color='blueviolet')
    plt.xlabel(r'Thermal Energy, $k_B T$, J=1')
    plt.ylabel('Average Magnetisation, M')
    plt.title('Average Magnetisation vs Thermal Energy (Glauber Dynamics)')

    plt.subplot(2, 1, 2)                                 # Plot susceptibility
    plt.plot(df['kBT'], df['chi'], 'o-', color='orangered')
    plt.xlabel(r'Thermal Energy, $k_B T$, J=1')
    plt.ylabel(r'Susceptibility, $\chi$')
    plt.title('Susceptibility vs Thermal Energy (Glauber Dynamics)')
    
    plt.tight_layout()
    plt.show()

def plot_task56(filepath):
    """plot average magnetisation and susceptibility data from file.
       filepath: {str} file path of data file to be plotted"""
    if "task5" in filepath:                              # Determine dynamics type from filename
        dynamics = 'Glauber'
    else:
        dynamics = 'Kawasaki'

    df = pd.read_csv(filepath)                           # Read data from file
    
    plt.figure(figsize=(8, 8))                           # Initialise figure
    
    plt.subplot(2, 1, 1)                                 # Plot average magnetisation
    plt.plot(df['kBT'], df['E'], 'o-', color='forestgreen')
    plt.xlabel(r'Thermal Energy, $k_B T$, J=1')
    plt.ylabel('Average Energy, E')
    plt.title(f'Average Energy vs Thermal Energy ({dynamics} Dynamics)')

    plt.subplot(2, 1, 2)                                 # Plot heat capacity
    plt.errorbar(df['kBT'], df['C'], yerr=df['sigma'], fmt='o-', color='teal', capsize=5)
    plt.xlabel(r'Thermal Energy, $k_B T$, J=1')
    plt.ylabel(r'Heat Capacity, $C$ [$k_B$]')
    plt.title(f'Heat Capacity vs Thermal Energy ({dynamics} Dynamics)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filepath = "./task4.txt"
    plot_task4(filepath)