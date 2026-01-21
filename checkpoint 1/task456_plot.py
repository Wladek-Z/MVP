from ising import Ising
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_task4(filepath):
    """plot average magnetisation and susceptibility data from file.
       filepath: {str} file path of data file to be plotted"""
    df = pd.read_csv(filepath)                           # Read data from file
    
    plt.figure(figsize=(10, 4))                          # Initialise figure
    
    plt.subplot(1, 2, 1)                                 # Plot average magnetisation
    plt.plot(df['kBT'], df['M'], color='magenta')
    plt.xlabel('Temperature [J]')
    plt.ylabel('Average Total Magnetisation, M')
    plt.title('Average Magnetisation vs Temperature')
    
    plt.subplot(1, 2, 2)                                 # Plot susceptibility
    plt.plot(df['kBT'], df['chi'], 'o-', color='orange')
    plt.xlabel('Temperature [J]')
    plt.ylabel(r'Susceptibility, $\chi$')
    plt.title('Susceptibility vs Temperature')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filepath = "./task4_2.txt"
    plot_task4(filepath)