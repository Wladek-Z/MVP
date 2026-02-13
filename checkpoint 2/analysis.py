import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.optimize import curve_fit

def histogram(filename):
        """
        Plot a histogram of equilibration times from data in file.

        Arguments:
            filename: Name of file containing equilibration time data
        """
        # Read equilibration time data from file
        data = pd.read_csv(filename, header=None)
        # Plot histogram of equilibration times, discarding outliers above 4000 timesteps
        plt.hist(data, bins=40, color='orangered', range=[0, 4000])
        plt.xlabel("Equilibration Timesteps")
        plt.ylabel("Occurrences")
        plt.show()

def glider_speed(filename):
        """
        Analyse the speed of the glider. Use curve fit to find gradient for multiple segments of motion,
        then plot histogram of speed distribution. Display the individual speeds, the mean, and the standard error on the mean.
        
        Arguments:
            filename: Name of file containing glider position data
        """
        # Read glider position data from file
        data = pd.read_csv(filename)
        t = data['t'].values
        x = data['x'].values
        y = data['y'].values
        # Combine x and y position data to get total distance travelled at each time step
        d = np.sqrt(x**2 + y**2)
        # Define a linear function for curve fitting
        linear_func = lambda t, m, c: m * t + c
        # Split the data for crossing of periodic boundary conditions
        splits = np.where(np.diff(t) > 1)[0] + 1
        t_split = np.split(t, splits)
        d_split = np.split(d, splits)
        # Fit a linear function to each segment of motion and extract the gradient (speed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--analyse", type=str, choices=['equ', 'com'], default=None, help="Select which analysis to perform (default: None)")
    parser.add_argument("-f", "--filename", type=str, default=None, help="Name of file containing data to analyse (default: None)")
    args = parser.parse_args()
    
    if args.analyse == 'equ':
        histogram(args.filename)
    elif args.analyse == 'com':
         glider_speed(args.filename)