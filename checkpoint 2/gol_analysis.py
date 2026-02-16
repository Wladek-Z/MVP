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
        data = pd.read_csv(filename)['time'].values
        # Plot histogram of equilibration times, discarding outliers above 4000 timesteps
        plt.hist(data, bins=40, color='orangered', range=[0, 4000], label=r'$p = 0.5$')
        plt.title("Distribution of Equilibration Times")
        plt.xlabel("Equilibration Timesteps")
        plt.ylabel("Occurrences")
        plt.legend()
        plt.show()

def glider_speed(filename):
        """
        Analyse the speed of the glider. Plot glider distance from origin against timestep.
        Use curve fit to find gradient for multiple segments of motion, then plot histogram of speed distribution. 
        Display the mean speed, and the standard error on the mean.
        
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
        # Plot distance from origin vs time
        upper_limit = 500
        upper_t = t[upper_limit]
        plt.plot(t[0:upper_limit], d[0:upper_limit], color='orangered')
        plt.title(f"Glider Distance from Origin vs Time\n (first {upper_t} timesteps)")
        plt.xlabel("Time [timesteps]")
        plt.ylabel("Distance from Origin [cells]")
        plt.show()
        # Define a linear function for curve fitting
        linear_func = lambda t, m, c: m * t + c
        # Split the data for crossing of periodic boundary conditions
        splits = np.where(np.diff(t) > 1)[0] + 1
        t_split = np.split(t, splits)
        d_split = np.split(d, splits)
        # Fit a linear function to each segment of motion and extract the gradient (speed)
        speeds = []
        for times, distances in zip(t_split, d_split):
            popt, _ = curve_fit(linear_func, times, distances)
            # Divide by sqrt(2) to get speed in cells/timestep
            speeds.append(popt[0]/np.sqrt(2))
        # Calculate mean and standard error of the mean for the speeds
        mean_speed = np.mean(speeds)
        sem_speed = np.std(speeds, ddof=1) / np.sqrt(len(speeds))
        # Print results
        for i, speed in enumerate(speeds):
            print(f'Speed {i+1} = {speed:.3f} cells/timestep')

        print(f'Mean speed = {mean_speed:.3f} \u00B1 {sem_speed:.1f} cells/timestep')
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--analyse", type=str, choices=['equ', 'com'], default=None, help="Select which analysis to perform (default: None)")
    parser.add_argument("-f", "--filename", type=str, default=None, help="Name of file containing data to analyse (default: None)")
    args = parser.parse_args()
    
    if args.analyse == 'equ':
        histogram(args.filename)
    elif args.analyse == 'com':
         glider_speed(args.filename)