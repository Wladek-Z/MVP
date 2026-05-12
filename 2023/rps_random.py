import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mc
import argparse
import random
from numba import njit
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def update_rps(old, sweep, L, p1, p2, p3):
        """
        Apply a sweeps of the RPS update scheme.

        Arguments:
            old: current state of the board
            sweep: number of updates to perform (one sweep)
            L: system size
            p1: probability of scissors turning to rock
            p2: probability of rock turning to paper
            p3: probability of paper turning to scissors

        Returns:
            new: updated state of the board
        """
        # Loop over one sweep
        for _ in range(sweep):
            # Copy old board to new board
            new = old.copy()
            # Choose a random cell
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            # Compute list of neighbours
            up = (i - 1) % L
            down = (i + 1) % L
            left = (j - 1) % L
            right = (j + 1) % L
            neighbours = [
                old[up, j], 
                old[down, j], 
                old[i, left], 
                old[i, right], 
                old[up, left], 
                old[up, right], 
                old[down, left], 
                old[down, right]
            ]
            # Check state of cell and update accordingly
            if (old[i, j] == 0) and np.random.binomial(1, p2) and (1 in set(neighbours)):
                # Turn rock to paper
                new[i, j] = 1
            elif (old[i, j] == 1) and np.random.binomial(1, p3) and (2 in set(neighbours)):
                # Turn paper to scissors
                new[i, j] = 2
            elif (old[i, j] == 2) and np.random.binomial(1, p1) and (0 in set(neighbours)):
                # Turn rock scissors to rock
                new[i, j] = 0
            # Update old board to new board for next iteration
            old = new.copy()
            
        return new

class RockPaperScissors:
    """Class for simulating a random and sequential Rock-Paper-Scissors on a 2D lattice"""

    def __init__(self, L, p1, p2, p3, task, collect, plot):
        """
        Initialise the SIRS board.
        
        Arguments:
            L: system size
            p1: probability of scissors turning to rock
            p2: probability of rock turning to paper
            p3: probability of paper turning to scissors
            task: task for the SIRS model to perform
            collect: set True to collect data
            plot: set True to plot data
        """
        # Initialise variables
        self.L = L
        self.sweep = L**2
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        # Populate RPS board
        self.board = np.random.choice(3, (self.L, self.L))

        # Choose which task to run
        if task == 'animation':
            # Run with animation
            self.run = self.animate
        elif task == 'd':
            if collect:
                # collect data
                self.run = self.collect_d
            elif plot:
                # plot results
                self.run = self.plot_d
        elif task == 'e':
            if collect:
                # collect data
                self.run = self.collect_e
            elif plot:
                # plot results
                self.run = self.plot_e

    def animate(self):
        """
        Run the Rock-Paper-Scissors simulation with an animation.
        """
        # Create custom cmap
        colours = ['blue', 'red', 'green']
        self.cmap = mc.ListedColormap(colours)

        fig, ax = plt.subplots(figsize=[5, 4])
        ani = FuncAnimation(fig, self.update, cache_frame_data=False, interval=100)

        img = plt.imshow(self.board, cmap=self.cmap, vmin=0, vmax=2)
        plt.title("Rock Paper Scissors\n"
                  + rf"$p_1$ = {self.p1}, $p_2$ = {self.p2}, $p_3$ = {self.p3}",
                  fontsize=16)
        # Define discrete boundaries for cmap/colorbar
        boundaries = np.linspace(0, 2, 4)
        # Create colorbar
        cbar = plt.colorbar(img, ax=ax, boundaries=boundaries)
        cbar.set_ticks([1/3, 1, 5/3])
        cbar.set_ticklabels([r'$R$', r'$P$', r'$S$'], fontsize=16)

        plt.tight_layout()
        plt.show()

    def update(self, _):
        """
        Update the animation.

        Returns:
            img: figure displaying the (old) game board grid
        """
        # Clear the figure
        plt.cla()
        # Set figure to previous configuration
        img = plt.imshow(self.board, cmap=self.cmap, vmin=0, vmax=2)   
        # Add title and remove axes
        plt.title("Rock Paper Scissors\n"
                  + rf"$p_1$ = {self.p1}, $p_2$ = {self.p2}, $p_3$ = {self.p3}",
                  fontsize=16)
        plt.axis('off')
        # Create copy of game board
        old = self.board.copy()
        # Update the game board           
        self.board = update_rps(old, self.sweep, self.L, self.p1, self.p2, self.p3) 
        
        return img
    
    def collect_d(self):
        """
        Collect data pertaining to the average fraction of the minority phase and its variance
        in equilibrium over varying p3, fixed p1 and p2. Save to file.
        """
        self.p1 = 0.5
        self.p2 = 0.5
        p3_list = np.arange(0, 0.105, 0.005) 
        p3_list = np.round(p3_list, 3)

        with open('task_d.txt', 'w') as f:
            f.write("p3,frac,var\n")

            for p3 in p3_list:
                print(f"Progress: p3 = {p3}\n", end="\r")
                # Update p3 and clean game board
                self.p3 = p3
                # Populate RPS board
                self.board = np.random.choice(3, (self.L, self.L))
                # Reset minority fraction
                frac = np.zeros(10000)
                # Equilibrate
                for i in range(100):
                    old = self.board.copy()
                    self.board = update_rps(old, self.sweep, self.L, self.p1, self.p2, self.p3)
                # Run 10000 sweeps
                for i in range(10000):
                    old = self.board.copy()
                    self.board = update_rps(old, self.sweep, self.L, self.p1, self.p2, self.p3)
                    # Count faction of minority states
                    frac[i] = np.min(np.bincount(self.board.ravel(), minlength=3)) / self.sweep
                    # Print live progress
                    print(f"{i+1}/10000", end="\r")
                # Calculate variance of minority fraction
                var = self.variance(frac)
                # Append data to file
                f.write(f"{p3},{np.mean(frac)},{var}\n")


    def collect_e(self):
        """
        Collect data pertaining to the average fraction of the minority phase over varying
        p2, p3, with fixed p1. Save data to file.
        """
        self.p1 = 0.5
        p2_list = np.arange(0, 0.32, 0.02) 
        p2_list = np.round(p2_list, 2)
        p3_list = np.arange(0, 0.32, 0.02) 
        p3_list = np.round(p3_list, 2)

        with open('task_e.txt', 'w') as f:
            f.write("p2,p3,frac\n")

            for p2 in p2_list:
                # Update p2
                self.p2 = p2

                for p3 in p3_list:
                    print(f"Progress: p2 = {p2}, p3 = {p3}\n", end="\r")
                    # Update p3 and clean game board
                    self.p3 = p3
                    # Populate RPS board
                    self.board = np.random.choice(3, (self.L, self.L))
                    # Reset minority fraction
                    frac = np.zeros(1000)
                    # Equilibrate
                    for i in range(100):
                        old = self.board.copy()
                        self.board = update_rps(old, self.sweep, self.L, self.p1, self.p2, self.p3)
                    # Run 1000 sweeps
                    for i in range(1000):
                        old = self.board.copy()
                        self.board = update_rps(old, self.sweep, self.L, self.p1, self.p2, self.p3)
                        # Count faction of minority states
                        frac[i] = np.min(np.bincount(self.board.ravel(), minlength=3)) / self.sweep
                        # Print live progress
                        print(f"{i+1}/1000", end="\r")
                    # Append data to file
                    f.write(f"{p2},{p3},{np.mean(frac)}\n")
    
    def variance(self, frac):
        """
        Calculate the variance of the fraction of the minority phase
        
        Arguments:
            frac: list of minority phase fraction over time
        
        Returns:
            variance of the minority fraction
        """
        # Calculate mean, mean squared of infected sites
        mean_f = np.mean(frac)
        mean_f2 = np.mean(frac**2)
        # Return variance
        return (mean_f2 - mean_f**2) / self.L**2
    
    def plot_d(self):
        """
        Plot the average fraction of the minority phase over varying p3, then plot its variance.
        """
        p, f, var = np.loadtxt('task_d.txt', delimiter=',', skiprows=1, unpack=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[8, 8], sharex=True)

        ax1.plot(p, f)
        ax1.set_title("Average fraction and variance of the minority phase")
        ax1.set_ylabel("fraction")

        ax2.plot(p, var)
        ax2.set_ylabel("variance")
        ax2.set_xlabel(r"$p_3$")

        plt.tight_layout()
        plt.show()

    def plot_e(self):
        """
        Obtain the phase diagram of the RPS system in the p2-p3 plane, with constant 
        p1. Plot the average fraction of the minority phase as a heat map.
        """
        # Read in data
        p2, p3, frac = np.loadtxt('task_e.txt', delimiter=',', skiprows=1, unpack=True)
        # Reshape data for plotting
        n_p2 = len(set(p2))
        n_p3 = len(set(p3))
        frac_grid = frac.reshape((n_p2, n_p3))
        # Plot heat map of the average fraction of the minority for varying p2 and p3
        fig = plt.figure(figsize=[8, 6])
        plt.imshow(frac_grid, origin='lower', extent=[p3.min(), p3.max(), p2.min(), p2.max()], cmap='viridis', interpolation='none')
        plt.colorbar(label='fraction')
        plt.xlabel(r'$p_3$')
        plt.ylabel(r'$p_2$')
        plt.title(r'Phase diagram of RPS system ($p_1 = 0.5$)')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-p1', type=float, default=0.5, help='Probability of scissors turning to rock (default: 0.5)')
    parser.add_argument('-p2', type=float, default=0.5, help='Probability of rock turning to paper (default: 0.5)')
    parser.add_argument('-p3', type=float, default=0.5, help='Probability of paper turning to scissors (default: 0.5)')
    parser.add_argument('-t', '--task', type=str, default='animation', choices=['animation', 'd', 'e'], help='Select a task for the simulation (default: animation)')
    parser.add_argument('--collect', action='store_true', help='Collect data for a given task')
    parser.add_argument('--plot', action='store_true', help='Plot results for a given task')
    args = parser.parse_args()

    p1, p2, p3 = args.p1, args.p2, args.p3

    rps = RockPaperScissors(args.size, p1, p2, p3, args.task, args.collect, args.plot)
    rps.run()
