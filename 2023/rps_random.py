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
def update_rps(old, sweep, L, p0, p1, p2):
        """
        Apply a sweeps of the RPS update scheme.

        Arguments:
            old: current state of the board
            sweep: number of updates to perform (one sweep)
            L: system size
            p0: probability of scissors turning to rock
            p1: probability of rock turning to paper
            p2: probability of paper turning to scissors

        Returns:
            new: updated state of the board
        """
        # Loop over one sweep
        for _ in range(sweep):
            # Copy old board to new board
            new = old.copy()
            # Choose a random cell
            i = random.randint(0, L-1)
            j = random.randint(0, L-1)
            # Compute list of neighbours
            up = i - 1 % L
            down = i + 1 % L
            left = j - 1 % L
            right = j + 1 % L
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
            if (old[i, j] == 0) and np.random.binomial(1, p1) and (1 in set(neighbours)):
                # Turn rock to paper
                new[i, j] = 1
            elif (old[i, j] == 1) and np.random.binomial(1, p2) and (2 in set(neighbours)):
                # Turn paper to scissors
                new[i, j] = 2
            elif (old[i, j] == 2) and np.random.binomial(1, p0) and (0 in set(neighbours)):
                # Turn rock scissors to rock
                new[i, j] = 0
            # Update old board to new board for next iteration
            old = new.copy()
            
        return new

class RockPaperScissors:
    """Class for simulating a random and sequential Rock-Paper-Scissors on a 2D lattice"""

    def __init__(self, L, p0, p1, p2, task):
        """
        Initialise the SIRS board.
        
        Arguments:
            L: system size
            p0: probability of scissors turning to rock
            p1: probability of rock turning to paper
            p2: probability of paper turning to scissors
            task: task for the SIRS model to perform
            immune: fraction of the population with permanent immunity
            filename: filepath for writing data, if applicable
        """
        # Initialise variables
        self.L = L
        self.sweep = L**2
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        # Populate RPS board
        self.board = np.random.choice(3, (self.L, self.L))

        # Choose which task to run
        if task == 'animation':
            # Run with animation
            self.run = self.animate

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
                  + rf"$p_0$ = {self.p0}, $p_1$ = {self.p1}, $p_2$ = {self.p2}",
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
                  + rf"$p_0$ = {self.p0}, $p_1$ = {self.p1}, $p_2$ = {self.p2}",
                  fontsize=16)
        plt.axis('off')
        # Create copy of game board
        old = self.board.copy()
        # Update the game board           
        self.board = update_rps(old, self.sweep, self.L, self.p0, self.p1, self.p2) 
        
        return img
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-p0', type=float, default=0.5, help='Probability of scissors turning to rock (default: 0.5)')
    parser.add_argument('-p1', type=float, default=0.5, help='Probability of rock turning to paper (default: 0.5)')
    parser.add_argument('-p2', type=float, default=0.5, help='Probability of paper turning to scissors (default: 0.5)')
    parser.add_argument('-t', '--task', type=str, default='animation', choices=['animation'], help='Select a task for the simulation (default: animation)')
    args = parser.parse_args()

    p0, p1, p2 = args.p0, args.p1, args.p2

    rps = RockPaperScissors(args.size, p0, p1, p2, args.task)
    rps.run()
