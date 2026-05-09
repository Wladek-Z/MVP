import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import matplotlib.colors as mc
from numba import njit
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def apply_rules(L, old, neighbours):
    """Apply RPS rules to update the lattice.
    
    Arguments:
        L: system size
        old: old lattice
        neighbours: list of neighbours for each lattice site
    
    Returns:
        new: the updated lattice
    """
    new = old.copy()
    # Loop over lattice
    for i in range(L):
        for j in range(L):
            c0 = 0
            c1 = 0
            c2 = 0
            for k in range(8):
                if neighbours[i, j, k] == 0:
                    c0 += 1
                elif neighbours[i, j, k] == 1:
                    c1 += 1
                else:
                    c2 += 1
            if (old[i, j] == 0) and (c1 > 2):
                new[i, j] = 1
            elif (old[i, j] == 1) and (c2 > 2):
                new[i, j] = 2
            elif (old[i, j] == 2) and (c0 > 2):
                new[i, j] = 0
    
    return new

class RockPaperScissors:
    """Class for simulating Rock-Paper-Scissors on a 2D lattice"""

    def __init__(self, L, task):
        """
        Generate the initial state of the Rock-Paper-Scissors board.
            
        Arguments:
            L: Lattice size
            task: Task with which to run the simulation
        """
        self.L = L

        # Run the simulation with an animation
        if task == 'animation':
            self.run = self.animate
        elif task == 'collect':
            self.run = self.collect_data
        elif task == 'plot':
            self.run = self.plot_data
        
        self.board = np.zeros((self.L, self.L))

        # Populate board with three equal 'pie wedges'
        wedge = 2 * np.pi / 3

        for i in range(self.L):
            for j in range(self.L):
                dy = i - self.L // 2
                dx = j - self.L // 2
                angle = np.arctan2(dy, dx)
                if angle < 0:
                    angle += 2 * np.pi
                sector = int(angle // wedge)
                self.board[i, j] = sector  # 0, 1, or 2

    def animate(self):
        """
        Run the Rock-Paper-Scissors simulation with an animation.
        """
        # Create custom cmap
        colours = ['blue', 'red', 'green']
        self.cmap = mc.ListedColormap(colours)

        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.update, cache_frame_data=False, interval=100)

        img = plt.imshow(self.board, cmap=self.cmap, vmin=0, vmax=2)
        plt.title("Rock Paper Scissors")
        # Define discrete boundaries for cmap/colorbar
        boundaries = np.linspace(0, 2, 4)
        # Create colorbar
        cbar = plt.colorbar(img, ax=ax, boundaries=boundaries)
        cbar.set_ticks([1/3, 1, 5/3])
        cbar.set_ticklabels([r'$R$', r'$P$', r'$S$'], fontsize=16)

        plt.show()

    def update(self, _):
        """
        Update the game board.

        Returns:
            img: figure displaying the (old) game board grid
        """
        # Clear the figure
        plt.cla()
        # Set figure to previous configuration
        img = plt.imshow(self.board, cmap=self.cmap, vmin=0, vmax=2)   
        # Add title and remove axes
        plt.title("Rock Paper Scissors")
        plt.axis('off')
        # Create copy of game board
        old = self.board.copy()
        # Update the game board           
        self.board = self.new_board(old) 
        
        return img
    
    def new_board(self, old):
        """
        Generate the new game board by applying RPS rules.
        
        Arguments:
            old: LxL array containing the old board
        
        Returns:
            new: LxL array containing the new board
        """
        # Compute list of neighbours for each cell
        up = np.roll(old, 1, axis=0)
        down = np.roll(old, -1, axis=0)

        shifts = [
            up, 
            down,                        
            np.roll(old, 1, axis=1),     
            np.roll(old, -1, axis=1),    
            np.roll(up, 1, axis=1),      
            np.roll(up, -1, axis=1),
            np.roll(down, 1, axis=1),    
            np.roll(down, -1, axis=1)
        ]

        neighbours = np.stack(shifts, axis=-1)
        # Apply RPS rules
        new = apply_rules(self.L, old, neighbours)
        return new
    
    def collect_data(self):
        """
        Record the number of R states at a fixed point over time and save to R_states.txt.
        """
        with open("R_states.txt", "w") as f:
            f.write("t,R\n")

            for i in range(1000):
                # Create copy of game board
                old = self.board.copy()
                # Record value of R in the corner
                if old[0, 0] == 0:
                    f.write(f"{i},1\n")
                else:
                    f.write(f"{i},0\n")
                # Update the game board           
                self.board = self.new_board(old) 

    def plot_data(self):
        """
        Plot the data for task b.
        """
        t, R = np.loadtxt('R_states.txt', delimiter=',', skiprows=1, unpack=True)

        fig = plt.figure(figsize=(8, 6))

        plt.plot(t, R)
        plt.title("Number of R states at a fixed point over time")
        plt.xlabel("timestep")
        plt.ylabel("number of R states")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=100, help='System size (default: 100)')
    parser.add_argument('-t', '--task', type=str, default='animation', choices=['animation', 'collect', 'plot'], help='Select a task for the simulation (default: animation)')
    args = parser.parse_args()

    RPS = RockPaperScissors(args.size, args.task)
    RPS.run()