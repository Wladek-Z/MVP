import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from numba import njit

class GameOfLife:
    """Class for simulating Conway's Game of Life on a 2D lattice"""

    def __init__(self, L, init, task, f):
        """
        Generate the initial state of the Game of Life board.
            
        Arguments:
            L   : Lattice size
            init: Initial condition of the board. Choices [random, blinker, glider] 
            task: Which task to run on the simulator
            f   : Filename for writing data, if applicable
        """
        self.L = L
        self.init = init
        self.filename = f

        # Run the simulation with an animation
        if task == 'animation':
            self.run = self.animate
        # Collect equilibration time data
        elif task == '2':
            while not(self.filename):
                self.filename = input("Enter filepath to save equilibration time data: ") 
            self.run = self.task2
        # Collect glider centre-of-mass data
        elif task == '3':
            while not(self.filename):
                self.filename = input("Enter filepath to save glider centre-of-mass time data: ") 
            self.run = self.task3

    def animate(self):
        """
        Run the Game of Life with an animation.
        """
        # Setup initial state of game board
        if self.init == 'random':
            self.board = np.random.choice([0, 1], size=(self.L, self.L))
        elif self.init == 'blinker':
            blinker = np.array([[0,0,0],[1,1,1],[0,0,0]])
            self.board = np.zeros((self.L, self.L))
            self.board[(self.L // 2 - 1):(self.L // 2 + 2), (self.L // 2 - 1):(self.L // 2 + 2)] = blinker
        else:
            glider = np.array([[0,0,1],[1,0,1],[0,1,1]])
            self.board = np.zeros((self.L, self.L))
            self.board[(self.L // 2 - 1):(self.L // 2 + 2), (self.L // 2 - 1):(self.L // 2 + 2)] = glider

        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.update, cache_frame_data=False, interval=100)
        plt.show()

    def task2(self):
        """
        Run n simulations and record how long each takes to equilibrate.
        Write the data to file.
        """
        # Set number of timesteps required to accept equilibration
        equ = 10
        # Open file for appending data
        with open(self.filename, 'a') as f:
            f.write("time\n")
            # Repeat for n simulations
            for i in range(self.n):
                # Print progress to terminal
                print(f"Simulation {i+1}/{self.n}", end='\r')
                # Setup initial state of the game board as random
                self.board = np.random.choice([0, 1], size=(self.L, self.L))
                # Initialise array to monitor the number of active sites over the 'equ' most recent timesteps
                # Equilibration is achieved when all elements of active_sites are equal
                active_sites = np.arange(-equ, 0, 1)
                # Initialise counter to count number of timesteps elapsed
                t = 0
                # Record initial state of game board
                active_sites[t % equ] = np.sum(self.board)
                # Run the GoL until it equilibrates and record the time taken
                # Max iterations set to 20009 to avoid infinite loop
                while t < 20009:
                    # Increment counter
                    t += 1
                    # Create copy of game board
                    old = self.board.copy()
                    # Update the game board           
                    self.board = self.new_board(old)
                    # Sequentially replace elements of active_sites with the current number of active sites
                    # Such that active_sites always contains data for 'equ' most recent timesteps
                    active_sites[t % equ] = np.sum(self.board)
                    # Check for equilibration (all elements equal) and break if achieved
                    if len(set(active_sites)) == 1:
                        break
                # Write the number of timesteps elapsed to file
                f.write(f"{t-9}\n")

    def task3(self):
        """
        Run the simulation and record the position of the centre of mass of a single glider over time.
        """
        # Initialise the game board with a glider
        glider = np.array([[0,0,1],[1,0,1],[0,1,1]])
        self.board = np.zeros((self.L, self.L))
        self.board[(self.L // 2 - 1):(self.L // 2 + 2), (self.L // 2 - 1):(self.L // 2 + 2)] = glider
        # Open file to write data
        with open(self.filename, 'a') as f:
            f.write("t,x,y\n")
            # Repeat for 999 timesteps
            for i in range(1, 10000):
                # Update the game board
                old = self.board.copy()
                self.board = self.new_board(old)
                # Find locations of glider elements
                one_positions = np.where(self.board == 1)
                x, y = one_positions[0], one_positions[1]
                # Discard data when glider is crossing PBCs
                if ((np.max(x) - np.min(x)) > 2) or ((np.max(y) - np.min(y)) > 2):
                    continue
                # Calculate centre of mass
                com_x, com_y = np.mean(one_positions[0]), np.mean(one_positions[1])
                # Write centre of mass to file
                f.write(f"{i},{com_x},{com_y}\n")

    def update(self, _):
        """
        Update the game board.

        Returns:
            img: figure displaying the (old) game board grid
        """
        # Clear the figure
        plt.cla()
        # Set figure to previous configuration
        img = plt.imshow(self.board, cmap='magma_r', vmin=0, vmax=1)   
        # Add title and remove axes
        plt.title(f"Game of Life: {self.init}")
        plt.axis('off')
        # Create copy of game board
        old = self.board.copy()
        # Update the game board           
        self.board = self.new_board(old) 
        
        return img
    
    
    def new_board(self, old):
        """
        Generate the new game board by applying Conway's GoL rules.
        
        Arguments:
            old: LxL array containing the old game board
        
        Returns:
            new: LxL array containing the new game board
        """
        # Compute number of alive neighbours for each cell
        up = np.roll(old, 1, axis=0)
        down = np.roll(old, -1, axis=0)

        N = (up 
             + down                        
             + np.roll(old, 1, axis=1)     
             + np.roll(old, -1, axis=1)    
             + np.roll(up, 1, axis=1)      
             + np.roll(up, -1, axis=1)     
             + np.roll(down, 1, axis=1)    
             + np.roll(down, -1, axis=1)
            )
        # Apply rules to flip dead cells to alive and make new game board
        new = (N == 3) | ((old == 1) & ((N == 2)))
        # Convert True/False array into array of 1s and 0s
        return new.astype(int)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-i', '--initialstate', type=str, choices=['random', 'blinker', 'glider'], default='random', help="Initial state of the game board (default: random)")
    parser.add_argument('-t', '--task', type=str, default='animation', choices=['animation', '2', '3'], help='Select a task for the simulation (default: animation)')
    parser.add_argument('-f', '--filename', type=str, default=None, help='Filepath to save data, if applicable (default: None)')
    args = parser.parse_args()

    GoL = GameOfLife(args.size, args.initialstate, args.task, args.filename)
    GoL.run()