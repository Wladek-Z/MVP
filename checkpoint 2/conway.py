import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

class GameOfLife:
    """Class for simulating Conway's Game of Life on a 2D lattice"""

    def __init__(self, L, init):
        """
        Generate the initial state of the Game of Life board.
            
        Arguments:
            L   : Lattice size
            init: Initial condition of the board. Choices [random, blinker, glider] 
        """
        self.L = L

        # Enforce legal inputs
        while init not in {'random', 'blinker', 'glider'}:
            init = input("Please choose from {'random', 'blinker', 'glider'}: ")

        self.init = init

        # Setup initial state of game board
        if init == 'random':
            self.board = np.random.choice([0, 1], size=(L, L))
        elif init == 'blinker':
            blinker = np.array([[0,0,0],[1,1,1],[0,0,0]])
            self.board = np.zeros((L, L))
            self.board[(L // 2 - 1):(L // 2 + 2), (L // 2 - 1):(L // 2 + 2)] = blinker
        else:
            glider = np.array([[0,0,1],[1,0,1],[0,1,1]])
            self.board = np.zeros((L, L))
            self.board[(L // 2 - 1):(L // 2 + 2), (L // 2 - 1):(L // 2 + 2)] = glider

    def play(self):
        """
        Run the Game of Life with an animation.
        """
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.update, cache_frame_data=False, interval=100)
        plt.show()

    def update(self, frame):
        """
        Update the game board.

        Arguments:
            frame: required for FuncAnimation to work properly (not used)

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
        new = ((old == 1) & ((N == 2))) | (N == 3)
        # Convert True/False array into array of 1s and 0s
        return new.astype(int)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-i', '--initialstate', type=str, choices=['random', 'blinker', 'glider'], default='random', help="Initial state of the game board (default: 'random')")
    args = parser.parse_args()
    GoL = GameOfLife(args.size, args.initialstate)
    GoL.play()