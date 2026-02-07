import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GoL:
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
            self.board = np.zeros(L, L)
            self.board[(L // 2 - 1):(L // 2 + 1), (L // 2 - 1):(L // 2 + 1)] = blinker
        else:
            glider = np.array([[0,0,1],[1,0,1],[0,1,1]])
            self.board = np.zeros(L, L)
            self.board[(L // 2 - 1):(L // 2 + 1), (L // 2 - 1):(L // 2 + 1)] = glider

    def run_ani(self):
        """
        Run the Game of Life with an animation.
        """
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.update, cache_frame_data=False)
        plt.show()

    def update(self):
        """
        Update the game board.
        """
        # Clear the figure
        plt.cla()
        # Set figure to previous configuration
        img = plt.imshow(self.board, cmap='magma', vmin=0, vmax=1)   
        # Add title and remove axes
        plt.title(f"Game of Life: {self.init}")
        plt.axis('off')
        # Create copy of game board
        old = self.board.copy()
        # Update the game board           
        self.apply_rules(old) 
        
        return img