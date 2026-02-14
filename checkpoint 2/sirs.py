import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random

class SIRS:
    """Class for simulating the SIRS model on a 2D lattice"""

    def __init__(self, L, pS_I, pI_R, pR_S):
        """
        Initialise the SIRS board.
        
        Arguments:
            L: system size
            pS_I: probability of susceptible cell to become infected
            pI_R: probability of infected cell to become recovered
            pR_S: probability of recovered cell to become susceptible
        """
        self.L = L
        self.sweep = L**2
        self.pS_I = pS_I
        self.pI_R = pI_R
        self.pR_S = pR_S

        # Let -1 <- R; 0 <- S; 1 <- I
        self.board = np.random.choice([-1, 0, 1], size=(L, L))

    def animation(self):
        """
        Run and animate the SIRS model simulation.
        """
        fig, ax = plt.subplots(figsize=[10, 8])
        img = plt.imshow(self.board, cmap='bwr', vmin=-1, vmax=1)
        plt.title('SIRS Model\n' +\
            rf'$p_{{S \rightarrow I}} = {self.pS_I}$, $p_{{I \rightarrow R}} = {self.pI_R}$,' +\
            rf' $p_{{R \rightarrow S}} = {self.pR_S}$', fontsize = 16)
        cbar = plt.colorbar(img)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['R', 'S', 'I'], fontsize=16)
        plt.xticks([])
        plt.yticks([])  
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)
        plt.show()
    
    def frame(self, _):
        """
        Run 10 sweeps of the simulation and return the next frame of the animation.
        """
        # Run 10 sweeps of the updating scheme
        for i in range(10 * self.sweep):                                       
            self.update()   
        # Clear the figure
        plt.cla()      
        # Update the figure                                                         
        img = plt.imshow(self.board, cmap='bwr', vmin=-1, vmax=1)     
        plt.title('SIRS Model\n' +\
            rf'$p_{{S \rightarrow I}} = {self.pS_I}$, $p_{{I \rightarrow R}} = {self.pI_R}$,' +\
            rf' $p_{{R \rightarrow S}} = {self.pR_S}$', fontsize = 16)
        plt.xticks([])
        plt.yticks([])                                                            
        
        return img
    
    def update(self):
        """
        Apply the SIRS model updating scheme to edit a single cell.
        """
        # Choose a random cell
        i = random.randint(0, self.L-1)
        j = random.randint(0, self.L-1)
        # Check state of cell and update accordingly
        if (self.board[i, j] == -1) and np.random.binomial(1, self.pR_S):
            # Change recovered to susceptible
            self.board[i, j] = 0
        elif (self.board[i, j] == 1) and np.random.binomial(1, self.pI_R):
            # Change infected to recovered
            self.board[i, j] = -1
        elif (self.board[i, j] == 0) and np.random.binomial(1, self.pS_I):
            # Check for any infected nearest neighbours
            NN = np.array([np.roll(self.board,  1, axis=0)[i, j],
                           np.roll(self.board, -1, axis=0)[i, j],
                           np.roll(self.board,  1, axis=1)[i, j],
                           np.roll(self.board, -1, axis=1)[i, j]])
            if 1 in set(NN):
                # Change susceptible to infected
                self.board[i, j] = 1

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-p1', '--probabilitySI', type=float, default=0.2, help='Probability of susceptible becoming infected (default: 0.2)')
    parser.add_argument('-p2', '--probabilityIR', type=float, default=0.5, help='Probability of infected becoming recovered (default: 0.5)')
    parser.add_argument('-p3', '--probabilityRS', type=float, default=0.5, help='Probability of recoverd becoming susceptible (default: 0.5)')
    parser.add_argument('-s', '--state', type=str, choices=['absorbing', 'dynamic', 'cyclic'], default=None, help='Select one of three preset states (default: None)')
    args = parser.parse_args()

    if args.state == 'absorbing':
        p1, p2, p3 = 0.2, 0.5, 0.5
    elif args.state == 'dynamic':
        p1, p2, p3 = 0.5, 0.5, 0.5
    elif args.state == 'cyclic':
        p1, p2, p3 = 0.4, 0.05, 0.005
    else:
        p1, p2, p3 = args.probabilitySI, args.probabilityIR, args.probabilityRS

    sirs = SIRS(args.size, p1, p2, p3)
    sirs.animation()