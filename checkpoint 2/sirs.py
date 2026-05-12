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
def jackknife(I, I_var, L):
        """
        Compute the standard error via the jackknife method.
        
        Arguments:
            I: full list of number of infected sites over time
            I_var: the true variance of the number of infected sites
            L: system size
            
        Returns:
            jackknife standard error on the variance
        """
        I_jack = np.empty(len(I)-1)
        I_var_jack = np.empty(len(I))


        for i in range(len(I)):  
            for j in range(len(I)-1):                           
                if j < i:
                    I_jack[j] = I[j]
                else:
                    I_jack[j] = I[j+1]

            I_var_jack[i] = (np.mean(I_jack**2) - np.mean(I_jack)**2) / L**2
        
        return np.sqrt(np.sum((I_var_jack - I_var)**2))    

@njit
def update(old, sweep, L, pS_I, pI_R, pR_S):
        """
        Apply a sweep of the SIRS model update scheme.

        Arguments:
            old: current state of the board
            sweep: number of updates to perform (one sweep)
            L: system size
            pS_I: probability of susceptible cell to become infected
            pI_R: probability of infected cell to become recovered
            pR_S: probability of recovered cell to become susceptible

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
            # Check state of cell and update accordingly
            if (old[i, j] == -1) and np.random.binomial(1, pR_S):
                # Change recovered to susceptible
                new[i, j] = 1
            elif (old[i, j] == 0) and np.random.binomial(1, pI_R):
                # Change infected to recovered
                new[i, j] = -1
            elif (old[i, j] == 1) and np.random.binomial(1, pS_I):
                # Check for any infected nearest neighbours
                NN = np.array([old[i, (j+1) % L], 
                            old[i, (j-1) % L], 
                            old[(i+1) % L, j], 
                            old[(i-1) % L, j]])
                if 0 in set(NN):
                    # Change susceptible to infected
                    new[i, j] = 0
            # Update old board to new board for next iteration
            old = new.copy()
            
        return new

class SIRS:
    """Class for simulating the SIRS model on a 2D lattice"""

    def __init__(self, L, pS_I, pI_R, pR_S, task, immune, filename):
        """
        Initialise the SIRS board.
        
        Arguments:
            L: system size
            pS_I: probability of susceptible cell to become infected
            pI_R: probability of infected cell to become recovered
            pR_S: probability of recovered cell to become susceptible
            task: task for the SIRS model to perform
            immune: fraction of the population with permanent immunity
            filename: filepath for writing data, if applicable
        """
        # Initialise variables
        self.L = L
        self.sweep = L**2
        self.pS_I = pS_I
        self.pI_R = pI_R
        self.pR_S = pR_S
        self.immune = immune
        self.filename = filename
        # Populate SIRS board
        self.populate(immune)
        # Choose which task to run
        if task == 'animation':
            # Run with animation
            self.run = self.animation
        elif task == '3':
            # Collect task 3 data
            while not(self.filename):
                self.filename = input("Enter filepath to save data: ")
            self.run = self.task3
        elif task == '4':
            # Collect task 4 data
            while not(self.filename):
                self.filename = input("Enter filepath to save data: ")
            self.run = self.task4
        elif task == '5':
            # Collect task 5 data
            while not(self.filename):
                self.filename = input("Enter filepath to save data: ")
            self.run = self.task5

    def populate(self, immune):
        """
        Populate board with susceptible, recovered, infected, and immune
        (if applicable) cells. Let -1 <- R; 0 <- I; 1 <- S; -2 <- immune.

        Arguments: 
            immune: fraction of permanently immune sites
        """
        # Create 1D array of length L**2 with random cells within set (-1, 0, 1)
        arr = np.random.choice([-1, 0, 1], self.L**2)
        # Insert immune sites from fraction, rounded down
        num_immune = int(immune * self.L**2)
        arr[:num_immune] = -2
        # Shuffle array
        random.shuffle(arr)
        # Reshape 1D array into 2D SIRS board
        self.board = arr.reshape(self.L, self.L)

    def animation(self):
        """
        Run and animate the SIRS model simulation.
        """
        # Create custom cmap
        colours = ['black', 'blue', 'red', 'white']
        self.cmap = mc.ListedColormap(colours)
        # Create figure and image
        fig, ax = plt.subplots(figsize=[6, 6])
        img = plt.imshow(self.board, cmap=self.cmap, vmin=-2, vmax=1)
        plt.title('SIRS Model\n' +\
            rf'$p_{{S \rightarrow I}} = {self.pS_I}$, $p_{{I \rightarrow R}} = {self.pI_R}$, $p_{{R \rightarrow S}} = {self.pR_S}$' + '\n' +\
            f'$f_{{imm}} = {self.immune}$', fontsize = 16)
        # Define discrete boundaries for cmap/colorbar
        boundaries = np.linspace(-2, 1, 5)
        # Add custom colour bar
        cbar = plt.colorbar(img, ax=ax, boundaries=boundaries)
        cbar.set_ticks([-1.625, -0.875, -0.125, 0.625])
        cbar.set_ticklabels([r'$immune$', r'$R$', r'$I$', r'$S$'], fontsize=16)

        plt.xticks([])
        plt.yticks([])  
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)

        plt.show()
    
    def frame(self, _):
        """
        Run 10 sweeps of the simulation and return the next frame of the animation.
        """
        # Run 10 sweeps of the updating scheme
        for i in range(10):                                       
            old = self.board.copy()
            self.board = update(old, self.sweep, self.L, self.pS_I, self.pI_R, self.pR_S) 
        # Clear the figure
        plt.cla()      
        # Update the figure                                                         
        img = plt.imshow(self.board, cmap=self.cmap, vmin=-2, vmax=1)     
        plt.title('SIRS Model\n' +\
            rf'$p_{{S \rightarrow I}} = {self.pS_I}$, $p_{{I \rightarrow R}} = {self.pI_R}$, $p_{{R \rightarrow S}} = {self.pR_S}$' + '\n' +\
            f'$f_{{imm}} = {self.immune}$', fontsize = 16)
        plt.xticks([])
        plt.yticks([]) 
        # Return image of board for animation
        return img
    
    def task3(self):
        """
        Collect data pertaining to the average fraction of infected sites
        over varying pS_I and pR_S, with constant pI_R. Save to file.
        """
        self.pI_R = 0.5
        p_list = np.arange(0, 1.05, 0.05)
        p_list = np.round(p_list, 2)

        with open(self.filename, 'a') as f:
            f.write("pS_I,pR_S,I_frac\n")

            for pS_I in p_list:
                # Update pS_I
                self.pS_I = pS_I

                for pR_S in p_list:
                    print(f"Progress: pS_I = {pS_I}, pR_S = {pR_S}")
                    # Initialise variables for new SIRS run
                    # Update pR_S
                    self.pR_S = pR_S
                    # More efficient board population for 0 immunity case
                    self.board = np.random.choice([-1, 0, 1], size=(self.L, self.L))
                    I = np.zeros(1000)
                    # Equilibrate
                    for i in range(100):
                        old = self.board.copy()
                        self.board = update(old, self.sweep, self.L, self.pS_I, self.pI_R, self.pR_S)
                    # Run 1000 sweeps
                    for i in range(1000):
                        old = self.board.copy()
                        self.board = update(old, self.sweep, self.L, self.pS_I, self.pI_R, self.pR_S)
                        # Count number of infected sites
                        I[i] = np.sum((self.board == 0).astype(int))
                    # Calculate average fraction of infected sites
                    I_frac = np.mean(I) / self.sweep
                    # Append data to file
                    f.write(f"{pS_I},{pR_S},{I_frac}\n")

    def task4(self):
        """
        Collect data pertaining to the variance of the fraction of infected sites
        over varying pS_I, with constant pI_R and pR_S. Compute errors and save to file.
        """
        self.pI_R = 0.5
        self.pR_S = 0.5
        p_list = np.arange(0.2, 0.51, 0.01) 
        p_list = np.round(p_list, 2)

        with open(self.filename, 'a') as f:
            f.write("pS_I,I_var,I_err\n")

            for pS_I in p_list:
                print(f"Progress: pS_I = {pS_I}\n", end="\r")
                # Update pS_I and clean game board
                self.pS_I = pS_I
                # More efficient board population for 0 immunity case
                self.board = np.random.choice([-1, 0, 1], size=(self.L, self.L))
                I = np.zeros(10000)
                # Equilibrate
                for i in range(100):
                    old = self.board.copy()
                    self.board = update(old, self.sweep, self.L, self.pS_I, self.pI_R, self.pR_S)
                # Run 10000 sweeps
                for i in range(10000):
                    old = self.board.copy()
                    self.board = update(old, self.sweep, self.L, self.pS_I, self.pI_R, self.pR_S)
                    # Count number of infected sites
                    I[i] = np.sum((self.board == 0).astype(int))
                    # Print live progress
                    print(f"Sweep {i+1}/10000", end="\r")
                # Calculate variance of infected sites
                I_var = self.I_variance(I)
                # Calculate error on the variance
                I_err = jackknife(I, I_var, self.L)
                # Append data to file
                f.write(f"{pS_I},{I_var},{I_err}\n")

    def task5(self):
        """
        Collect data pertaining to the fraction of immunity required to prevent the
        infection from spreading at pS_I = pI_R = pR_S = 0.5.
        """
        self.pI_R = 0.5
        self.pR_S = 0.5
        self.pS_I = 0.5
        f_list = np.arange(0, 1.02, 0.02)
        f_list = np.round(f_list, 2)

        with open(self.filename, 'a') as f:
            f.write("f_imm,I_frac\n")

            for f_imm in f_list:
                print(f"Progress: f = {f_imm}\n", end="\r")
                # Update immune fraction and clean game board
                self.populate(f_imm)
                I = np.zeros(1000)
                # Equilibrate
                for i in range(100):
                    old = self.board.copy()
                    self.board = update(old, self.sweep, self.L, self.pS_I, self.pI_R, self.pR_S)
                # Run 1000 sweeps
                for i in range(1000):
                    old = self.board.copy()
                    self.board = update(old, self.sweep, self.L, self.pS_I, self.pI_R, self.pR_S)
                    # Count number of infected sites
                    I[i] = np.sum((self.board == 0).astype(int))
                    # Print live progress
                    print(f"Sweep {i+1}/1000", end="\r")
                # Calculate average fraction of infected sites
                I_frac = np.mean(I) / self.sweep
                # Append data to file
                print(f"Progress: I = {I_frac}")
                f.write(f"{f_imm},{I_frac}\n")

    def I_variance(self, I):
        """
        Calculate the variance of the number of infected sites over time
        
        Arguments:
            I: list of number of infected sites over time
        
        Returns:
            variance of the number of infected sites
        """
        # Calculate mean, mean squared of infected sites
        mean_I = np.mean(I)
        mean_I2 = np.mean(I**2)
        # Return variance
        return (mean_I2 - mean_I**2) / self.L**2

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-p1', '--probabilitySI', type=float, default=0.5, help='Probability of susceptible becoming infected (default: 0.5)')
    parser.add_argument('-p2', '--probabilityIR', type=float, default=0.5, help='Probability of infected becoming recovered (default: 0.5)')
    parser.add_argument('-p3', '--probabilityRS', type=float, default=0.5, help='Probability of recoverd becoming susceptible (default: 0.5)')
    parser.add_argument('-s', '--state', type=str, choices=['absorbing', 'dynamic', 'cyclic'], default=None, help='Select one of three preset states (default: None)')
    parser.add_argument('-t', '--task', type=str, default='animation', choices=['animation', '3', '4', '5'], help='Select a task for the simulation (default: animation)')
    parser.add_argument('-i', '--immune', type=float, default=0, help='Choose fraction of the population with permanent immunity to the infection (default: 0)')
    parser.add_argument('-f', '--filename', type=str, default=None, help='Filepath to save data, if applicable (default: None)')
    args = parser.parse_args()

    if args.state == 'absorbing':
        p1, p2, p3 = 0.25, 0.5, 0.5
    elif args.state == 'dynamic':
        p1, p2, p3 = 0.5, 0.5, 0.5
    elif args.state == 'cyclic':
        p1, p2, p3 = 0.4, 0.05, 0.0056
    else:
        p1, p2, p3 = args.probabilitySI, args.probabilityIR, args.probabilityRS

    sirs = SIRS(args.size, p1, p2, p3, args.task, args.immune, args.filename)
    sirs.run()
