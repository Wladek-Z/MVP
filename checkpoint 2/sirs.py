import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random


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
        self.L = L
        self.sweep = L**2
        self.pS_I = pS_I
        self.pI_R = pI_R
        self.pR_S = pR_S
        self.immune = immune
        self.filename = filename

        # Let -1 <- R; 0 <- S; 1 <- I
        self.board = np.random.choice([-1, 0, 1], size=(L, L))
        # Introduce permanently immune cells
        self.immunise()
        # Choose which task to run
        if task == 'animation':
            self.run = self.animation
        elif task == '3':
            while not(self.filename):
                self.filename = input("Enter filepath to save data: ")
            self.run = self.task3
        elif task == '4':
            while not(self.filename):
                self.filename = input("Enter filepath to save data: ")
            self.run = self.task4
        elif task == '5':
            while not(self.filename):
                self.filename = input("Enter filepath to save data: ")
            self.run = self.task5


    def animation(self):
        """
        Run and animate the SIRS model simulation.
        """
        # Equilibrate board before displaying animation
        print("Wait: equilibrating")
        for i in range(100 * self.sweep):
            self.update()

        fig, ax = plt.subplots(figsize=[10, 8])
        img = plt.imshow(self.board, cmap='bwr', vmin=-1, vmax=1)
        plt.title('SIRS Model\n' +\
            rf'$p_{{S \rightarrow I}} = {self.pS_I}$, $p_{{I \rightarrow R}} = {self.pI_R}$,' +\
            rf' $p_{{R \rightarrow S}} = {self.pR_S}$, $f_{{imm}} = {self.immune}$', fontsize = 16)
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
            rf' $p_{{R \rightarrow S}} = {self.pR_S}$, $f_{{imm}} = {self.immune}$', fontsize = 16)
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
        # First check if cell is permanently immune
        while (i, j) in self.i_cells:
            # Exit function
            return
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
                    self.pR_S = pR_S
                    self.board = np.random.choice([-1, 0, 1], size=(self.L, self.L))
                    I = np.zeros(1000)
                    # Equilibrate
                    for i in range(100):
                        for j in range(self.sweep):
                            self.update()
                    # Run 1000 sweeps
                    for i in range(1000):
                        for j in range(self.sweep):
                            self.update()
                        # Count number of infected sites
                        I[i] = np.sum((self.board == 1).astype(int))
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
                self.board = np.random.choice([-1, 0, 1], size=(self.L, self.L))
                I = np.zeros(10000)
                # Equilibrate
                for i in range(100):
                    for j in range(self.sweep):
                        self.update()
                # Run 10000 sweeps
                for i in range(10000):
                    for j in range(self.sweep):
                        self.update()
                    # Count number of infected sites
                    I[i] = np.sum((self.board == 1).astype(int))
                    # Print live progress
                    print(f"Sweep {i+1}/10000", end="\r")
                # Calculate variance of infected sites
                I_var = self.I_variance(I, 0)
                # Calculate error on the variance
                I_err = self.jackknife(I, I_var)
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
                self.immune = f_imm
                self.board = np.random.choice([-1, 0, 1], size=(self.L, self.L))
                self.immunise()
                I = np.zeros(1000)
                # Equilibrate
                for i in range(100):
                    for j in range(self.sweep):
                        self.update()
                # Run 1000 sweeps
                for i in range(1000):
                    for j in range(self.sweep):
                        self.update()
                    # Count number of infected sites
                    I[i] = np.sum((self.board == 1).astype(int))
                    # Print live progress
                    print(f"Sweep {i+1}/1000", end="\r")
                # Calculate average fraction of infected sites
                I_frac = np.mean(I) / self.sweep
                # Append data to file
                f.write(f"{f_imm},{I_frac}\n")


    def jackknife(self, I, I_var):
        """
        Compute the standard error via the jackknife method.
        
        Arguments:
            I: full list of number of infected sites over time
            I_var: the true variance of the number of infected sites
            
        Returns:
            jackknife standard error on the variance
        """
        I_jack = [] 

        for i in range(len(I)):                             
            I_jack.append(np.delete(I, i)) 

        I_jack = np.array(I_jack)
        I_var_jack = self.I_variance(I_jack, 1)
        
        return np.sqrt(np.sum((I_var_jack - I_var)**2))                        
    
    def I_variance(self, I, axis):
        """
        Calculate the variance of the number of infected sites over time
        
        Arguments:
            I   : list of number of infected sites over time
            axis: axis along which means should be calculated. 0 for 1d array and 1 for 2d array (Jackknife)
        
        Returns:
            variance of the number of infected sites
        """
        # Calculate mean, mean squared of infected sites
        mean_I = np.mean(I, axis=axis)
        mean_I2 = np.mean(I**2, axis=axis)
        # Return variance
        return (mean_I2 - mean_I**2) / self.L**2
    

    def immunise(self):
        """
        Introduce permanently immune cells
        """
        self.i_cells = self.immune_cells()

        if len(self.i_cells) > 0:
            i_cells = np.array(list(self.i_cells))
            i = i_cells[:, 0]
            j = i_cells[:, 1]
            # Set immune cells to recovered
            self.board[i, j] = -1 

        
    def immune_cells(self):
        """
        Generate unique [i, j] index pairs for each cell with immunity to the infection.

        Returns:
            list of [i, j] index pairs corresponding to permanently immune sites
        """
        # Compute number of immune cells
        N = int(self.L**2 * self.immune)
        # Choose immune sites
        indices = np.random.choice(self.L**2, size=N, replace=False)
        # Create arrays of row indices, array of col indices
        i, j = np.unravel_index(indices, (self.L, self.L))
        # Create array of immune cell positions in the SIRS model lattice
        i_cells = np.c_[i, j]
        # Return as set of tuples
        if len(i_cells) > 0:
            return set(map(tuple, i_cells))
        else:
            return set()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-p1', '--probabilitySI', type=float, default=0.25, help='Probability of susceptible becoming infected (default: 0.2)')
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
