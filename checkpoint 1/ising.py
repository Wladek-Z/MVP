import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

class Ising:
    """Class to represent a 2D Ising model"""

    def __init__(self, L, kBT, dynamics='G'):
        """
               L: {int} system size
             kBT: {float} thermal energy (J=1)
        dynamics: {str} 'G' or 'K' for Glauber/Kawasaki dynamics, respectively
        
        Note: J has been set to 1 and not included as a parameter
        """
        self.L = L
        self.kBT = kBT
        self.sweep = L * L                                       # Define a unique unit of time for an LxL system                                         # Keep track of how many sweeps have passed
        self.M = np.empty(0)                                     # Initialise empty list to track magnetisation

        while dynamics not in {'G', 'K'}:
            dynamics = input("Please enter 'G' or 'K' ")

        self.S = np.random.choice([-1, 1], size=(L, L))

        if dynamics == 'G':
            self.update = self.Glauber
        else:
            self.update = self.Kawasaki
    
    def apply_PBCs(self):
        """apply periodic boundary conditions after updating/generating new system.
        Note: currently not used as PBCs are implemented using modulo arithmetic"""
        self.S[0, 0:-1] = self.S[-2, 0:-1] # Copy last lattice row to first ghost row
        self.S[-1, 0:-1] = self.S[1, 0:-1] # Copy first lattice row to last ghost row
        self.S[0:-1, 0] = self.S[0:-1, -2] # Copy last lattice column to first ghost column
        self.S[0:-1, -1] = self.S[0:-1, 1] # Copy first lattice column to last ghost column

    def run(self, t):
        """run the simulation for t sweeps.
           t: {int} number of sweeps for which to run the simulation"""
        count = 0                                               # Keep track of how many sweeps have passed
        for i in range(t + 100):         
            for j in range(self.sweep):                         # Perform a sweep of the algorithm
                self.update()                                   # Update the lattice

            count += 1                                          # Add 1 to the sweep counter

            if count < 100:                                    # First 100 sweeps are for equilibriation
                continue
            elif count % 10 == 0:                               # Take measurements every 10 sweeps
                self.M = np.append(self.M, np.sum(self.S))
                #print(f"Progress: kBT = {self.kBT}   count = {count}/{t}", end='\r')
      
    def run_ani(self):
        """run the simulation with an animated grid. blue corresponds to S=-1 and yellow corresponds to S=+1"""
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)
        plt.show()
    
    def frame(self, t):
        """run ten sweeps of the simulation using Glauber or Kawasaki dynamics and update the image.
           t: {none} not used, required for animation"""
        plt.cla()                                                             # Clear the axis
        img = plt.imshow(self.S[1:-1, 1:-1], cmap='plasma', vmin=-1, vmax=1)  # Set fixed color scale
        plt.title(f"Ising Model: {self.update.__name__} dynamics \n kBT = {self.kBT}, L = {self.L}")
        plt.axis('off')

        for i in range(self.sweep * 10):                                      # Run 10 sweeps of the algorithm
            self.update()                                                     # Update the lattice
        
        return img

    def Glauber(self):
        """update the system using Glauber dynamics"""
        """choose random state i"""
        i_row = random.randint(0, self.L-1)
        i_col = random.randint(0, self.L-1)

        dE = self.delta_E_G(i_row, i_col)

        if self.metropolis(dE):
            self.S[i_row, i_col] *= -1

    def Kawasaki(self):
        """update the system using Kawasaki dynamics"""
        """choose random states i and j"""
        i_row = random.randint(0, self.L-1)
        i_col = random.randint(0, self.L-1)
        j_row = random.randint(0, self.L-1)
        j_col = random.randint(0, self.L-1)

        """continue choosing j state until it is distinct from the i state"""
        while ([i_row, i_col] == [j_row, j_col]) or (self.S[i_row, i_col] == self.S[j_row, j_col]):
            j_row = random.randint(0, self.L-1)
            j_col = random.randint(0, self.L-1)

        dE = self.delta_E_K(i_row, i_col, j_row, j_col)

        if self.metropolis(dE):
            self.S[i_row, i_col], self.S[j_row, j_col] = self.S[j_row, j_col], self.S[i_row, i_col]

    def delta_E_G(self, i_row, i_col):
        """calculate the energy change upon flipping spin state i in Glauber dynamics.
           i_row: {int} position of state i along first dimension
           i_col: {int} position of state i along second dimension"""
        NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]                  # Nearest neighbours in 2 dimensions
        I_sum = 0                                                # Initialise sum over pairs

        for drow, dcol in NN:                                    # Loop over nearest neighbours 'k'
            k_row = (i_row + drow) % self.L
            k_col = (i_col + dcol) % self.L
            I_sum += self.S[i_row, i_col] * self.S[k_row, k_col] # Add contribution due to pair

        return 2 * I_sum                                         # Shortcut energy change calculation

    def delta_E_K(self, i_row, i_col, j_row, j_col):
        """calculate the energy change upon switching spin states i and j in Kawasaki dynamics.
           i_row: {int} position of state i along first dimension
           i_col: {int} position of state i along second dimension
           j_row: {int} position of state j along first dimension
           j_col: {int} position of state j along second dimension"""
        NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]                      # Nearest neighbours in 2 dimensions
        I_sum = 0                                                    # Initialise sum over pairs
        J_sum = 0                                                    # Initialise sum over pairs

        for drow, dcol in NN:                                        # Loop over nearest neighbours 'k' for each state
            k_row = (i_row + drow) % self.L
            k_col = (i_col + dcol) % self.L

            if [k_row, k_col] != [j_row, j_col]:                     # Swapping neighbouring i and j has no effect
                I_sum += self.S[j_row, j_col] * self.S[k_row, k_col] # Swap states i and j

            k_row = (j_row + drow) % self.L
            k_col = (j_col + dcol) % self.L

            if [k_row, k_col] != [i_row, i_col]:                     # Swapping neighbouring i and j has no effect
                J_sum += self.S[i_row, i_col] * self.S[k_row, k_col] # Swap states i and j

        return -2 * (I_sum + J_sum)                                  # Calculate total energy change

    def metropolis(self, dE):
        """use the Metropolis algorithm to decide whether to flip the spin state"""
        if dE <= 0:
            return True                                      # Always accept energy-lowering flip
        elif random.uniform(0, 1) < np.exp(-dE / self.kBT):  # Spin flips with probability
            return True
        else:
            return False

    def avg_M(self):
        """return average magnetisation and average magnetisation squared"""
        return np.mean(self.M), np.mean(np.square(self.M))
    
    def susceptibility(self, M, M2):
        """return susceptibility.
            M: {float} expectation value of total magnetisation
           M2: {float} expectation value of total magnetisation squared"""
        return (M2 - M**2) / (self.L * self.L * self.kBT)
 

if __name__ == "__main__":
    #L, kBT, dynamics = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    L, kBT, dynamics = 50, 1, 'G'
    I = Ising(L, kBT, dynamics)
    #I.run(10000)
    I.run_ani()
    # testing
