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
             kBT: {real} thermal energy (J=1)
        dynamics: {str} 'G' or 'K' for Glauber/Kawasaki dynamics, respectively
        """
        self.L = L
        self.kBT = kBT
        self.sweep = L**2                                       # Define a unique unit of time for an LxL system
        self.count = 0                                          # Keep track of how many sweeps have passed
        self.M = np.empty(0)                                     # Initialise empty list to track magnetisation

        while dynamics not in {'G', 'K'}:
            dynamics = input("Please enter 'G' or 'K' ")

        self.S = np.random.choice([-1, 1], size=(L + 2, L + 2))
        self.apply_PBCs()

        if dynamics == 'G':
            self.update = self.glauber
        else:
            self.update = self.kawasaki
    
    def apply_PBCs(self):
        """apply periodic boundary conditions after updating/generating new system"""
        self.S[0, 0:-1] = self.S[-2, 0:-1] # Copy last lattice row to first ghost row
        self.S[-1, 0:-1] = self.S[1, 0:-1] # Copy first lattice row to last ghost row
        self.S[0:-1, 0] = self.S[0:-1, -2] # Copy last lattice column to first ghost column
        self.S[0:-1, -1] = self.S[0:-1, 1] # Copy first lattice column to last ghost column

    def run(self, t):
        """run ten sweeps of the simulation using Glauber or Kawasaki dynamics and update the image.
            t: {int} number of sweeps for which to run the simulation"""
        for i in range((t + 100) * self.sweep):                          
            self.update()                                         # Update the lattice
            self.apply_PBCs()                                     # Refresh ghost cells
            self.count += 1                                       # Add 10 to the sweep counter

            if self.count <= 100:
                continue
            elif self.count % 10 == 0:
                self.M = np.append(self.M, np.sum(self.S))
      
    def run_ani(self):
        """run the simulation with an animated grid. blue corresponds to S=-1 and yellow corresponds to S=+1"""
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)
        plt.show()
    
    def frame(self, t):
        """run ten sweeps of the simulation using Glauber or Kawasaki dynamics and update the image.
           t: {none} not used, required for animation"""
        plt.cla()                                                 # Clear the axis
        img = plt.imshow(self.S[1:-1, 1:-1], cmap='plasma')       # Save previous image

        for i in range(self.sweep*10):                            # Run 10 sweeps of the algorithm
            self.update()                                         # Update the lattice
            self.apply_PBCs()                                     # Refresh ghost cells
        
        self.count += 10                                          # Add 10 to the sweep counter

        return img

    def glauber(self):
        """update the system using Glauber dynamics"""
        """choose random state i"""
        i_row = random.randint(1, self.L)
        i_col = random.randint(1, self.L)

        dE = self.delta_E_G(i_row, i_col)

        if self.metropolis(dE):
            self.S[i_row, i_col] *= -1

    def kawasaki(self):
        """update the system using Kawasaki dynamics"""
        """choose random states i and j"""
        i_row = random.randint(1, self.L)
        i_col = random.randint(1, self.L)
        j_row = random.randint(1, self.L)
        j_col = random.randint(1, self.L)

        """continue choosing j state until it is distinct from the i state"""
        while ([i_row, i_col] == [j_row, j_col]) or (self.S[i_row, i_col] == self.S[j_row, j_col]):
            j_row = random.randint(1, self.L)
            j_col = random.randint(1, self.L)

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
            k_row = i_row + drow
            k_col = i_col + dcol
            I_sum -= self.S[i_row, i_col] * self.S[k_row, k_col] # Negative sign accounts for spin flip

        return -2 * I_sum

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
            k_row = i_row + drow
            k_col = i_col + dcol

            if [k_row, k_col] != [j_row, j_col]:                     # Energy change unaffected by contribution from neighbouring i and j states
                I_sum += self.S[j_row, j_col] * self.S[k_row, k_col] # Swap states i and j

            k_row = j_row + drow
            k_col = j_col + dcol

            if [k_row, k_col] != [i_row, i_col]:                     # Energy change unaffected by contribution from neighbouring i and j states
                J_sum += self.S[i_row, i_col] * self.S[k_row, k_col] # Swap states i and j

        return -2 * (I_sum + J_sum)                                  # Total energy change contribution

    def metropolis(self, dE):
        """use the Metropolis algorithm to decide whether to flip the spin state"""
        if dE <= 0:
            return True                                      # Always accept energy-lowering flip
        elif random.uniform(0, 1) < np.exp(-dE / self.kBT):  # Spin flips with probability
            return True
        else:
            return False


if __name__ == "__main__":
    #L, kBT, dynamics = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    L, kBT, dynamics = 50, 2, 'G'
    I = Ising(L, kBT, dynamics)
    #I.run(10000)
    I.run_ani