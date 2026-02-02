import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

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
        self.sweep = L * L                                    # Define a unique unit of time for an LxL system                                         # Keep track of how many sweeps have passed
        self.M = np.empty(0)                                  # Initialise empty list for magnetisation measurements
        self.E = np.empty(0)                                  # Initialise empty list for energy measurements

        while dynamics not in {'G', 'K'}:
            dynamics = input("Please enter 'G' or 'K' ")

        self.S = np.random.choice([-1, 1], size=(L, L))       # Initialise random spin configuration
        self.E_now = self.total_E()                           # Record initial energy; E_now = "what is the current energy?"

        if dynamics == 'G':
            self.update = self.Glauber
        else:
            self.update = self.Kawasaki
  
    def run(self, t):
        """run the simulation for t sweeps.
           t: {int} number of sweeps for which to run the simulation"""
        for i in range(100):                                        # Equilibriate the system
            for j in range(self.sweep):                             # Perform a sweep of the algorithm
                self.update()                                       # Update the lattice
                                              
        for i in range(1, t + 1):                                   # Run the simulation for t sweeps
            for j in range(self.sweep):                             # Perform a sweep of the algorithm
                self.update()                                       # Update the lattice
            
            if i % 10 == 0:                                         # Take measurements every 10 sweeps
                S_sum = np.sum(self.S)                              # Calculate current magnetisation                                    
                self.M = np.append(self.M, S_sum)                   # Record current magnetisation         
                self.E = np.append(self.E, self.E_now)              # Record current energy 
                
    def run_ani(self):
        """run the simulation with an animated grid. blue corresponds to S=-1 and yellow corresponds to S=+1"""
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)
        plt.show()
    
    def frame(self, t):
        """run ten sweeps of the simulation using Glauber or Kawasaki dynamics and update the image.
           t: {none} not used, required for animation"""
        plt.cla()                                                                                     # Clear the axis
        img = plt.imshow(self.S[1:-1, 1:-1], cmap='plasma', vmin=-1, vmax=1)                          # Set fixed color scale
        plt.title(f"Ising Model: {self.update.__name__} dynamics \n kBT = {self.kBT}, L = {self.L}")
        plt.axis('off')

        for i in range(10):                                                                           # Run 10 sweeps of the algorithm
            for j in range(self.sweep):                                       
                self.update()                                                                         # Update the lattice
        
        return img

    def Glauber(self):
        """update the system using Glauber dynamics"""
        """choose random state i"""
        i_row = random.randint(0, self.L-1)
        i_col = random.randint(0, self.L-1)

        dE = self.delta_E_G(i_row, i_col)

        if self.metropolis(dE):
            self.S[i_row, i_col] *= -1
            self.E_now = self.E_now + dE        # Update total energy whilst avoiding total recalculation

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
            self.E_now = self.E_now + dE         # Update total energy while avoiding recalculation

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
        NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]                      # Nearest neighbours
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

    def avg_M(self, M):
        """return average magnetisation and average magnetisation squared.
           M: {arr} magnetisation measurements array"""
        return np.mean(M), np.mean(np.square(M))
    
    def susceptibility(self, M, M2):
        """return susceptibility.
            M: {float} expectation value of total magnetisation
           M2: {float} expectation value of total magnetisation squared"""
        return (M2 - M**2) / (self.L * self.L * self.kBT)
    
    def total_E(self):
        """calculate the total energy of the system"""
        E_sum = 0                                                 # Initialise energy sum
        NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]                   # Nearest neighbours

        for i in range(self.L):                                   # Loop over all spins                       
            for j in range(self.L):
                for drow, dcol in NN:                             # Loop over nearest neighbours 'k'
                    k_row = (i + drow) % self.L
                    k_col = (j + dcol) % self.L
                    E_sum += -self.S[i, j] * self.S[k_row, k_col] # Add contribution due to pair

        return E_sum / 2                                          # Avoid double counting
    
    def avg_E(self, E):
        """return average energy and average energy squared.
           E: {arr} energy measurements array"""
        return np.mean(E), np.mean(np.square(E))
    
    def heat_capacity(self, E, E2):
        """return heat capacity.
            E: {float} expectation value of total energy
           E2: {float} expectation value of total energy squared"""
        return (E2 - E**2) / (self.L * self.L * self.kBT * self.kBT)   
    
    def jackknife(self, C):
        """compute the standard error on the heat capacity via the jackknife method.
           C: {float} measured heat capacity per spin"""
        n = len(self.E)                                                # Number of data points
        C_i = np.empty(0)                                              # Initialise empty array for jackknife samples

        for i in range(n):                                             # Loop over all data points
            E_jack = np.delete(self.E, i)                              # Sample distribution with i-th element removed
            E, E2 = self.avg_E(E_jack)                                 # Calculate average energy and average energy squared
            C_jack = self.heat_capacity(E, E2)                         # Calculate heat capacity for jackknife sample
            C_i = np.append(C_i, C_jack)                               # append heat capacity for jackknife sample

        return np.sqrt(np.sum((C_i - C)**2))                           # Jackknife standard error calculation
            

 

if __name__ == "__main__":
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-T', '--temperature', type=float, default=2, help='Thermal energy (default: 2)')
    parser.add_argument('-d', '--dynamics', type=str, choices=['G', 'K'], default='G', help="Dynamics type: 'G' for Glauber, 'K' for Kawasaki (default: 'G')")
    args = parser.parse_args()
    I = Ising(args.size, args.temperature, args.dynamics)
    I.run_ani()         # Run the simulation with an animated grid

