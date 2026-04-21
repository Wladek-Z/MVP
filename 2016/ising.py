import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from numba import njit

@njit
def delta_E_G(i_row, i_col, L, S, J):
    """
    Calculate the energy change upon flipping spin state i in Glauber dynamics.

    Arguments:
        i_row: position of state i along first dimension
        i_col: position of state i along second dimension
        L: system size
        S: spin lattice
        J: coupling constant
        h: external magnetic field

    Returns:
        energy change upon flipping spin state i
    """
    # Nearest neighbours in 2 dimensions
    NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]       
    # Initialise sum over pairs           
    I_sum = 0                                                
    # Loop over nearest neighbours 'k'
    for drow, dcol in NN:                                    
        k_row = (i_row + drow) % L
        k_col = (i_col + dcol) % L
        # Add contribution due to pair
        I_sum += S[i_row, i_col] * S[k_row, k_col] 
    # Shortcut energy change calculation
    return 2 * J * I_sum                           

@njit
def delta_E_K(i_row, i_col, j_row, j_col, L, S, J):
    """
    Calculate the energy change upon switching spin states i and j in Kawasaki dynamics.

    Arguments:
        i_row: position of state i along first dimension
        i_col: position of state i along second dimension
        j_row: position of state j along first dimension
        j_col: position of state j along second dimension
        L: system size
        S: spin lattice
        J: coupling constant

    Returns:
        energy change upon switching spin states i and j
        """
    # Nearest neighbours
    NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Initialise sum over pairs                      
    I_sum = 0                                                    
    J_sum = 0                                                   
    # Loop over nearest neighbours 'k' for each state
    for drow, dcol in NN:                                        
        k_row = (i_row + drow) % L
        k_col = (i_col + dcol) % L
        # Swapping neighbouring i and j has no effect
        if [k_row, k_col] != [j_row, j_col]:            
            # Compute contribution due to pair             
            I_sum += S[j_row, j_col] * S[k_row, k_col] 

        k_row = (j_row + drow) % L
        k_col = (j_col + dcol) % L
        # Swapping neighbouring i and j has no effect
        if [k_row, k_col] != [i_row, i_col]:                     
            # Compute contribution due to pair
            J_sum += S[i_row, i_col] * S[k_row, k_col]
    # Calculate total energy change
    return -2 * J * (I_sum + J_sum)   

@njit
def magnetisation(S, L):
        """
        Calculate and return the magnetisation and staggered magnetisation.

        Arguments:
            S: spin lattice
            L: system size 

        Returns:
            magnetisation, staggered magnetisation
        """
        # Initialise magnetisation and staggered magnetisation sums
        M = 0
        M_s = 0

        for i in range(L):
            for j in range(L):
                M += S[i, j]
                M_s += S[i, j] * (-1)**(i + j)

        return M, M_s

class Ising:
    """Class to represent a 2D Ising model"""

    def __init__(self, L, kBT, J, h):
        """
        Constructor method for Ising class.

        Arguments:
            L: system size
            kBT: thermal energy (J=1)
            J:  coupling constant
            h: external magnetic field
        """
        self.L = L
        self.kBT = kBT
        # Modify coupling constant for different equilibrium states
        self.J = J              
        # Define external magnetic field
        self.h = h
        # Define a unique unit of time for an LxL system 
        # Keep track of how many sweeps have passed
        self.sweep = L * L                                               
        # Initialise empty list for magnetisation measurements                            
        self.M = np.empty(0)                           
        # Initialise empty list for energy measurements       
        self.E = np.empty(0)                       
        # Initialise random spin configuration
        self.S = np.random.choice([-1, 1], size=(L, L))       
        # Record initial energy; E_now = "what is the current energy?"
        self.E_now = self.total_E()                           

        self.update = self.Glauber
  
    def run(self, t):
        """
        Run the simulation for t sweeps.

        Arguments:
           t: number of sweeps for which to run the simulation
        """
        # Equilibriate the system
        for i in range(100):                           
            # Perform a sweep of the algorithm             
            for j in range(self.sweep):                         
                # Update the lattice    
                self.update()                                       
                                              
        # Run the simulation for t sweeps
        for i in range(1, t + 1):                       
            # Perform a sweep of the algorithm            
            for j in range(self.sweep):                       
                # Update the lattice      
                self.update()                                       
            
            # Take measurements every 10 sweeps
            if i % 10 == 0:                       
                # Calculate current magnetisation                  
                S_sum = np.sum(self.S)                                    
                # Record current magnetisation                                 
                self.M = np.append(self.M, S_sum)                         
                # Record current energy  
                self.E = np.append(self.E, self.E_now)             
                
    def run_ani(self):
        """
        Run the simulation with an animated grid. blue corresponds to S=-1 and yellow corresponds to S=+1.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)
        plt.show()
    
    def frame(self, t):
        """
        Run ten sweeps of the simulation using Glauber or Kawasaki dynamics and update the image.
        """
        # Calculate the magnetisation and staggered magnetisation for the current configuration
        M, M_s = magnetisation(self.S, self.L)
        # Clear the axis
        plt.cla()                                                                                     
        img = plt.imshow(self.S, cmap='plasma', vmin=-1, vmax=1)      
        # Include the parameters and current magnetisation values in the title                    
        plt.title(f"Ising Model: Antiferromagnet\n $L$ = {self.L}, $k_BT$ = {self.kBT}, $J$ = {self.J}, $h$ = {self.h}\n $M$ = {M}, $M_s$ = {M_s}", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        # Run 10 sweeps of the algorithm
        for i in range(10):                                                                           
            for j in range(self.sweep):                                       
                # Update the lattice
                self.update()    

        return img

    def Glauber(self):
        """
        Update the system using Glauber dynamics.
        """
        #choose random state i
        i_row = random.randint(0, self.L-1)
        i_col = random.randint(0, self.L-1)
        # Calculate energy change upon flipping spin state i
        # Include contribution from external magnetic field h
        dE = delta_E_G(i_row, i_col, self.L, self.S, self.J) - self.h * self.S[i_row, i_col]
        # Apply the Metropolis algorithm to decide whether to flip the spin state i
        if self.metropolis(dE):
            self.S[i_row, i_col] *= -1
            # Update total energy whilst avoiding total recalculation
            self.E_now = self.E_now + dE        

    def Kawasaki(self):
        """
        Update the system using Kawasaki dynamics.
        """
        #choose random states i and j
        i_row = random.randint(0, self.L-1)
        i_col = random.randint(0, self.L-1)
        j_row = random.randint(0, self.L-1)
        j_col = random.randint(0, self.L-1)
        #continue choosing j state until it is distinct from the i state
        while ([i_row, i_col] == [j_row, j_col]) or (self.S[i_row, i_col] == self.S[j_row, j_col]):
            j_row = random.randint(0, self.L-1)
            j_col = random.randint(0, self.L-1)
        # Calculate energy change upon switching spin states i and j
        dE = delta_E_K(i_row, i_col, j_row, j_col, self.L, self.S, self.J)
        # Apply the Metropolis algorithm to decide whether to switch the spin states i and j
        if self.metropolis(dE):
            self.S[i_row, i_col], self.S[j_row, j_col] = self.S[j_row, j_col], self.S[i_row, i_col]
            # Update total energy while avoiding recalculation
            self.E_now = self.E_now + dE         

    def metropolis(self, dE):
        """
        Use the Metropolis algorithm to decide whether to flip the spin state.
        """
        if dE <= 0:
            # Always accept energy-lowering flip
            return True                                      
        elif random.uniform(0, 1) < np.exp(-dE / self.kBT):  
            # Spin flips with probability
            return True
        else:
            # Spin flip is rejected
            return False

    def avg_M(self, M):
        """
        Return average magnetisation and average magnetisation squared.

        Arguments:
           M: magnetisation measurements array
        
        Returns:
           average magnetisation, average magnetisation squared
        """
        return np.mean(M), np.mean(np.square(M))
    
    def susceptibility(self, M, M2):
        """
        Calculate and return the magnetic susceptibility.

        Arguments:
            M: expectation value of total magnetisation
           M2: expectation value of total magnetisation squared
           
        Returns:
           susceptibility
        """
        return (M2 - M**2) / (self.L * self.L * self.kBT)
    
    def total_E(self):
        """
        Calculate the total energy of the system.

        Returns:
            total energy of the system
        """
        # Initialise energy sum
        E_sum = 0                                          
        # Nearest neighbours       
        NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]                   
        # Loop over all spins
        for i in range(self.L):                                                          
            for j in range(self.L):
                # Loop over nearest neighbours 'k'
                for drow, dcol in NN:                             
                    k_row = (i + drow) % self.L
                    k_col = (j + dcol) % self.L
                    # Add contribution due to pair
                    E_sum += -self.S[i, j] * self.S[k_row, k_col] 
        # Avoid double counting and add external magnetic field contribution
        return self.J * E_sum / 2 - self.h * np.sum(self.S)                          
    
    def avg_E(self, E):
        """
        Calculate and return average energy and average energy squared.

        Arguments:
           E: energy measurements array
        
        Returns:
           average energy, average energy squared
        """
        return np.mean(E), np.mean(np.square(E))
    
    def heat_capacity(self, E, E2):
        """
        Calculate and return heat capacity.

        Arguments:
            E: expectation value of total energy
           E2: expectation value of total energy squared

        Returns:
            heat capacity
        """
        return (E2 - E**2) / (self.L * self.L * self.kBT * self.kBT)   
    
    def jackknife(self, C):
        """
        Compute the standard error on the heat capacity via the jackknife method.

        Arguments:
           C: measured heat capacity per spin
        
        Returns:
           Jackknife standard error on the heat capacity
        """
        # Number of data points
        n = len(self.E)                          
        # Initialise empty array for jackknife samples                      
        C_i = np.empty(0)                                              
        # Loop over all data points
        for i in range(n):                                             
            # Sample distribution with i-th element removed
            E_jack = np.delete(self.E, i)                              
            # Calculate average energy and average energy squared
            E, E2 = self.avg_E(E_jack)                                
            # Calculate heat capacity for jackknife sample
            C_jack = self.heat_capacity(E, E2)                         
            # append heat capacity for jackknife sample
            C_i = np.append(C_i, C_jack)                               
        # Jackknife standard error calculation
        return np.sqrt(np.sum((C_i - C)**2))                          
 

if __name__ == "__main__":
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-T', '--temperature', type=float, default=1, help='Thermal energy (default: 1)')
    parser.add_argument('-J', '--coupling', type=float, default=-1, help='Coupling constant (default: -1)')
    parser.add_argument('-H', '--field', type=float, default=0, help='External magnetic field (default: 0)')
    args = parser.parse_args()

    I = Ising(args.size, args.temperature, args.coupling, args.field)
    I.run_ani()  