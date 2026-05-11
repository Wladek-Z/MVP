import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from numba import njit
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def Glauber(L, S, J, kBT):
    """
    Run a sweep of the update procedure using Glauber dynamics.

    Arguments:
        L: system size
        S: spin lattice
        J: coupling constant
        kBT: the thermal energy

    Returns:
        S: the updated spin lattice
        dE: the energy change
    """
    # Choose random state. x and y denote rows and columns, respectively
    x = random.randint(0, L-1)
    y = random.randint(0, L-1)
    
    if S[x, y] == 0:
        # No effect if He3 chosen in Glauber update
        return S, 0
    else:
        # Nearest neighbours in 2 dimensions
        NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]       
        # Initialise sum over pairs           
        I_sum = 0                                                
        # Loop over nearest neighbours 'k'
        for dx, dy in NN:                                    
            xk = (x + dx) % L
            yk = (y + dy) % L
            # Add contribution due to pair
            I_sum += S[x, y] * S[xk, yk] 
        # Shortcut energy change calculation
        dE = 2 * (J * I_sum)

        # Apply the Metropolis algorithm to decide whether to flip the spin state i
        if metropolis(dE, kBT):
            # Flip the spin
            S[x, y] *= -1

        return S, dE

@njit
def Kawasaki(L, S, J, kBT):
    """
    Run a sweep of the update procedure using Kawasaki dynamics.

    Arguments:
        L: system size
        S: spin lattice
        J: coupling constant
        kBT: the thermal energy

    Returns:
        S: the updated spin lattice
        dE: the energy change
    """
    #choose random states i and j
    xi = random.randint(0, L-1)
    yi = random.randint(0, L-1)
    xj = random.randint(0, L-1)
    yj = random.randint(0, L-1)
    #continue choosing j state until it is distinct from the i state
    while ([xi, yi] == [xj, yj]) or (S[xi, yi] == S[xj, yj]):
        xj = random.randint(0, L-1)
        yj = random.randint(0, L-1)

    # Nearest neighbours
    NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Initialise sum over pairs                      
    I_sum = 0                                                    
    J_sum = 0                                                   
    # Loop over nearest neighbours 'k' for each state
    for dx, dy in NN:                                        
        xk = (xi + dx) % L
        yk = (yi + dy) % L
        # Swapping neighbouring i and j has no effect
        if [xk, yk] != [xj, yj]:            
            # Compute contribution due to pair             
            I_sum += S[xj, yj] * S[xk, yk] 

        xk = (xj + dx) % L
        yk = (yj + dy) % L
        # Swapping neighbouring i and j has no effect
        if [xk, yk] != [xi, yi]:                     
            # Compute contribution due to pair
            J_sum += S[xi, yi] * S[xk, yk]

    # Calculate total energy change
    dE = -2 * J * (I_sum + J_sum) 

    # Apply the Metropolis algorithm to decide whether to switch the spin states i and j
    if metropolis(dE, kBT):
        # Swap the spin states i and j
        S[xi, yi], S[xj, yj] = S[xj, yj], S[xi, yi]

    return S, dE

@njit
def metropolis(dE, kBT):
        """
        Use the Metropolis algorithm to decide whether to flip the spin state.

        Arguments:
            dE: energy change upon flipping the spin state
            kBT: thermal energy
        """
        if dE <= 0:
            # Always accept energy-lowering flip
            return True                                      
        elif random.uniform(0, 1) < np.exp(-dE / kBT):  
            # Spin flips with probability
            return True
        else:
            # Spin flip is rejected
            return False  

@njit        
def total_E(S, L, J):
        """
        Calculate the total energy of the system.

        Arguments:
            S: spin lattice
            L: system size
            J: coupling constant

        Returns:
            total energy of the system
        """
        # Initialise energy sum
        E_sum = 0                                          
        # Nearest neighbours       
        NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]                   
        # Loop over all spins
        for i in range(L):                                                          
            for j in range(L):
                # Loop over nearest neighbours 'k'
                for drow, dcol in NN:                             
                    k_row = (i + drow) % L
                    k_col = (j + dcol) % L
                    # Add contribution due to pair
                    E_sum += -S[i, j] * S[k_row, k_col] 
        # Avoid double counting
        return J * E_sum / 2 


@njit
def jackknife(E, n, C, L, kBT):
        """
        Compute the standard error on the heat capacity via the jackknife method.

        Arguments:
            E_array: energy measurements array
            n: number of data points
            C: measured heat capacity per spin
            L: system size
            kBT: thermal energy
        
        Returns:
            Jackknife standard error on the heat capacity
        """                       
        # Initialise empty array for jackknife samples 
        E_jack = np.empty(n-1)                     
        C_jack = np.empty(n)  
        # Loop over all data points
        for i in range(n):
            # Sample distribution with i-th element removed
            for j in range(n-1):                           
                if j < i:
                    E_jack[j] = E[j]
                else:
                    E_jack[j] = E[j+1]                         
            # Calculate and append the heat capacity for the jackknife sample
            C_jack[i] = (np.mean(np.square(E_jack)) - np.mean(E_jack)**2) / (L*L*kBT*kBT)                                            
        # Jackknife standard error calculation
        return np.sqrt(np.sum((C_jack - C)**2)) 

@njit
def update(sweep, L, S, J, kBT):
    """
    Perform a sweep of update moves, choosing randomly between Glauber and Kawasaki
    dynamics at every step.
    
    Arguments:
        sweep: length of one sweep
        L: system size
        S: spin lattice
        J: coupling constant
        kBT: thermal energy

    Returns:
        S_new: the updated spin lattice
        dE_total: the total energy change
    """
    dE_total = 0
    S_old = S.copy()
    for _ in range(sweep):

        if np.random.binomial(1, 0.5):
            S_new, dE = Glauber(L, S_old, J, kBT)
        else:
            S_new, dE = Kawasaki(L, S_old, J, kBT)
        
        S_old = S_new.copy()
        dE_total += dE
    
    return S_new, dE_total


class BEG:
    """Class to represent a 2D Blume-Emery-Griffiths model"""

    def __init__(self, L, T, c):
        """
        Constructor method for Ising class.

        Arguments:
            L: system size
            T: temperature
            c: fraction of sites occupied by He4
        """
        self.L = L
        self.kBT = T
        self.c = c
        self.J = 1            
        # Define a unique unit of time for an LxL system 
        self.sweep = L * L                                               
        # Initialise empty list for magnetisation measurements                            
        self.M = np.empty(1000)                          
        # Initialise empty list for energy measurements       
        self.E = np.empty(1000)                                  

        # Initialise random spin configuration
        self.S = np.random.choice([-1, 0, 1], p=[c/2, (1-c), c/2], size=(L, L))       
        # Record initial energy
        self.totalE = total_E(self.S, self.L, self.J)                       

  
    def run(self, t):
        """
        Run the simulation for t sweeps.

        Arguments:
           t: number of sweeps for which to run the simulation
        """
        # Equilibriate the system
        for i in range(100):                           
            # Perform a sweep of the algorithm             
            S_old = self.S.copy()                                                             
            self.S, dE = update(self.sweep, self.L, S_old, self.J, self.kBT)
            # Update total energy
            self.totalE += dE                                       
                                              
        # Run the simulation for t sweeps
        for i in range(1, t + 1):                       
            # Perform a sweep of the algorithm             
            S_old = self.S.copy()                                                             
            self.S, dE = update(self.sweep, self.L, S_old, self.J, self.kBT)
            # Update total energy
            self.totalE += dE                                        
            
            # Take measurements every 10 sweeps
            if i % 10 == 0:                       
                # Calculate current magnetisation                  
                S_sum = np.sum(self.S)                                    
                # Record current magnetisation                                 
                self.M[i//10 - 1] = S_sum                         
                # Record current energy  
                self.E[i//10 - 1] = self.totalE              
                
    def run_ani(self):
        """
        Run the simulation with an animated grid.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        img = plt.imshow(self.S, cmap='plasma', vmin=-1, vmax=1)  
        # Add colour bar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels([-1, 0, 1], fontsize=16)
        cbar.set_label(r'spin $S_i$', size=16)

        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
    
    def frame(self, _):
        """
        Run ten sweeps of the simulation using Glauber or Kawasaki dynamics and update the image.
        """
        # Clear the axis
        plt.cla()                                       
        plt.title(f"BEG Model\n $L$ = {self.L}, $T$ = {self.kBT}, $c$ = {self.c}")
        plt.axis('off')
        # Run 10 sweeps of the algorithm
        for i in range(10):                                                                           
            # Perform a sweep of the algorithm             
            S_old = self.S.copy()                                                             
            self.S, dE = update(self.sweep, self.L, S_old, self.J, self.kBT)
            # Update total energy
            self.totalE += dE  
        # Update the image                                                         
        img = plt.imshow(self.S, cmap='plasma', vmin=-1, vmax=1)                                                                      
        return img
          

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
                          
 

if __name__ == "__main__":
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-T', '--temperature', type=float, default=1, help='Temperature (default: 1)')
    parser.add_argument('-c', '--fraction', type=float, default=0.5, help='Fraction of sites occupied by helium 4 (default: 0.5)')
    args = parser.parse_args()

    beg = BEG(args.size, args.temperature, args.fraction)
    beg.run_ani()  