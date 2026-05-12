import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import pandas as pd
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
        # Sum up nearest neighbours
        NN_I = neighbour_sum(S, x, y, L)
        # Shortcut energy change calculation
        dE = 2 * J * S[x, y] * NN_I

        # Apply the Metropolis algorithm to decide whether to flip the spin state i
        if metropolis(dE, kBT):
            # Flip the spin
            S[x, y] *= -1
            return S, dE
        else: 
            return S, 0

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
    # Choose random states i and j
    xi = random.randint(0, L-1)
    yi = random.randint(0, L-1)
    xj = random.randint(0, L-1)
    yj = random.randint(0, L-1)

    # Continue choosing j until it is distinct from i and not a nearest neighbour.
    while ((xi == xj) and (yi == yj)) or nearest_neighbor(xi, yi, xj, yj, L) or (S[xi, yi] == S[xj, yj]):
        xj = random.randint(0, L-1)
        yj = random.randint(0, L-1)
    
    # Compute the neighbour sums for sites i and j.
    NN_I = neighbour_sum(S, xi, yi, L)
    NN_J = neighbour_sum(S, xj, yj, L)

    # Compute resulting energy change
    dE = J * (S[xi, yi] - S[xj, yj]) * (NN_I - NN_J)

    # Apply the Metropolis algorithm to decide whether to swap the spin states i and j
    if metropolis(dE, kBT):
        # Swap the spin states i and j
        S[xi, yi], S[xj, yj] = S[xj, yj], S[xi, yi]
        return S, dE
    else:
        # No change
        return S, 0

@njit
def nearest_neighbor(xi, yi, xj, yj, L):
    """
    Return True when two sites are nearest neighbours on the periodic lattice.
    """
    return ((xi == (xj - 1) % L) and (yi == yj)) \
        or ((xi == (xj + 1) % L) and (yi == yj)) \
        or ((xi == xj) and (yi == (yj - 1) % L)) \
        or ((xi == xj) and (yi == (yj + 1) % L))
    
@njit
def neighbour_sum(S, x, y, L):
    """
    Sum up the contributions from the 4 nearest neighbours of site (x, y) of S.
    
    Arguments:
        S: spin lattice
        x: coordinate along axis 0
        y: coordinate along axis 1
        L: system size
        
    Returns:
        the nearest neighbour sum
    """
    return S[(x-1) % L, y] + S[(x+1) % L, y] + S[x, (y-1) % L] + S[x, (y+1) % L]

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
        # Loop over all spins
        for i in range(L):                                                          
            for j in range(L):
                E_sum -= S[i, j] * neighbour_sum(S, i, j, L)
        # Avoid double counting
        return J * E_sum / 2 


@njit
def jackknife(E, n, C, L, kBT):
        """
        Compute the standard error on the heat capacity via the jackknife method.

        Arguments:
            E: energy measurements array
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
            # Print live progress
            print(f"Progress: {i}/{t}", end="\r", flush=True)                                   
            
            # Take measurements every 10 sweeps
            if i % 10 == 0:                       
                # Calculate current magnetisation                  
                S_sum = np.sum(self.S)                                    
                # Record current magnetisation                                 
                self.M[i//10 - 1] = S_sum                         
                # Record current energy  
                self.E[i//10 - 1] = self.totalE
        # Move to next line after sweeps complete
        print()              
                
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
        if self.totalE != total_E(self.S, self.L, self.J):
            print("fail")
        else:
            print("pass")
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
            heat capacity in units kB
        """
        return (E2 - E**2) / (self.L * self.L * self.kBT * self.kBT)   
    
    def collect4_5(self):
        """
        Collect data for tasks 4 and 5 at the same time. Save results to file.
        """
        self.c = 0.8
        T_list = np.arange(3, 0, -0.1)
        T_list = np.round(T_list, 1)
        data4 = []
        data5 = []

        for T in T_list:
            print(f"T = {T}")
            # Choose new temperature
            self.kBT = T
            self.E = np.empty(1000)
            self.M = np.empty(1000)
            # Collect data
            self.run(10000)
            # Calculate heat capacity and susceptibility
            E, E2 = self.avg_E(self.E)
            C = self.heat_capacity(E, E2)
            sigma = jackknife(self.E, 1000, C, self.L, self.kBT)
            M, M2 = self.avg_M(self.M)
            chi = self.susceptibility(M, M2)
            # Append to data arrays
            data4.append([T, C, sigma])
            data5.append([T, chi])
        
        # Save to files
        df4 = pd.DataFrame(data4, columns=['T', 'C', 'sigma'])
        df4.to_csv("task4.txt", mode='w', index=False, header=True)
        df5 = pd.DataFrame(data5, columns=['T', 'chi'])
        df5.to_csv("task5.txt", mode='w', index=False, header=True)

    def plot4(self):
        """
        Plot heat capacity results for task 4.
        """
        df = pd.read_csv("task4.txt")
        fig = plt.figure(figsize=[8, 6])
        plt.plot(df['T'], df['C'], '-', color='deepskyblue')
        plt.errorbar(df['T'], df['C'], yerr=df['sigma'], fmt='.', color='deepskyblue', capsize=5, ecolor='crimson')
        plt.xlabel(r'Temperature, $T$ [$J/k_B$]')
        plt.ylabel(r'Heat Capacity, $C$ [$k_B$]')
        plt.title(f'Heat Capacity vs Temperature')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-T', '--temperature', type=float, default=1, help='Temperature (default: 1)')
    parser.add_argument('-c', '--fraction', type=float, default=0.5, help='Fraction of sites occupied by helium 4 (default: 0.5)')
    parser.add_argument('-t', '--task', choices=['animation', '4', '5', '6'], default='animation', help='Select a task for the simulation (default: animation)')
    parser.add_argument('--collect', action='store_true', help='Collect data required for a given task')
    parser.add_argument('--plot', action='store_true', help='Plot results for a given task')
    args = parser.parse_args()

    beg = BEG(args.size, args.temperature, args.fraction)

    if args.task == 'animation':
        beg.run_ani()  
    elif args.collect:
        if (args.task == '4') or (args.task == '5'):
            beg.collect4_5()
        elif args.task == '6':
            beg.collect6()
    elif args.plot:
        if args.task == '4':
            beg.plot4()
        elif args.task == '5':
            beg.plot5()
        elif args.task == '6':
            beg.plot6()
