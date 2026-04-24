import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from numba import njit
import pandas as pd
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def delta_E_G(i_row, i_col, L, S, J, h):
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
    return 2 * (J * I_sum + h * S[i_row, i_col])                       

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
        C_i = np.zeros(n)                                             
        # Loop over all data points
        for i in range(n):                                             
            # Sample distribution with i-th element removed
            E_jack = np.delete(E, i)                              
            # Calculate average energy and average energy squared
            Ej, Ej2 = np.mean(E_jack), np.mean(np.square(E_jack))                              
            # Calculate heat capacity for jackknife sample
            C_jack = (Ej2 - Ej**2) / (L * L * kBT * kBT)                          
            # append heat capacity for jackknife sample
            C_i[i] = C_jack                               
        # Jackknife standard error calculation
        return np.sqrt(np.sum((C_i - C)**2))  

@njit        
def total_E(S, L, J, h):
        """
        Calculate the total energy of the system.

        Arguments:
            S: spin lattice
            L: system size
            J: coupling constant
            h: external magnetic field

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
        # Avoid double counting and add external magnetic field contribution
        return J * E_sum / 2 - h * np.sum(S)      

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
        self.M = np.zeros(1000)                     
        # Initialise empty list for staggered magnetisation measurements
        self.M_s = np.zeros(1000)     
        # Initialise empty list for energy measurements       
        self.E = np.zeros(1000)                       
        # Initialise random spin configuration
        self.S = np.random.choice([-1, 1], size=(L, L))       
        # Record initial total energy
        self.totalE = total_E(self.S, self.L, self.J, self.h)                           
        # Relic from when the user could choose between Glauber and Kawasaki dynamics
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
        for i in range(1, t+1):                       
            # Perform a sweep of the algorithm            
            for j in range(self.sweep):                       
                # Update the lattice      
                self.update()                                       
            # Take measurements every 10 sweeps
            if i % 10 == 0:  
               # Append current energy to array
               self.E[i//10 - 1] = self.totalE                     
               # Calculate current magnetisation and staggered magnetisation
               M, M_s = magnetisation(self.S, self.L)
               # Append magnetisation and staggered magnetisation measurements to arrays
               self.M[i//10 - 1] = M
               self.M_s[i//10 - 1] = M_s

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
        dE = delta_E_G(i_row, i_col, self.L, self.S, self.J, self.h)
        # Apply the Metropolis algorithm to decide whether to flip the spin state i
        if metropolis(dE, self.kBT):
            self.S[i_row, i_col] *= -1
            # Update total energy whilst avoiding total recalculation
            self.totalE += dE        

    def avg_M(self, M):
        """
        Return average magnetisation and average magnetisation squared.

        Arguments:
           M: magnetisation measurements array
        
        Returns:
           average magnetisation, average magnetisation squared
        """
        return np.mean(M), np.mean(np.square(M))

    def avg_E(self, E):
        """
        Calculate and return average energy and average energy squared.

        Arguments:
           E: energy measurements array
        
        Returns:
           average energy
        """
        return np.mean(E)
    
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
    
    def variance(self, M, M2):
        """
        Calculate and return the variance of the magnetisation.

        Arguments:
            M: magnetisation measurements array
            M2: squared magnetisation measurements array

        Returns:
            variance of the magnetisation
        """
        return M2 - M**2

def task_c_data(I):
    """
    Generate data for task c and write to file. (i) the average and the variance of the
    magnetisation, (ii) the average and variance of the staggered magnetisation, and (iii)
    the average of the energy.

    Arguments:
        I: Ising class instance
    """ 
    # Create list of external magnetic fields to probe
    h_list = np.arange(0, 10.5, 0.5)
    h_list = np.round(h_list, 1) 
    # Initialise empty data array
    data = []
    # Equilibriate for an additional 900 sweeps on first data point
    # Note: 100 equilibration sweeps + 't' additional sweeps per 'run' function call
    I.run(800)

    for h in h_list:
        # Set new value of external magnetic field
        print(f"h = {h}")
        I.h = h
        # Reset energy and magnetisation arrays
        I.E = np.zeros(1000)
        I.M = np.zeros(1000)
        I.M_s = np.zeros(1000)
        # Run simulation for 10000 sweeps
        I.run(10000)
        # Calculate the average magnetisation, average magnetisation squared
        M, M2 = I.avg_M(I.M)
        # Repeat for staggered magnetisation
        M_s, M_s2 = I.avg_M(I.M_s)
        # Calculate variance of magnetisation and staggered magnetisation
        var_M = I.variance(M, M2)
        var_M_s = I.variance(M_s, M_s2)
        # Calculate average energy
        E = I.avg_E(I.E)
        # Append data to array
        data.append([h, M, var_M, M_s, var_M_s, E])
    
    # Write all data to file
    df = pd.DataFrame(data, columns=['h', 'M', 'M_var', 'M_s', 'M_s_var', 'E'])
    df.to_csv('task_c_data.txt', mode='a', index=False, header=True)

def task_c_plot():
    """
    Read in and plot the data generated for task c.
    """
    df = pd.read_csv('task_c_data.txt')

    # Plot the average of the magnetisation against the external magnetic field
    fig = plt.figure(figsize=(8, 6))
    plt.plot(df['h'], df['M'])
    plt.title('Average magnetisation vs external magnetic field')
    plt.xlabel('External magnetic field, $h$')
    plt.ylabel('Average magnetisation, $M$')
    plt.tight_layout()
    plt.show()

    # Plot the variance of the magnetisation against the external magnetic field
    fig = plt.figure(figsize=(8, 6))
    plt.plot(df['h'], df['M_var'])
    plt.title('Variance of magnetisation vs external magnetic field')
    plt.xlabel('External magnetic field, $h$')
    plt.ylabel('Variance of magnetisation, Var($M$)')
    plt.tight_layout()
    plt.show()

    # Plot the average of the staggered magnetisation against the external magnetic field
    fig = plt.figure(figsize=(8, 6))
    plt.plot(df['h'], df['M_s'])
    plt.title('Average staggered magnetisation vs external magnetic field')
    plt.xlabel('External magnetic field, $h$')
    plt.ylabel('Average staggered magnetisation, $M_s$')
    plt.tight_layout()
    plt.show()

    # Plot the variance of the staggered magnetisation against the external magnetic field
    fig = plt.figure(figsize=(8, 6))
    plt.plot(df['h'], df['M_s_var'])
    plt.title('Variance of staggered magnetisation vs external magnetic field')
    plt.xlabel('External magnetic field, $h$')
    plt.ylabel('Variance of staggered magnetisation, Var($M_s$)')
    plt.tight_layout()
    plt.show()

    # Plot the average of the energy against the external magnetic field
    fig = plt.figure(figsize=(8, 6))
    plt.plot(df['h'], df['E'])
    plt.title('Average energy vs external magnetic field')
    plt.xlabel('External magnetic field, $h$')
    plt.ylabel('Average energy, $E$')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-T', '--temperature', type=float, default=1, help='Thermal energy (default: 1)')
    parser.add_argument('-J', '--coupling', type=float, default=-1, help='Coupling constant (default: -1)')
    parser.add_argument('-H', '--field', type=float, default=0, help='External magnetic field (default: 0)')
    parser.add_argument('-t', '--task', type=str, choices=['animation', 'c'], default='animation', help="Task to run: 'animation' for animation, or 'c' for task c (default: 'animation')")
    parser.add_argument('--collect', action='store_true', help='Collect data for a given task')
    parser.add_argument('--plot', action='store_true', help='Display plots for a given task')
    args = parser.parse_args()

    if args.task == 'c':
        if args.collect:
            I = Ising(50, 1, -1, 0)
            task_c_data(I)
        elif args.plot:
            task_c_plot()
    elif args.task == 'animation':
        I = Ising(args.size, args.temperature, args.coupling, args.field)
        I.run_ani()  