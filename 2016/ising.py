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

@njit
def Glauber(sweep, L, h0, P, tau, n, S, J, kBT, new_field):
    """
    Run a sweep of the update procedure using Glauber dynamics.

    Arguments:
        sweep: length of one sweep
        L: system size
        h0: amplitude of external magnetic field
        P: spatial period of magnetic field
        tau: temporal period of magnetic field
        n: number of timesteps elapsed
        S: spin lattice
        J: coupling constant
        kBT: the thermal energy
        new_field: whether to use the space/time-dependent magnetic field (True/False)

    Returns:
        S_new: the updated spin lattice
        dE: the energy change
    """
    # Initialise net energy change over one sweep
    dE_total = 0
    for i in range(sweep):
        # Copy the spin lattice
        S_new = S.copy()
        # Choose random state. x and y denote rows and columns, respectively
        x = random.randint(0, L-1)
        y = random.randint(0, L-1)
        # Calculate current magnetic field
        if new_field:
            h = h0 * np.cos(2*np.pi*x/P) * np.cos(2*np.pi*y/P) * np.sin(2*np.pi*n/tau) 
        else:
            h = h0
        
        # Nearest neighbours in 2 dimensions
        NN = [(-1, 0), (1, 0), (0, -1), (0, 1)]       
        # Initialise sum over pairs           
        I_sum = 0                                                
        # Loop over nearest neighbours 'k'
        for dx, dy in NN:                                    
            kx = (x + dx) % L
            ky = (y + dy) % L
            # Add contribution due to pair
            I_sum += S_new[x, y] * S_new[kx, ky] 
        # Shortcut energy change calculation
        dE = 2 * (J * I_sum + h * S_new[x, y])

        # Apply the Metropolis algorithm to decide whether to flip the spin state i
        if metropolis(dE, kBT):
            # Flip the spin
            S_new[x, y] *= -1
            S = S_new.copy()
            # Update the net energy change
            dE_total += dE

    return S_new, dE_total

class Ising:
    """Class to represent a 2D Ising model"""

    def __init__(self, L, kBT, J, h0, P):
        """
        Constructor method for Ising class.

        Arguments:
            L: system size
            kBT: thermal energy (J=1)
            J:  coupling constant
            h: external magnetic field
            P: spatial period
        """
        self.L = L
        self.kBT = kBT
        # Modify coupling constant for different equilibrium states
        self.J = J              
        # Define external magnetic field
        self.h0 = h0
        # time-dependent field is initially 0
        self.h = 0
        # Define a unique unit of time for an LxL system 
        self.sweep = L * L   
        # Keep track of how many sweeps have passed
        self.n = 0                
        # Fix spatial and time periods
        self.P = P
        self.tau = 10000               
        # Initialise empty list for magnetisation measurements                            
        self.M = np.empty(1000)                     
        # Initialise empty list for staggered magnetisation measurements
        self.M_s = np.empty(1000)     
        # Initialise empty list for energy measurements       
        self.E = np.empty(1000)        
        # Initialise empty list to record time
        self.timestep = np.empty(1000)               
        # Initialise random spin configuration
        self.S = np.random.choice([-1, 1], size=(L, L))       
        # Record initial total energy
        self.totalE = total_E(self.S, self.L, self.J, self.h)                           
        # Relic from when the user could choose between Glauber and Kawasaki dynamics
        self.update = Glauber
  
    def run(self, t, new_field=True, freq=10, equ=100):
        """
        Run the simulation for t sweeps.

        Arguments:
           t: number of sweeps for which to run the simulation
           new_field: whether to use the space/time-dependent magnetic field (True/False)
           freq: frequency of data collection
           equ: number of equilibration sweeps
        """
        # Equilibrate the system
        for i in range(equ):                           
            # Perform a sweep of the algorithm             
            S_old = self.S.copy()                                                                         
            self.S, dE = self.update(self.sweep, self.L, self.h0, self.P, self.tau, self.n, S_old, self.J, self.kBT, new_field)
            # Update total energy
            self.totalE += dE
            # Increment time
            self.n += 1                                     
                                              
        # Run the simulation for t sweeps
        for i in range(1, t+1):                       
            # Perform a sweep of the algorithm            
            S_old = self.S.copy()                                                                         
            self.S, dE = self.update(self.sweep, self.L, self.h0, self.P, self.tau, self.n, S_old, self.J, self.kBT, new_field)
            # Update total energy
            self.totalE += dE     
            # Increment time
            self.n += 1

            # Take measurements every freq sweeps
            if i % freq == 0:  
               # Append current energy to array
               self.E[i//freq - 1] = self.totalE                     
               # Calculate current magnetisation and staggered magnetisation
               M, M_s = magnetisation(self.S, self.L)
               # Append magnetisation and staggered magnetisation measurements to arrays
               self.M[i//freq - 1] = M
               self.M_s[i//freq - 1] = M_s
               # Record current time
               self.timestep[i//freq - 1] = self.n

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
        plt.title(f"Ising Model: Antiferromagnet\n $L$ = {self.L}, $k_BT$ = {self.kBT}, $J$ = {self.J}, $h_0$ = {self.h0}\n $M$ = {M}, $M_s$ = {M_s}", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        # Run 10 sweeps of the algorithm
        for i in range(10):  
            S_old = self.S.copy()                                                                         
            self.S, dE = self.update(self.sweep, self.L, self.h0, self.P, self.tau, self.n, S_old, self.J, self.kBT, True)
            # Update total energy
            self.totalE += dE
            # Increment time
            self.n += 1
            
        # Print the current value of sin(2 pi n / tau)
        print(f"sin = {np.sin(2 * np.pi * self.n / self.tau)}", end='\r')

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
    I.run(800, False)

    for h in h_list:
        # Set new value of external magnetic field
        print(f"h = {h}")
        I.h = h
        # Reset energy and magnetisation arrays
        I.E = np.empty(1000)
        I.M = np.empty(1000)
        I.M_s = np.empty(1000)
        # Run simulation for 10000 sweeps
        I.run(10000, False)
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

def task_d_data():
    """
    Generate data for task d and write to file. Measure value of maximal field strength over time
    and the instantaneous staggered magnetisation over time.
    """
    # Collect data for P = 25
    I_25 = Ising(50, 1, -1, 10, 25)
    # Collect data over 1000 sweeps
    I_25.run(20000, True, 20, 1000)
    # Save data to file
    data_25 = {
        't': I_25.timestep,
        'max_h': I_25.h0 * np.sin(2*np.pi*I_25.timestep/I_25.tau),
        'M_s': I_25.M_s
    }
    df_25 = pd.DataFrame(data_25)
    df_25.to_csv('task_d_data_P_25.txt', index=False, header=True)

    # Collect data for P = 10
    I_10 = Ising(50, 1, -1, 10, 10)
    # Collect data over 1000 sweeps
    I_10.run(20000, True, 20, 1000)
    # Save data to file
    data_10 = {
        't': I_10.timestep,
        'max_h': I_10.h0 * np.sin(2*np.pi*I_10.timestep/I_10.tau),
        'M_s': I_10.M_s
    }
    df_10 = pd.DataFrame(data_10)
    df_10.to_csv('task_d_data_P_10.txt', index=False, header=True)


def task_d_plot():
    """
    Read in and plot the data generated for task d.
    """
    df_25 = pd.read_csv('task_d_data_P_25.txt')
    df_10 = pd.read_csv('task_d_data_P_10.txt')

    # Plot the maximal field strength and the staggered magnetisation over time for P = 25
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"$P$ = 25")
    ax[0].plot(df_25['t'], df_25['M_s'])
    ax[0].set_title('Staggered magnetisation vs time')
    ax[0].set_xlabel('Timestep, $n$')
    ax[0].set_ylabel(r'Staggered magnetisation, $M_s$')
    ax[1].scatter(df_25['max_h'], df_25['M_s'], marker='.', s=10)
    ax[1].set_title('Staggered magnetisation vs maximal field strength')
    ax[1].set_xlabel(r'Maximal field strength, $h = h_0 \sin(2\pi n/\tau)$')
    ax[1].set_ylabel('Staggered magnetisation, $M_s$')
    plt.tight_layout()
    plt.show()

    # Plot the maximal field strength and the staggered magnetisation over time for P = 10
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"$P$ = 10")
    ax[0].plot(df_10['t'], df_10['M_s'])
    ax[0].set_title('Staggered magnetisation vs time')
    ax[0].set_xlabel('Timestep, $n$')
    ax[0].set_ylabel(r'Staggered magnetisation, $M_s$')
    ax[1].scatter(df_10['max_h'], df_10['M_s'], marker='.', s=10)
    ax[1].set_title('Staggered magnetisation vs maximal field strength')
    ax[1].set_xlabel(r'Maximal field strength, $h = h_0 \sin(2\pi n/\tau)$')
    ax[1].set_ylabel('Staggered magnetisation, $M_s$')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--size', type=int, default=50, help='System size (default: 50)')
    parser.add_argument('-T', '--temperature', type=float, default=1, help='Thermal energy (default: 1)')
    parser.add_argument('-J', '--coupling', type=float, default=-1, help='Coupling constant (default: -1)')
    parser.add_argument('-H', '--field', type=float, default=0, help='External magnetic field (default: 0)')
    parser.add_argument('-P', '--spatialperiod', type=int, default=25, help='Spatial period for space/time-dependent external magnetic field (default: 25)')
    parser.add_argument('-t', '--task', type=str, choices=['animation', 'c', 'd'], default='animation', help="Task to run: 'animation' for animation, 'c' for task c, or 'd' for task d (default: 'animation')")
    parser.add_argument('--collect', action='store_true', help='Collect data for a given task')
    parser.add_argument('--plot', action='store_true', help='Display plots for a given task')
    args = parser.parse_args()

    if args.task == 'c':
        if args.collect:
            I = Ising(50, 1, -1, 0, args.spatialperiod)
            task_c_data(I)
        elif args.plot:
            task_c_plot()
    elif args.task == 'd':
        if args.collect:
            task_d_data()
        elif args.plot:
            task_d_plot()
    elif args.task == 'animation':
        I = Ising(args.size, args.temperature, args.coupling, args.field, args.spatialperiod)
        I.run_ani()  