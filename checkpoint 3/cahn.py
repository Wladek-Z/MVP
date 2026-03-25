import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from numba import njit

@njit
def laplacian(x, L):
    """
    Calculate the laplacian of x.
    
    Arguments:
        x: the chemical potential or the order parameter field
        L: system size
    
    Returns:
        laplacian: the lapacian of x
    """
    laplacian = np.zeros_like(x)
    for i in range(L):
        for j in range(L):
            up = (i - 1) % L
            down = (i + 1) % L
            left = (j - 1) % L
            right = (j + 1) % L
            laplacian[i, j] = x[up, j] + x[down, j] + x[i, left] + x[i, right] - 4 * x[i, j]

    return laplacian

@njit
def free_energy(phi, L, dx):
    """
    Calculate the free energy via CFD.

    Arguments:
        phi: the current order parameter field
        L: system size
        dx: spatial step
    
    Returns:
        the free energy of the system
    """
    dphi2 = np.zeros_like(phi)
    for i in range(L):
        for j in range(L):
            up = (i - 1) % L
            down = (i + 1) % L
            left = (j - 1) % L
            right = (j + 1) % L
            # Apply central finite difference scheme
            dphi2[i, j] = (phi[up, j]  - phi[down, j])**2 + (phi[i, left] - phi[i, right])**2
    # Calculate free energy density at each position
    f = -0.5 * phi**2 + 0.25 * phi**4 + 0.5 * 1 / (4 * dx**2) * dphi2
    # return free energy
    return np.sum(f) 

class CahnHilliard:
    """
    Class for discretising the Cahn-Hilliard equation, to describe phase
    separation in a physical system.
    """
    def __init__(self, L, phi_0, dt):
        """
        Arguments:
            L: system size
            phi_0: the order parameter at time = 0
            dt: time step
        """
        self.L = L
        self.dt = dt
        self.dx = 1
        # Set initial state of board to phi_0 plus some small random noise
        self.phi = (np.ones((L, L)) * phi_0) + np.random.uniform(-0.01, high=0.01, size=(L, L))

    def update(self, phi):
        """
        Calculate the next order parameter field.
        
        Arguments:
            phi: the current order parameter field
        
        Returns:
            the updated order parameter field
        """
        mu = - phi * (1 - phi**2) - self.dx**(-2) * laplacian(phi, self.L)
        return phi + self.dt / self.dx**2 * laplacian(mu, self.L)
    
    def animation(self):
        """
        Run and animate the simulation.
        """
        # Create figure and image
        fig, ax = plt.subplots(figsize=[10, 8])
        img = plt.imshow(self.phi, cmap='ocean', vmin=-1, vmax=1)
        plt.title('Phase Separation', fontsize = 16)
        # Add colour bar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels([-1, 0, 1], fontsize=16)
        cbar.set_label(r'order parameter $\phi$', size=16)

        plt.xticks([])
        plt.yticks([])  
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)

        plt.show()

    def frame(self, _):
        """
        Update the animation.

        Returns:
            img: figure displaying the (old) system
        """
        # Update 1000 times
        for i in range(1000):
            phi = self.phi.copy()
            self.phi = self.update(phi)

        # Clear the figure
        plt.cla()      
        # Update the figure                                                         
        img = plt.imshow(self.phi, cmap='ocean', vmin=-1, vmax=1) 

        plt.title('Phase Separation with Cahn-Hilliard Equation', fontsize = 16)
        plt.xticks([])
        plt.yticks([]) 
        # Return image of board for animation
        return img
    
    def task5(self):
        """
        Collect data for task 5.
        """
        file1 = 'cahn_phi_0.0.txt'
        file2 = 'cahn_phi_0.5.txt'

        self.collect_data(file1, 0)
        self.collect_data(file2, 0.5)
        
    def collect_data(self, filename, phi_0):
        """
        Collect data pertaining to the time required to equilibrate the free energy.

        Arguments:
            filename: file to save the data
            phi_0: initial value for the order parameter field
        """
        with open(filename, 'w') as file:
            file.write("time,f\n")
            # Initialise the order parameter field
            self.phi = (np.ones((self.L, self.L)) * phi_0) \
                       + np.random.uniform(-0.01, high=0.01, size=(self.L, self.L))
            # Initialise time
            t = 0
            # Record initial state
            phi = self.phi.copy()
            f = free_energy(phi, self.L, self.dx)
            file.write(f"{t},{f}\n")
            # Run the simulation until equilibrium
            while True:
                # increment time
                t += self.dt
                # Run the update procedure
                phi = self.phi.copy()
                self.phi = self.update(phi)
                # Calculate the free energy density
                f = free_energy(phi, self.L, self.dx)
                # Write to file
                file.write(f"{t},{f}\n")
                # Check if sufficient time has passed
                if t > 8000:
                    break
                
                    

    def plot(self):
        """
        Plot the free energy against time.
        """
        file1 = 'cahn_phi_0.0.txt'
        file2 = 'cahn_phi_0.5.txt'

        data1 = np.loadtxt(file1, delimiter=',', skiprows=1)
        data2 = np.loadtxt(file2, delimiter=',', skiprows=1)
        # Infer time step
        dt = np.round(data1[-1, 0] / len(data1[:, 0]), 6)

        plt.figure(figsize=[10, 8])
        plt.plot(data1[:, 0], data1[:, 1], label=r'$\phi_0 = 0.0$')
        plt.plot(data2[:, 0], data2[:, 1], label=r'$\phi_0 = 0.5$')
        plt.xlabel(r'time [$\kappa/Ma^2$]', fontsize=16)
        plt.ylabel(r'free energy [$a$]', fontsize=16)
        plt.title(rf'Free Energy vs Time for Cahn-Hilliard Simulation ($\delta t$ = {dt})', fontsize=16)
        plt.legend(fontsize=16)
        plt.show()

        
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Cahn-Hilliard simulation')
    argparser.add_argument('--data', action='store_true', help='Collect data for task 5')
    argparser.add_argument('--animation', action='store_true', help='Run the animation')
    argparser.add_argument('--plot', action='store_true', help='Plot the free energy for task 5')
    argparser.add_argument('-L', '--size', type=int, default=50, help='Systems size (default: 50)')
    argparser.add_argument('-dt', '--timestep', type=float, default=0.01, help='Time step (default: 0.01)')
    argparser.add_argument('-phi0', '--initialphi', type=float, default=0.0, help='Initial state of the order parameter field (default: 0.0)')
    args = argparser.parse_args()
    
    CH = CahnHilliard(args.size, args.initialphi, args.timestep)

    if args.data:
        CH.task5()
    elif args.animation:
        CH.animation()
    elif args.plot:
        CH.plot()