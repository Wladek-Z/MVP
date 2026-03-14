import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

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

    def order_parameter(self, phi):
        """
        Calculate the next order parameter field.
        
        Arguments:
            phi: the current order parameter field
        
        Returns:
            the updated order parameter field
        """
        mu = self.chemical_potential(phi)
        laplacian = self.laplacian(mu)
        return phi + self.dt / self.dx**2 * laplacian
    
    def chemical_potential(self, phi):
        """
        Calculate the chemical potential.

        Arguments:
            phi: the order parameter field
        
        Returns:
            the chemical potential
        """
        laplacian = self.laplacian(phi)
        return - phi * (1 - phi**2) - self.dx**(-2) * laplacian
    
    def laplacian(self, x):
        """
        Calculate the laplacian of x.
        
        Arguments:
            x: the chemical potential or the order parameter field
        
        Returns:
            laplacian: the lapacian of x
        """
        laplacian = np.roll(x, -1, axis=0)  \
                    + np.roll(x, 1, axis=0) \
                    + np.roll(x, -1, axis=1)\
                    + np.roll(x, 1, axis=1) \
                    - 4 * x
        return laplacian
    
    def update(self):
        """
        Update the order parameter field.
        """
        phi = self.phi.copy()
        self.phi = self.order_parameter(phi)

    
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
        # Run the update procedure 1000 times
        for i in range(1000):
            self.update()

        # Clear the figure
        plt.cla()      
        # Update the figure                                                         
        img = plt.imshow(self.phi, cmap='ocean', vmin=-1, vmax=1) 

        plt.title('Phase Separation with Cahn-Hilliard Equation', fontsize = 16)
        plt.xticks([])
        plt.yticks([]) 
        # Return image of board for animation
        return img
    

    def free_energy(self):
        """
        Calculate the free energy.
        
        Returns:
            the free energy of the system
        """
        # Get current order parameter field
        phi = self.phi.copy()
        # Calculate gradient squared via CFD
        dphi2 = 1 / (4 * self.dx**2)\
                *   ((np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0))**2\
                +    (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))**2)
        # Calculate free energy density at each position
        f = -0.5 * phi**2 + 0.25 * phi**4 + dphi2
        # return free energy
        return np.sum(f)
    
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
        # Set tolerance for equilibrium
        tol = 1e-4

        with open(filename, 'w') as file:
            file.write("time,f\n")
            # Initialise the order parameter field
            self.phi = (np.ones((self.L, self.L)) * phi_0) \
                       + np.random.uniform(-0.2, high=0.2, size=(self.L, self.L))
            # Initialise time and free energy
            t = 0
            f = self.free_energy()
            # Write initial values to file
            file.write(f"{t},{f}\n")
            # Run the simulation until equilibrium
            while True:
                t += 1
                # Run the update procedure
                self.update()
                # Calculate new free energy density
                f_new = self.free_energy()
                # Check if the system has equilibrated
                if (np.abs(f_new - f) < tol) and (t > 49999):
                    break
                else:
                    f = f_new
                    file.write(f"{t},{f}\n")

    def plot(self):
        """
        Plot the free energy against time.
        """
        file1 = 'cahn_phi_0.0.txt'
        file2 = 'cahn_phi_0.5.txt'

        data1 = np.loadtxt(file1, delimiter=',', skiprows=1)
        data2 = np.loadtxt(file2, delimiter=',', skiprows=1)

        plt.figure(figsize=[10, 8])
        plt.plot(data1[:, 0], data1[:, 1], label=r'$\phi_0 = 0.0$')
        plt.plot(data2[:, 0], data2[:, 1], label=r'$\phi_0 = 0.5$')
        plt.xlabel('time step', fontsize=16)
        plt.ylabel(r'free energy ($a$)', fontsize=16)
        plt.title('Free Energy vs Time for Cahn-Hilliard Simulation', fontsize=16)
        plt.legend(fontsize=16)
        plt.show()

        
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Cahn-Hilliard simulation')
    argparser.add_argument('--data', action='store_true', help='Collect data for task 5')
    argparser.add_argument('--animation', action='store_true', help='Run the animation')
    argparser.add_argument('--plot', action='store_true', help='Plot the free energy for task 5')
    argparser.add_argument('-L', '--size', type=int, default=50, help='Systems size (default: 50)')
    argparser.add_argument('-dt', '--timestep', type=float, default=0.001, help='Time step (default: 0.001)')
    argparser.add_argument('-phi0', '--initialphi', type=float, default=0.0, help='Initial state of the order parameter field (default: 0.0)')
    args = argparser.parse_args()
    
    CH = CahnHilliard(args.size, args.initialphi, args.timestep)

    if args.data:
        CH.task5()
    elif args.animation:
        CH.animation()
    elif args.plot:
        CH.plot()