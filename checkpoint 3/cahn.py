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
        self.sweep = L**2
        self.dt = dt
        self.dx = 1
        # Set initial state of board to phi_0 plus some small random noise
        self.phi = (np.ones((L, L)) * phi_0) + np.random.uniform(-0.25, high=0.25, size=(L, L))

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
    
    def animation(self):
        """
        Run and animate the simulation.
        """
        # Create figure and image
        fig, ax = plt.subplots(figsize=[10, 8])
        img = plt.imshow(self.phi, cmap='vanimo', vmin=-1, vmax=1)
        plt.title('Phase Separation', fontsize = 16)
        # Add colour bar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels([-1, 0, 1], fontsize=16)

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
        # Run the update procedure
        self.update()
        # Clear the figure
        plt.cla()      
        # Update the figure                                                         
        img = plt.imshow(self.phi, cmap='vanimo', vmin=-1, vmax=1) 

        plt.title('Phase Separation', fontsize = 16)
        plt.xticks([])
        plt.yticks([]) 
        # Return image of board for animation
        return img
    
    def update(self):
        """
        Run a sweep of the Euler algorithm.
        """
        for _ in range(self.sweep):
            # Copy the current order parameter field
            phi = self.phi.copy()
            # Update the order parameter field
            self.phi = self.order_parameter(phi)
    
if __name__ == "__main__":
    CH = CahnHilliard(50, 0, 0.001)
    CH.animation()