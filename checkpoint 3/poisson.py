import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

class Poisson:
    """
    Class containing simulation for solving Poisson's equation.
    For convenience, set dx = episilon = 1.
    """
    def __init__(self, L, tol):
        """
        Let the potential follow the Dirichlet boundary condition
        with phi = 0.

        Arguments:
            L: system size
            tol: accuracy of final solution
        """
        self.L = L
        self.tol = tol
        self.converged = False
        self.iters = 0
        # Initialise the charge density for a random charge distribution around the centre of the middle-z slice
        self.rho = np.zeros((L, L, L))
        self.rho[L//4:3*L//4, L//4:3*L//4, L//2] = np.random.choice([0, 1], size=(L//2, L//2), p=[0.99, 0.01])
        # Initialise potential
        self.phi = np.zeros((L, L, L))

    def Jacobi(self, phi):
        """
        Calculate the updated potential using the Jacobi algorithm.
        
        Arguments:
            phi: the current potential
        
        Returns:
            the updated potential
        """
        phi_new = (np.roll(phi, -1, axis=0)\
                    + np.roll(phi, 1, axis=0)\
                        + np.roll(phi, -1, axis=1)\
                            + np.roll(phi, 1, axis=1)\
                                + np.roll(phi, -1, axis=2)\
                                    + np.roll(phi, 1, axis=2)\
                                        + self.rho) / 6
        # Apply boundary conditions
        phi_new[0, :, :] = \
            phi_new[-1, :, :] = \
                phi_new[:, 0, :] = \
                    phi_new[:, -1, :] = \
                        phi_new[:, :, 0] = \
                            phi_new[:, :, -1] = 0
        # Return the updated potential
        return phi_new
    
    def update(self):
        """
        Update the potential and check for convergence.
        """
        # Obtain updated potential and update phi, increment iters
        phi_old = self.phi.copy()
        self.phi = self.Jacobi(phi_old)
        self.iters += 1
        # Check for convergence
        if np.max(np.abs(self.phi - phi_old)) <= self.tol:
            self.converged = True
            print(f"Convergence achieved in {self.iters} iteration(s)!")
    
    # DELETE ANIMATION STUFF
    def frame(self, _):
        """
        Update the animation, stop when convergence is achieved.
        """
        # Terminate animation in case of convergence
        if self.converged:
            return []
        # Run update procedure
        self.update()
        # Take slice through middle of potential for plot
        slice = self.phi[:, :, self.L//2]
        # Clear and update the figure
        plt.cla()
        img = plt.imshow(slice, cmap='plasma', vmin=0, vmax=np.max(slice)) 

        plt.title('Electrostatic Potential', fontsize = 16)
        plt.xlabel(r'$x$', fontsize=16)
        plt.ylabel(r'$y$', fontsize=16)
        plt.xticks([])
        plt.yticks([]) 
        # Return image of field for animation
        return img  
          
    def animation(self):
        """
        Animate the solution to the boundary value problem.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=[10, 8])
        # Take slice through middle of potential for plot
        slice = self.phi[:, :, self.L//2]
        img = plt.imshow(slice, cmap='plasma', vmin=0, vmax=np.max(slice))

        plt.title('Electrostatic Potential', fontsize = 16)
        # Add colour bar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(r'electrostatic potential $\phi$', size=16)

        plt.xlabel(r'$x$', fontsize=16)
        plt.ylabel(r'$y$', fontsize=16)
        plt.xticks([])
        plt.yticks([])  
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)

        plt.show()

    def run_arb(self):
        """
        Calculate the potential and resultant electric field due to
        an arbitrary charge distribution.
        """
        # Converge the potential
        while not(self.converged):
            self.update()

        # Plot the electrostatic potential
        self.e_potential()

    def monopole(self):
        """
        Calculate the potential and resultant electric field due to
        a single charge at the centre.
        """
        # Initialise the charge density for a single charge at the centre
        self.rho = np.zeros((self.L, self.L, self.L))
        self.rho[self.L//2, self.L//2, self.L//2] = 1
        # Converge the potential
        while not(self.converged):
            self.update()

        # Plot the electrostatic potential
        self.e_potential()

    def e_potential(self):
        """
        Plot the electrostatic potential.
        """
        fig, ax = plt.subplots(figsize=[10, 8])
        # Take slice through middle of potential for plot
        slice = self.phi[:, :, self.L//2]
        img = plt.imshow(slice, cmap='plasma', vmin=0, vmax=np.max(slice), origin='lower')

        plt.title('Electrostatic Potential', fontsize = 16)
        # Add colour bar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(r'electrostatic potential $\phi$', size=16)

        plt.xlabel(r'$x$', fontsize=16)
        plt.ylabel(r'$y$', fontsize=16)
        plt.show()

        



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Poisson equation simulation")
    argparser.add_argument('-L', '--size', type=int, default=49, help="System size (default: 49)")
    argparser.add_argument('-t', '--tolerance', type=float, default=1e-6, help="Accuracy of final solution (default: 1e-6)")
    args = argparser.parse_args()

    P = Poisson(args.size, args.tolerance)
    P.monopole()