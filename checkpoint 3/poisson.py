import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from numba import njit

@njit
def Gauss_Seidel(phi, rho, L):
    """
    Calculate the updated potential using the Gauss-Seidel algorithm.
    
    Arguments:
        phi: the current potential
        rho: the charge density distribution
        L: system size
    
    Returns:
        phi: the updated potential
    """
    for i in range(1, L-1):
        for j in range(1, L-1):
            for k in range(1, L-1):
                phi[i, j, k] = (phi[i-1, j, k]\
                                + phi[i+1, j, k]\
                                    + phi[i, j-1, k]\
                                        + phi[i, j+1, k]\
                                            + phi[i, j, k-1]\
                                                + phi[i, j, k+1]\
                                                    + rho[i, j, k]) / 6
    return phi

@njit
def E_field(phi, L):
        """
        Calculate the electric field from the potential.

        Arguments:
            phi: a slice of the potential
            L: system size
        
        Returns:
            the electric field components
        """
        # Initialise components of electric field
        Ex = np.zeros((L, L))
        Ey = np.zeros((L, L))
        # Calculate the gradient of phi through central finite difference (CFD)
        for i in range(L):
            for j in range(L):
                up = (i - 1) % L
                down = (i + 1) % L
                left = (j - 1) % L
                right = (j + 1) % L
                Ey[i, j] = -0.5 * (phi[down, j] - phi[up, j])
                Ex[i, j] = -0.5 * (phi[i, right] - phi[i, left])
        return Ex, Ey


class Poisson:
    """
    Class containing simulation for solving Poisson's equation.
    For convenience, set dx = episilon = 1.
    """
    def __init__(self, L, tol, method, w):
        """
        Let the potential follow the Dirichlet boundary condition with phi = 0.

        Arguments:
            L: system size
            tol: accuracy of final solution
            method: method for solving Poisson's equation
            w: relaxation parameter for SOR method
        """
        # Initialise parameters
        self.L         = L
        self.tol       = tol
        self.converged = False
        self.iters     = 0
        self.w         = w
        # Initialise the charge density for a random charge distribution around the centre of the middle-z slice
        self.rho = np.zeros((L, L, L))
        self.rho[L//4:3*L//4, L//4:3*L//4, L//2] = np.random.choice([0, 1], size=(L//2, L//2), p=[0.99, 0.01])
        # Initialise potential
        self.phi = np.zeros((L, L, L))
        # Choose method
        if method == 'Jacobi':
            self.algo = self.Jacobi
        elif method == 'Gauss-Seidel':
            self.algo = Gauss_Seidel
        elif method == 'SOR':
            self.algo = self.SOR

    def Jacobi(self, phi, _=None, __=None):
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
    
    def SOR(self, phi_old, _=None, __=None):
        """
        Calculate the updated potential using the SOR algorithm.
        
        Arguments:
            phi: the current potential
        
        Returns:
            the updated potential
        """
        phi = phi_old.copy()
        phi_new = Gauss_Seidel(phi, self.rho, self.L)
        delta = phi_new - phi_old
        return phi_old + self.w * delta
    
    def update(self):
        """
        Update the potential and check for convergence.
        """
        # Obtain updated potential and update phi, increment iters
        phi = self.phi.copy()
        phi_old = self.phi.copy()
        self.phi = self.algo(phi, self.rho, self.L)
        self.iters += 1
        # Check for convergence
        if np.max(np.abs(self.phi - phi_old)) <= self.tol:
            self.converged = True
            print(f"Convergence achieved in {self.iters} iteration(s)!")

    
    def run_arb(self):
        """
        Calculate the potential and resultant electric field due to
        an arbitrary charge distribution.
        """
        # Converge the potential
        while not(self.converged):
            self.update()

        # Plot the electrostatic potential
        self.plot_potential()
        # Plot the electric field
        self.plot_field()

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
        self.plot_potential()
        # Plot the electric field
        self.plot_field()
        # Save potential and E-field to file
        slice = self.phi[:, :, self.L//2]
        Ex, Ey = E_field(slice, self.L)
        with open('monopole.txt', 'w') as f:
            f.write("i,j,phi,Ex,Ey\n")
            for i in range(self.L):
                for j in range(self.L):
                    f.write(f"{i},{j},{slice[i, j]},{Ex[i, j]},{Ey[i, j]}\n")

    def plot_potential(self):
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

    def plot_field(self):
        """
        Plot the electric field.
        """
        fig, ax = plt.subplots(figsize=[8, 8])
        # Take slice through middle of potential for plot
        slice = self.phi[:, :, self.L//2]
        Ex, Ey = E_field(slice, self.L)
        # Plot the electric field as a quiver plot
        plt.quiver(Ex, Ey, scale=0.1)

        plt.title('Electric Field', fontsize = 16)
        plt.xlabel(r'$x$', fontsize=16)
        plt.ylabel(r'$y$', fontsize=16)
        plt.show()


        



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Poisson equation simulation")
    argparser.add_argument('-L', '--size', type=int, default=49, help="System size (default: 49)")
    argparser.add_argument('-t', '--tolerance', type=float, default=1e-6, help="Accuracy of final solution (default: 1e-6)")
    argparser.add_argument('--monopole', action='store_true', help="Calculate potential due to a single charge at the centre")
    argparser.add_argument('--wire', action='store_true', help="Calculate potential due to a straight wire through the centre")
    argparser.add_argument('-m', '--method', choices=['Jacobi', 'Gauss-Seidel', 'SOR'], default='Jacobi', help="Method for solving Poisson's equation (default: Jacobi)")
    argparser.add_argument('-w', '--relaxation', type=float, default=1.5, help="Relaxation parameter for SOR method (default: 1.5)")
    args = argparser.parse_args()

    P = Poisson(args.size, args.tolerance, args.method, args.relaxation)

    if args.monopole:
        P.monopole()
    #elif args.wire:
    #    P.wire()
    else:
        P.run_arb()