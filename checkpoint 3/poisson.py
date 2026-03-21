import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from numba import njit
from scipy.optimize import curve_fit

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
        # Initialise the charge density for a single charge at the centre
        self.rho = np.zeros((L, L, L))
        self.rho[self.L//2, self.L//2, self.L//2] = 1
        # Initialise potential
        self.phi = np.zeros((L, L, L))
        # Choose method
        if method == 'Jacobi':
            self.alg = self.Jacobi
        elif method == 'Gauss-Seidel':
            self.alg = Gauss_Seidel
        elif method == 'SOR':
            self.alg = self.SOR

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
        self.phi = self.alg(phi, self.rho, self.L)
        self.iters += 1
        # Check for convergence
        if np.max(np.abs(self.phi - phi_old)) <= self.tol:
            self.converged = True
            print(f"Convergence achieved in {self.iters} iteration(s)!")

    
    def monopole(self):
        """
        Calculate the potential and resultant electric field due to
        a single charge at the centre, then save to file.
        """
        # Converge the potential
        while not(self.converged):
            self.update()

        # Take slice through middle of potential for plot
        slice = self.phi[:, :, self.L//2]
        Ex, Ey = E_field(slice, self.L)
        E = np.sqrt(Ex**2 + Ey**2)

        # Plot the electrostatic potential
        self.plot_potential(slice)
        # Plot the electric field
        self.plot_field(Ex, Ey, E)

        # Save potential and E-field to file if desired
        save = input("Save results to file? [y/n]: ")
        if save == "y":
            self.save_monopole()

    def task7(self):
        """
        Read in data regarding potential and electric field due to a single
        charge from file and plot results. Also fit the potential and electric 
        field strengths as a function of the distance to the charge. (task 7)
        """
        # Read in data to plot phi and E
        i, r, phi, Ex, Ey = np.loadtxt('poisson_monopole.txt', skiprows=1, \
                                    usecols=[0, 2, 3, 4, 5], unpack=True, delimiter=',')
        E = np.sqrt(Ex**2 + Ey**2)

        # Recreate potential and E-field on 2D plane from datafile
        L = int(np.sqrt(len(i)))
        slice_phi = np.zeros((L, L))
        slice_E = np.zeros((L, L))
        slice_Ex = np.zeros((L, L))
        slice_Ey = np.zeros((L, L))

        for m in range(L):
            slice_phi[m, :] = phi[(m * L):((m + 1) * L)]
            slice_E[m, :] = E[(m * L):((m + 1) * L)]
            slice_Ex[m, :] = Ex[(m * L):((m + 1) * L)]
            slice_Ey[m, :] = Ey[(m * L):((m + 1) * L)]

        # Plot the electrostatic potential
        self.plot_potential(slice_phi)
        # Plot the electric field
        self.plot_field(slice_Ex, slice_Ey, slice_E)

        # Perform the curve fit
        self.fit_curves(r, phi, E)

    def save_monopole(self):
        """
        Save the monopole electrostatic potential and electric field data to file.
        """
        slice = self.phi[:, :, self.L//2]
        Ex, Ey = E_field(slice, self.L)
        with open('poisson_monopole.txt', 'w') as f:
            f.write("i,j,r,phi,Ex,Ey\n")
            for i in range(self.L):
                for j in range(self.L):
                    # Calculate distance to monopole
                    r = np.linalg.norm([(self.L // 2) - j, (self.L // 2) - i])
                    f.write(f"{i},{j},{r},{slice[i, j]},{Ex[i, j]},{Ey[i, j]}\n")

    def fit_curves(self, r, phi, E):
        """
        Read in data from monopole.txt and fit electrostatic potential/electric
        field strength to Gauss's law.

        Arguments:
            r: magnitude of separation from monopole
            phi: electrostatic potential at each value of separation
            E: electric field strength at each value of separation
        """
        # Delete r=0 elements for curve fitting
        phi = np.delete(phi, r==0)
        E   = np.delete(E,   r==0)
        r   = np.delete(r,   r==0)

        # Sort the data
        sort = np.argsort(r)
        r = r[sort]
        phi = phi[sort]
        E = E[sort]

        # Create curves from Gauss's law
        phi_Gauss = 1 / (4 * np.pi * r)
        E_Gauss = 1 / (4 * np.pi * r**2)

        # Plot the data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[18, 8])

        ax1.scatter(r, phi, marker=".", color="orange", label="data")
        ax1.plot(r, phi_Gauss, color="blue", label=r"$1/4\pi r$")
        ax1.set_xlabel(r'$r$', fontsize=12)
        ax1.set_ylabel(r'$\phi$', fontsize=12)
        ax1.set_yscale("log")
        ax1.set_xscale("log")
        ax1.set_title("Electrostatic potential", fontsize=16)
        ax1.legend(loc="upper right", fontsize=12)

        ax2.scatter(r, E, marker=".", color="orange", label="data")
        ax2.plot(r, E_Gauss, color="blue", label=r"$1/4\pi r^2$")
        ax2.set_xlabel(r'$r$', fontsize=12)
        ax2.set_ylabel(r'$|E|$', fontsize=12)
        ax2.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_title("Electric field strength", fontsize=16)
        ax2.legend(loc="upper right", fontsize=12)

        plt.show()

    def plot_potential(self, slice):
        """
        Plot the electrostatic potential.

        Arguments:
            slice: the potential on a 2D plane
        """
        fig, ax = plt.subplots(figsize=[10, 8])
        img = plt.imshow(slice, cmap='plasma', vmin=0, vmax=np.max(slice), origin='lower')

        plt.title('Electrostatic Potential', fontsize = 16)
        # Add colour bar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(r'electrostatic potential $\phi$', size=16)

        plt.xlabel(r'$x$', fontsize=16)
        plt.ylabel(r'$y$', fontsize=16)
        plt.show()

    def plot_field(self, Ex, Ey, E):
        """
        Plot the electric field.

        Arguments:
            Ex: the x-component of the electric field on a 2D midplane
            Ey: the y-component of the electric field on a 2D midplane
            E: the magnitude of the electric field on a 2D midplane
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[18, 8])

        # Plot the electric field as a quiver plot
        ax1.quiver(Ex, Ey, scale=0.3)
        ax1.set_title('Electric field vector', fontsize=16)
        ax1.set_xlabel(r'$x$', fontsize=12)
        ax1.set_ylabel(r'$y$', fontsize=12)

        # Also plot the electric field magnitude as a contour plot
        img = ax2.imshow(E, cmap='plasma', vmin=0, vmax=np.max(E), origin='lower')
        ax2.set_title('Electric field strength', fontsize=16)
        ax2.set_xlabel(r'$x$', fontsize=12)
        ax2.set_ylabel(r'$y$', fontsize=12)
        # Add colour bar
        cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
        cbar.set_label(r'Electric field $|E|$', size=12)

        plt.show()

    

        



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Poisson equation simulation")
    argparser.add_argument('-L', '--size', type=int, default=49, help="System size (default: 49)")
    argparser.add_argument('-t', '--tolerance', type=float, default=1e-6, help="Accuracy of final solution (default: 1e-6)")
    argparser.add_argument('--monopole', action='store_true', help="Calculate potential due to a single charge at the centre")
    argparser.add_argument('--task7', action='store_true', help="Plot the potential and electric field due to a monopole from datafile, fit as function of r. (task 7)")
    argparser.add_argument('--wire', action='store_true', help="Calculate potential due to a straight wire through the centre")
    argparser.add_argument('-m', '--method', choices=['Jacobi', 'Gauss-Seidel', 'SOR'], default='Jacobi', help="Method for solving Poisson's equation (default: Jacobi)")
    argparser.add_argument('-w', '--relaxation', type=float, default=1.5, help="Relaxation parameter for SOR method (default: 1.5)")
    args = argparser.parse_args()

    P = Poisson(args.size, args.tolerance, args.method, args.relaxation)

    if args.monopole:
        P.monopole()
    elif args.task7:
        P.task7()
    else:
        print("Error: no action input")