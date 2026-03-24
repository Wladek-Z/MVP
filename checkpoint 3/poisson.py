import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from numba import njit
from scipy.optimize import curve_fit

@njit
def Gauss_Seidel_electric(phi, rho, L, dx):
    """
    Calculate the updated potential using the Gauss-Seidel algorithm.
    
    Arguments:
        phi: the current potential
        rho: the charge density distribution
        L: system size
        dx: spatial resolution
    
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
                                                    + dx**2 * rho[i, j, k]) / 6
    return phi

@njit
def Gauss_Seidel_magnetic(phi, rho, L, dx):
    """
    Calculate the updated potential using the Gauss-Seidel algorithm,
    this time for the magnetic problem.
    
    Arguments:
        phi: the current potential
        rho: the charge density distribution
        L: system size
        dx: spatial resolution
    
    Returns:
        phi: the updated potential
    """
    for i in range(1, L-1):
        for j in range(1, L-1):
            for k in range(0, L):
                phi[i, j, k] = (phi[i-1, j, k]\
                                + phi[i+1, j, k]\
                                    + phi[i, j-1, k]\
                                        + phi[i, j+1, k]\
                                            + phi[i, j, (k-1) % L]\
                                                + phi[i, j, (k+1) % L]\
                                                    + dx**2 * rho[i, j, k]) / 6
    return phi

@njit
def E_field(phi, L, dx):
        """
        Calculate the electric field from the potential.

        Arguments:
            phi: a slice of the potential
            L: system size
            dx: spatial resolution
        
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
                Ey[i, j] = -0.5 * (phi[down, j] - phi[up, j]) / dx
                Ex[i, j] = -0.5 * (phi[i, right] - phi[i, left]) / dx

        return Ex, Ey

@njit
def B_field(A, L, dx):
        """
        Calculate the magnetic field from the potential.

        Arguments:
            A: a slice of the vector potential
            L: system size
            dx: spatial resolution
        
        Returns:
            the magnetic field components
        """
        # Initialise components of magnetic field
        Bx = np.zeros((L, L))
        By = np.zeros((L, L))
        # Calculate the curl of A through central finite difference (CFD)
        for i in range(L):
            for j in range(L):
                up = (i - 1) % L
                down = (i + 1) % L
                left = (j - 1) % L
                right = (j + 1) % L
                Bx[i, j] = 0.5 * (A[down, j] - A[up, j]) / dx
                By[i, j] = -0.5 * (A[i, right] - A[i, left]) /dx

        return Bx, By



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
        self.method = method
        self.dx = 1
        # Initialise potential
        self.phi = np.zeros((L, L, L))

    def Jacobi(self, old_phi, _=None, __=None, ___=None):
        """
        Calculate the updated potential using the Jacobi algorithm.
        
        Arguments:
            old_phi: the current potential
        
        Returns:
            the updated potential
        """
        phi = (np.roll(old_phi, -1, axis=0)\
                    + np.roll(old_phi, 1, axis=0)\
                        + np.roll(old_phi, -1, axis=1)\
                            + np.roll(old_phi, 1, axis=1)\
                                + np.roll(old_phi, -1, axis=2)\
                                    + np.roll(old_phi, 1, axis=2)\
                                        + self.dx * self.rho) / 6
        # Apply boundary conditions
        phi_new = self.boundary_conditions(phi)
        # Return the updated potential
        return phi_new

    def electric_BC(self, phi):
        """
        Apply the boundary conditions for the electric field problem.
        
        Arguments:
            phi: the current potential
        
        Returns:
            phi: the potential with updated boundary conditions
        """
        phi[0, :, :] = \
            phi[-1, :, :] = \
                phi[:, 0, :] = \
                    phi[:, -1, :] = \
                        phi[:, :, 0] = \
                            phi[:, :, -1] = 0
        # Return the updated potential
        return phi

    def magnetic_BC(self, phi):
        """
        Apply the boundary conditions for the magnetic field problem.
        
        Arguments:
            phi: the current potential
        
        Returns:
            phi: the potential with updated boundary conditions
        """
        phi[0, :, :] = \
            phi[-1, :, :] = \
                phi[:, 0, :] = \
                    phi[:, -1, :] = 0
        # Return the updated potential
        return phi

    
    def SOR(self, phi_old, _=None, __=None, ___=None):
        """
        Calculate the updated potential using the SOR algorithm.
        
        Arguments:
            phi: the current potential
        
        Returns:
            the updated potential
        """
        phi = phi_old.copy()
        phi_new = self.Gauss_Seidel(phi, self.rho, self.L, self.dx)
        delta = phi_new - phi_old
        return phi_old + self.w * delta
    
    def update(self):
        """
        Update the potential and check for convergence.
        """
        # Obtain updated potential and update phi, increment iters
        phi = self.phi.copy()
        phi_old = self.phi.copy()
        self.phi = self.alg(phi, self.rho, self.L, self.dx)
        self.iters += 1
        # Check for convergence
        if np.max(np.abs(self.phi - phi_old)) <= self.tol:
            self.converged = True
            print(f"Convergence achieved in {self.iters} iteration(s)!")

    
    def monopole(self):
        """
        Calculate the potential and resultant electric field due to a single charge at the centre, 
        display Gaussian fits. Save to file. (task 7)
        """
        # Initialise the charge density for a single charge at the centre
        self.rho = np.zeros((self.L, self.L, self.L))
        self.rho[self.L//2, self.L//2, self.L//2] = 1
        # Set electric boundary conditions method
        self.boundary_conditions = self.electric_BC
        # Use electric Gauss-Seidel
        self.Gauss_Seidel = Gauss_Seidel_electric
        # Choose method
        if self.method == 'Jacobi':
            self.alg = self.Jacobi
        elif self.method == 'Gauss-Seidel':
            self.alg = self.Gauss_Seidel
        elif self.method == 'SOR':
            self.alg = self.SOR

        # Converge the potential
        while not(self.converged):
            self.update()

        # Take slice through middle of potential for plot
        slice = self.phi[:, :, self.L//2]
        Ex, Ey = E_field(slice, self.L, self.dx)
        E = np.sqrt(Ex**2 + Ey**2)

        # Choose titles and colour bar labels for plotting
        title_pot = "Electrostatic Potential"
        cbar_label_pot = r'electrostatic potential $\phi$'
        title1_field = 'Electric field vector'
        title2_field = 'Electric field strength'
        cbar_label_field = r'Electric field $|E|$'

        # Plot the electrostatic potential
        self.plot_potential(slice, title_pot, cbar_label_pot)
        # Plot the electric field
        self.plot_field(Ex, Ey, E, title1_field, title2_field, cbar_label_field)
        # Save potential and E-field to file 
        self.save_data("poisson_monopole.txt", Ex, Ey)
        # Read in data to get r and phi
        r, phi = np.loadtxt('poisson_monopole.txt', skiprows=1, usecols=[2, 3], unpack=True, delimiter=',')
        # Perform the curve fit
        self.fit_curves_electric(r, phi, E)

    def save_data(self, filename, Fx, Fy):
        """
        Save the potential and field data to file.

        Arguments:
            filename: desired name of the output file
            Fx: the x-component of the field on a 2D midplane
            Fy: the y-component of the field on a 2D midplane
        """
        slice = self.phi[:, :, self.L//2]
        with open(filename, 'w') as f:
            f.write("i,j,r,pot,Fieldx,Fieldy\n")
            for i in range(self.L):
                for j in range(self.L):
                    # Calculate distance to monopole
                    r = np.linalg.norm([((self.L // 2) - j) * self.dx, ((self.L // 2) - i) * self.dx])
                    f.write(f"{i},{j},{r},{slice[i, j]},{Fx[i, j]},{Fy[i, j]}\n")

    def fit_curves_electric(self, r, phi, E):
        """
        Read in data from poisson_monopole.txt and fit electrostatic potential/electric
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

    def fit_curves_magnetic(self, r, A, B):
        """
        Read in data from poisson_wire.txt and fit magnetic potential/electric
        field strength to Ampere's law.

        Arguments:
            r: magnitude of separation from monopole
            A: magnetic potential at each value of separation
            B: magnetic field strength at each value of separation
        """
        # Delete r=0 elements for curve fitting
        A = np.delete(A, r==0)
        B   = np.delete(B,   r==0)
        r   = np.delete(r,   r==0)

        # Sort the data
        sort = np.argsort(r)
        r = r[sort]
        A = A[sort] 
        B = B[sort]

        # Create curves from Ampere's law
        r0 = np.min(r[A == 0])
        A_Ampere =  np.log(r0 / r) / (2 * np.pi)
        B_Ampere = 1 / (2 * np.pi * r)

        # Plot the data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[18, 8])

        ax1.scatter(r, A, marker=".", color="orange", label="data")
        ax1.plot(r, A_Ampere, color="blue", label=r"$-ln(r)/2\pi + C$")
        ax1.set_xlabel(r'$r$', fontsize=12)
        ax1.set_ylabel(r'$A$', fontsize=12)
        ax1.set_yscale("log")
        ax1.set_xscale("log")
        ax1.set_title("Magnetic potential", fontsize=16)
        ax1.legend(loc="upper right", fontsize=12)

        ax2.scatter(r, B, marker=".", color="orange", label="data")
        ax2.plot(r, B_Ampere, color="blue", label=r"$1/2\pi r$")
        ax2.set_xlabel(r'$r$', fontsize=12)
        ax2.set_ylabel(r'$|B|$', fontsize=12)
        ax2.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_title("Magnetic field strength", fontsize=16)
        ax2.legend(loc="upper right", fontsize=12)

        plt.show()

    def plot_potential(self, slice, title, cbar_label):
        """
        Plot the electrostatic potential.

        Arguments:
            slice: the potential on a 2D plane
            title: desired title of the plot
            cbar_label: desired label for the colour bar
        """
        fig, ax = plt.subplots(figsize=[10, 8])
        img = plt.imshow(slice, cmap='plasma', origin='lower')

        plt.title(title, fontsize = 16)
        # Add colour bar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(cbar_label, size=16)

        plt.xlabel(r'$x$', fontsize=16)
        plt.ylabel(r'$y$', fontsize=16)
        plt.show()

    def plot_field(self, Fx, Fy, F, title1, title2, cbar_label):
        """
        Plot the field.

        Arguments:
            Fx: the x-component of the field on a 2D midplane
            Fy: the y-component of the field on a 2D midplane
            F: the magnitude of the field on a 2D midplane
            title1: title of vector plot
            title2: title of magnitude plot
            cbar_label: desired colour bar label
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[18, 8])

        if title1 == "Magnetic field vector":
            scale = 0.8
        else:
            scale = 0.3

        # Plot the electric field as a quiver plot
        ax1.quiver(Fx, Fy, scale=scale)
        ax1.set_title(title1, fontsize=16)
        ax1.set_xlabel(r'$x$', fontsize=12)
        ax1.set_ylabel(r'$y$', fontsize=12)

        # Also plot the electric field magnitude as a contour plot
        img = ax2.imshow(F, cmap='plasma', origin='lower')
        ax2.set_title(title2, fontsize=16)
        ax2.set_xlabel(r'$x$', fontsize=12)
        ax2.set_ylabel(r'$y$', fontsize=12)
        # Add colour bar
        cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, size=12)

        plt.show()

    
    def wire(self):
        """
        Calculate the potential and resultant magnetic field due to a single wire at the centre
        extending through z, fit to Ampere's law. Save to file. (task 9)
        """
        # Initialise the charge density for a wire at the centre of the x-y plane along the z-axis
        self.rho = np.zeros((self.L, self.L, self.L))
        self.rho[self.L//2, self.L//2, :] = 1
        # Set magnetic boundary conditions method (periodic along z-axis)
        self.boundary_conditions = self.magnetic_BC
        # Use magnetic Gauss-Seidel
        self.Gauss_Seidel = Gauss_Seidel_magnetic
        # Choose method
        if self.method == 'Jacobi':
            self.alg = self.Jacobi
        elif self.method == 'Gauss-Seidel':
            self.alg = self.Gauss_Seidel
        elif self.method == 'SOR':
            self.alg = self.SOR

        # Converge the potential
        while not(self.converged):
            self.update()

        # Take slice through middle of potential for plot
        slice = self.phi[:, :, self.L//2]
        Bx, By = B_field(slice, self.L, self.dx)
        B = np.sqrt(Bx**2 + By**2)

        # Choose titles and colour bar labels for plotting
        title_pot = r'Magnetic Potential ($z$-component)'
        cbar_label_pot = r'magnetic potential $A_z$'
        title1_field = 'Magnetic field vector'
        title2_field = 'Magnetic field strength'
        cbar_label_field = r'Magnetic field $|B|$'

        # Plot the magnetic potential
        self.plot_potential(slice, title_pot, cbar_label_pot)
        # Plot the magnetic field
        self.plot_field(Bx, By, B, title1_field, title2_field, cbar_label_field)
        # Save potential and B-field to file 
        self.save_data("poisson_wire.txt", Bx, By)
        # Read in data to get r and A
        r, A = np.loadtxt('poisson_wire.txt', skiprows=1, usecols=[2, 3], unpack=True, delimiter=',')
        # Perform the curve fit
        self.fit_curves_magnetic(r, A, B)

    def task10(self):
        """
        Find the optimal value of the relaxation parameter (w), such as to minimise the number of
        iterations required for convergence in the SOR method. Save results to file (task 10)
        """
        # Initialise the charge density for a single charge at the centre
        self.rho = np.zeros((self.L, self.L, self.L))
        self.rho[self.L//2, self.L//2, self.L//2] = 1
        # Set electric boundary conditions method
        self.boundary_conditions = self.electric_BC
        # Use electric Gauss-Seidel
        self.Gauss_Seidel = Gauss_Seidel_electric
        # Choose method
        self.alg = self.SOR

        # collect convergence data over different w
        w_list = np.linspace(1.622, 1.627, 20)

        with open("poisson_task10.txt", 'w') as f:
            f.write("w,iters\n")
            for w in w_list:
                # Initialise relaxation parameter, potential, iters, and convergence
                self.w = w
                self.phi = np.zeros((self.L, self.L, self.L))
                self.iters = 0
                self.converged = False

                # Converge the electrostatic potential
                while not(self.converged):
                    self.update()
                
                # Write to file
                f.write(f"{w},{self.iters}\n")
                        
        # Plot number of iterations vs relaxation parameter
        w, iters = np.loadtxt("poisson_task10.txt", skiprows=1, unpack=True, delimiter=',')

        best_w = np.round(w[np.argmin(iters)], 6)
        
        plt.plot(w, iters, color='black')
        plt.axvline(best_w, color='black', linestyle='--', label=rf'$\omega_0 =$ {best_w}')
        plt.xlabel("relaxation parameter", fontsize=16)
        plt.ylabel("iteration to convergence", fontsize=16)
        plt.legend(fontsize=12)

        plt.show()

                



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Poisson equation simulation")
    argparser.add_argument('-L', '--size', type=int, default=49, help="System size (default: 49)")
    argparser.add_argument('-t', '--tolerance', type=float, default=1e-6, help="Accuracy of final solution (default: 1e-6)")
    argparser.add_argument('--monopole', action='store_true', help="Calculate potential due to a single charge at the centre")
    argparser.add_argument('--task10', action='store_true', help="Find optimal value of w in SOR method. (task 10)")
    argparser.add_argument('--wire', action='store_true', help="Calculate potential due to a straight wire through the centre")
    argparser.add_argument('-m', '--method', choices=['Jacobi', 'Gauss-Seidel', 'SOR'], default='Jacobi', help="Method for solving Poisson's equation (default: Jacobi)")
    argparser.add_argument('-w', '--relaxation', type=float, default=1.5, help="Relaxation parameter for SOR method (default: 1.5)")

    args = argparser.parse_args()

    P = Poisson(args.size, args.tolerance, args.method, args.relaxation)

    if args.monopole:
        P.monopole()
    elif args.wire:
        P.wire()
    elif args.task10:
        P.task10()
    else:
        print("Error: no action input")