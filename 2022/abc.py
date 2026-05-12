import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import matplotlib.colors as mc
from numba import njit
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def laplacian(x, L):
    """
    Calculate the laplacian of x.
    
    Arguments:
        x: the concentration
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

@njit
def type_field(a, b, c, L):
    """
    Calculate the type field for a given set of concentrations.
    
    Arguments:
        a: concentration of chemical species a
        b: concentration of chemical species b
        c: concentration of chemical species c
        L: system size
    
    Returns:
        tau: the type field
    """
    # Define (1 - a - b - c) field
    abc = 1 - a - b - c
    # Initialise the type field
    tau = np.empty(shape=(L, L))
    # Calculate each component of the type field
    for i in range(L):
        for j in range(L):
            tau[i, j] = np.argmax(np.array([abc[i, j], a[i, j], b[i, j], c[i, j]]))

    return tau

@njit
def update(a, b, c, L, D, q, p, dt):
    """
    Calculate the next concentrations of chemical species a, b, and c.
    
    Arguments:
        a: concentration of chemical species a
        b: concentration of chemical species b
        c: concentration of chemical species c
        L: system size
        D: diffusion coefficient
        q: reaction parameter q
        p: reaction parameter p
        dt: timestep
    
    Returns:
        a_new: the updated concentration of a
        b_new: the updated concentration of b
        c_new: the updated concentration of c
    """
    # Calculation laplacians
    lap_a = laplacian(a, L)
    lap_b = laplacian(b, L)
    lap_c = laplacian(c, L)
    # Update concentrations
    a_new = a + dt * (D * lap_a + q * a * (1 - a - b - c) - p * a * c)
    b_new = b + dt * (D * lap_b + q * b * (1 - a - b - c) - p * a * b)
    c_new = c + dt * (D * lap_c + q * c * (1 - a - b - c) - p * b * c)
    # Return updated concentrations
    return a_new, b_new, c_new

class Chemicals:
    """
    Class for solving a set of coupled partial differential equations for the concentrations of 
    the reactive chemical species a, b, and c.
    """
    def __init__(self, L, dt, D, q, p):
        """
        Arguments:
            L: system size
            dt: time step
            D: diffusion coefficient
            q: reaction parameter q
            p: reaction parameter p
        """
        self.L = L
        self.dt = dt
        self.dx = 1
        self.D = D
        self.q = q
        self.p = p
        # Initialise concentrations of a, b, and c
        self.a = np.random.uniform(0, 1/3, (L, L))
        self.b = np.random.uniform(0, 1/3, (L, L))
        self.c = np.random.uniform(0, 1/3, (L, L))
        # Calculate initial type field
        self.tau = type_field(self.a, self.b, self.c, L)
    
    def animation(self):
        """
        Run and animate the simulation.
        """
        # Define custom colormap
        colours = ['gray', 'red', 'green', 'blue']
        self.cmap = mc.ListedColormap(colours)
        # Create figure and image
        fig, ax = plt.subplots(figsize=[8, 6])
        img = plt.imshow(self.tau, cmap=self.cmap, vmin=0, vmax=3)
        # Set title
        plt.title('Chemical Concentrations', fontsize = 16)
        # Define discrete boundaries for colour bar
        boundaries = np.linspace(0, 3, 5)
        # Add custom colour bar
        cbar = plt.colorbar(img, ax=ax, boundaries=boundaries)
        cbar.set_ticks([3/8, 9/8, 15/8, 21/8])
        cbar.set_ticklabels([0, 1, 2, 3], fontsize=16)
        cbar.set_label(r'type field $\tau$', size=16)
        # Clear axis labels
        plt.xticks([])
        plt.yticks([]) 
        
        ani = FuncAnimation(fig, self.frame, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

    def frame(self, _):
        """
        Update the animation.

        Returns:
            img: figure displaying the (old) system
        """
        # Update 100 times
        for i in range(100):
            a, b, c = self.a.copy(), self.b.copy(), self.c.copy()
            self.a, self.b, self.c = update(a, b, c, self.L, self.D, self.q, self.p, self.dt)
            self.tau = type_field(self.a, self.b, self.c, self.L)
        # Clear the figure
        plt.cla()      
        # Update the figure                                                         
        img = plt.imshow(self.tau, cmap=self.cmap, vmin=0, vmax=3) 
        plt.title('Chemical Concentrations', fontsize = 16)
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
    argparser.add_argument('--collect', action='store_true', help='Collect data for a given task')
    argparser.add_argument('--animation', action='store_true', help='Run the animation')
    argparser.add_argument('--plot', action='store_true', help='Plot the results for a given task')
    argparser.add_argument('-L', '--size', type=int, default=50, help='Systems size (default: 50)')
    argparser.add_argument('-dt', '--timestep', type=float, default=0.01, help='Time step (default: 0.01)')
    argparser.add_argument('-D', type=float, default=1, help='Diffusion coefficient (default: 1)')
    argparser.add_argument('-q', type=float, default=1, help='Reaction parameter q (default: 1)')
    argparser.add_argument('-p', type=float, default=0.5, help='Reaction parameter p (default: 0.5)')
    args = argparser.parse_args()
    
    abc = Chemicals(args.size, args.timestep, args.D, args.q, args.p)

    if args.animation:
        abc.animation()