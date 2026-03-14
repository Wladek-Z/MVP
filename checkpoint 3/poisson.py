import numpy as np
import matplotlib.pyplot as plt

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

        # Initialise the charge density and potential
        self.rho = np.zeros((L, L, L))
        self.rho[L//2][L//2][L//2] = 1

        self.phi = np.zeros((L, L, L))

    def Jacobi(self, phi):
        """
        Update the potential using the Jacobi algorithm.
        
        Arguments:
            phi: the current potential
        
        Returns:
            the updated potential
        """
        phi_next = (np.roll(phi, -1, axis=0)\
                    + np.roll(phi, 1, axis=0)\
                        + np.roll(phi, -1, axis=1)\
                            + np.roll(phi, 1, axis=1)\
                                + np.roll(phi, -1, axis=2)\
                                    + np.roll(phi, 1, axis=2)\
                                        + self.rho) / 6
        # Apply boundary conditions
        phi_next[0, :, :] = \
            phi_next[-1, :, :] = \
                phi_next[:, 0, :] = \
                    phi_next[:, -1, :] = \
                        phi_next[:, :, 0] = \
                            phi_next[:, :, -1] = 0
        # Return the updated potential
        return phi_next
