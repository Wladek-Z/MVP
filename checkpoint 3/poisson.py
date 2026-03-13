import numpy as np
import matplotlib.pyplot as plt

class Poisson:
    """
    Class containing simulation for solving Poisson's equation.
    """
    def __init__(self, L):
        self.L = L
