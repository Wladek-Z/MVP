import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

class SIRS:
    """Class for simulating the SIRS model on a 2D lattice"""

    def __init__(self, pS_I, p)