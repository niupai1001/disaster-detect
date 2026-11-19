"""Support screening for SpaceNet 3 WorldView-3 multispectral source tiles."""
import numpy as np
from scipy.ndimage import binary_propagation


MIN_SUPPORT = .5


def edge_zero_padding(values):
    """Only classify zero fill connected to the outer edge in every band."""
    zeros = np.logical_and.reduce([values_i == 0 for values_i in values])
    boundary = np.zeros_like(zeros)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    return binary_propagation(boundary & zeros, mask=zeros)
