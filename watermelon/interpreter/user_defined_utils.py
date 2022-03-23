import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython.display import display


def imshow(x):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(x)

    plt.show()
