"""Implementation of a Gaussian Smothing filter with scipy"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve
from scipy.signal.windows import gaussian


def fdata(x, L):
    A = L/10.0
    return 2*np.sin(2*np.pi*x/L) + x*(L-x)**2/L**3 * np.cos(x) + \
           5*x*(L-x)/L**2 + A/2 + 0.1*A*np.sin(13*np.pi*x/L)



if __name__=="__main__":

    pars = argparse.ArgumentParser()
    pars.add_argument("--std", type=float, default=30, help="Standard Deviation of the gaussian smoothing filter")

    args = pars.parse_args()
    std  = args.std

    N = 2048
    L = 50.0

    #define the input data and add some noise
    x = np.linspace(0, L, N, endpoint=False)
    original = fdata(x, L)
    noisy = original + 0.5*np.random.randn(N)

    #define the gaussian smoothing window
    window = gaussian(N, std=std)

    #convolve the signal with the window and normalize the output
    conv = convolve(original, window, mode="same") / window.sum()

    #do the plots
    plt.rcParams.update({'font.size': 13, 
                         'text.usetex': True, 
                         'font.family': "Computer Modern"})

    fig, ax = plt.subplots()
    ax.plot(x, noisy, alpha = 0.5, label="$f(x) + n$")
    ax.plot(x, original, label="$f(x)$")
    ax.plot(x, conv, label="smoothed")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.title("Convolution with Gaussian smoothing window")
    plt.legend()
    plt.show()