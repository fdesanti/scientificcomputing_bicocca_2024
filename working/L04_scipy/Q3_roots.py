import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams.update({'font.size': 13, 
                        'text.usetex': True, 
                        'font.family': "Computer Modern"})
def q(x):
    return x**3 -2*x**2 -11*x + 12

if __name__ == "__main__":
    # Define the polynomial q(x) = x^3 - 2x^2 - 11x + 12
    x = np.linspace(-4.5, 4.5, 100)
    y = q(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="$q(x)$")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.minorticks_on()
    ax.legend(frameon=False, loc="best")

    # find the roots with brentq
    crossing_indices = y[1:] * y[:-1] < 0
    crossing_points = x[:-1][crossing_indices] + (x[1] - x[0]) / 2
    print("Approximate roots:", crossing_points)

    roots = []
    for n, xroot in enumerate(crossing_points):
        # Use brentq to find the root close to each crossing point
        root = brentq(q, xroot - 0.5, xroot + 0.5, full_output=False)
        print("Root found:", root)
        roots.append(root)

        # Plot roots 
        ax.scatter(root, q(root), color='red', label='root' if n==0 else "_nolegend_")

    # plot the roots
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Roots of the polynomial $q(x) = x^3 -2x^2-11x+12$")
    plt.show()
