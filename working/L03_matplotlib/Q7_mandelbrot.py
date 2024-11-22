"""Script to plot a mandelbrot set"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_mandelbrot(N=1000, itmax=1000, xmin=-2, xmax=2, ymin=-2, ymax=2):
    N = 1000
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)

    xv, yv = np.meshgrid(x, y, indexing="ij")
    c = xv + 1j*yv
    z = np.zeros((N, N), dtype=np.complex128)
    Counts = np.ones((N, N))

    it=0
    while it<itmax:
        print(f"Iteration {it+1}/{itmax}", end="\r")
        #compute z_{k+1} = z_k**2 + c 
        z = z**2 + c

        #avoid overflows
        z = np.where(np.abs(z) > 1e10, 1e10, z)

        #avoid nans
        z = np.nan_to_num(z)

        Counts[(np.abs(z)>2) * (Counts==1)] = it+1

        it+=1
    print(f"Iteration {it}/{itmax}")
    
    #let's do the plot
    fig, ax = plt.subplots()
    ax.imshow(Counts, norm="log", cmap="magma")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar(ax.imshow(Counts.T, extent=(xmin, xmax, ymin, ymax), norm="log",  cmap="magma", origin="lower"), label="Iterations")
    plt.show()
    return fig, Counts



if __name__ == "__main__":  

    pars = argparse.ArgumentParser()
    pars.add_argument("--N", type=int, default=1000, help="Size of the grid NxX. (Default: 1000)")
    pars.add_argument("--itmax", type=int, default =100, help="Max number of iteraions. (default 100)")
    pars.add_argument("--xmin", type=float, default=-2, help="Min value for x. (Default: -2)")
    pars.add_argument("--xmax", type=float, default= 2, help="Max value for x. (Default: 2)")
    pars.add_argument("--ymin", type=float, default=-2, help="Min value for y. (Default: -2)")
    pars.add_argument("--ymax", type=float, default= 2, help="Max value for y. (Default: 2)")
    
    args = pars.parse_args()

    N     = args.N
    itmax = args.itmax
    xmin  = args.xmin
    xmax  = args.xmax
    ymin  = args.ymin
    ymax  = args.ymax

    _ = plot_mandelbrot(N=N, itmax=itmax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)