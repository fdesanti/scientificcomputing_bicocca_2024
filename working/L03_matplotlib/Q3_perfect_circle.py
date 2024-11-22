import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def circle(x0=[0], y0=[0], R=[1], N=int(1e3)):
    """
        Functions that generates coordinate points for circle(s).
        
        Args:
        -----
            x0 (list or numpy.ndarray): x-coordinate of circle center(s).
            y0 (list or numpy.ndarray): y-coordinate of circle center(s).
            R  (list or numpy.ndarray): radius of circle(s).
            N  (int)                  : number of points to generate on the circle.

        Returns:
        --------
            x (numpy.ndarray): x-coordinate of circle(s) points sorted according to R.
            y (numpy.ndarray): y-coordinate of circle(s) points sorted according to R.
            R (numpy.ndarray): sorted radius of circle(s).
    
    """
    #broadcast to numpy.ndarrays
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    R  = np.asarray(R)

    #sort according to radius value
    isort = np.argsort(R)
    x0 = x0[isort]
    y0 = y0[isort]
    R  = R[isort]

    #determine if we have multiple circles
    n_circs = max(len(x0), len(y0), len(R))

    #generate theta values
    theta = np.linspace(0, 2*np.pi, N)
    if n_circs :
        theta = np.expand_dims(theta, 1)
        
    #compute x and y coordinates
    x = R * np.cos(theta) + x0
    y = R * np.sin(theta) + y0

    return x.T, y.T, R #we transpose so that they have shape [ncirc, N]

def plot_circles(x, y, R, cmap="magma"):
    """
        Function that plots filled circles in a single figure
        according to a specified colormap.

        Args:
        -----
            x (numpy.ndarray): x-coordinate of circle(s) points.
            y (numpy.ndarray): y-coordinate of circle(s) points.
            R (numpy.ndarray): radius of circle(s).
            cmap (str)       : name of colormap to use.

        Returns:
        --------
            fig, ax (tuple): matplotlib figure and axis objects
    """

    # get the colormap
    color_map = plt.get_cmap(cmap)  
    norm = Normalize(vmin=R.min(), vmax=R.max())
    
    # Get the colors based on radii
    colors = [color_map(norm(r)) for r in R]

    #create the figure
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box') #Tuft would be upset with squeezed circles

    #plot the various circles
    for xval, yval, color in zip(x, y, colors):        
        ax.plot(xval, yval, color=color )
        ax.fill(xval, yval, color=color, alpha=0.1)
    
    # Add a colorbar
    sm = ScalarMappable(cmap=color_map, norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("$R$")
    
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.show()
    return fig, ax


if __name__ == '__main__':
    #update rc params
    plt.rcParams.update({
        "font.size":12,
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    #draw some random centers/radii
    x0 = np.random.uniform(-5, 5, 10)
    y0 = np.random.uniform(-5, 5, 10)
    R  = np.random.uniform(1, 10, 10)

    #get the circles
    xc, yc, rc = circle(x0, y0, R)

    #plot them
    _ = plot_circles(xc, yc, rc)

