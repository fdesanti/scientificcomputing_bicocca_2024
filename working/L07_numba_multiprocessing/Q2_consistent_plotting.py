import corner
import shutil
import matplotlib.pyplot as plt



def my_plot(plot_func):
    """A decorator to apply LaTeX styling to matplotlib plots.
    """
    def wrapper_plot(*args, **kwargs):
        plt.rcParams["font.size"] = 18

        #use latex if available on the machine
        if shutil.which("latex"): 
            plt.rcParams.update({"text.usetex": True, 
                                 "font.family": "serif",
                                 "text.latex.preamble": r"\usepackage{amsmath}"
                                 })
            
        fig = plot_func(*args, **kwargs)
        fig.savefig("saved_plot.pdf", bbox_inches="tight")
        return fig
    return wrapper_plot


@my_plot
def plot_corner(samples, labels):
    """Plot a corner plot using the corner package.
    
    Args:
        samples    (ndarray): The samples to plot.
        labels (list of str): The labels for each parameter.
        
    Returns:
        fig: The matplotlib figure object.
    """
    fig = corner.corner(samples, labels=labels, show_titles=True)
    return fig


if __name__ == "__main__":
    import numpy as np

    # Generate samples from a multivariate gaussian distribution
    np.random.seed(42)
    samples = np.random.randn(10000, 3)
    labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]

    # Generate the corner plot
    plot_corner(samples, labels)