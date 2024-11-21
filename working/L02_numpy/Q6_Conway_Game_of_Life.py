"""
    Conway's Game of Life
"""
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


class GameofLife:
    """"
        Args:
        -----
            N (int): the size of the grid (NxN)
    """

    def __init__(self, N=10, M=None, boundary='wall'):

        self.N = N

        self._pad_mode = {"wall": "constant", "pacman": "wrap"}
        self.boundary = boundary
        assert boundary in self._pad_mode.keys(),f"Invalid boundary condition. You can choose from {self._pad_mode.keys()}"


    def _initialize_grid(self):

        self.grid  = np.zeros((N, N))

        i = np.arange(self.N)

        Nmax = np.random.choice(np.arange(self.N**2))
        ixs = np.random.choice(i, Nmax, replace=True)
        jxs = np.random.choice(i, Nmax, replace=True)
        self.grid[ixs, jxs] = 1.0

        return self.grid
        
    

    @staticmethod
    def conv2d(grid, kernel, pad_mode):
        """
            2D convolution of the grid. 
            Implements a 2d convolution of the input grid with a 2d kernel.
            First we pad the input grid with the suited boundary condition.
            We use then the numpy's np.lib.stride_tricks.sliding_window_view function
            to create a view of the input grid. (See https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html)
            (This avoids ugly for loops)

            This creates a new matrix X of shape [N, M, K, L] where (N, M) and (K, L) are the dimensions of the input grid
            and kernel respectively

            Then Conv is just Conv = Sum_k Sum_l (X_ijkl * W_kl)

            Args:
            -----
                grid (numpy.ndarray)   : The input grid of shape (N, M)
                kernel (numpy.ndarray) : The kernel of shape (K, L) to be convolved with the grid

            Returns:
            --------
                conv (numpy.ndarray) : The convolution of the grid with the kernel
        """
        #pad the grid according to the boundary condition
        padded = np.pad(grid, 1, mode=pad_mode)

        #create the sliding windows
        slidinng_windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
        
        #compute the convolution
        convolved = np.sum(slidinng_windows*kernel, axis=(2, 3))
        
        return convolved



    def count_neighbours(self):
        """
            Counts the alive neighbours of each cell. 
            We use a Conv2D approach where the gris is convolved with a kernel 
            K = [[1, 1, 1], 
                 [1, 0, 1], 
                 [1, 1, 1]]
            
            This function accounts for the boundary condition specified in the constructor
        """

        kernel = np.array([[1, 1, 1], 
                           [1, 0, 1], 
                           [1, 1, 1]])

        return self.conv2d(self.grid, kernel, pad_mode=self._pad_mode[self.boundary])
        
    
    def evolve(self):
        """
            Computes the evolutionary step in the Game of Life. 
            First we count the alive neighbours of each cell
            Then, based on the game's rules we determine who dies and who becomes alive
        """
        #copy the current grid 
        grid = self.grid.copy()

        #count neighbours
        neighbours = self.count_neighbours()

        #select who dies
        dies = (neighbours>=3) | (neighbours<2)

        #select who comes alive
        birth = (neighbours==3)

        grid[dies]  = 0
        grid[birth] = 1
        
        return grid
        

    def play(self):
        """
            Main function that evolves the cell grid and plots the animated result. 
            We use a while loop which is interrupted if the Game reaches a stationary state
        """

        #the current grid is the initialized one
        prev_grid = self._initialize_grid()
        
        #create the figure for the grid plot
        fig, ax = plt.subplots()
        plt.title("Welcome to the Conway's Game of Life!")
        im = ax.imshow(self.grid, cmap="binary_r")
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        
        #evolution loop
        while True:
            #get the new evolved grid
            new_grid = self.evolve()

            #plot the new grid
            im.set_array(new_grid)
            plt.draw()
            plt.pause(0.1)

            #check wether we reached a stationary state
            if prev_grid is new_grid:
                break
            else:
                prev_grid = new_grid
                self.grid = new_grid.copy()
            
            #time.sleep(0.1)


if __name__=='__main__':

    pars = argparse.ArgumentParser()
    pars.add_argument("--N", type=int, default=10, help="Size of the grid NxN. (Default: 10)")
    pars.add_argument("--boundary", type=str, default="wall", help="Boundary condition. (Default: wall)")
    pars.add_argument("--seed", default=None, help="Random seed for reproducibility. (Default: None)")
    
    args = pars.parse_args()

    if args.seed:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
    
    N        = args.N
    boundary = args.boundary


    Game = GameofLife(N=N, boundary=boundary)
    Game.play()