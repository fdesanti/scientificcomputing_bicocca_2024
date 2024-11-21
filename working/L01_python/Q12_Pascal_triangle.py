"""
    Print a center formatted Pascal Triangle
"""

import sys

def print_pascal(n):
    """
        Function that prints the Pascal Triangle given N.
        
        Printing is done by computing the maximum width of the last row, 
        then each line is centered accordingly

        Args:
        -----
            n (int) : first n row to print
    """

    def pascal(n, L=None, prev_output=None):
        """
            Computes the Padcal Triangle with a recursive function
        """
        assert n > 0, "n must be greater than 0"
        
        if prev_output is None:
            prev_output = []
            
        if L is None:
            L = []

        if n == 1:
            L = [1]   
        else:
            prev_L, prev_output = pascal(n - 1)
            L = [1] + [prev_L[i] + prev_L[i+1] for i in range(len(prev_L) - 1)] + [1]
        prev_output.append(L)
        return L, prev_output
    
    _, output = pascal(n)
    
    # Calculate the maximum width for formatting
    max_width = len(" ".join(map(str, output[-1]))) + n
    
    # Print each row, centered
    for row in output:
        line = " ".join(map(str, row))
        print(line.center(max_width))

    return output

if __name__ == '__main__':
    N = int(sys.argv[1])

    _=print_pascal(N)
