#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Tinatin Baratashvili'
__email__ = 'tinatin.baratashvili@kuleuven.be'

from math import factorial
from math import pi


def cos_approx(x, accuracy=10):
    """
    x (float): to evaluate cosine of
    accuracy (int): Number of Taylor series coefficient to use
    Retuns (float): approximate cosine of *x*.

        This function will calculate the taylor approximation
        with the given accuracy.
    """

    # He defined the function separetely to calculate the approximation

    return sum([(-1)**n/factorial(2*n)*x**(2*n) for n in range(accuracy+1)])



# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    assert cos_approx(0) < 1+1.e-2 and cos_approx(0) > 1-1.e-2, "cos(0) is not 1"
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
