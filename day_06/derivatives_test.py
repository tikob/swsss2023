#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Take first derivative of a function
# ----------------------------------------------------------------------





def first_derivative(f, x):

    """ Function that takes the first derivative

    Parameters
    ----------
    f - values of a function that is dependent on x -> f(x)
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    dfdx - the first derivative of f(x)

    Notes
    -----
    take the first derivative of f(x) here

    """

    nPts = len(f)

    # dx = np.diff(x)[0]
    dfdx = np.zeros(nPts)
    # The boundary cells are treated with 1-sided derivative
    dfdx[0] = (f[1]-f[0])/dx
    dfdx[-1] = (f[-1]-f[-2])/dx

    dfdx[1:-1] = [(f[x+1]-f[x-1])/(2*dx) for x in range(1, nPts-1)]
    # dfdx[0] = dfdx[1]
    # dfdx[-1] = dfdx[-2]


    return dfdx

# ----------------------------------------------------------------------
# Take second derivative of a function
# ----------------------------------------------------------------------

def second_derivative(f, x):

    """ Function that takes the second derivative

    Parameters
    ----------
    f - values of a function that is dependent on x -> f(x)
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    d2fdx2 - the second derivative of f(x)

    Notes
    -----
    take the second derivative of f(x) here

    """

    nPts = len(f)
    # dx = np.diff(x)[0]


    d2fdx2 = np.zeros(nPts)
    # The boundary cells are treated with 1-sided derivative

    d2fdx2[1:-1] =[(f[x+1]+f[x-1]-2*f[x])/dx**2 for x in range(1, nPts-1)]
    # d2fdx2[0] = (f[1]+f[0]-2*f[0])/dx**2
    # d2fdx2[-1] = (f[-1]+f[-2]-2*f[-2])/dx**2
    # d2fdx2[0] = (d2fdx2[2] - d2fdx2[1])/(x[2]-x[1])*(x[0]-x[1])+d2fdx2[1]
    d2fdx2[-1] = d2fdx2[-2]
    d2fdx2[0] = d2fdx2[1]
    # d2fdx2[-1] = (d2fdx2[-2] - d2fdx2[-3])/(x[-2]-x[-3])*(x[-1]-x[-2])+d2fdx2[-2]


    return d2fdx2

# ----------------------------------------------------------------------
# Get the analytic solution to f(x), dfdx(x) and d2fdx2(x)
# ----------------------------------------------------------------------

def analytic(x):

    """ Function that gets analytic solutions

    Parameters
    ----------
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    f - the function evaluated at x
    dfdx - the first derivative of f(x)
    d2fdx2 - the second derivative of f(x)

    Notes
    -----
    These are analytic solutions!

    """

    # f = x**2*np.cos(x)+x*np.exp(x)
    #
    # dfdx = x*(2*np.cos(x)-x*np.sin(x))+np.exp(x)*(1+x)
    # d2fdx2 = np.cos(x)*(2-x**2) - 4*x*np.sin(x) +np.exp(x)*(x+2)

    f = 4 * x ** 2 - 3 * x -7
    dfdx = 8 * x - 3
    d2fdx2 = np.zeros(len(f)) + 8.0

    return f, dfdx, d2fdx2

def integral_analytic(x):
    f = x**2
    integral_of_f = x[-1]**3/3-x[0]**3/3

    return f, integral_of_f


def numerical_integration(f, x):
    numerical_integral = np.sum([(f[1:]+f[:-1])/2*dx])
    return numerical_integral
# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # get figures:
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    fig, (ax4,ax5) = plt.subplots(2, figsize = (10,10))

    # define dx:
    nPts = np.array([10*2**i for i in range(0, 8)])
    print (nPts)
    error_arr = []
    for n in nPts:
        # dx = np.pi/4
        x = np.linspace(-2.0 * np.pi, 2.0 * np.pi, n)
        dx = x[0]-x[1]
        f, a_dfdx, a_d2fdx2 = analytic(x)
        n_dfdx = first_derivative(f, x)
        n_d2fdx2 = second_derivative(f, x)
        error = np.sum(np.abs(n_dfdx - a_dfdx)) / len(n_dfdx)
        error_arr.append(error)


    ax4.plot(nPts, error_arr)
    # ax4.set_yscale('log')
    # ax4.set_xscale('log')
    ax4.set_xlabel("Number of Points")
    ax4.set_ylabel("Error Values")
    ax5.plot(np.log(nPts), np.log(error_arr))
    # ax5.set_yscale('log')
    # ax5.set_xscale('log')
    ax5.set_xlabel("Number of Points")
    ax5.set_ylabel("Error Values")
    ax5.set_title("Log-Log plot")
    fig.savefig('errors.png')

    # arange doesn't include last point, so add explicitely:


    # get analytic solutions:


    # function, integral_function = integral_analytic(x)

    # numerical_integral_f = numerical_integration(function, x)

    # print ("analytical integral ", integral_function, "numerical integral: ", numerical_integral_f)

    # get numeric first derivative:


    # get numeric first derivative:



    # n_d2fdx2_num = first_derivative(n_dfdx, x)
    # plot:

    ax1.plot(x, f, label = "f(x) = $x^2cos(x)+xe^x$")
    ax1.legend()
    # ax1.set_yscale('log')

    # plot first derivatives:

    sError = ' (Err: %5.1f)' % error
    ax2.plot(x, a_dfdx, color = 'black', label = 'Analytic')
    ax2.plot(x, n_dfdx, color = 'red', label = 'Numeric'+ sError)
    ax2.scatter(x, n_dfdx, color = 'red')
    # ax2.set_yscale('log')
    ax2.legend()

    # plot second derivatives:
    error2 = np.sum(np.abs(n_d2fdx2 - a_d2fdx2)) / len(n_d2fdx2)
    sError2 = ' (Err: %5.1f)' % error2
    # error3 = np.sum(np.abs(n_d2fdx2_num - a_d2fdx2)) / len(n_d2fdx2_num)
    # sError3 = ' (Err: %5.1f)' % error3
    ax3.plot(x, a_d2fdx2, color = 'black', label = 'Analytic')
    ax3.plot(x, n_d2fdx2, color = 'red', label = 'Numeric'+ sError2)
    # ax3.plot(x, n_d2fdx2_num, color = 'blue', label = '2 times first derivative'+sError3)
    ax3.scatter(x, n_d2fdx2, color = 'red')
    # ax3.scatter(x, n_d2fdx2_num, color = 'blue')
    # ax3.set_yscale('log')
    ax3.legend()

    plotfile = 'plot.png'
    print('writing : ',plotfile)
    plt.show()
    # fig.savefig(plotfile)
    # plt.close()
