#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:41:10 2023

@author: holtorf
"""

import numpy as np
import matplotlib.pyplot as plt

# Use Euler's method with different stepsizes to solve the IVP:
# dx/dt = -2*x, with x(0) = 3 over the time-horizon [0,2]

# Compare the numerical approximation of the IVP solution to its analytical
# solution by plotting both solutions in the same figure.

def numerical_dxdt(t, x0, h):
    x = np.zeros(nPts)
    x[0] = x0

    for i in range(len(t)-1):
        x[i+1] = x[i]+h*(-2*x[i])

    return x



def analytic(t):
    x_func = 3*np.exp(-2*t)

    dxdt = -2*x_func
    return x_func, dxdt

if __name__=="__main__":
    fig, ax1 = plt.subplots(figsize=(10,10))
    t_final = 2
    h = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    x0 = 3
    # t = np.arange(0,t_final, h)
    # nPts = len(t)
    # x_func, dxdt = analytic(t)
    # dxdt_numeric = numerical_dxdt(t, x0, h)
    error_array = []
    for i in range(len(h)):
        t = np.arange(0,t_final, h[i])
        nPts = len(t)
        x_func, dxdt = analytic(t)
        dxdt_numeric = numerical_dxdt(t, x0, h[i])
        error = np.sum(np.abs(dxdt_numeric - dxdt)) / len(dxdt_numeric)
        error_array.append(error)
        print ("Error and timestep size: ", h[i], error)
        sError = ' (Err: %5.1f)' % error
        ax1.plot(t, x_func, label= ("Stepsize = "+ str(h[i]) + "; Analytical;"))
        ax1.plot(t, dxdt_numeric, label =("Stepsize = "+ str(h[i])+ "; Numerical; Error: "+sError))
        ax1.set_xlabel("Time")
        ax1.set_ylabel("x(t)")

    ax1.legend()
    plt.show()
