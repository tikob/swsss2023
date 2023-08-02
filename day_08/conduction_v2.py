#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tridiagonal import solve_tridiagonal

# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

if __name__ == "__main__":
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)

    dx = 0.25

    # set x with 1 ghost cell on both sides:
    x = np.arange(-dx, 10. + 2 * dx, dx)
    alt = 100+40*x
    dalt = alt[1]-alt[0]
    # fr = dalt/dx
    # print ("fraction", fr,  100./fr)

    x = alt
    dx = dalt


    t_lower = 200.0
    t_upper = 1500.0

    nPts = len(x)
    nDays = 27
    dt = 1
    times = np.arange(0, nDays*24.0, dt)
    f10_7 = 100 + 50*times/(365.*24.)+ 25.*np.sin(times/(27*24)*2*np.pi)
    # print ()
    # ax.plot(times, f10_7)

    # ax.set_xlabel("times")
    # ax.set_ylabel("F 10.7")
    AmpDi = 10.
    AmpSd = 5.
    PhaseDi = np.pi/2
    PhaseSd = 3*np.pi/2

    lon = 44.8015

    temperatures = np.zeros([len(times), nPts])
    for counter,hour in enumerate(times):

        ut = hour%24
        LT = lon/15+ut
        t_lower = 200.0 + AmpDi*np.sin(LT/24*2*np.pi+PhaseDi) + AmpSd*np.sin(LT/24*np.pi*2*2 + PhaseSd)
        # f10_7 = 100+50*hour/(365.*24.)+ 25.np.sin(hour/27*2*np.pi)
        # print (ut)
        factor = -np.cos(LT/24*np.pi*2)
        # print (factor)
        if factor<0: factor = 0
        a = np.zeros(nPts) - 1
        b = np.zeros(nPts) + 2
        c = np.zeros(nPts) - 1
        d = np.zeros(nPts)

        # Add a source term:
        lam = 80. #10.
        dz = x[1]-x[0]
        dz2 = dz*dz

        qback = np.zeros(nPts)
        qback[(x>200)&(x<400)] = 0.4
        Sunheat =0.4*f10_7[counter]/100  #100./fr**2
        qeuv = np.zeros(nPts)
        qeuv[(x>200)&(x<400)] = (factor*Sunheat)
        d = (qback+qeuv)*dz2/lam

        # boundary conditions (bottom - fixed):
        a[0] = 0
        b[0] = 1
        c[0] = 0
        d[0] = t_lower

        # top - fixed:
        a[-1] = 1
        b[-1] = -1
        c[-1] = 0
        d[-1] = 0#t_upper


        # solve for Temperature:
        t = solve_tridiagonal(a, b, c, d)
        temperatures[counter]=t


        # plot:


        # ax.plot(x, t, label = hour)
        # ax.set_xlabel("x")
        # ax.set_ylabel("Temperature")

    CS = ax.contourf(times/24.,x, temperatures.T, cmap = 'viridis')
    cbat = fig.colorbar(CS)
    cbat.ax.set_ylabel('T')
    ax.set_ylabel("Alt")
    ax.set_xlabel("Time")
    ax.set_title("Temperature above longitude "+ str(lon)+"$^\circ$")
    plt.show()
    # fig.savefig(plotfile)
    # plt.close()
