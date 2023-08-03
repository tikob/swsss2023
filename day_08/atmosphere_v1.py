#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tridiagonal import solve_tridiagonal
# from generate_density_from_temperature import generate_gensity_from_temp
import generate_density_from_temperature

# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------


def calc_gravity(alt):
    radius_array = alt+r_earth
    return generate_density_from_temperature.generate_g_array(radius_array, len(radius_array))

def calc_sc_height(mass, alt, temperature):
    k_boltz = 1.38e-23
    gravity = calculate_gravity(alt)
    return generate_density_from_temperature.generate_H_array(k_boltz, mass, temperature, gravity, len(alt))


def build_hydro(species, temperature, alt):
    mass, n_0 = species
    return generate_density_from_temperature.generate_gensity_from_temp(species, temperature, alt, len(alt))

def plot_density(ax,times, alt, density, m, n_0, species):

    CS = ax.contourf(times,alt, np.log10(density.T), cmap = 'viridis')
    cbar = fig.colorbar(CS)
    ax.set_ylabel("Alt [km]")
    ax.set_xlabel("Time [days]")
    ax.set_title("Density of "+species)
    return ax


def generate_species_dict():
    m_n2 = 28*1.67e-27
    n_n2 = 1.e19
    m_o2 = 32*1.67e-27
    n_o2 = 0.3e19
    m_o = 16*1.67e-27
    n_o = 1.e18
    dict = {'n2': [m_n2, n_n2], 'o2':[m_o2, n_o2], 'o': [m_o, n_o]}

    return dict


if __name__ == "__main__":
    fig, ax = plt.subplots(3, figsize=(12,18))

    dx = 0.25

    # set x with 1 ghost cell on both sides:
    x = np.arange(-dx, 10. + 2 * dx, dx)
    alt = 100+40*x
    dalt = alt[1]-alt[0]

    t_lower = 200.0

    nPts = len(alt)
    nDays = 3
    dt = 0.5
    dt_s = dt*3600.
    times = np.arange(0, nDays*24.0, dt)
    f10_7 = 100 + 50*times/(365.*24.)+ 25.*np.sin(times/(27*24)*2*np.pi)
    AmpDi = 10.
    AmpSd = 5.
    PhaseDi = np.pi/2
    PhaseSd = 3*np.pi/2

    lon = 44.8015

    m_N2 = 28*1.67e-27

    n_0_N2 = 1.e19
    r_earth = 6370e3
    k_boltz = 1.38e-23
    m_o2 = 32*1.67e-27
    n_0_o2 = 0.3e19
    m_o = 16*1.67e-27
    n_0_o = 1.e18


    temperatures = np.zeros([len(times), nPts])
    n2_densities = np.zeros([len(times), nPts])
    o2_densities = np.zeros([len(times), nPts])
    o_densities = np.zeros([len(times), nPts])
    t = np.linspace(200, 1000, len(alt))
    print (len(alt), len(t))
    for counter,hour in enumerate(times):
        ut = hour%24
        LT = lon/15+ut
        t_lower = 200.0 + AmpDi*np.sin(LT/24*2*np.pi+PhaseDi) + AmpSd*np.sin(LT/24*np.pi*2*2 + PhaseSd)
        factor = -np.cos(LT/24*np.pi*2)
        if factor<0: factor = 0


        # Add a source term:
        lam = 80. #10.
        dz = alt[1]-alt[0]
        dz2 = dz*dz

        qback = np.zeros(nPts)
        qback[(alt>200)&(alt<400)] = 0.4
        Sunheat =0.4*f10_7[counter]/100  #100./fr**2
        qeuv = np.zeros(nPts)
        qeuv[(alt>200)&(alt<400)] = (factor*Sunheat)
        q = (qback+qeuv)
        # d = (qback+qeuv)*dz2/lam


        k = dt_s*lam/dz2




        a = np.zeros(nPts) - k
        b = np.zeros(nPts) + 2*k + 1
        c = np.zeros(nPts) - k
        d = np.zeros(nPts) + t + dt_s*q
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
        species = generate_species_dict()
        density_n2=build_hydro(species['n2'], t, alt)
        # density_n2 = generate_density_from_temperature.generate_gensity_from_temp(species['n2'], t, alt, nPts)
        density_o2 = generate_density_from_temperature.generate_gensity_from_temp(species['o2'], t, alt, nPts)
        density_o = generate_density_from_temperature.generate_gensity_from_temp(species['o'], t, alt, nPts)

        temperatures[counter]=t
        n2_densities[counter] = density_n2
        o2_densities[counter] = density_o2
        o_densities[counter] = density_o

    ax[0] = plot_density(ax[0],times/24, alt, n2_densities, m_N2, n_0_N2, "N2")
    ax[1] = plot_density(ax[1],times/24, alt, o2_densities, m_o2, n_0_o2, "O2")
    ax[2] = plot_density(ax[2],times/24, alt, o_densities, m_o, n_0_o, "O")




    plt.show()
    # fig.savefig(plotfile)
    # plt.close()
