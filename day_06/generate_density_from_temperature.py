import numpy as np
import matplotlib.pyplot as plt


def generate_temperature_array(T_0, T_top):
    # T_array = np.zeros(nPts)
    # Here we define a linear function for temperature
    T_array = np.linspace(T_0, T_top, nPts)
    return T_array


def generate_g_array(radius_array):
    g_array = np.zeros(nPts)
    g_array = 3.99e14/(radius_array)**2

    return g_array


def generate_H_array(k_boltz, m,temperature_array, g_array):
    print (temperature_array[0], g_array[0])
    # temp_average = (temperature_array[1:]+temperature_array[-1])/2
    H_array = k_boltz*temperature_array/(g_array*m)
    # print (H_array[0])
    return (k_boltz*temperature_array/(g_array*m))


def generate_n(T_0, temperature_array, alt_array, H_array):
    density_array =  np.zeros(nPts)
    delta_z = alt_array[1]-alt_array[0]
    # delta_z_arr = np.array([alt_array[0]+delta_z*i for i in range(1,len(alt_array))])
    # print (delta_z)
    # print ("last delta z", delta_z[-1])
    density_array = np.zeros(nPts)
    density_array[0] = n_0
    # print (delta_z, H_array[0])
    for i in range(1,nPts):
        density_array[i] =density_array[i-1]* temperature_array[i-1]/temperature_array[i]*np.exp(-delta_z/H_array[i-1])

    # print (density_array)

    # print (len(temperature_array))
    # print ("z step ", delta_z)
    # density_array[1:] = temperature_array[:-1]/temperature_array[1:]*n_0*np.exp(-delta_z/H_array[1:])

    return density_array


def generate_density_profile(T_0, n_0, alt_array, nPts):
    radius_array = alt_array+r_earth
    temperature_array = generate_temperature_array(T_0, T_top)
    g_array = generate_g_array(radius_array)
    H_array = generate_H_array(k_boltz,m,temperature_array, g_array)
    density_array = generate_n(T_0, temperature_array, alt_array, H_array)

    return density_array



if __name__ == "__main__":
    T_0 = 200
    T_top = 1000
    m = 28*1.67e-27
    alt_0 = 100e3
    alt_top = 500e3
    nPts = 100
    n_0 = 1.e19
    r_earth = 6370e3
    k_boltz = 1.38e-23
    alt_array = np.linspace(alt_0, alt_top, nPts)
    density_array_n2 = generate_density_profile(T_0, n_0, alt_array, nPts)
    m = 32*1.67e-27
    n_0 = 0.3e19
    density_array_o2 = generate_density_profile(T_0, n_0, alt_array, nPts)
    m = 16*1.67e-27
    n_0 = 1.e18
    density_array_o = generate_density_profile(T_0, n_0, alt_array, nPts)

    fig, ax1 = plt.subplots(1,figsize = (10,10))
    ax1.plot(np.log(density_array_n2), alt_array, label="Density profile N2")
    ax1.plot(np.log(density_array_o2), alt_array, label="Density profile O2")
    ax1.plot(np.log(density_array_o), alt_array, label="Density profile O")
    ax1.set_ylabel("Altitudes")
    ax1.set_xlabel("Density")
    # ax1.set_xscale('log')
    ax1.legend()
    plt.show()
