#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%%
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
# print ("Hello World")

#%%
"""
Creating a random numpy array
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
# print(data_arr)

#%%
"""
TODO: Writing and reading numpy file
"""
# Save the data_arr variable into a .npy file
np.save('test_np_save.npy', data_arr)


# Load data from a .npy file
data_arr_loaded = np.load('test_np_save.npy')


# Verify that the loaded data matches the initial data
# print (np.equal(data_arr,data_arr_loaded))
# print(data_arr==data_arr_loaded)

#%%
"""
TODO: Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
# print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez('test_savez.npz', data_arr,data_arr2)

# Load the numpy zip file
npzfile = np.load('test_savez.npz')
# print(npzfile)
# print ("Variable names within this file: ", sorted(npzfile.files))

# Verify that the loaded data matches the initial data
# print(npzfile['arr_0'])
# print((data_arr==npzfile['arr_0']).all())
# print((data_arr2==npzfile['arr_1']).all())

# #%%
# """
# Error and exception
# """
# # Exception handling, can be use with assertion as well
# try:
#     # Python will try to execute any code here, and if there is an exception
#     # skip to below
#     print(np.equal(data_arr,npzfile).all())
# except:
#     # Execute this code when there is an exception (unable to run code in try)
#     print("The codes in try returned an error.")
#     print(np.equal(data_arr,npzfile['arr_0']).all())
#
# # #%%
# """
# TODO: Error solving 1
# """
# # What is wrong with the following line?
# try:
#     np.equal(data_arr,data_arr2)
# except:
#     print ("This arrays are different.")
#
# #%%
# """
# TODO: Error solving 2
# """
# # What is wrong with the following line?
# try:
#     np.equal(data_arr2,npzfile['data_arr2'])
# except:
#     print("This is not the key in npz file")
#
# # #%%
# """
# TODO: Error solving 3
# """
# # What is wrong with the following line?
# try:
#     numpy.equal(data_arr2,npzfile['arr_1'])
# except:
#     print("We have imported numpy as np.")
#     np.equal(data_arr2,npzfile['arr_1'])


#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
from scipy.io import loadmat

dir_density_Jb2008 = 'JB2008/2002_JB2008_density.mat'

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
except:
    print("File not found. Please check your directory")

# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']

# The shape command now works
print(JB2008_dens.shape)

#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""
# Import required packages
import matplotlib.pyplot as plt

# Before we can visualize our density data, we first need to generate the
# discretization grid of the density data in 3D space. We will be using
# np.linspace to create evenly sapce data between the limits.

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0] #nof stands for number of
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]


# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = np.linspace(0,8759,5, dtype = int)
# print(time_array_JB2008)
# For the dataset that we will be working with today, you will need to reshape
# them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,
                                               nofAlt_JB2008,8760), order='F') # Fortran-like index order
# print(JB2008_dens_reshaped.shape)
#%%
"""
TODO: Plot the atmospheric density for 400 KM for the first time index in
      time_array_JB2008 (time_array_JB2008[0]).
"""

import matplotlib.pyplot as plt



# Look for data that correspond to an altitude of 400 KM
alt = 400
hi = np.where(altitudes_JB2008==alt)

# fig, axs = plt.subplots(4, figsize=(15,30), sharex=True)
# # help(plt.contourf)
# # print (JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]])
# axs[0].contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]].squeeze().T)
# print ((JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]].squeeze().T).shape)
# cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]].squeeze().T)
# cbar = fig.colorbar(cs,ax=axs[0])
# axs[0].set_title("JB2008 density at 400 km, t = "+str(time_array_JB2008[0])+" hrs")
# cbar.ax.set_ylabel('Density')
# for i in range(len(time_array_JB2008)):
#
#     axs[i].contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_array_JB2008[i]].squeeze().T)
#     cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_array_JB2008[i]].squeeze().T)
#     cbar = fig.colorbar(cs,ax=axs[i])
#     axs[i].set_title("JB2008 density at 400 km, t = "+str(time_array_JB2008[i])+" hrs")
#     cbar.ax.set_ylabel('Density')
# # plt.show()


#%%
"""
TODO: Plot the atmospheric density for 300 KM for all time indexes in
      time_array_JB2008
"""

#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002.
# Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]

print('The dimension of the data are as followed (local solar time,latitude,altitude):', dens_data_feb1.shape)


# # mean_densities = [dens_data_feb1[:,:,i].mean() for i in range(nofAlt_JB2008)]
# mean_densities_JB2008 = np.mean(dens_data_feb1,axis=(0,1))
# fig, ax = plt.subplots()
# ax.plot(altitudes_JB2008, mean_densities_JB2008,'-.', label ="JB2008")
# ax.set_xlabel("Altitude")
# ax.set_ylabel("Density")
# ax.set_title("Mean densities VS Altitude")

# plt.show()
#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density
field at 310km

"""
# Import required packages
import h5py
loaded_data = h5py.File('TIEGCM/2002_TIEGCM_density.mat')
# This is a HDF5 dataset object, some similarity with a dictionary
print('Key within dataset:',list(loaded_data.keys()))
tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]

# We will be using the same time index as before.
time_array_tiegcm = time_array_JB2008

# Each data correspond to the density at a point in 3D space.
# We can recover the density field by reshaping the array.
# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')

print(JB2008_dens_reshaped.shape, tiegcm_dens_reshaped.shape)

#%%
"""
TODO: Plot the atmospheric density for 310 KM for all time indexes in
      time_array_tiegcm
"""
alt = 310
hi_tiegcm = np.where(altitudes_tiegcm==alt)
# fig, axes = plt.subplots(5, figsize=(15,30), sharex=True)
#
# for i in range(len(time_array_JB2008)):
#     axes[i].contourf(localSolarTimes_tiegcm,latitudes_tiegcm,tiegcm_dens_reshaped[:,:,hi_tiegcm,time_array_JB2008[i]].squeeze().T)
#     cs = plt.contourf(localSolarTimes_tiegcm,latitudes_tiegcm,tiegcm_dens_reshaped[:,:,hi_tiegcm,time_array_JB2008[i]].squeeze().T)
#     cbar = fig.colorbar(cs,ax=axes[i])
#     axes[i].set_title("TIE-GCM density at 310 km, t = "+str(time_array_JB2008[i])+" hrs")
#     cbar.ax.set_ylabel('Density')
# # plt.show()


# #%%
# """
# Assignment 1.5
#
# Can you plot the mean density for each altitude at February 1st, 2002 for both
# models (JB2008 and TIE-GCM) on the same plot?
# """
#
# # First identidy the time index that corresponds to  February 1st, 2002.
# # Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1_tiegcm = tiegcm_dens_reshaped[:,:,:,time_index]
tiegcm_dens_feb1_mean = np.mean(dens_data_feb1_tiegcm, axis=(0,1))
# ax.plot(altitudes_tiegcm, tiegcm_dens_feb1_mean, label ="TIE GCM")
# ax.semilogy()
# ax.legend()
# plt.grid()
# plt.show()
#
# # #%%
# """"
# # Data Interpolation (1D)
#
# # Now, let's us look at how to do data interpolation with scipy
# """
# # Import required packages
# from scipy import interpolate
#
# # Let's first create some data for interpolation
# x = np.arange(0, 12, 3)
# y = np.exp(-x/3.0)
#
#
# interp_func_1d = interpolate.interp1d(x,y)
# xnew = np.arange(0,9, 0.1)
# ynew = interp_func_1d(xnew)
#
# interp_func_1d_cubic = interpolate.interp1d(x,y,kind='cubic')
# ycubic = interp_func_1d_cubic(xnew)
#
# interp_func_1d_quadratic = interpolate.interp1d(x,y,kind='quadratic')
# yquadratic = interp_func_1d_quadratic(xnew)
#
# plt.subplots(1,figsize=(10,6))
# plt.plot(x,y,'o', xnew, ynew,'*', xnew,ycubic, '--', xnew, yquadratic,'--', linewidth=2)
# plt.legend(['Initial Points', 'Interpolated line-linear', 'Interpolated line-cubic', 'Interpolated line-quadratic'],fontsize=16)
# plt.xlabel('x', fontsize=18)
# plt.ylabel('y', fontsize=18)
# plt.show()

# #%%
# """
# Data Interpolation (3D)
#
# Now, let's us look at how to do data interpolation with scipy
# """
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)

interpolated_function_1 = RegularGridInterpolator((x,y,z), sample_data)

pts = np.array([[2.1, 6.2, 8.3], [3.3,5.2,7.1]])
print('Using interpolation method:',interpolated_function_1(pts))
print('From true function:',function_1(pts[:,0],pts[:,1],pts[:,2]))

#%%
# """
# Saving mat file
#
# Now, let's us look at how to we can save our data into a mat file
# """
# # Import required packages
# from scipy.io import savemat
#
# a = np.arange(20)
# mdic = {"a": a, "label": "experiment"} # Using dictionary to store multiple variables
# savemat("matlab_matrix.mat", mdic)

# #%%
# """
# Assignment 2 (a)
#
# The two data that we have been working on today have different discretization
# grid.
#
# Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on
# February 1st, 2002, with the discretized grid used for the JB2008
# ((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).
# """
time_index=31*24
dens_data_feb1 = tiegcm_dens_reshaped[:,:,:,time_index]

"""
    The interpolationg of a 3d grid for TIE GCM data. the (x,y,z) coordinates are LST, LAT and ALT
    The value is the density data at all lst, latitudes and altitudes, for the given time_index.

"""
interpolated_function_1 = RegularGridInterpolator((localSolarTimes_tiegcm,latitudes_tiegcm,altitudes_tiegcm), dens_data_feb1, bounds_error=False,fill_value=None)

"""
    Now, we generate the 2d grid for LST vs LAT coordinates at alt = 400,
    and set the value in this 2x2 grid, with the interpolated value from
    the previously 3D interpolated grid for TIE GCM. we are getting the values for the JB2008 locations,
    to compare at altitude 400km, beause TIE GCM data does not have exactly 400 km.

"""

# dens_400 = np.meshgrid(interpolated_function_1((localSolarTimes_JB2008, latitudes_JB2008,400)))
# xyz_grid = np.meshgrid(localSolarTimes_JB2008,latitudes_JB2008, 400)
# #
# dens_400 = interpolated_function_1(xyz_grid.squeeze())



dens_400 = np.zeros((nofLst_JB2008,nofLat_JB2008))

for lst in range(nofLst_JB2008):
    for lat in range(nofLat_JB2008):
        dens_400[lst, lat] = interpolated_function_1((localSolarTimes_JB2008[lst], latitudes_JB2008[lat],400))


print (dens_400.shape)
# interpolation_tiegcm_dens = interpolated_function_1(generated_location_points)

fig, axs = plt.subplots(4, figsize=(15,30), sharex=True)
# help(plt.contourf)
# print (JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]])
axs[0].contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T)

cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T)
cbar = fig.colorbar(cs,ax=axs[0])
axs[0].set_title("JB2008 density at 400 km, t = "+str(time_index)+" hrs")
cbar.ax.set_ylabel('Density')

axs[1].contourf(localSolarTimes_JB2008,latitudes_JB2008,dens_400.T)
cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,dens_400.T)
cbar = fig.colorbar(cs,ax=axs[1])
axs[1].set_title("JB2008 density at 400 km, t = "+str(time_index)+" hrs")
cbar.ax.set_ylabel('Density')
# plt.show()
#
# #%%
# """
# Assignment 2 (b)
#
# Now, let's find the difference between both density models and plot out this
# difference in a contour plot.
# """
JB2008_dens = JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T
tiegcm_difference = (dens_400.T - JB2008_dens)
print (tiegcm_difference)
axs[2].contourf(localSolarTimes_JB2008,latitudes_JB2008,tiegcm_difference,vmin = -3.2e-12, vmax=3.2e-12)
cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,tiegcm_difference,vmin = -3.2e-12, vmax=3.2e-12)
cbar = fig.colorbar(cs,ax=axs[2])
axs[2].set_title("The difference between JB2008 and TIE GCM data at 400 km, t = "+str(time_index)+" hrs")
cbar.ax.set_ylabel('Density')
#
#
#
#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in
terms of absolute percentage difference/error (APE). Let's plot the APE
for this scenario.

APE = abs(tiegcm_dens-JB2008_dens)/tiegcm_dens
"""

JB2008_dens = JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T
tiegcm_APE = abs(dens_400.T - JB2008_dens)/dens_400.T
axs[3].contourf(localSolarTimes_JB2008,latitudes_JB2008,tiegcm_APE)
cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,tiegcm_APE)
cbar = fig.colorbar(cs,ax=axs[3])
axs[3].set_title("The APE between JB2008 and TIE GCM data at 400 km, t = "+str(time_index)+" hrs")
cbar.ax.set_ylabel('Density')
plt.show()
