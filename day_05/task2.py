import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import h5py

dir_density_Jb2008 = '../day_03/JB2008/2002_JB2008_density.mat'

# Load Density Data





# alt = 400
# hi = np.where(altitudes_JB2008==alt)

# fig, axs = plt.subplots(3, figsize=(15,30), sharex=True)
# # help(plt.contourf)
# # print (JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]])
# axs[0].contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index_start].squeeze().T)
# # print ((JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]].squeeze().T).shape)
# cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index_start].squeeze().T)
# cbar = fig.colorbar(cs,ax=axs[0])
# axs[0].set_title("JB2008 density at 400 km, t = "+str(time_index_start)+" hrs")
# cbar.ax.set_ylabel('Density')
#
# axs[1].contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index_dst].squeeze().T)
# cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index_dst].squeeze().T)
# cbar = fig.colorbar(cs,ax=axs[1])
# axs[1].set_title("JB2008 density at 400 km, t = "+str(time_index_dst)+" hrs")
# cbar.ax.set_ylabel('Density')
#
# axs[2].contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index_end].squeeze().T)
# cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index_end].squeeze().T)
# cbar = fig.colorbar(cs,ax=axs[2])
# axs[2].set_title("JB2008 density at 400 km, t = "+str(time_index_end)+" hrs")
# cbar.ax.set_ylabel('Density')
# plt.show()
#
# for i in range(len(time_array_JB2008)):
#
#     axs[i].contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_array_JB2008[i]].squeeze().T)
#     cs = plt.contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_array_JB2008[i]].squeeze().T)
#     cbar = fig.colorbar(cs,ax=axs[i])
#     axs[i].set_title("JB2008 density at 400 km, t = "+str(time_array_JB2008[i])+" hrs")
#     cbar.ax.set_ylabel('Density')
# # plt.show()
#

# density_20_april_start =JB2008_dens_reshaped[:,:,:,time_index_start]
# density_20_april_dst =JB2008_dens_reshaped[:,:,:,time_index_dst]
# density_20_april_end =JB2008_dens_reshaped[:,:,:,time_index_end]

# # mean_densities = [dens_data_feb1[:,:,i].mean() for i in range(nofAlt_JB2008)]
# mean_densities_JB2008_start = np.mean(density_20_april_start,axis=(0,1))
# fig, ax = plt.subplots()
# ax.plot(altitudes_JB2008, mean_densities_JB2008,'-.', label ="JB2008")
# ax.set_xlabel("Altitude")
# ax.set_ylabel("Density")
# ax.set_title("Mean densities VS Altitude")
#
# plt.show()

#
# loaded_data = h5py.File('../day_03/TIEGCM/2002_TIEGCM_density.mat')
# # This is a HDF5 dataset object, some similarity with a dictionary
# print('Key within dataset:',list(loaded_data.keys()))
# tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
# altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
# latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
# localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
# nofAlt_tiegcm = altitudes_tiegcm.shape[0]
# nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
# nofLat_tiegcm = latitudes_tiegcm.shape[0]
#
# time_array_tiegcm = time_array_JB2008
#
# tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')
#
# print(JB2008_dens_reshaped.shape, tiegcm_dens_reshaped.shape)
#
# alt = 310
# hi_tiegcm = np.where(altitudes_tiegcm==alt)
# # fig, axes = plt.subplots(5, figsize=(15,30), sharex=True)
#
# for i in range(len(time_array_JB2008)):
#     axes[i].contourf(localSolarTimes_tiegcm,latitudes_tiegcm,tiegcm_dens_reshaped[:,:,hi_tiegcm,time_array_JB2008[i]].squeeze().T)
#     cs = plt.contourf(localSolarTimes_tiegcm,latitudes_tiegcm,tiegcm_dens_reshaped[:,:,hi_tiegcm,time_array_JB2008[i]].squeeze().T)
#     cbar = fig.colorbar(cs,ax=axes[i])
#     axes[i].set_title("TIE-GCM density at 310 km, t = "+str(time_array_JB2008[i])+" hrs")
#     cbar.ax.set_ylabel('Density')
# plt.show()

# print ("Altitudes JB2008: ", altitudes_JB2008)

def create_lat_lon_meanplots_JB2008(filename, alt_value, time_index):
    try:
        loaded_data = loadmat(filename)
    except:
        print("File not found. Please check your directory")

    # Uses key to extract our data of interest
    JB2008_dens = loaded_data['densityData']
    localSolarTimes_JB2008 = np.linspace(0,24,24)
    latitudes_JB2008 = np.linspace(-87.5,87.5,20)
    altitudes_JB2008 = np.linspace(100,800,36)
    nofAlt_JB2008 = altitudes_JB2008.shape[0] #nof stands for number of
    nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
    nofLat_JB2008 = latitudes_JB2008.shape[0]
    time_array_JB2008 = np.linspace(0,8759,5, dtype = int)
    JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,
                                                   nofAlt_JB2008,8760), order='F')

    hi = np.where(altitudes_JB2008==alt_value)
    density_data_JB2008 = JB2008_dens_reshaped[:,:,hi,time_index].squeeze()
    print (len(density_data_JB2008[0,:]))
    mean_densities_JB2008_lon = np.array([density_data_JB2008[i,:].mean() for i in range(nofLst_JB2008)])
    std_densities_JB2008_lon = np.array([density_data_JB2008[i,:].std() for i in range(nofLst_JB2008)])
    mean_densities_JB2008_lat = np.array([density_data_JB2008[:,i].mean() for i in range(nofLat_JB2008)])
    std_densities_JB2008_lat = np.array([density_data_JB2008[:,i].std() for i in range(nofLat_JB2008)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1.plot(localSolarTimes_JB2008, mean_densities_JB2008_lon+3*std_densities_JB2008_lon,'r--', label ="JB2008 mean along lon + 3 $\sigma$")
    ax1.plot(localSolarTimes_JB2008, mean_densities_JB2008_lon,'k-', linewidth = 2, label ="JB2008 mean")
    ax1.plot(localSolarTimes_JB2008, mean_densities_JB2008_lon-3*std_densities_JB2008_lon,'b--', label ="JB2008 mean - 3 $\sigma$")
    ax1.set_xlabel("Local solar times")
    ax1.set_ylabel("Density")
    ax1.set_ylim(0.1e-12, 3.e-11)
    ax1.legend()
    ax2.plot(latitudes_JB2008, mean_densities_JB2008_lat+3*std_densities_JB2008_lat,'r--', label ="JB2008 mean along lat + 3 $\sigma$")
    ax2.plot(latitudes_JB2008, mean_densities_JB2008_lat,'-', label ="JB2008 mean")
    ax2.plot(latitudes_JB2008, mean_densities_JB2008_lat-3*std_densities_JB2008_lat,'b--', label ="JB2008 mean along lat - 3 $\sigma$")
    ax2.set_xlabel("Latitudes")
    ax2.set_ylabel("Density")
    ax2.set_ylim(0.1e-12, 3.e-11)
    ax2.legend()
    plt.savefig("JB2008_"+str(time_index)+".png")
    return fig, (ax1, ax2)

def create_lat_lon_meanplots_tiegcm(filename, alt_value, time_index):
    import h5py
    loaded_data = h5py.File('../day_03/TIEGCM/2002_TIEGCM_density.mat')
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
    time_array_tiegcm = np.linspace(0,8759,5, dtype = int)

    # Each data correspond to the density at a point in 3D space.
    # We can recover the density field by reshaping the array.
    # For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
    tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')

    hi= np.where(altitudes_tiegcm==alt_value)
    dens_data_feb1_tiegcm = tiegcm_dens_reshaped[:,:,hi,time_index].squeeze()

    print (len(dens_data_feb1_tiegcm[0,:]))
    mean_densities_tiegcm_lon = np.array([dens_data_feb1_tiegcm[i,:].mean() for i in range(nofLst_tiegcm)])
    std_densities_tiegcm_lon = np.array([dens_data_feb1_tiegcm[i,:].std() for i in range(nofLst_tiegcm)])
    mean_densities_tiegcm_lat = np.array([dens_data_feb1_tiegcm[:,i].mean() for i in range(nofLat_tiegcm)])
    std_densities_tiegcm_lat = np.array([dens_data_feb1_tiegcm[:,i].std() for i in range(nofLat_tiegcm)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1.plot(localSolarTimes_tiegcm, mean_densities_tiegcm_lon+3*std_densities_tiegcm_lon,'r--', label ="TIE GCM mean along lon + 3 $\sigma$")
    ax1.plot(localSolarTimes_tiegcm, mean_densities_tiegcm_lon,'k-', linewidth = 2, label ="TIE GCM mean densities")
    ax1.plot(localSolarTimes_tiegcm, mean_densities_tiegcm_lon-3*std_densities_tiegcm_lon,'b--', label ="TIE GCM mean - 3 $\sigma$")
    ax1.set_xlabel("Local solar times")
    ax1.set_ylabel("Density")
    ax1.set_ylim(0.4e-11, 6.e-11)
    ax1.legend()
    ax2.plot(latitudes_tiegcm, mean_densities_tiegcm_lat+3*std_densities_tiegcm_lat,'r--', label ="TIE GCM mean along lat + 3 $\sigma$")
    ax2.plot(latitudes_tiegcm, mean_densities_tiegcm_lat,'-', label ="TIE GCM mean")
    ax2.plot(latitudes_tiegcm, mean_densities_tiegcm_lat-3*std_densities_tiegcm_lat,'b--', label ="TIE GCM mean along lat - 3 $\sigma$")
    ax2.set_xlabel("Latitudes")
    ax2.set_ylabel("Density")
    ax2.set_ylim(0.4e-11, 6.e-11)
    ax2.legend()
    plt.savefig("TIEGCM_"+str(time_index)+".png")
    return fig, (ax1, ax2)

time_index_start=2603
time_index_end=2638
print ((time_index_end - time_index_start))

time_array = np.linspace(time_index_start, time_index_end, (time_index_end - time_index_start+1))
print (time_array)
for time_index in time_array:
    # fig, (ax1, ax2) = create_lat_lon_meanplots_tiegcm('../day_03/TIEGCM/2002_TIEGCM_density.mat', 310, int(time_index))
    fig, (ax1, ax2) = create_lat_lon_meanplots_JB2008(dir_density_Jb2008, 400, int(time_index))
# alt_jb = 400
# time_index_start=2603
# time_index_end=2638
# hi = np.where(altitudes_JB2008==alt_jb)
# density_data_JB2008 = JB2008_dens_reshaped[:,:,hi,time_index_start]
# print (density_data_JB2008.squeeze().shape)
# mean_densities_JB2008_lon = [density_data_JB2008[i,:,:].mean() for i in range(nofLst_JB2008)]
# mean_densities_JB2008_lat = [density_data_JB2008[:,i,:].mean() for i in range(nofLat_JB2008)]
# # density_mean_JB2008_longitudes = np.mean(density_data_JB2008, axis=(1,2))
# print (mean_densities_JB2008_lon)


# ax[0].plot(localSolarTimes_JB2008, mean_densities_JB2008_lon,'-.', label ="JB2008 mean")
# ax[0].set_xlabel("Local solar times")
# ax[0].set_ylabel("Density")
# ax[1].plot(latitudes_JB2008, mean_densities_JB2008_lat,'-.', label ="JB2008 mean")
# ax[1].set_xlabel("Latitudes")
# ax[1].set_ylabel("Density")

# ax.set_title("Mean densities VS LST")

# plt.show()
