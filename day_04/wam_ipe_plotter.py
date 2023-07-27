__author__ = 'Tinatin Baratashvili'
__email__ = 'tinatin.baratashvili@kuleuven.be'

import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def save_figure(infilename, variable_name):
    """
        This function generates the dataset from the given filename
        and creates, and saves a figure for the given variable_name.
    """
    dataset = nc.Dataset(infilename)
    fig, ax = plot_ipe(dataset, variable_name)
    # outfilename =  infilename +'_'+variable_name+ '.png'
    outfilename =  infilename +'.png'
    fig.savefig(outfilename)



def plot_ipe(dataset, variable_name, figsize=(12,6)):
    """
        This function generates the colorplot of the TEC for the given dataset.
        Inputs:
            dataset (netCDF4 format file): dataset for a particular time (5 minute cadence)
            figsize : fixed figsize to add the plot

        The code gets the longitudes and latitudes and creates a pcolormesh of TEC.
        It also creates the colorbar. Sets the x and y axis on ax and sets the title.
        Returns:
            returns axis with pcolormesh of the TEC on Lon vs Lat grid.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if (variable_name in dataset.variables.keys()):

        lats = dataset['lat'][:]
        lons = dataset['lon'][:]
        variable_to_plot = dataset[variable_name][:]
        # These lines below are to include the colormap
        levels = MaxNLocator(nbins=(max(variable_to_plot.shape))).tick_values(variable_to_plot.min(), variable_to_plot.max())
        cmap = plt.colormaps['viridis']
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im = ax.pcolormesh(lons, lats, variable_to_plot, cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(variable_name + ' ['+ dataset[variable_name].units + ']')
        ax.set_xlabel("Longitudes [degrees]")
        ax.set_ylabel("Latitudes [degrees]")
        ax.set_title(variable_name)
    return fig, ax



# This is the main body of the code
if __name__ == '__main__':
    infilename = 'wfs.t06z.ipe05.20230725_050000.nc'
    variable_name = 'HmF2'
    # save_figure(infilename, variable_name)


    # plt.show()
