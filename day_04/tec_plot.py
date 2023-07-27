from wam_ipe_plotter import plot_ipe, save_figure
import sys


print (sys.argv)
# filename = 'wfs.t06z.ipe05.20230725_050000.nc'
# variable_name = 'HmF2'
command_arguments = sys.argv[1:]
filenames = command_arguments[:-1]
variable_name = command_arguments[-1]

for filename in filenames:
    save_figure(filename,variable_name)
