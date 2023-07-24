import matplotlib.pyplot as plt
from swmfpy.web import get_omni_data
from datetime import datetime
from matplotlib.dates import DateFormatter




def plot_al_data(start_time, end_time):
    """
    This function uses the swmfpy package and plots al data for the given date.
    Inputs:
        start_time: datetime object to start the plots
        end_time: datetime object that is the end time.

    It plots the figure with dates on x axis and al data on y axis.
    """
    data=get_omni_data(start_time, end_time)

    time_axis = data['times']
    al_data = data['al']

    plt.plot(time_axis, al_data)

    plt.xlabel("Dates")
    plt.ylabel('al [nT]')
    axis = plt.gca()
    axis.xaxis.set_major_formatter(DateFormatter('%d-%H'))

    plt.show()


start_time = datetime(1996, 5, 27)
end_time = datetime(1996, 5, 28)

plot_al_data(start_time, end_time)
